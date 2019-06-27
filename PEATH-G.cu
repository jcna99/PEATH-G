#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <thrust/sort.h>


#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>
#include <string.h>


//#define __DEBUG__	// to create files for debugging
//#define __DTIME__		// print execution time

#define __PRINT_RESULT__	// to create output files

////////////////////////////////

#define CU_BSIZE	32 	// cuda block dimension - x

#define CU_DEVICE	0	// cuda device number runnig the program

////////////////////////

#define G_ITER		50		// GA iteration 
//int G_ITER;

#define TOG_ITER	10		// toggling iteration

#define DEFAULT_PHASING_ITER 50

/////////////////////////////////


#define EPSILON 0.00000000001		// error threshold of computing floating point

#define POPSIZE 100			// population size in GA
#define OFFSIZE 50			// offsping size in GA

////////////////////////////////


using namespace std;

typedef unsigned int uint;


typedef struct {			// subfragments
	uint start_pos = 0;
	uint end_pos = 0;		
	uint Moffset = 0;		// offset (inside block) of starting position in Allele & Qual Matrix Data
}SubFragType;

typedef struct {			// fragment(read)
	uint start_pos = 0;		// start_pos in block
	uint end_pos = 0;		// end_pos in block
	uint subFrag0 = 0;		// start_subfrag in block
	uint num_subFrag = 0;
}FragType;

typedef struct {			// block
	uint start_pos = 0;
	uint start_subfrag = 0;			// Index of the 0th subframent in SubFragments vector
	uint start_frag = 0;
	uint num_Frag = 0;
	uint length = 0;
	uint Moffset = 0;				// offset of starting position in Allele & Qual Matrix Data
}BlockType;

typedef struct {			// fragment inforamtion for each position
	uint start_frag = 0;
	uint end_frag = 0;
}FragsForPosType;

typedef struct {			// Weighted MEC (distance) for each fragment
	double D_hf = -1.0;			// It is used in toggling stage
	double D_hfstar = -1.0;		// in order to speed up calculation of weighted MEC
}DFragType;

typedef struct {				// individual in GA
	double sumD; 			// Weighted MEC (distance)
	uint stx_pos;			// starting position in Pop_seq, not unsed in Toggling
}IndvDType;

typedef struct {			// Haplotype sequeucne
	char *seq = NULL, *seq2 = NULL;
	double sumD = -1.0;			// Weighted MEC (distance)
}HaploType;

typedef double QualType;		// Quality data type

/////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////


#ifdef __INTELLISENSE__
void __syncthreads();
#endif

void cudaCheckSync(const char func[])
{
	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "[%s] Kernel launch failed: %s\n", func, cudaGetErrorString(cudaStatus));
		exit(1);
	}
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "[%s] cudaDeviceSynchronize returned error code %d after launching Kernel!\n", func, cudaStatus);
		exit(1);
	}
}

////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////

// compare function for sorting
__device__ __host__
bool compare_sumD_val(IndvDType left, IndvDType right)
{
	return left.sumD < right.sumD;
}

bool compare_frag_pos(FragType left, FragType right)
{
	return left.start_pos < right.start_pos;
}

////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////

//
// calculating non-weighted MEC (using one sequence)
//
uint calc_MEC1(const char *haplo_seq, const BlockType& block,
	const char *AlleleData, const SubFragType *SubFragments, const FragType *Fragments)
{
	uint sumD = 0;

	AlleleData += block.Moffset;
	Fragments += block.start_frag;
	SubFragments += block.start_subfrag;

	for (uint i = 0; i < block.num_Frag; i++) {	// for each fragment

		FragType Fragi = Fragments[i];
		const SubFragType *SubFrag = SubFragments + Fragi.subFrag0; // 0th subfragments in the fragment[i]
		uint D_hf = 0, D_hstarf = 0;

		for (uint j = 0; j < Fragi.num_subFrag; j++) {	// for each subfragment

			const SubFragType SubFragj = SubFrag[j];
			const char *subfrag_str = AlleleData + SubFragj.Moffset;

			uint pos_sf = 0;	// pos in subfragment
			uint pos_blk = SubFragj.start_pos;		// pos in block (haplotype)
			for (; pos_blk <= SubFragj.end_pos; pos_blk++, pos_sf++) {	// for each position

				if (haplo_seq[pos_blk] == '-' || subfrag_str[pos_sf] == '-')
					continue;

				if (haplo_seq[pos_blk] != subfrag_str[pos_sf])
					++D_hf;
				else
					++D_hstarf;

			}  // end_for k (each position)
		}  // end_for j (each subfragment)

		if (D_hf < D_hstarf)
			sumD += D_hf;
		else
			sumD += D_hstarf;

	}// end_for i (each fragment)

	return sumD;
}

//
// calculating w-MEC values (normal version)
// : if 'update' is true, DFrag array and haplo (except for seq/seq_star) are updated
//
// this function is run by one thread
__device__
double calc_sumD(double *haplo_sumD, const char *haplo_seq, DFragType *DFrag, const uint num_Frag_blk, const bool update,
	const char *AlleleData, const QualType *QualData, const SubFragType *SubFragments, const FragType *Fragments)
{
	double sumD = 0.0;

	for (uint i = 0; i < num_Frag_blk; i++) {	// for each fragment

		FragType Fragi = Fragments[i];
		const SubFragType *SubFrag = SubFragments + Fragi.subFrag0; // 0th subfragments in the fragment[i]
		double D_hf = 0.0, D_hfstar = 0.0;

		for (uint j = 0; j < Fragi.num_subFrag; j++) {	// for each subfragment

			const SubFragType SubFragj = SubFrag[j];

			const char *subfrag_str = AlleleData + SubFragj.Moffset;
			const QualType *subfrag_qual = QualData + SubFragj.Moffset;

			uint pos_sf = 0;	// pos in subfragment
			uint pos_blk = SubFragj.start_pos;		// pos in block (haplotype)
			for (; pos_blk <= SubFragj.end_pos; pos_blk++, pos_sf++) {	// for each position

				QualType q_j = subfrag_qual[pos_sf];
				QualType q_j_star = 1 - q_j;

				// calculating distance for a position
				if (haplo_seq[pos_blk] != subfrag_str[pos_sf]) {
					D_hf += q_j_star;
					D_hfstar += q_j;
				}
				else {
					D_hf += q_j;
					D_hfstar += q_j_star;
				}

			} // end_for (each position)

		} // end_for j (each subfragment)

		if (D_hf < D_hfstar)	// select min( D_h, D_h*)
			sumD += D_hf;
		else
			sumD += D_hfstar;

		if (update) {					// *** if update is tree  ***
			DFrag[i].D_hf = D_hf;			// *** the calculated values are stored in DFrag ***
			DFrag[i].D_hfstar = D_hfstar;
		}
	}// end_for i (each fragment)

	*haplo_sumD = sumD;

	return sumD;
}

//
// calculating w-MEC values (position version) used only in range_switch_procedure( )
// : calculating weighted MEC with assumption that seq[0..i] is toggled
// : recalculating the distance of only fragments covering at position pos
//
// this function is run by one thread
__device__ __host__
double calc_sumD_range_tog(double *haplo_sumD, const char *haplo_seq, DFragType *DFrag, const BlockType block, const uint tog_pos, // const bool update,
	const char *AlleleData, const QualType *QualData, const SubFragType *SubFragments, const FragType *Fragments,
	const FragsForPosType *FragsForPos)
{
	double sumD = *haplo_sumD;

	const uint startFrag = FragsForPos[tog_pos].start_frag ;	// first fragment located at pos
	const uint endFrag = FragsForPos[tog_pos].end_frag ;		// last fragment located at pos

	for (uint i = startFrag; i <= endFrag; i++) {	// for each fragment covering pos

		const FragType Fragi = Fragments[i];

		// 1. check if the fragment includes tog_pos
		if (Fragi.end_pos < tog_pos)	// |---Fragi---|  < tog_pos
			continue;						// skip

		// 2. substract sumD of the current fragments
		double D_hf = DFrag[i].D_hf, D_hfstar = DFrag[i].D_hfstar;  // previous DFrag values

		if (D_hf < D_hfstar) sumD -= D_hf;		// substract the previous DFrag value
		else sumD -= D_hfstar;

		D_hf = D_hfstar = 0.0;

		// 3. compute sumD of the current fragments for toggled haplotype sequence

		const SubFragType *SubFrag = SubFragments + Fragi.subFrag0; // 0th subfragments in the fragment[i]

		for (uint j = 0; j < Fragi.num_subFrag; j++) {	// for each subfragment

			const SubFragType SubFragj = SubFrag[j];

			const char *subfrag_str = AlleleData + SubFragj.Moffset;
			const QualType *subfrag_qual = QualData + SubFragj.Moffset;

			// for toggled positions
			uint pos_sf = 0;	// pos in subfragment
			uint pos_blk = SubFragj.start_pos;		// pos in block (haplotype)
			for (; pos_blk <= SubFragj.end_pos; pos_blk++, pos_sf++) {	// for each position

				if (pos_blk > tog_pos)	break;		// if  pos_k is not a toggled position

				QualType q_j = subfrag_qual[pos_sf];
				QualType q_j_star = 1 - q_j;

				// calculating distance for a position
				// computing under the assumption that the haplo_seq[pos_blk] is toggled. So, != -> ==	
				if (haplo_seq[pos_blk] == subfrag_str[pos_sf]) {	// 
					D_hf += q_j_star;
					D_hfstar += q_j;
				}
				else {
					D_hf += q_j;
					D_hfstar += q_j_star;
				}

			} // end_for (each position)

			  // for not toggled positions
			for (; pos_blk <= SubFragj.end_pos; pos_blk++, pos_sf++) {	// for each position

				QualType q_j = subfrag_qual[pos_sf];
				QualType q_j_star = 1 - q_j;

				// calculating distance for a position
				if (haplo_seq[pos_blk] != subfrag_str[pos_sf]) {
					D_hf += q_j_star;
					D_hfstar += q_j;
				}
				else {
					D_hf += q_j;
					D_hfstar += q_j_star;
				}
			}

		} // end_for j (each subfragment)


		if (D_hf < D_hfstar)		// select min( D_h, D_h*)
			sumD += D_hf;			// add the new DFrag value
		else
			sumD += D_hfstar;

		//if (update) {				// *** if update is true  ***
		//	DFrag[i].D_hf = D_hf;			// *** the calculated values are stored in DFrag ***
		//	DFrag[i].D_hfstar = D_hfstar;
		//}

	}// end_for i (each fragment)

	//if (update) 					// *** if update is true  ***
	//	*haplo_sumD = sumD;

	return sumD;
}

//
// calculating w-MEC values (position version) used only in single_switch_procedure( )
// : calculating weighted MEC with assumption that seq[i] is toggled
// : recalculating the distance of only fragments covering at position pos
//
// this function is run by ONE thread
__device__ __host__
double calc_sumD_single_tog(double *haplo_sumD, const char *haplo_seq, DFragType *DFrag, const BlockType block, const uint tog_pos, // const bool update,
	const char *AlleleData, const QualType *QualData, const SubFragType *SubFragments, const FragType *Fragments,
	const FragsForPosType *FragsForPos)
{
	double sumD = *haplo_sumD;

	const uint startFrag = FragsForPos[tog_pos].start_frag;	// first fragment located at pos
	const uint endFrag = FragsForPos[tog_pos].end_frag ;		// last fragment located at pos


	for (uint i = startFrag; i <= endFrag; i++) {	// for each fragment covering pos

		const FragType Fragi = Fragments[i];

		// 1. check if the fragment includes tog_pos
		if (Fragi.end_pos < tog_pos)	// |---Fragi---|  < tog_pos
			continue;						// skip

		// 2. finding subfragment located at pos
		const SubFragType *SubFrag = SubFragments + Fragi.subFrag0; // 0th subfragments in the fragment[i]

		uint j = 0;
		while (j < Fragi.num_subFrag && SubFrag[j].end_pos  < tog_pos )
			++j;								// skip subfragments before tog_pos

		const SubFragType SubFragj = SubFrag[j];

		if ( tog_pos < SubFragj.start_pos)  // |--- subfrag_j-1---| < tog_pos  <  |-- subfrag_j---| 
			continue;							// no subfragment is located at pos


		// 3. update sumD : subFragment[j] is located at pos
		double D_hf = DFrag[i].D_hf, D_hfstar = DFrag[i].D_hfstar;  // previous DFrag values

		if (D_hf < D_hfstar) sumD -= D_hf;		// substract the previous DFrag value
		else sumD -= D_hfstar;

		const char *subfrag_str = AlleleData + SubFragj.Moffset;
		const QualType *subfrag_qual = QualData + SubFragj.Moffset;

		uint ps_sf = tog_pos - SubFragj.start_pos;
		QualType q_j = subfrag_qual[ps_sf];
		QualType q_j_star = 1 - q_j;

		// computing under the assumption that the the bit is toggled. So, != -> ==	
		if (haplo_seq[tog_pos] == subfrag_str[ps_sf]) {
			D_hf += q_j_star - q_j;			// + new value - previous value
			D_hfstar += q_j - q_j_star;
		}
		else {
			D_hf += q_j - q_j_star;			// + new value - previous value
			D_hfstar += q_j_star - q_j;
		}

		if (D_hf < D_hfstar)		// select min( D_h, D_h*)
			sumD += D_hf;			// add the new DFrag value
		else
			sumD += D_hfstar;

		//if (update) {				// *** if update is true  ***
		//	DFrag[i].D_hf = D_hf;			// *** the calculated values are stored in DFrag ***
		//	DFrag[i].D_hfstar = D_hfstar;
		//}

	}// end_for i (each fragment)

	//if (update) 					// *** if update is true  ***
	//	*haplo_sumD = sumD;

	return sumD;
}


////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////


__device__ void GA_1stGen(
	uint global_seed, curandState *b_randstate,
	IndvDType *Population, char *Pop_seq, const uint BlkLen)
{
		__shared__	uint local_Seed;		// seed used in phasing_instance

		if (threadIdx.x == 0) {			// init blk_Seed from global_seed
			curand_init(global_seed, blockIdx.x, 0, b_randstate);
			local_Seed = curand(b_randstate);
		}

		__syncthreads();


		curandState localState;
		curand_init(local_Seed, threadIdx.x, 0, &localState);

		for (uint i = threadIdx.x; i < POPSIZE; i += blockDim.x) 	// relative indv_id in cur pop
			Population[i].stx_pos = i * BlkLen;
		__syncthreads();

		for (uint indv = 0; indv < POPSIZE; ++indv) {
			char *indv_seq = Pop_seq + Population[indv].stx_pos;
				
			for (uint i = threadIdx.x; i < BlkLen; i += blockDim.x) {
				if (curand_uniform(&localState) < 0.5)
					indv_seq[i] = '1';
				else
					indv_seq[i] = '0';
			}
		}
}

__device__ void GA_nextGen(
	IndvDType *Population, char *Pop_seq, curandState *b_randstate, const uint BlkLen)
{
	__shared__	uint localSeed;

	if (threadIdx.x == 0)
		localSeed = curand(b_randstate);
	__syncthreads();

	curandState localState;
	curand_init(localSeed, threadIdx.x, 0, &localState);


	for (uint i = threadIdx.x; i < BlkLen; i += blockDim.x) {	// position inside of haplotype block
		uint cnt = 0;
		for (uint j = 0; j < POPSIZE - OFFSIZE; ++j) {				// compute cnt with selected individuals
			char *indv_seq = Pop_seq + Population[j].stx_pos;
			cnt += indv_seq[i] - '0';
		}

		double prob = (double)cnt / (POPSIZE - OFFSIZE);

		for (uint indv = (POPSIZE - OFFSIZE); indv < POPSIZE; ++indv) {		// generate new offsprings
			char *indv_seq = Pop_seq + Population[indv].stx_pos;

			if (curand_uniform(&localState) < prob)
				indv_seq[i] = '1';
			else
				indv_seq[i] = '0';
		}
	}
}


__device__ void GA_pop_eval(
	IndvDType *Population, char *Pop_seq,
	const char *AlleleData, const QualType *QualData, const SubFragType *SubFragments, const FragType *Fragments,
	const uint num_Frag_blk, const uint num_indv_per_ph)
{
	// __golbal__ memory version
	Population += (POPSIZE - num_indv_per_ph);

	for (uint i = threadIdx.x; i < num_indv_per_ph; i += blockDim.x) 	// relative indv_id in cur pop
		calc_sumD(&Population[i].sumD, Pop_seq + Population[i].stx_pos, NULL, num_Frag_blk, false,
			AlleleData, QualData, SubFragments, Fragments);

	//// __shared__ memory version
	//uint loop = (num_indv_per_ph - 1) / blockDim.x + 1;
	//for (uint i = 0; i < loop; ++i) 	// relative indv_id in cur pop
	//{
	//	uint stx_indv_idx = POPSIZE - num_indv_per_ph + blockDim.x * i;
	//	uint num_indv = (POPSIZE - stx_indv_idx < blockDim.x) ? POPSIZE - stx_indv_idx : blockDim.x;
	//	calc_sumD_multi(stx_indv_idx, num_indv, Population, Pop_seq, NULL, num_Frag_blk, false,
	//		AlleleData, QualData, SubFragments, Fragments);
	//}

}


//gathering the best haplotype in the population of each phasing iteration
__device__ void gather_pop0(
	double *Tog_sumD, char *Tog_seq,
	const IndvDType *Population, const char *Pop_seq, const uint BlkLen)
{

	const char *pop0_seq = Pop_seq + Population[0].stx_pos;

	for (uint i = threadIdx.x; i < BlkLen; i += blockDim.x)
		Tog_seq[i] = pop0_seq[i];

	if (threadIdx.x == 0)
		*Tog_sumD = Population[0].sumD;			// copy sumD

}


//
// Genetic algorithm
//
__device__ void GA(
	const uint seed, uint g_iter, BlockType block,
	double *sh_haplo_sumD, char *sh_haplo_seq, char *d_Pop_seq,
	const char *b_AlleleData, const QualType *b_QualData, const SubFragType *b_SubFragments, const FragType *b_Fragments)
{
	// gridDim = { phasing_iter, 1, 1};		// blockIdx.x = ph_id;
	// blockDim = {CU_BSIZE, 1, 1};

	__shared__ IndvDType sh_Population[POPSIZE];
	__shared__ curandState_t sh_b_randstate;

	
	GA_1stGen(		// faster 
		seed, &sh_b_randstate,
		sh_Population, d_Pop_seq, block.length);

	__syncthreads();


	GA_pop_eval(			// slower (a little)
		sh_Population, d_Pop_seq,
		b_AlleleData, b_QualData, b_SubFragments, b_Fragments, block.num_Frag, POPSIZE);

	__syncthreads();


	if (threadIdx.x == 0)
		thrust::sort(sh_Population, sh_Population + POPSIZE, compare_sumD_val);

	__syncthreads();


	//	uint g_iter = G_ITER;		// generation number of GA
	while (g_iter--) {

		GA_nextGen(				// faster (very much X6)
			sh_Population, d_Pop_seq, &sh_b_randstate, block.length);

		__syncthreads();


		GA_pop_eval(
			sh_Population, d_Pop_seq,
			b_AlleleData, b_QualData, b_SubFragments, b_Fragments, block.num_Frag, OFFSIZE);

		__syncthreads();


		if (threadIdx.x == 0)
			thrust::sort(sh_Population, sh_Population + POPSIZE, compare_sumD_val);

		__syncthreads();

	} // end_while (g_iter--)


	gather_pop0(
		sh_haplo_sumD, sh_haplo_seq, sh_Population, d_Pop_seq, block.length);

	__syncthreads();


}

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

// 
// Toggling for range switch
// return 1 if better soultion is found
//
// this function is run by MULTI thread
__device__
int range_switch_toggling(
	double *haplo_sumD, char *haplo_seq, DFragType *DFrag, const BlockType &block,
	double sh_bestSum[], int sh_bestPos[],
	const char *b_AlleleData, const QualType *b_QualData, const SubFragType *b_SubFragments, const FragType *b_Fragments,
	const FragsForPosType *b_FragsForPos)
{
	uint thIdx = threadIdx.x;	// index for shared memory
	double bestSum, tmpSum;
	int bestPos, rval = 0;
	bool isImp;

	do {
		// 1. compute sumD & DFrag;
		isImp = false;
		if (thIdx == 0) {
			calc_sumD(haplo_sumD, haplo_seq, DFrag, block.num_Frag, true,
				b_AlleleData, b_QualData, b_SubFragments, b_Fragments);	// update DFrag array for the current haplotype sequence
		}

		__syncthreads();

		// 2. compute sumD in each toggled position계산 해서 최소값 찾기 (paralled by thread)
		bestSum = *haplo_sumD;
		bestPos = -1;

		for (uint i = thIdx; i < block.length; i += blockDim.x) {

			tmpSum = calc_sumD_range_tog(haplo_sumD, haplo_seq, DFrag, block, i,
				b_AlleleData, b_QualData, b_SubFragments, b_Fragments, b_FragsForPos); // DFrag array & haplo are NOT updated

			if (bestSum > tmpSum) {
				bestSum = tmpSum;
				bestPos = i;
			}
		}

		sh_bestSum[thIdx] = bestSum;
		sh_bestPos[thIdx] = bestPos;

		__syncthreads();

		// 3. find the best sumD in all threads
		for (uint i = blockDim.x / 2; i != 0; i /= 2) {
			if (thIdx < i)
				if (sh_bestSum[thIdx] > sh_bestSum[thIdx + i]) {
					sh_bestSum[thIdx] = sh_bestSum[thIdx + i];
					sh_bestPos[thIdx] = sh_bestPos[thIdx + i];
				}

			__syncthreads();
		}

		// 4. update haplotype with new best haplo
		bestSum = sh_bestSum[0];
		bestPos = sh_bestPos[0];
		if (*haplo_sumD - bestSum > EPSILON) {
			isImp = true;
			rval = 1;
			for (int i = thIdx; i <= bestPos; i += blockDim.x)
				haplo_seq[i] = (haplo_seq[i] == '0') ? '1' : '0';	// toggling
			// haplo_sumD is upated in calc_sumD() of the next while loop
		}

		__syncthreads();

	} while (isImp);		// same values in all threads

	return rval;			// the same values in all threads
}

// 
// Toggling for range switch
// return 1 if better soultion is found
//
// this function is run by MULTI thread
__device__
int single_switch_toggling(
	double *haplo_sumD, char *haplo_seq, DFragType *DFrag, const BlockType &block,
	double sh_bestSum[], int sh_bestPos[],
	const char *b_AlleleData, const QualType *b_QualData, const SubFragType *b_SubFragments, const FragType *b_Fragments,
	const FragsForPosType *b_FragsForPos)
{
	uint thIdx = threadIdx.x;	// index for shared memory
	double bestSum, tmpSum;
	int bestPos, rval = 0;
	bool isImp;

	do {
		// 1. compute sumD & DFrag;
		isImp = false;
		if (thIdx == 0) {
			calc_sumD(haplo_sumD, haplo_seq, DFrag, block.num_Frag, true,
				b_AlleleData, b_QualData, b_SubFragments, b_Fragments);	// update DFrag array for the current haplotype sequence
		}

		__syncthreads();

		// 2. compute sumD in each toggled position계산 해서 최소값 찾기 (paralled by thread)
		bestSum = *haplo_sumD;
		bestPos = -1;

		for (uint i = thIdx; i < block.length; i += blockDim.x) {

			tmpSum = calc_sumD_single_tog(haplo_sumD, haplo_seq, DFrag, block, i, 
				b_AlleleData, b_QualData, b_SubFragments, b_Fragments, b_FragsForPos); // DFrag array & haplo are NOT updated

			if (bestSum > tmpSum) {
				bestSum = tmpSum;
				bestPos = i;
			}
		}

		sh_bestSum[thIdx] = bestSum;
		sh_bestPos[thIdx] = bestPos;

		__syncthreads();

		// 3. find the best sumD in all threads
		for (uint i = blockDim.x / 2; i != 0; i /= 2) {
			if (thIdx < i)
				if (sh_bestSum[thIdx] > sh_bestSum[thIdx + i]) {
					sh_bestSum[thIdx] = sh_bestSum[thIdx + i];
					sh_bestPos[thIdx] = sh_bestPos[thIdx + i];
				}
			__syncthreads();
		}

		// 4. update haplotype with new best haplo
		bestSum = sh_bestSum[0];
		bestPos = sh_bestPos[0];
		if (*haplo_sumD - bestSum > EPSILON) {
			isImp = true;
			rval = 1;
			if (thIdx == 0)
				haplo_seq[bestPos] = (haplo_seq[bestPos] == '0') ? '1' : '0';	// toggling
		}
		__syncthreads();

	} while (isImp);		// same values in all threads

	return rval;			// the same values in all threads
}

__device__ void Toggling(
	BlockType block, double *sh_haplo_sumD, char *sh_haplo_seq, DFragType *sh_DFrag,
	const char *b_AlleleData, const QualType *b_QualData, const SubFragType *b_SubFragments, const FragType *b_Fragments,
	const FragsForPosType *b_FragsForPos)
{
	// gridDim = { phasing_iter, 1, 1};		// blockIdx.x = ph_id;
	// blockDim = {CU_BSIZE, 1, 1};

	__shared__ double sh_bestSum[CU_BSIZE];	// best sumD for each thread
	__shared__ int sh_bestPos[CU_BSIZE];		// position with best sumD for each thread

	uint tog_iter = TOG_ITER;			// toggling iteration number
	while (tog_iter--) {
		uint imp = 0;

		// Step 1: Range Switch Toggling
		imp += range_switch_toggling(	// faster (quite)
			sh_haplo_sumD, sh_haplo_seq, sh_DFrag, block,
			sh_bestSum, sh_bestPos,
			b_AlleleData, b_QualData, b_SubFragments, b_Fragments, b_FragsForPos);

		__syncthreads();

		// Step 2: Single Switch Toggling
		imp += single_switch_toggling(
			sh_haplo_sumD, sh_haplo_seq, sh_DFrag, block,
			sh_bestSum, sh_bestPos,
			b_AlleleData, b_QualData, b_SubFragments, b_Fragments, b_FragsForPos);

		__syncthreads();

		if (!imp) break;				// if not improved, stop
	}
}


////////////////////////////////////////////////////////////////////////////
// This function processes each phasing-instance for a phasing-uint in parallel
// 
__global__ void Phasing_instance(
	const uint seed, uint g_iter, BlockType block,
	double *b_Tog_sumD, char *d_Tog_seq, char *d_Pop_seq,
	const char *b_AlleleData, const QualType *b_QualData, const SubFragType *b_SubFragments, const FragType *b_Fragments,
	const FragsForPosType *b_FragsForPos,
	const uint NumPos, const uint phasing_iter)
{
	// gridDim = { phasing_iter, 1, 1};		// blockIdx.x = ph_id;
	// blockDim = {CU_BSIZE, 1, 1};

	b_Tog_sumD += blockIdx.x;
	d_Tog_seq += blockIdx.x * NumPos + block.start_pos;
	d_Pop_seq += (blockIdx.x * NumPos + block.start_pos) * POPSIZE;;


	// Allocating shared memory 

	__shared__ double sh_haplo_sumD;		// static

	extern __shared__ char s[];				// dynamic

	int sh_mem_size = 0;				// in bytes
	char *sh_haplo_seq = (char *)s;			// size : block.length
	sh_mem_size += block.length * sizeof(char);

	sh_mem_size = ((sh_mem_size - 1) / sizeof(DFragType) + 1) * sizeof(DFragType);
	DFragType *sh_DFrag = (DFragType *)(s + sh_mem_size);
	sh_mem_size += block.num_Frag * sizeof(DFragType);		 // size : Nfrag_blk

	// Genetic algorithm
	GA(seed, g_iter, block,
		&sh_haplo_sumD, sh_haplo_seq, d_Pop_seq,
		b_AlleleData, b_QualData, b_SubFragments, b_Fragments);

	// Toggling heuristic
	Toggling( block, &sh_haplo_sumD, sh_haplo_seq, sh_DFrag,
		b_AlleleData, b_QualData, b_SubFragments, b_Fragments, b_FragsForPos);

	// copy data from shared memory to global memory
	if (threadIdx.x == 0)
		*b_Tog_sumD = sh_haplo_sumD;
	for (uint i = threadIdx.x; i < block.length; i += blockDim.x)
		d_Tog_seq[i] = sh_haplo_seq[i];

}


// Fing the best among phasing-instances for a phasing-unit
__device__ void Find_Best_Haplo_instance(
	double *d_Best_sumD, char *d_Best_seq, double *b_Tog_sumD, char *d_Tog_seq,
	const BlockType block,
	const uint NumPos, const uint phasing_iter)
{
	uint thIdx = threadIdx.x;	// index for shared memory
	uint blk_id = blockIdx.x;

	__shared__ double min_sumD[CU_BSIZE];		// blockDim.x == CU_BSIZE;
	__shared__ uint min_phid[CU_BSIZE];

	if (thIdx < phasing_iter) {
		min_sumD[thIdx] = b_Tog_sumD[thIdx];
		min_phid[thIdx] = thIdx;
	}
	else {
		min_sumD[thIdx] = b_Tog_sumD[0];			// dummy data
		min_phid[thIdx] = 0;
	}
	__syncthreads();

	// find the best sumD in all threads
	for (uint i = thIdx + blockDim.x; i < phasing_iter; i += blockDim.x)
		if (min_sumD[thIdx] > b_Tog_sumD[i]) {
			min_sumD[thIdx] = b_Tog_sumD[i];
			min_phid[thIdx] = i;
		}
	__syncthreads();

	for (uint i = blockDim.x / 2; i != 0; i /= 2) {		// reduction
		if (thIdx < i)
			if (min_sumD[thIdx] > min_sumD[thIdx + i]) {
				min_sumD[thIdx] = min_sumD[thIdx + i];
				min_phid[thIdx] = min_phid[thIdx + i];
			}
	}
	__syncthreads();


	if (thIdx == 0)
		d_Best_sumD[blk_id] = min_sumD[0];	// copy sumD

	d_Best_seq += block.start_pos;
	d_Tog_seq += NumPos * min_phid[0] + block.start_pos;
	for (int i = thIdx; i < block.length; i += blockDim.x)
		d_Best_seq[i] = d_Tog_seq[i];		// copy sequence
	__syncthreads();

}


////////////////////////////////////////////////////////////////////////////
// This function runs phasing each phasing-unit in parallel
// Phasing-instances of each phasing-unit are processed by a grid of size phasing_iter (dynamic parallelism)
__global__ void Phasing_unit(
	uint seed, uint g_iter, const BlockType *d_blocks,
	double *d_Best_sumD, char *d_Best_seq, double *d_Tog_sumD, char *d_Tog_seq, char *d_Pop_seq,
	const char *d_AlleleData, const QualType *d_QualData, const SubFragType *d_SubFragments, const FragType *d_Fragments,
	const FragsForPosType *d_FragsForPos,
	const uint NumPos, const uint NumFrag, const uint NumBlk, const uint phasing_iter)
{
	// gridDim = { NumBlk, 1, 1};  // blockIdx.x = blk_id
	// blockDim = { 1, 1, 1};

	BlockType block = d_blocks[blockIdx.x];

	d_Fragments += block.start_frag;
	d_SubFragments += block.start_subfrag;
	d_AlleleData += block.Moffset;
	d_QualData += block.Moffset;
	d_FragsForPos += block.start_pos;

	d_Tog_sumD += blockIdx.x * phasing_iter;

	// dynamic shared memory size
	uint sh_mem_size = 0;		// in bytes

	sh_mem_size += block.length * sizeof(char);	// for sh_haplo_seq[]

	sh_mem_size = ((sh_mem_size - 1) / sizeof(DFragType) + 1) * sizeof(DFragType);
	sh_mem_size += (block.num_Frag) * sizeof(DFragType);	// for sh_DFrag[]

	if (threadIdx.x == 0)
		Phasing_instance << < phasing_iter, CU_BSIZE, sh_mem_size >> > (	// dynamic parallelism
			seed, g_iter, block,
			d_Tog_sumD, d_Tog_seq, d_Pop_seq,
			d_AlleleData, d_QualData, d_SubFragments, d_Fragments, d_FragsForPos,
			NumPos, phasing_iter);

	cudaDeviceSynchronize();

	Find_Best_Haplo_instance(
		d_Best_sumD, d_Best_seq,
		d_Tog_sumD, d_Tog_seq, block,
		NumPos, phasing_iter);


}



////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////

//
// haplotype phasing procedure for a block
//

void haplotype_phasing(double *h_BestHaplo_sumD, char *h_BestHaplo_seq,
	const char *d_AlleleData, const QualType *d_QualData,
	const SubFragType *d_SubFragments, const FragType *d_Fragments,	const BlockType *d_blocks,
	const FragsForPosType *d_FragsForPos,
	const uint NumPos, const uint NumFrag, const uint NumBlk, const uint MaxBlkLen, const uint phasing_iter)
{
	cudaError_t  cudaStatus;

	cudaSetDevice(CU_DEVICE);

	///////////////////////////////////////////
	// Step 1: Allocating global memory for temporary space used in device
	///////////////////////////////////////////
#ifdef __DTIME__
	std::chrono::system_clock::time_point time0 = std::chrono::system_clock::now();
	printf("Allocation 2: ");
#endif
	double *d_Tog_sumD;				// haplotype sumD used in toggling
	cudaStatus = cudaMalloc((void**)&d_Tog_sumD, phasing_iter * NumBlk * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "[Haplo_sumD]cudaMalloc failed!\n");
	}

	char *d_Tog_seq;					// haplotype sequence used in toggling
	cudaStatus = cudaMalloc((void**)&d_Tog_seq, phasing_iter * NumPos * sizeof(char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "[Haplo_seq]cudaMalloc failed!\n");
	}

	char *d_Pop_seq;					// haplotype sequence in population of GA
	cudaStatus = cudaMalloc((void**)&d_Pop_seq, phasing_iter * POPSIZE * NumPos * sizeof(char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "[Pop_seq]cudaMalloc failed!");
	}

	double *d_Best_sumD;
	cudaStatus = cudaMalloc((void**)&d_Best_sumD, NumBlk * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "[Haplo_sumD]cudaMalloc failed!");
	}

	char *d_Best_seq;
	cudaStatus = cudaMalloc((void**)&d_Best_seq, NumPos * sizeof(char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "[Haplo_seq]cudaMalloc failed!");
	}

	//size_t heap_size = phasing_iter * POPSIZE * NumPos * sizeof(char)/8;
	//cudaStatus = cudaDeviceSetLimit(cudaLimitMallocHeapSize, heap_size);
	//if (cudaStatus != cudaSuccess) {
	//	fprintf(stderr, "cudaLimitMallocHeapSize set failed!\n");
	//}

#ifdef __DTIME__
	std::chrono::system_clock::time_point time1 = std::chrono::system_clock::now();
	std::chrono::duration<double> DefaultSec1 = time1 - time0;
	std::cout << DefaultSec1.count() << "secs" << endl;

	printf("Phasing :");
#endif

	///////////////////////////////////////////
	//Step 2: Phasing in device
	///////////////////////////////////////////

	uint seed = (uint)chrono::system_clock::now().time_since_epoch().count();

	Phasing_unit << < NumBlk, CU_BSIZE >> > (
		seed, G_ITER, d_blocks,
		d_Best_sumD, d_Best_seq, d_Tog_sumD, d_Tog_seq, d_Pop_seq,
		d_AlleleData, d_QualData, d_SubFragments, d_Fragments, d_FragsForPos,
		NumPos, NumFrag, NumBlk, phasing_iter);

	cudaCheckSync("Phasing");

	cudaFree(d_Tog_sumD);
	cudaFree(d_Tog_seq);
	cudaFree(d_Pop_seq);

#ifdef __DTIME__
	std::chrono::system_clock::time_point time2 = std::chrono::system_clock::now();
	std::chrono::duration<double> DefaultSec2 = time2 - time1;
	std::cout << DefaultSec2.count() << "secs" << endl;
#endif

	///////////////////////////////////////////
	// Step 3: Copy Best Haplotype
	///////////////////////////////////////////

	// memory copy h_BestHaplo <- d_Best
	cudaStatus = cudaMemcpy(h_BestHaplo_sumD, d_Best_sumD, NumBlk * sizeof(double), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "[phasing-Haplo_sumD] cudaMemcpy failed!");
	}
	cudaStatus = cudaMemcpy(h_BestHaplo_seq, d_Best_seq, NumPos * sizeof(char), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "[phasing-Haplo_seq] cudaMemcpy failed!");
	}

	cudaFree(d_Best_sumD);
	cudaFree(d_Best_seq);
}


///////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////

// 
// finding position uncovered by matrix (Covered[])
// return : the number of uncovered position by matrix
//
void find_covered_sites(const BlockType &block,
	const SubFragType *SubFragments, const FragType *Fragments, char *Covered)
{
	uint startFrag, endFrag;

	for (uint i = 0; i < block.length; i++)
		Covered[i] = 0;

	startFrag = block.start_frag;
	endFrag = startFrag + block.num_Frag - 1;

	// finding uncovered positions
	for (uint i = startFrag; i <= endFrag; i++) 	// for each fragment
		for (uint j = 0; j < Fragments[i].num_subFrag; j++) { // for each subfragment

			uint subfragidx = Fragments[i].subFrag0 + j + block.start_subfrag;
			uint pos_blk = SubFragments[subfragidx].start_pos;

			for (; pos_blk <= SubFragments[subfragidx].end_pos; pos_blk++) 	// for each position
				Covered[pos_blk] = 1;
		}
}

// 
// Creating complement sequence
// return : the number of phased position
//
uint create_complement(HaploType haplo, const BlockType &block, const char *Covered)
{
	uint phased = 0;

	for (uint i = 0; i < block.length; i++)
		if (Covered[i]) {
			haplo.seq2[i] = (haplo.seq[i] == '0') ? '1' : '0';
			++phased;
		}
		else
			haplo.seq[i] = haplo.seq2[i] = '-';

	haplo.seq[block.length] = haplo.seq2[block.length] = '\0';

	return phased;
}


///////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////

//
// load input matrix file
//
void load_matrixFile(const char matrixFileName[],
	char *&AlleleData, QualType *&QualData, vector <SubFragType> *SubFragments, FragType *&Fragments,
	uint *MatrixDataSize, uint *NumPos, uint *NumSubFrag, uint *NumFrag)
{
	FILE * matrixFile = fopen(matrixFileName, "r");  	// open input file

	if (!matrixFile) {
		cout << "Inputfile \"" << matrixFileName << "\" does not exist." << endl;
		exit(1);
	}

	fseek(matrixFile, 0, SEEK_END);
	uint fileSize = ftell(matrixFile);
	fseek(matrixFile, 0, SEEK_SET);

	// 1. load naive matrix data

	char *FileData = new char[fileSize + 10];

	size_t fragDataSize = fread(FileData, sizeof(char), fileSize + 10, matrixFile);	// read raw data in input file
	FileData[fragDataSize] = '\0';

	fclose(matrixFile);

	// 2. get # of fragments and # of positions (columns in input matrix)

	const char *token = " \t\n";	// token used in strtok
	char * p;

	p = strtok(FileData, token);
	*NumFrag = atoi(p);				// number of fragments

	p = strtok(NULL, token);
	*NumPos = atoi(p);				// number of positions

	// 3. build structure for matrix data

	AlleleData = new char[fragDataSize / 2];
	QualData = new QualType[fragDataSize / 2];
	uint offset = 0;		// offset in AlleleData & QualData;
	uint subfrag_idx = 0;	// index in SubFragments vector

	Fragments = new FragType[*NumFrag];
	
	SubFragments->clear();
	SubFragType subfrag;			// temp
	
	for (uint i = 0; i < *NumFrag; i++) {	// storing fragment data in data structures

		if (!(p = strtok(NULL, token))) {
			*NumFrag = i;			// set NumFrag to real number of frags
			break;
		}

		uint num_subFrag = atoi(p); // number of subfragments in fragment[i]

		Fragments[i].num_subFrag = num_subFrag;
		Fragments[i].subFrag0 = subfrag_idx;

		p = strtok(NULL, token);		// skip fragment name

		// storing subfragment metadata
		for (uint j = 0; j < num_subFrag; j++) {

			p = strtok(NULL, token);
			uint s_pos = atoi(p) - 1;

			p = strtok(NULL, token);
			strcpy(AlleleData + offset, p);

			uint len = (uint)strlen(p);

			subfrag.Moffset = offset;
			subfrag.start_pos = s_pos;
			subfrag.end_pos = s_pos + len - 1;

			SubFragments->push_back(subfrag);
			++subfrag_idx;

			offset += len;	//for next allele sequence
		} // end_for j

		// storing fragment metadata 
		uint first_subfrag = Fragments[i].subFrag0;
		uint last_subfrag = Fragments[i].subFrag0 + Fragments[i].num_subFrag - 1;
		Fragments[i].start_pos = (*SubFragments)[first_subfrag].start_pos;
		Fragments[i].end_pos = (*SubFragments)[last_subfrag].end_pos;

		p = strtok(NULL, "\n");

		// if fragment contains only one site, exclude the fragment in phasing
		if (Fragments[i].start_pos == Fragments[i].end_pos) {
			SubFragments->erase(SubFragments->begin() + first_subfrag, SubFragments->begin() + last_subfrag + 1);
			subfrag_idx -= last_subfrag - first_subfrag + 1;
			i--;
			continue;
		}

		// storing quality scores

		for (uint j = 0; j < num_subFrag; j++)
		{
			uint start_offset = (*SubFragments)[first_subfrag + j].Moffset;
			uint k;
			// precomputing error probability
			uint subfrag_len = (*SubFragments)[first_subfrag + j].end_pos - (*SubFragments)[first_subfrag + j].start_pos + 1;
			for (k = 0; k < subfrag_len; k++) {
				uint qual_j = (uint)(*(p++));
				qual_j -= 33;
				QualData[start_offset + k] = pow(10.0, ((-1 * (QualType)qual_j) / 10.0));
			}

			//QualData[start_offset + k] = -1.0;	 // delimiter 

		} // end_for j

	} // end i

	*MatrixDataSize = offset;
	*NumSubFrag = (uint)SubFragments->size();

//	sort(Fragments, Fragments + *NumFrag, compare_frag_pos);	// sorting with starting positions

	delete[] FileData;

#ifdef __DEBUG__
/*
	// recording fragment data in file for checking if fragment data is well loaded
	ofstream fragmentOutFile;
	fragmentOutFile.open("ZfragmentOut.txt");

	for (uint i = 0; i < *NumFrag; i++) {
		fragmentOutFile << "start: " << Fragments[i].start_pos
			<< " end: " << Fragments[i].end_pos
			<< " num_subFrag: " << Fragments[i].num_subFrag
			<< " subFrag0: " << Fragments[i].subFrag0 << endl;

		uint first_subfrag = Fragments[i].subFrag0;

		for (uint j = 0; j < Fragments[i].num_subFrag; j++) {
			fragmentOutFile << "    start: " << (*SubFragments)[first_subfrag + j].start_pos
				<< " end: " << (*SubFragments)[first_subfrag + j].end_pos
				//				<< " offset:" << SubFragments[first_subfrag + j].Moffset
				<< " str:" << AlleleData + (*SubFragments)[first_subfrag + j].Moffset
				<< " qual:" << QualData + (*SubFragments)[first_subfrag + j].Moffset << endl;
		}
		fragmentOutFile << "==============================================================" << endl;
	}
	fragmentOutFile.close();
*/

	// checking order of start_pos in Fragments and SubFragments
	for (uint i = 0; i < *NumFrag; i++) {
		if (i > 0 && Fragments[i - 1].start_pos > Fragments[i].start_pos) {
			printf("Start positions of Fragments are not sorted  ~!!!\n");
			printf("start_pos of Frag %d: %d,  start_pos of Frag %d: %d\n",
				i - 1, Fragments[i - 1].start_pos, i, Fragments[i].start_pos);
			exit(1);
		}

		uint first_subfrag = Fragments[i].subFrag0;

		for (uint j = 1; j < Fragments[i].num_subFrag; j++) {
			if ((*SubFragments)[first_subfrag + j - 1].start_pos > (*SubFragments)[first_subfrag + j].start_pos) {
				printf("Start positions of SubFragments are not sorted in Fragment %d ~!!!\n", i);
				printf("start_pos of subFrag %d: %d,  start_pos of subFrag %d: %d\n",
					j - 1, (*SubFragments)[first_subfrag + j - 1].start_pos, j, (*SubFragments)[first_subfrag + j].start_pos);
				exit(1);
			}
		}
	}

#endif
}

//
// build auxilary data structures
// 1. fragments info. for each position
// 2. blocks of fragments
//
// build auxilary data structures
void build_aux_struct(
	vector <BlockType> *Blocks, FragsForPosType *&FragsForPos, uint *NumBlk, uint *MaxBlkLen,
	SubFragType *Subfragments, FragType *Fragments, const uint NumPos, const uint NumFrag )
{

	// 1. build an array of fragment numbers locating at each position

	FragsForPos = new FragsForPosType[NumPos];

	//for (int i = 0; i < NumPos; i++)
	//FragsForPos[i] = { -1,-1 };

	int cutStart = 0;
	for (int i = 0; i < (int)NumFrag; i++) 		// storing starting fragment #
		for (; cutStart <= (int)Fragments[i].end_pos; cutStart++)
			FragsForPos[cutStart].start_frag = i;

	int cutEnd = NumPos - 1;
	for (int i = NumFrag - 1; i >= 0; i--) 	// storing ending fragment #
		for (; cutEnd >= (int)Fragments[i].start_pos; cutEnd--)
			FragsForPos[cutEnd].end_frag = i;

#ifdef __DEBUG__
	// recording range positions in file
	ofstream rangePosFile;
	rangePosFile.open("ZrangePos.txt");
	for (uint i = 0; i < NumPos; i++) {
		rangePosFile << "[" << i << "] " << FragsForPos[i].start_frag << " " << FragsForPos[i].end_frag << endl;
	}
	rangePosFile.close();
#endif


	// 2. divide fragments into several blocks (sets of fragments whose covered positions were overlapped)

	uint blkStartFrag = 0;	// set starting fragment
	uint blkEndPos = Fragments[0].end_pos;

	BlockType block;	// temp

	Blocks->clear();

	*MaxBlkLen = 0;

	uint i;
	for (i = 1; i < NumFrag; i++) {
		if (Fragments[i].start_pos > blkEndPos) {	// if frag[i] is NOT overlapped with the previous frags

			block.start_frag = blkStartFrag;
			block.num_Frag = i - blkStartFrag;
			block.start_pos = Fragments[blkStartFrag].start_pos;
			block.length = blkEndPos - Fragments[blkStartFrag].start_pos + 1;
			block.start_subfrag = Fragments[blkStartFrag].subFrag0;
			block.Moffset = Subfragments[block.start_subfrag].Moffset;

			Blocks->push_back(block); // stroing current block info
			blkStartFrag = i;						// set the new starting fragment

			if (*MaxBlkLen < block.length)	*MaxBlkLen = block.length;
		}
		if( blkEndPos < Fragments[i].end_pos)
			blkEndPos = Fragments[i].end_pos; // update end position
	}

	block.start_frag = blkStartFrag;
	block.num_Frag = i - blkStartFrag;
	block.start_pos = Fragments[blkStartFrag].start_pos;
	block.length = blkEndPos - Fragments[blkStartFrag].start_pos + 1;
	block.start_subfrag = Fragments[blkStartFrag].subFrag0;
	block.Moffset = Subfragments[block.start_subfrag].Moffset;

	Blocks->push_back(block); // for the last block
	if (*MaxBlkLen < block.length)	*MaxBlkLen = block.length;

	*NumBlk = (uint)Blocks->size();

#ifdef __DEBUG__
	// recording fragment numbers in each block
	ofstream blockFile;
	blockFile.open("ZBlockPos.txt");
	for (i = 0; i < Blocks->size(); i++) {
		blockFile << "[" << i + 1 << "] " << (*Blocks)[i].start_frag << " " << (*Blocks)[i].start_frag + (*Blocks)[i].num_Frag
			<< " " << (*Blocks)[i].start_pos << " " << (*Blocks)[i].length << endl;

	}
	blockFile.close();
#endif

	// 3. change indexes into local indexes inside block
	
	for (uint b = 0; b < *NumBlk; ++b) {
		BlockType block = Blocks->at(b);
		FragType last_frag = Fragments[block.start_frag + block.num_Frag - 1];	// global index
		uint last_subfrag = last_frag.subFrag0 + last_frag.num_subFrag - 1;  // global index
		for (uint k = block.start_subfrag; k <= last_subfrag; ++k) {
			Subfragments[k].start_pos -= block.start_pos;
			Subfragments[k].end_pos -= block.start_pos;
			Subfragments[k].Moffset -= block.Moffset;
		}
	}

	for (uint b = 0; b < *NumBlk; ++b) {
		BlockType block = Blocks->at(b);
		uint block_end_frag = block.start_frag + block.num_Frag - 1;
		for (uint k = block.start_frag; k <= block_end_frag; ++k) {
			Fragments[k].start_pos -= block.start_pos;
			Fragments[k].end_pos -= block.start_pos;
			Fragments[k].subFrag0 -= block.start_subfrag;
		}
	}


	for (uint b = 0; b < *NumBlk; ++b) {
		BlockType block = Blocks->at(b);
		for (uint pos = 0; pos < block.length; ++pos) {
			FragsForPos[pos+ block.start_pos].start_frag -= block.start_frag;
			FragsForPos[pos+ block.start_pos].end_frag -= block.start_frag;
		}
	}

	#ifdef __DEBUG__
	// recording range positions in file
	ofstream rangePosBlkFile;
	rangePosBlkFile.open("ZrangePos_block.txt");
	for (uint i = 0; i < NumPos; i++) {
	rangePosBlkFile << "[" << i << "] " << FragsForPos[i].start_frag << " " << FragsForPos[i].end_frag << endl;
	}
	rangePosBlkFile.close();
	#endif
	
}

#ifdef __DEBUG__
// recording fragment data in matrix form
void debug_build_matrix_map_file(
	char * g_AlleleData, SubFragType *g_SubFragments, FragType *g_Fragments, BlockType *g_Blocks,
	const uint NumPos, const uint NumBlk, const uint MaxBlkLen)
{
	ofstream fragmentForm("ZFragmentForm.SORTED");

	for (uint b = 0; b<NumBlk; b++) {
		BlockType block = g_Blocks[b];

		FragType *Fragments = g_Fragments + block.start_frag;
		SubFragType *SubFragments = g_SubFragments + block.start_subfrag;
		char * AlleleData = g_AlleleData + block.Moffset;


		fragmentForm << "block # : " << b + 1 << endl;

		for (uint i = 0 ; i < block.num_Frag; i++) {		// For each fragment

			char * frag = new char[(block.length + 1) * 2];
			for (uint k = 0; k<block.length; k++)
				frag[2 * k] = ' ', frag[2 * k + 1] = '-';
			frag[block.length * 2] = '\0';

			FragType Fragi = Fragments[i];
			SubFragType *SubFrag = SubFragments + Fragi.subFrag0; // 0th subfragments in the fragment[i]

			for (uint j = 0; j < Fragi.num_subFrag; j++) {

				SubFragType SubFragj = SubFrag[j];
				char *subfrag_str = AlleleData + SubFragj.Moffset;

				uint pos_blk = SubFragj.start_pos;
				for ( ; pos_blk <= SubFragj.end_pos; pos_blk++) {
					frag[2 * (pos_blk) + 1] = subfrag_str[pos_blk - SubFragj.start_pos];
				}
			}
			fragmentForm << frag << endl;

			delete[] frag;
		}
	}

	fragmentForm.close();
}
#endif


void allocate(double *&h_BestHaplo_sumD, char *&h_BestHaplo_seq, char *&h_Covered, HaploType &h_BestHaplo2,

	char *&d_AlleleData, QualType *&d_QualData, SubFragType *&d_SubFragments, FragType *&d_Fragments,
	BlockType *&d_Blocks, FragsForPosType *&d_FragsForPos,
	char *h_AlleleData, QualType *h_QualData, SubFragType *h_SubFragments, FragType *h_Fragments,
	BlockType *h_Blocks, FragsForPosType *h_FragsForPos,
	uint MatrixDataSize, uint NumPos, uint NumSubFrag, uint NumFrag,
	uint NumBlk, uint MaxBlkLen, uint phasing_iter)

{

	// Memory allocation for host
	h_BestHaplo_sumD = new double[NumBlk];
	h_BestHaplo_seq = new char[NumPos];

	h_Covered = new char[MaxBlkLen + 1];
	h_BestHaplo2.seq = new char[(MaxBlkLen + 1)];
	h_BestHaplo2.seq2 = new char[(MaxBlkLen + 1)];




	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
	}

	//
	// Allocate GPU memory and
	// Copy data from host memory to GPU buffers for members of structures
	//

	cudaStatus = cudaMalloc((void**)&d_AlleleData, MatrixDataSize * sizeof(char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "[AlleleData] cudaMalloc failed!");
	}
	cudaStatus = cudaMemcpy(d_AlleleData, h_AlleleData, MatrixDataSize * sizeof(char), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "[AlleleData] cudaMemcpy failed!");
	}

	cudaStatus = cudaMalloc((void**)&d_QualData, MatrixDataSize * sizeof(QualType));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "[QualData] cudaMalloc failed!");
	}
	cudaStatus = cudaMemcpy(d_QualData, h_QualData, MatrixDataSize * sizeof(QualType), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "[QualData] cudaMemcpy failed!");
	}

	cudaStatus = cudaMalloc((void**)&d_SubFragments, NumSubFrag * sizeof(SubFragType));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "[SubFrags] cudaMalloc failed!");
	}
	cudaStatus = cudaMemcpy(d_SubFragments, h_SubFragments, NumSubFrag * sizeof(SubFragType), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "[SubFrags]cudaMemcpy failed!");
	}

	cudaStatus = cudaMalloc((void**)&d_Fragments, NumFrag * sizeof(FragType));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "[Fragmts]cudaMalloc failed!");
	}
	cudaStatus = cudaMemcpy(d_Fragments, h_Fragments, NumFrag * sizeof(FragType), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "[Fragmts]cudaMemcpy failed!");
	}

	cudaStatus = cudaMalloc((void**)&d_Blocks, NumBlk * sizeof(BlockType));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "[Blocks]cudaMalloc failed!");
	}
	cudaStatus = cudaMemcpy(d_Blocks, h_Blocks, NumBlk * sizeof(BlockType), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "[Blocks]cudaMemcpy failed!");
	}

	cudaStatus = cudaMalloc((void**)&d_FragsForPos, NumPos * sizeof(FragsForPosType));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "[FragsForPos]cudaMalloc failed!");
	}
	cudaStatus = cudaMemcpy(d_FragsForPos, h_FragsForPos, NumPos * sizeof(FragsForPosType), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "[FragsForPos]cudaMemcpy failed!");
	}

	//------ GPU memory used locally

	//cudaStatus = cudaMalloc((void**)&d_Tog_sumD, phasing_iter * NumBlk * sizeof(double));
	//if (cudaStatus != cudaSuccess) {
	//	fprintf(stderr, "[Haplo_sumD]cudaMalloc failed!");
	//}

	//cudaStatus = cudaMalloc((void**)&d_Tog_seq, phasing_iter * NumPos * sizeof(char));
	//if (cudaStatus != cudaSuccess) {
	//	fprintf(stderr, "[Haplo_seq]cudaMalloc failed!");
	//}

	//cudaStatus = cudaMalloc((void**)&d_Population, phasing_iter * NumBlk * POPSIZE * sizeof(IndvDType));
	//if (cudaStatus != cudaSuccess) {
	//	fprintf(stderr, "[PopSumD]cudaMalloc failed!");
	//}

	//cudaStatus = cudaMalloc((void**)&d_Pop_seq, phasing_iter * POPSIZE * NumPos * sizeof(char));
	//if (cudaStatus != cudaSuccess) {
	//	fprintf(stderr, "[Pop_seq]cudaMalloc failed!");
	//}

	//cudaStatus = cudaMalloc((void**)&d_GAcnt, phasing_iter * NumPos * sizeof(uint));
	//if (cudaStatus != cudaSuccess) {
	//	fprintf(stderr, "[GAcnt]cudaMalloc failed!");
	//}

	//cudaStatus = cudaMalloc((void**)&d_DFrag, phasing_iter * NumFrag * sizeof(DFragType));
	//if (cudaStatus != cudaSuccess) {
	//	fprintf(stderr, "[DFrag]cudaMalloc failed!");
	//}

	//cudaStatus = cudaMalloc((void**)&d_randstates, phasing_iter * NumBlk * sizeof(curandState));
	//if (cudaStatus != cudaSuccess) {
	//	fprintf(stderr, "[RandStates]cudaMalloc failed!");
	//}

}


void deallocate(char *h_AlleleData, QualType *h_QualData, FragType *h_Fragments, FragsForPosType *h_FragsForPos,
	double *h_BestHaplo_sumD, char *h_BestHaplo_seq, char *h_Covered, HaploType h_BestHaplo2,
	char *d_AlleleData, QualType *d_QualData, SubFragType *d_SubFragments, FragType *d_Fragments,
	BlockType *d_Blocks, FragsForPosType *d_FragsForPos)
{
	// memory deallocation for host
	delete[] h_AlleleData, h_QualData, h_Fragments, h_FragsForPos;

	delete[] h_BestHaplo_sumD, h_BestHaplo_seq;

	delete[] h_BestHaplo2.seq, h_BestHaplo2.seq2;

	delete[] h_Covered;


	// memory deallocation for device
	cudaFree(d_AlleleData);
	cudaFree(d_QualData);

	cudaFree(d_SubFragments);
	cudaFree(d_Fragments);
	cudaFree(d_Blocks);
	cudaFree(d_FragsForPos);

	//cudaFree(d_Tog_sumD);
	//cudaFree(d_Tog_seq);
//	cudaFree(d_Population);
//	cudaFree(d_Pop_seq);

//	cudaFree(d_GAcnt);
//	cudaFree(d_DFrag);

//	cudaFree(d_randstates);
}


////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////

//
// main procedure
//
void procedure(const char matrixFileName[], const char outputFileName[], const uint phasing_iter)
{
	// Constant data produced in host

	char *h_AlleleData, *d_AlleleData;				// matrix allele data
	QualType *h_QualData, *d_QualData;				// matrix quality data

	vector <SubFragType> h_SubFragments;	// vector of subframents
	SubFragType *d_SubFragments;			// array of subframents
	FragType *h_Fragments, *d_Fragments;	// array of fragments
	vector <BlockType> h_Blocks;			// haplo phasing blocks
	BlockType *d_Blocks;					// array of haplo phasing blocks

	FragsForPosType *h_FragsForPos, *d_FragsForPos;		// aux. data structures

	double *h_BestHaplo_sumD;		// Best haplotype sumD
	char *h_BestHaplo_seq;			// Best haplotype sequence

	char *h_Covered;				// positions covered by fragments
	HaploType h_BestHaplo2;			// Best Haplotype storing 2 sequences


	// data produced in device

//	IndvDType *d_Population;		// haplotype sumD in population of GA
//	char *d_Pop_seq;					// haplotype sequence in population of GA
//	uint *d_GAcnt;					// array for storing the frequency of 1's for each position in GA

//	double *d_Tog_sumD;				// haplotype sumD used in toggling
//	char *d_Tog_seq;					// haplotype sequence used in toggling
//	DFragType *d_DFrag;				// array for storing weighted MEC for each frament

//	curandState *d_randstates = NULL;		// random number states used in GA

	// Variables for Size

	uint MatrixDataSize;				// matrix data size
	uint NumPos;						// number of positions
	uint NumSubFrag;					// total number of subfragments
	uint NumFrag;					// number of fragments
	uint NumBlk;						// number of blocks
	uint MaxBlkLen;					// maximum block length

	////////////////////////////////////////////////////////////////////////////

	// for running time measurement
	std::chrono::system_clock::time_point time0 = std::chrono::system_clock::now();

	// load matrix data in input file into data structures
#ifdef __DTIME__
	printf("\nLoading :");
#endif

	load_matrixFile(matrixFileName,
		h_AlleleData, h_QualData, &h_SubFragments, h_Fragments, 
		&MatrixDataSize, &NumPos, &NumSubFrag, &NumFrag);

	// build auxilary data structures
	build_aux_struct(&h_Blocks, h_FragsForPos, &NumBlk, &MaxBlkLen, &h_SubFragments[0], h_Fragments, NumPos, NumFrag);

#ifdef __DEBUG__
	debug_build_matrix_map_file(h_AlleleData, &h_SubFragments[0], h_Fragments, &h_Blocks[0],
		NumPos, NumBlk, MaxBlkLen);
#endif

#ifdef __DTIME__
	std::chrono::system_clock::time_point time1 = std::chrono::system_clock::now();
	std::chrono::duration<double> DefaultSec1 = time1 - time0;
	std::cout << DefaultSec1.count() << "secs" << endl;

	printf("Allocation : ");
#endif

	// Memory allocation & memory copy from host to device
	allocate(h_BestHaplo_sumD, h_BestHaplo_seq, h_Covered, h_BestHaplo2,
		d_AlleleData, d_QualData, d_SubFragments, d_Fragments, d_Blocks, d_FragsForPos,
		h_AlleleData, h_QualData, &h_SubFragments[0], h_Fragments, &h_Blocks[0], h_FragsForPos,
		MatrixDataSize, NumPos, NumSubFrag, NumFrag, NumBlk, MaxBlkLen, phasing_iter);

#ifdef __DTIME__
	std::chrono::system_clock::time_point time2 = std::chrono::system_clock::now();
	std::chrono::duration<double> DefaultSec2 = time2 - time1;
	std::cout << DefaultSec2.count() << "secs" << endl;
#endif


	// haplotype phasing procedure running in device
	haplotype_phasing(h_BestHaplo_sumD, h_BestHaplo_seq,
		d_AlleleData, d_QualData, d_SubFragments, d_Fragments, d_Blocks, d_FragsForPos,
		NumPos, NumFrag, NumBlk, MaxBlkLen, phasing_iter);


#ifdef __PRINT_RESULT__ 
	ofstream phasingResultFile;   // results file
	phasingResultFile.open(outputFileName);
	//	phasingResultFile.precision(5);

	uint totalBlockSize = 0;
	double totalWMEC = 0.0;		// weighted mec value
	uint totalMEC = 0;			// mec value
	uint totalPhased = 0;		// total number of phased positions
	uint totalReads = 0;			// total number of reads used for phasing
#endif


	for (uint i = 0; i< NumBlk; i++) {

		h_BestHaplo2.sumD = h_BestHaplo_sumD[i];
		memcpy(h_BestHaplo2.seq, h_BestHaplo_seq + h_Blocks[i].start_pos, h_Blocks[i].length * sizeof(char));

		find_covered_sites(h_Blocks[i], &h_SubFragments[0], h_Fragments, h_Covered);
		uint phased = create_complement(h_BestHaplo2, h_Blocks[i], h_Covered);


		uint mec = calc_MEC1(h_BestHaplo2.seq, h_Blocks[i], h_AlleleData, &h_SubFragments[0], h_Fragments);

#ifdef __PRINT_RESULT__ 
		totalBlockSize += h_Blocks[i].length;
		totalPhased += phased;
		totalReads += h_Blocks[i].num_Frag;
		totalWMEC += h_BestHaplo2.sumD;
		totalMEC += mec;

		// ---------------------
		// 1. Printing Block Header
		// ---------------------
		phasingResultFile << "Block Number: " << i + 1;
		phasingResultFile << "  Block Length: " << h_Blocks[i].length;
		phasingResultFile << "  Phased Length: " << phased;
		phasingResultFile << "  Number of Reads: " << h_Blocks[i].num_Frag;
		phasingResultFile << "  Start position: " << h_Blocks[i].start_pos + 1;
		phasingResultFile << "  Weighted MEC: " << h_BestHaplo2.sumD;
		phasingResultFile << "  MEC: " << mec << endl;

		// ---------------------
		// 2. Printg Haloptype (Format 1) : printing each phased site in each line
		// ---------------------
		uint blk_offset = h_Blocks[i].start_pos + 1;	// 1st index is 1 
		for (uint k = 0; k < h_Blocks[i].length; ++k)
			if (h_BestHaplo2.seq[k] != '-')
				phasingResultFile << blk_offset + k << "\t " << h_BestHaplo2.seq[k]
				<< "\t " << h_BestHaplo2.seq2[k] << endl;

		// ---------------------
		// 2. Printg Haloptype (Format 2) : printing phased sequence in one line (without blank)
		// ---------------------
		//phasingResultFile << BestHaplo.seq1 << endl;
		//phasingResultFile << BestHaplo.seq2 << endl << endl;

		// ---------------------
		// 2. Printg Haloptype (Format 3) : printing phased sequence in one line (with blank)
		// ---------------------
		//for (uint k = 0; k < Blocks[i].length; ++k)
		//	phasingResultFile << " " << BestHaplo.seq[k];
		//phasingResultFile << endl;
		//for (uint k = 0; k < Blocks[i].length; ++k)
		//	phasingResultFile << " " << BestHaplo.seq2[k];
		//phasingResultFile << endl << endl;

		// ---------------------
		// 3. Printg Block Closer
		// ---------------------
		phasingResultFile << "********" << endl;
#endif

	}

	// Memory deallocation
	deallocate(h_AlleleData, h_QualData, h_Fragments, h_FragsForPos,
		h_BestHaplo_sumD, h_BestHaplo_seq, h_Covered, h_BestHaplo2,
		d_AlleleData, d_QualData, d_SubFragments, d_Fragments, d_Blocks, d_FragsForPos);

	std::chrono::system_clock::time_point time9 = std::chrono::system_clock::now();
	std::chrono::duration<double> DefaultSec = time9 - time0;

#ifdef __PRINT_RESULT__ 
	// ==============================================
	// Printing Total Results
	// ==============================================
	//phasingResultFile << "-------------------------------------" << endl;
	//phasingResultFile << "Total Block Length : " << totalBlockSize << endl;
	//phasingResultFile << "Total Phased Length : " << totalPhased << endl;
	//phasingResultFile << "Total Number of used Reads : " << totalReads << endl;
	//phasingResultFile << "Total Weighted MEC : " << totalWMEC << endl;
	//phasingResultFile << "Total MEC : " << totalMEC << endl;
	//phasingResultFile << "Total Time : " << DefaultSec.count() << " seconds" << std::endl;
	//phasingResultFile << "-------------------------------------" << endl;
	//----------------------------------------------------------------
	phasingResultFile.close();

	//--------------------
	// Printing in stdout
	//--------------------
	// std::cout << "Block_len: "<<totalBlockSize << "   Phased_len: " << totalPhased
	//	<< "   Used_Reads: " << totalReads 
	//	<< "   w_MEC: " << totalWMEC << "   MEC: " << totalMEC
	//	<< "   Time: " << DefaultSec.count() << endl;


	std::cout << totalBlockSize << "   " << totalPhased << "   " << totalReads << "  "
		<< totalWMEC << "   " << totalMEC << "   " << DefaultSec.count() << endl;
#endif

	//	std::cout << "   " << DefaultSec.count();
	// std::cerr << "   " << DefaultSec.count();

}

////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////

//
// main function
// 1st argument : input file name
// 2nd argument : output filne name

//int peath_exp_main(int argc, char **argv, uint cbsize, uint p_iter, uint g_iter)
int main(int argc, char ** argv)
{
	if (argc != 3 && argc != 4) {
		cout << "usage: " << argv[0] << " <input_file> <output_file> <param>" << endl;
		cout << "<param> is a positive integer (optional, default : 50)" << endl;
		exit(1);
	}

	char matrixFileName[100], outputFileName[100];

	strcpy(matrixFileName, argv[1]);
	strcpy(outputFileName, argv[2]);

	uint phasing_iter = DEFAULT_PHASING_ITER;

	if (argc == 4) {
		phasing_iter = atoi(argv[3]);
		if (phasing_iter < 1) {
			cout << "usage: " << argv[0] << " <input_file> <output_file> <param>" << endl;
			cout << "<param> is a positive integer (optional, default : 50)" << endl;
			exit(1);
		}
	}

	//phasing_iter = p_iter;
	//G_ITER = g_iter;

	procedure(matrixFileName, outputFileName, phasing_iter);		//// main procedure

//	cout << endl << "-- Run by CUDA!!! -- cuda block size : " << CU_MAX_BSIZE << endl << endl;

	return 0;
}

