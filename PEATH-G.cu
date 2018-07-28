#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <device_functions.h>
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
#define __PRINT_RESULT__	// to create files for debugging

////////////////////////////////

#define __CUDA__	1		// CPU: 0 , CUDA: 1

#define CU_MAX_BSIZE	32		// cuda block dimension - x

////////////////////////

#define G_ITER		50			// GA iteration
#define STOPCNT		10			// GA termination condition

#define TOG_ITER	10			// toggling iteration

#define DEFAULT_PHASING_ITER 50

/////////////////////////////////


#define EPSILON 0.00000000001		// error threshold of computing floating point

#define POPSIZE 100			// population size in GA
#define OFFSIZE 50			// offsping size in GA

////////////////////////////////


using namespace std;

typedef unsigned int uint;


typedef struct {			// subfragments
	uint Moffset = 0;				// offset of starting position in Allele & Qual Matrix Data
	uint start_pos = 0;
	uint end_pos = 0;
	uint length = 0;
}SubFragType;

typedef struct {			// fragment(read)
	uint subFrag0 = 0;			// Index of the First subframent in SubFragments vector
	uint num_subFrag = 0;
	uint start_pos = 0;
	uint end_pos = 0;
	uint length = 0;
}FragType;

typedef struct {			// block
	uint start_frag = 0;
	uint end_frag = 0;
	uint start_pos = 0;
	uint length = 0;
}BlockType;

typedef struct {			// fragment inforamtion for each position
	uint start_frag = 0;
	uint end_frag = 0;
	//	uint endPos;
}FragsForPosType;

typedef struct {			// Weighted MEC (distance) for each fragment
	double D_hf = -1.0;			// It is used in toggling stage
	double D_hfstar = -1.0;		// in order to speed up calculation of weighted MEC
}DFragType;

typedef struct {				// individual in GA
	double sumD = -1.0;			// Weighted MEC (distance)
	uint stx_pos = 0;			// starting position in Pop_seq, not unsed in Toggling
}IndvDType;

typedef struct {			// Haplotype sequeucne
	char *seq = NULL, *seq2 = NULL;
	double sumD = -1.0;			// Weighted MEC (distance)
}HaploType;


/////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////


#ifdef __INTELLISENSE__
void __syncthreads();
#endif

void cudaCheckSync(const char func[])
{
#if __CUDA__ == 1
	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "[%s] addKernel launch failed: %s\n", func, cudaGetErrorString(cudaStatus));
		exit(1);
	}
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "[%s] cudaDeviceSynchronize returned error code %d after launching addKernel!\n", func, cudaStatus);
		exit(1);
	}
#endif
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
	const char *AlleleData, const SubFragType *SubFragments, const FragType *Fragments) {

	uint startFrag, endFrag;
	uint sumD = 0;
	uint pos_k;

	startFrag = block.start_frag;
	endFrag = block.end_frag;

	uint blk_offset = block.start_pos;

	for (uint i = startFrag; i <= endFrag; i++) {	// for each fragment

		uint D_hf = 0, D_hstarf = 0;

		for (uint j = 0; j<Fragments[i].num_subFrag; j++) {		// for each subfragment

			uint subfragidx = Fragments[i].subFrag0 + j;
			uint offset_in_block = SubFragments[subfragidx].start_pos - blk_offset;
			const char *subfrag_str = AlleleData + SubFragments[subfragidx].Moffset;

			for (uint k = 0; k<SubFragments[subfragidx].length; k++) {		// for each position

				pos_k = k + offset_in_block;		// position in block;

				if (haplo_seq[pos_k] == '-' || subfrag_str[k] == '-')
					continue;

				if (haplo_seq[pos_k] != subfrag_str[k])
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
__device__ __host__
double calc_sumD(double *haplo_sumD, const char *haplo_seq, DFragType *DFrag, const BlockType block, const bool update,
	const char *AlleleData, const double *QualData, const SubFragType *SubFragments, const FragType *Fragments)
{
	double sumD = 0.0;
	uint pos_k;

	const uint startFrag = block.start_frag;
	const uint endFrag = block.end_frag;

	const uint blk_offset = block.start_pos;

	for (uint i = startFrag; i <= endFrag; i++) {	// for each fragment

		double D_hf = 0.0, D_hfstar = 0.0;

		for (uint j = 0; j < Fragments[i].num_subFrag; j++) {	// for each subfragment

			const uint subfragidx = Fragments[i].subFrag0 + j;
			const uint offset_in_block = SubFragments[subfragidx].start_pos - blk_offset;
			const char *subfrag_str = AlleleData + SubFragments[subfragidx].Moffset;
			const double *subfrag_qual = QualData + SubFragments[subfragidx].Moffset;

			for (uint k = 0; k < SubFragments[subfragidx].length; k++) {	// for each position

				pos_k = k + offset_in_block;		// position in block;

				double q_j = subfrag_qual[k];
				double q_j_star = 1 - q_j;

				// calculating distance for a position
				if (haplo_seq[pos_k] != subfrag_str[k]) {
					D_hf += q_j_star;
					D_hfstar += q_j;
				}
				else {
					D_hf += q_j;
					D_hfstar += q_j_star;
				}

			} // end_for k (each position)

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
double calc_sumD_range_tog(double *haplo_sumD, const char *haplo_seq, DFragType *DFrag, const BlockType& block, const uint tog_pos, const bool update,
	const char *AlleleData, const double *QualData, const SubFragType *SubFragments, const FragType *Fragments,
	const FragsForPosType *FragsForPos)
{
	double sumD = *haplo_sumD;

	const uint blk_offset = block.start_pos;	// start position of block
	const uint border_pos = blk_offset + tog_pos;		// convert pos to real position (matrix position)

	const uint startFrag = FragsForPos[border_pos].start_frag;	// first fragment located at pos
	const uint endFrag = FragsForPos[border_pos].end_frag;		// last fragment located at pos

	if (startFrag > endFrag)	// no fragment is located at pos
		return sumD;

	for (uint i = startFrag; i <= endFrag; i++) {	// for each fragment covering pos

													// 1. substract sumD of the current fragments
		double D_hf = DFrag[i].D_hf, D_hfstar = DFrag[i].D_hfstar;  // previous DFrag values

		if (D_hf < D_hfstar) sumD -= D_hf;		// substract the previous DFrag value
		else sumD -= D_hfstar;

		// 2. compute sumD of the current fragments for toggled haplotype sequence
		D_hf = D_hfstar = 0.0;

		uint numSubFrag = Fragments[i].num_subFrag;
		for (uint j = 0; j < numSubFrag; j++) {	// for each subfragment

			const uint subfragidx = Fragments[i].subFrag0 + j;
			const uint offset_in_block = SubFragments[subfragidx].start_pos - blk_offset;
			const char *subfrag_str = AlleleData + SubFragments[subfragidx].Moffset;
			const double *subfrag_qual = QualData + SubFragments[subfragidx].Moffset;
			const uint subfraglen = SubFragments[subfragidx].length;

			// for toggled positions
			uint k;
			uint pos_k;
			for (k = 0; k < subfraglen; k++) {	// for each position

				pos_k = k + offset_in_block;		// position in block;

				if (pos_k > tog_pos)	break;		// if  pos_k is not a toggled position

				double q_j = subfrag_qual[k];
				double q_j_star = 1 - q_j;

				// calculating distance for a position
				// computing under the assumption that the haplo_seq[pos_k] is toggled. So, != -> ==	
				if (haplo_seq[pos_k] == subfrag_str[k]) {	// 
					D_hf += q_j_star;
					D_hfstar += q_j;
				}
				else {
					D_hf += q_j;
					D_hfstar += q_j_star;
				}

			} // end_for k (each position)

			  // for not toggled positions
			for (; k < subfraglen; k++) {	// for each position

				pos_k = k + offset_in_block;		// position in block;

				double q_j = subfrag_qual[k];
				double q_j_star = 1 - q_j;

				// calculating distance for a position
				if (haplo_seq[pos_k] != subfrag_str[k]) {
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

		if (update) {				// *** if update is true  ***
			DFrag[i].D_hf = D_hf;			// *** the calculated values are stored in DFrag ***
			DFrag[i].D_hfstar = D_hfstar;
		}

	}// end_for i (each fragment)

	if (update) 					// *** if update is true  ***
		*haplo_sumD = sumD;

	return sumD;
}

//
// calculating w-MEC values (position version) used only in single_switch_procedure( )
// : calculating weighted MEC with assumption that seq[i] is toggled
// : recalculating the distance of only fragments covering at position pos
//
// this function is run by ONE thread
__device__ __host__
double calc_sumD_single_tog(double *haplo_sumD, const char *haplo_seq, DFragType *DFrag, const BlockType& block, const uint tog_pos, const bool update,
	const char *AlleleData, const double *QualData, const SubFragType *SubFragments, const FragType *Fragments,
	const FragsForPosType *FragsForPos)
{
	double sumD = *haplo_sumD;

	const uint blk_offset = block.start_pos;	// start position of block
	const uint border_pos = blk_offset + tog_pos;		// convert pos to real position (matrix position)

	const uint startFrag = FragsForPos[border_pos].start_frag;	// first fragment located at pos
	const uint endFrag = FragsForPos[border_pos].end_frag;		// last fragment located at pos

	if (startFrag > endFrag)	// no fragment is located at pos
		return sumD;

	for (uint i = startFrag; i <= endFrag; i++) {	// for each fragment covering pos

													// 1. finding subfragment located at pos
		uint j = 0;
		uint subfragidx = Fragments[i].subFrag0 + j;
		uint numSubFrag = Fragments[i].num_subFrag;

		while (j < numSubFrag && border_pos > SubFragments[subfragidx].end_pos) {
			++j;								// skip subfragments before pos
			++subfragidx;
		}
		if (j >= numSubFrag || border_pos < SubFragments[subfragidx].start_pos)
			continue;							// no subfragment is located at pos

												// 2. update sumD : subFragment[j] is located at pos
		double D_hf = DFrag[i].D_hf, D_hfstar = DFrag[i].D_hfstar;  // previous DFrag values

		if (D_hf < D_hfstar) sumD -= D_hf;		// substract the previous DFrag value
		else sumD -= D_hfstar;

		const char *subfrag_str = AlleleData + SubFragments[subfragidx].Moffset;
		const double *subfrag_qual = QualData + SubFragments[subfragidx].Moffset;

		uint k = border_pos - SubFragments[subfragidx].start_pos;
		double q_j = subfrag_qual[k];
		double q_j_star = 1 - q_j;

		// computing under the assumption that the the bit is toggled. So, != -> ==	
		if (haplo_seq[border_pos - blk_offset] == subfrag_str[k]) {
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

		if (update) {				// *** if update is true  ***
			DFrag[i].D_hf = D_hf;			// *** the calculated values are stored in DFrag ***
			DFrag[i].D_hfstar = D_hfstar;
		}

	}// end_for i (each fragment)

	if (update) 					// *** if update is true  ***
		*haplo_sumD = sumD;

	return sumD;
}


////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////


#if __CUDA__ == 0
void init_randstates(const dim3 gridDim, const dim3 blockDim,
#elif __CUDA__ == 1
__global__ void init_randstates(
#endif
	uint seed, curandState *states)
{
#if __CUDA__ == 0
	dim3 blockIdx;		// gridDim = gDim_PBx{ phasing_iter, NumBlk, 1 };
	dim3 threadIdx;		// blockDim = {cu_bsize, 1, 1};
	//for (blockIdx.x = 0; blockIdx.x < gridDim.x; ++blockIdx.x) // FOR EACH phasing iteration
	//for (blockIdx.y = 0; blockIdx.y < gridDim.y; ++blockIdx.y) // FOR EACH haplotype block
	//for (blockIdx.z = 0; blockIdx.z < gridDim.z; ++blockIdx.z) // 1
	//	for (threadIdx.x = 0; threadIdx.x < blockDim.x; ++threadIdx.x) 	// one block is processed by multiple threads

	default_random_engine generator(seed);
	uniform_real_distribution<double> distribution(0.0, 1.0);

#elif __CUDA__ == 1

	uint id = blockIdx.x * gridDim.y + blockIdx.y;
	curand_init(seed, id, 0, &states[id]);

#endif

}


#if __CUDA__ == 0
void GA_init(const dim3 gridDim, const dim3 blockDim,
#elif __CUDA__ == 1
__global__ void GA_init(
#endif
	IndvDType *d_Population, char *d_Pop_seq,
	const BlockType *d_blocks, const uint NumPos, const uint NumBlk, const uint MaxBlkLen)
{
#if __CUDA__ == 0
	dim3 blockIdx;		// gridDim = gDim_PBx{ phasing_iter, NumBlk, 1 };
	dim3 threadIdx;		// blockDim = {cu_bsize, 1, 1};
	for (blockIdx.x = 0; blockIdx.x < gridDim.x; ++blockIdx.x) // FOR EACH phasing iteration
	for (blockIdx.y = 0; blockIdx.y < gridDim.y; ++blockIdx.y) // FOR EACH haplotype block
	for (blockIdx.z = 0; blockIdx.z < gridDim.z; ++blockIdx.z) // 1
		for (threadIdx.x = 0; threadIdx.x < blockDim.x; ++threadIdx.x) 	// one Population is processed by multiple threads
#endif
		{

			BlockType block = d_blocks[blockIdx.y];
			IndvDType *Population = d_Population + (blockIdx.x * NumBlk + blockIdx.y) * POPSIZE;

			for (uint i = threadIdx.x; i < POPSIZE; i += blockDim.x) 	// relative indv_id in cur pop
				Population[i].stx_pos = i * block.length;
			
		}

}

#if __CUDA__ == 0
void GA_1stGen(const dim3 gridDim, const dim3 blockDim,
#elif __CUDA__ == 1
__global__ void GA_1stGen(
#endif
	IndvDType *d_Population, char *d_Pop_seq, curandState *d_randstates,
	const BlockType *d_blocks, const uint NumPos, const uint NumBlk)
{
#if __CUDA__ == 0
	dim3 blockIdx;		// gridDim = gDim_PBx{ phasing_iter, NumBlk, 1 };
	dim3 threadIdx;		// blockDim = {cu_bsize, 1, 1};
	for (blockIdx.x = 0; blockIdx.x < gridDim.x; ++blockIdx.x) // FOR EACH phasing iteration
	for (blockIdx.y = 0; blockIdx.y < gridDim.y; ++blockIdx.y) // FOR EACH haplotype block
	for (blockIdx.z = 0; blockIdx.z < gridDim.z; ++blockIdx.z) // 1
		for (threadIdx.x = 0; threadIdx.x < blockDim.x; ++threadIdx.x) // one Population is processed by multiple threads
#endif
		{

			BlockType block = d_blocks[blockIdx.y];
			IndvDType *Population = d_Population + (blockIdx.x * NumBlk + blockIdx.y) * POPSIZE;
			char *Pop_seq = d_Pop_seq + (blockIdx.x * NumPos + block.start_pos) * POPSIZE;;

			uint rid = blockIdx.x * gridDim.y + blockIdx.y;

#if __CUDA__ == 1
			__shared__	uint localSeed;

			if( threadIdx.x == 0 )
				localSeed = curand(&d_randstates[rid]);

			__syncthreads();

			curandState localState;
			curand_init(localSeed, threadIdx.x, 0, &localState);
#endif

			for (uint indv = 0; indv < POPSIZE; ++indv) {
				char *indv_seq = Pop_seq + Population[indv].stx_pos;
				
				for (uint i = threadIdx.x; i < block.length; i += blockDim.x) {
					if (curand_uniform(&localState) < 0.5)
						indv_seq[i] = '1';
					else
						indv_seq[i] = '0';
				}
			}

		}
}

#if __CUDA__ == 0
void GA_nextGen(const dim3 gridDim, const dim3 blockDim,
#elif __CUDA__ == 1
__global__ void GA_nextGen(
#endif
	IndvDType *d_Population, char *d_Pop_seq, curandState *d_randstates,
	const uint *d_GAcnt, const BlockType *d_blocks,
	const uint NumPos, const uint NumBlk, const uint g_iter)
{
#if __CUDA__ == 0
	dim3 blockIdx;		// gridDim = gDim_PBx{ phasing_iter, NumBlk, 1 };
	dim3 threadIdx;		// blockDim = {cu_bsize, 1, 1};
	for (blockIdx.x = 0; blockIdx.x < gridDim.x; ++blockIdx.x) // FOR EACH phasing iteration
	for (blockIdx.y = 0; blockIdx.y < gridDim.y; ++blockIdx.y) // FOR EACH haplotype block
	for (blockIdx.z = 0; blockIdx.z < gridDim.z; ++blockIdx.z) // 1
		for (threadIdx.x = 0; threadIdx.x < blockDim.x; ++threadIdx.x) // one Population is processed by multiple threads
#endif
		{

			BlockType block = d_blocks[blockIdx.y];
			IndvDType *Population = d_Population + (blockIdx.x * NumBlk + blockIdx.y) * POPSIZE;
			char *Pop_seq = d_Pop_seq + (blockIdx.x * NumPos + block.start_pos) * POPSIZE;;
			const uint *GAcnt = d_GAcnt + (blockIdx.x * NumPos + block.start_pos);

			uint rid = blockIdx.x * gridDim.y + blockIdx.y;

#if __CUDA__ == 1
			__shared__	uint localSeed;

			if (threadIdx.x == 0)
				localSeed = curand(&d_randstates[rid]);

			__syncthreads();

			curandState localState;
			curand_init(localSeed, threadIdx.x, 0, &localState);
#endif

			for (uint indv = (POPSIZE - OFFSIZE); indv < POPSIZE; ++indv) {
				char *indv_seq = Pop_seq + Population[indv].stx_pos;

				for (uint i = threadIdx.x; i < block.length; i += blockDim.x) { // position inside of haplotype block
					double prob = (double)GAcnt[i] / (POPSIZE - OFFSIZE);

					if (curand_uniform(&localState) < prob)
						indv_seq[i] = '1';
					else
						indv_seq[i] = '0';
				}
			}

		}
}


#if __CUDA__ == 0
void GA_cnt_comp(const dim3 gridDim, const dim3 blockDim,
#elif __CUDA__ == 1
__global__ void GA_cnt_comp(
#endif
	IndvDType *d_Population, char *d_Pop_seq, uint *d_GAcnt,
	const BlockType *d_blocks, const int NumPos, const int NumBlk)
{
#if __CUDA__ == 0
	dim3 blockIdx;		// gridDim = gDim_PBx { phasing_iter, NumBlk, 1 };
	dim3 threadIdx;		// blockDim = {cu_bsize, 1, 1};
	for (blockIdx.x = 0; blockIdx.x < gridDim.x; ++blockIdx.x) // FOR EACH phasing iteration
	for (blockIdx.y = 0; blockIdx.y < gridDim.y; ++blockIdx.y) // FOR EACH haplotype block
	for (blockIdx.z = 0; blockIdx.z < gridDim.z; ++blockIdx.z) // 1
		for (threadIdx.x = 0; threadIdx.x < blockDim.x; ++threadIdx.x) // one block is processed by multiple threads
#endif
		{

			BlockType block = d_blocks[blockIdx.y];
			IndvDType *Population = d_Population + (blockIdx.x * NumBlk + blockIdx.y) * POPSIZE;
			char *Pop_seq = d_Pop_seq + (blockIdx.x * NumPos + block.start_pos) * POPSIZE;;
			uint *GAcnt = d_GAcnt + (blockIdx.x * NumPos + block.start_pos);


			for (uint i = threadIdx.x; i < block.length; i += blockDim.x) {	// position inside of haplotype block
				GAcnt[i] = 0;
				for (uint j = 0; j < POPSIZE - OFFSIZE; ++j) {
					char *indv_seq = Pop_seq + Population[j].stx_pos;
					GAcnt[i] += indv_seq[i] - '0';
				}
			}

		}
}

#if __CUDA__ == 0
void GA_pop_eval(const dim3 gridDim, const dim3 blockDim,
#elif __CUDA__ == 1
__global__ void GA_pop_eval(
#endif
	IndvDType *d_Population, char *d_Pop_seq,
	const char *AlleleData, const double *QualData, const SubFragType *SubFragments, const FragType *Fragments,
	const BlockType *d_blocks, const uint NumPos, const uint NumBlk, const uint num_indv_per_ph)
{
#if __CUDA__ == 0
	dim3 blockIdx;		// gridDim = gDim_PBpth{ phasing_iter, NumBlk, 1};
	dim3 threadIdx;		// blockDim = {cu_bsize, 1, 1};
	for (blockIdx.x = 0; blockIdx.x < gridDim.x; ++blockIdx.x) // FOR EACH phasing iteration
	for (blockIdx.y = 0; blockIdx.y < gridDim.y; ++blockIdx.y) // FOR EACH haplotype block 
	for (blockIdx.z = 0; blockIdx.z < gridDim.z; ++blockIdx.z) // 1
		for (threadIdx.x = 0; threadIdx.x < blockDim.x; ++threadIdx.x) 	// one Population is processed by multiple threads
#endif
		{

			BlockType block = d_blocks[blockIdx.y];
			IndvDType *Population = d_Population + (blockIdx.x * NumBlk + blockIdx.y) * POPSIZE;
			char *Pop_seq = d_Pop_seq + (blockIdx.x * NumPos + block.start_pos) * POPSIZE;;

			Population += (POPSIZE - num_indv_per_ph); 

			for (uint i = threadIdx.x; i < num_indv_per_ph; i += blockDim.x) 	// relative indv_id in cur pop
				calc_sumD(&Population[i].sumD, Pop_seq + Population[i].stx_pos,
					NULL, block, false,	AlleleData, QualData, SubFragments, Fragments);

		}
}

#if __CUDA__ == 0
void GA_pop_sort(const dim3 gridDim, const dim3 blockDim,
#elif __CUDA__ == 1
__global__ void GA_pop_sort(
#endif
	IndvDType *d_Population, const uint NumBlk)
{
#if __CUDA__ == 0
	dim3 blockIdx;		// gridDim = gDim_PBx{ phasing_iter, NumBlk, 1 };
	dim3 threadIdx;		// blockDim = {1, 1, 1};
	for (blockIdx.x = 0; blockIdx.x < gridDim.x; ++blockIdx.x) // FOR EACH phasing iteration
	for (blockIdx.y = 0; blockIdx.y < gridDim.y; ++blockIdx.y) // FOR EACH haplotype block 
	for (blockIdx.z = 0; blockIdx.z < gridDim.z; ++blockIdx.z) // 1
		for (threadIdx.x = 0; threadIdx.x < blockDim.x; ++threadIdx.x) 	// one Population is processed by multiple threads
#endif
		{

			IndvDType *Population = d_Population + (blockIdx.x * NumBlk + blockIdx.y) * POPSIZE;

			thrust::sort(Population, Population + POPSIZE, compare_sumD_val);

		}
}

//gathering the best haplotype in the population of each phasing iteration
#if __CUDA__ == 0
void gather_pop0(const dim3 gridDim, const dim3 blockDim,
#elif __CUDA__ == 1
__global__ void gather_pop0(
#endif
	double *d_Tog_sumD, char *d_Tog_seq,
	const IndvDType *d_Population, const char *d_Pop_seq, const BlockType *d_blocks,
	const uint NumPos, const uint NumBlk, const uint MaxBlkLen)
{
#if __CUDA__ == 0
	dim3 blockIdx;		// gridDim = gDim_PB_{ phasing_iter, NumBlk, 1};
	dim3 threadIdx;		// blockDim = {cu_bsize, 1, 1};
	for (blockIdx.x = 0; blockIdx.x < gridDim.x; ++blockIdx.x)  // FOR EACH phasing iteration
	for (blockIdx.y = 0; blockIdx.y < gridDim.y; ++blockIdx.y)  // FOR EACH haplotype block
	for (blockIdx.z = 0; blockIdx.z < gridDim.z; ++blockIdx.z) // 1
		for (threadIdx.x = 0; threadIdx.x < blockDim.x; ++threadIdx.x) 	// each block is grouped by thread
#endif
		{

			BlockType block = d_blocks[blockIdx.y];
			const IndvDType *Population = d_Population + (blockIdx.x * NumBlk + blockIdx.y) * POPSIZE;
			const char *Pop_seq = d_Pop_seq + (blockIdx.x * NumPos + block.start_pos) * POPSIZE;;
			double *Tog_sumD = d_Tog_sumD + blockIdx.x * NumBlk + blockIdx.y;
			char *Tog_seq = d_Tog_seq + blockIdx.x * NumPos + block.start_pos;

			const char *pop0_seq = Pop_seq + Population[0].stx_pos;

			for (uint i = threadIdx.x; i < block.length; i += blockDim.x)
				Tog_seq[i] = pop0_seq[i];

			if (threadIdx.x == 0)
				*Tog_sumD = Population[0].sumD;			// copy sumD

		}
}

//
// Genetic algorithm (EDA)
// A chromosome in GA is a haplotype sequence.
// the fitness values used in GA are w-MEC scores.
//
void GA(IndvDType *d_Population, char *d_Pop_seq, uint *d_GAcnt, curandState_t *d_randstates,
	double *d_Tog_sumD, char *d_Tog_seq,
	const char *d_AlleleData, const double *d_QualData,
	const SubFragType *d_SubFragments, const FragType *d_Fragments,	const BlockType *d_blocks,
	const uint NumPos, const uint NumBlk, const uint MaxBlkLen, const uint phasing_iter)
{

	uint cu_bsize = CU_MAX_BSIZE;

	dim3 bDim = { cu_bsize, 1 , 1 };

	dim3 gDim_PBx = { phasing_iter, NumBlk, 1 };

	uint seed = chrono::system_clock::now().time_since_epoch().count();

#if __CUDA__ == 0
	init_randstates(1, bDim,
#elif __CUDA__ == 1
	init_randstates << < gDim_PBx, 1 >> > (		// initialize random seeds
#endif
		seed, d_randstates);

	cudaCheckSync("GA_randstates");


#if __CUDA__ == 0
	GA_init(gDim_PBx, { 1,1,1 },
#elif __CUDA__ == 1
	GA_init << < gDim_PBx, bDim >> > (			// almost same
#endif
		d_Population, d_Pop_seq,
		d_blocks, NumPos, NumBlk, MaxBlkLen);

	cudaCheckSync("GA_init");


#if __CUDA__ == 0
	GA_1stGen(gDim_PBx, bDim,
#elif __CUDA__ == 1
	GA_1stGen << < gDim_PBx, bDim >> > (		// faster 
#endif
		d_Population, d_Pop_seq, d_randstates, d_blocks, NumPos, NumBlk);

	cudaCheckSync("GA_1stGen");


	gDim_PBx = { phasing_iter, NumBlk, 1 };
#if __CUDA__ == 0
	GA_pop_eval(gDim_PBx, bDim,
#elif __CUDA__ == 1
	GA_pop_eval << < gDim_PBx, bDim >> > (			// slower (a little)
#endif
		d_Population, d_Pop_seq,
		d_AlleleData, d_QualData, d_SubFragments, d_Fragments, d_blocks,
		NumPos, NumBlk, POPSIZE);

	cudaCheckSync("GA_pop_eval");


	gDim_PBx = { phasing_iter, NumBlk, 1 };
#if __CUDA__ == 0
	GA_pop_sort(gDim_PBx, { 1,1,1 },
#elif __CUDA__ == 1
	GA_pop_sort << < gDim_PBx, 1 >> > (		// faster
#endif
		d_Population, NumBlk);

	cudaCheckSync("GA_pop_sort");


	uint g_iter = G_ITER;		// generation number of GA
	while (g_iter--) {

			gDim_PBx = { phasing_iter, NumBlk, 1 };
#if __CUDA__ == 0
			GA_cnt_comp(gDim_PBx, bDim,
#elif __CUDA__ == 1
			GA_cnt_comp << < gDim_PBx, bDim >> > (		// faster (quite)
#endif
				d_Population, d_Pop_seq, d_GAcnt, d_blocks, NumPos, NumBlk);

			cudaCheckSync("GA_cnt_comp");


#if __CUDA__ == 0
			GA_nextGen(gDim_PBx, bDim,
#elif __CUDA__ == 1
			GA_nextGen << < gDim_PBx, bDim >> > (	// faster (very much X6)
#endif
				d_Population, d_Pop_seq, d_randstates,
				d_GAcnt, d_blocks, NumPos, NumBlk, g_iter );
			
			cudaCheckSync("GA_nextGen");


			gDim_PBx = { phasing_iter, NumBlk, 1 };
#if __CUDA__ == 0
			GA_pop_eval(gDim_PBx, bDim,
#elif __CUDA__ == 1
			GA_pop_eval << < gDim_PBx, bDim >> > (
#endif
				d_Population, d_Pop_seq, 
				d_AlleleData, d_QualData, d_SubFragments, d_Fragments, d_blocks,
				NumPos, NumBlk, OFFSIZE);

			cudaCheckSync("GA_pop_eval");


			gDim_PBx = { phasing_iter, NumBlk, 1 };
#if __CUDA__ == 0
			GA_pop_sort(gDim_PBx, { 1,1,1 },
#elif __CUDA__ == 1
			GA_pop_sort << < gDim_PBx, 1 >> > (		// faster
#endif
				d_Population, NumBlk);

			cudaCheckSync("GA_pop_sort");

		} // end_while (g_iter--)



	gDim_PBx = { phasing_iter, NumBlk, 1 };
#if __CUDA__ == 0
	gather_pop0(gDim_PBx, bDim,
#elif __CUDA__ == 1
	gather_pop0 << < gDim_PBx, bDim >> > (
#endif
		d_Tog_sumD, d_Tog_seq, d_Population, d_Pop_seq,
		d_blocks, NumPos, NumBlk, MaxBlkLen	);

	cudaCheckSync("GA_gater_pop0");

}

//////////////////////////////////////////////////////////////////////////


// 
// Toggling for range switch
// return 1 if better soultion is found
//
#if __CUDA__ == 1
// this function is run by MULTI thread
__device__
#endif
int range_switch_toggling_thread(
#if __CUDA__ == 0
	const dim3 gridDim, const dim3 blockDim,
#endif
	double *haplo_sumD, char *haplo_seq, DFragType *DFrag, const BlockType &block,
	double sh_bestSum[], int sh_bestPos[],
	const char *d_AlleleData, const double *d_QualData, const SubFragType *d_SubFragments, const FragType *d_Fragments,
	const FragsForPosType *d_FragsForPos, const uint NumFrag, const uint phasing_iter)
{
#if __CUDA__ == 0
	dim3 threadIdx = { 1, 1, 1 };
#endif

	uint thIdx = threadIdx.x;	// index for shared memory
	double bestSum, tmpSum;
	int bestPos, rval = 0;
	bool isImp;

	do {

#if __CUDA__ == 0
		for (threadIdx.x = 0; threadIdx.x < blockDim.x; ++threadIdx.x) {	// position is grouped by thread
			thIdx = threadIdx.x;
#endif

			// 1. compute sumD & DFrag;
			isImp = false;
			if (thIdx == 0) {
				calc_sumD(haplo_sumD, haplo_seq, DFrag, block, true,
					d_AlleleData, d_QualData, d_SubFragments, d_Fragments);	// update DFrag array for the current haplotype sequence
			}

#if __CUDA__ == 0
		}
#elif __CUDA__ == 1
			__syncthreads();
#endif


#if __CUDA__ == 0
		for (threadIdx.x = 0; threadIdx.x < blockDim.x; ++threadIdx.x) {	// position is grouped by thread
			thIdx = threadIdx.x;
#endif

			// 2. compute sumD in each toggled position계산 해서 최소값 찾기 (paralled by thread)
			bestSum = *haplo_sumD;
			bestPos = -1;

			for (uint i = thIdx; i < block.length; i += blockDim.x) {

				tmpSum = calc_sumD_range_tog(haplo_sumD, haplo_seq, DFrag, block, i, false,
					d_AlleleData, d_QualData, d_SubFragments, d_Fragments, d_FragsForPos); // DFrag array & haplo are NOT updated

				if (bestSum > tmpSum) {
					bestSum = tmpSum;
					bestPos = i;
				}
			}

			sh_bestSum[thIdx] = bestSum;
			sh_bestPos[thIdx] = bestPos;

#if __CUDA__ == 0
		}
#elif __CUDA__ == 1
			__syncthreads();
#endif


		// 3. find the best sumD in all threads
		for (uint i = blockDim.x / 2; i != 0; i /= 2) {
#if __CUDA__ == 0
			for (threadIdx.x = 0; threadIdx.x < blockDim.x; ++threadIdx.x) {	// position is grouped by thread
				thIdx = threadIdx.x;
#endif

				if (thIdx < i)
					if (sh_bestSum[thIdx] > sh_bestSum[thIdx + i]) {
						sh_bestSum[thIdx] = sh_bestSum[thIdx + i];
						sh_bestPos[thIdx] = sh_bestPos[thIdx + i];
					}

#if __CUDA__ == 0
			}
#elif __CUDA__ == 1
				__syncthreads();
#endif
		}

#if __CUDA__ == 0
		for (threadIdx.x = 0; threadIdx.x < blockDim.x; ++threadIdx.x) {	// position is grouped by thread
			thIdx = threadIdx.x;
#endif

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

#if __CUDA__ == 0
		}
#elif __CUDA__ == 1
			__syncthreads();
#endif

	} while (isImp);		// same values in all threads

	return rval;			// the same values in all threads
}

// 
// Toggling for range switch
// return 1 if better soultion is found
//
#if __CUDA__ == 1
__device__
#endif
int single_switch_toggling_thread(
#if __CUDA__ == 0
	const dim3 gridDim, const dim3 blockDim,
#endif
	double *haplo_sumD, char *haplo_seq, DFragType *DFrag, const BlockType &block,
	double sh_bestSum[], int sh_bestPos[],
	const char *d_AlleleData, const double *d_QualData, const SubFragType *d_SubFragments, const FragType *d_Fragments,
	const FragsForPosType *d_FragsForPos, const uint NumFrag, const uint phasing_iter)
{
#if __CUDA__ == 0
	dim3 threadIdx = { 1, 1, 1 };
#endif

	uint thIdx = threadIdx.x;	// index for shared memory
	double bestSum, tmpSum;
	int bestPos, rval = 0;
	bool isImp;

	do {

#if __CUDA__ == 0
		for (threadIdx.x = 0; threadIdx.x < blockDim.x; ++threadIdx.x) {	// position is grouped by thread
			thIdx = threadIdx.x;
#endif

			// 1. compute sumD & DFrag;
			isImp = false;
			if (thIdx == 0) {
				calc_sumD(haplo_sumD, haplo_seq, DFrag, block, true,
					d_AlleleData, d_QualData, d_SubFragments, d_Fragments);	// update DFrag array for the current haplotype sequence
			}

#if __CUDA__ == 0
		}
#elif __CUDA__ == 1
			__syncthreads();
#endif

#if __CUDA__ == 0
		for (threadIdx.x = 0; threadIdx.x < blockDim.x; ++threadIdx.x) {	// position is grouped by thread
			thIdx = threadIdx.x;
#endif

			// 2. compute sumD in each toggled position계산 해서 최소값 찾기 (paralled by thread)
			bestSum = *haplo_sumD;
			bestPos = -1;

			for (uint i = thIdx; i < block.length; i += blockDim.x) {

				tmpSum = calc_sumD_single_tog(haplo_sumD, haplo_seq, DFrag, block, i, false,
					d_AlleleData, d_QualData, d_SubFragments, d_Fragments, d_FragsForPos); // DFrag array & haplo are NOT updated

				if (bestSum > tmpSum) {
					bestSum = tmpSum;
					bestPos = i;
				}
			}

			sh_bestSum[thIdx] = bestSum;
			sh_bestPos[thIdx] = bestPos;

#if __CUDA__ == 0
		}
#elif __CUDA__ == 1
			__syncthreads();
#endif


		// 3. find the best sumD in all threads
		for (uint i = blockDim.x / 2; i != 0; i /= 2) {
#if __CUDA__ == 0
			for (threadIdx.x = 0; threadIdx.x < blockDim.x; ++threadIdx.x) {	// position is grouped by thread
				thIdx = threadIdx.x;
#endif

				if (thIdx < i)
					if (sh_bestSum[thIdx] > sh_bestSum[thIdx + i]) {
						sh_bestSum[thIdx] = sh_bestSum[thIdx + i];
						sh_bestPos[thIdx] = sh_bestPos[thIdx + i];
					}

#if __CUDA__ == 0
			}
#elif __CUDA__ == 1
				__syncthreads();
#endif
		}

#if __CUDA__ == 0
		for (threadIdx.x = 0; threadIdx.x < blockDim.x; ++threadIdx.x) {	// position is grouped by thread
			thIdx = threadIdx.x;
#endif

			// 4. update haplotype with new best haplo
			bestSum = sh_bestSum[0];
			bestPos = sh_bestPos[0];
			if (*haplo_sumD - bestSum > EPSILON) {
				isImp = true;
				rval = 1;
				if (thIdx == 0)
					haplo_seq[bestPos] = (haplo_seq[bestPos] == '0') ? '1' : '0';	// toggling
			}

#if __CUDA__ == 0
		}
#elif __CUDA__ == 1
			__syncthreads();
#endif

	} while (isImp);		// same values in all threads

	return rval;			// the same values in all threads
}

#if __CUDA__ == 0
void Toggling(const dim3 gridDim, const dim3 blockDim,
#elif __CUDA__ == 1
// One PEATH-block is run by OEN CUDA-block,
// this function is run by MULTI thread
__global__ void Toggling(
#endif
	double *d_Tog_sumD, char *d_Tog_seq, DFragType *d_DFrag,
	const char *d_AlleleData, const double *d_QualData, const SubFragType *d_SubFragments, const FragType *d_Fragments,
	const BlockType *d_blocks, const FragsForPosType *d_FragsForPos,
	const uint NumPos, const uint NumFrag, const uint NumBlk, const uint phasing_iter)
{
#if __CUDA__ == 0
	dim3 blockIdx;		// gridDim = gGim_PBx{ phasing_iter, NumBlk, 1};
//	dim3 threadIdx;		// blockDim = {cu_bsize, 1, 1};
	for (blockIdx.x = 0; blockIdx.x < gridDim.x; ++blockIdx.x) // FOR EACH phasing iteration
	for (blockIdx.y = 0; blockIdx.y < gridDim.y; ++blockIdx.y) // FOR EACH haplotype block
	for (blockIdx.z = 0; blockIdx.z < gridDim.z; ++blockIdx.z) // 1
#endif
		{
			BlockType block = d_blocks[blockIdx.y];	// blockIdx.y is haplotype blk_id
			uint ph_id = blockIdx.x;
			double *haplo_sumD = d_Tog_sumD + ph_id * NumBlk + blockIdx.y;
			char *haplo_seq = d_Tog_seq + ph_id * NumPos + block.start_pos;
			DFragType *DFrag = d_DFrag + ph_id * NumFrag;


#if __CUDA__ == 1
			__shared__
#endif
				double sh_bestSum[CU_MAX_BSIZE];	// best sumD for each thread
#if __CUDA__ == 1
			__shared__
#endif
				int sh_bestPos[CU_MAX_BSIZE];		// position with best sumD for each thread


			uint tog_iter = TOG_ITER;			// toggling iteration number
			while (tog_iter--) {
				uint imp = 0;

				// Step 1: Range Switch Toggling
				imp += range_switch_toggling_thread(	// faster (quite)
#if __CUDA__ == 0
					gridDim, blockDim,
#endif
					haplo_sumD, haplo_seq, DFrag, block,
					sh_bestSum, sh_bestPos,
					d_AlleleData, d_QualData, d_SubFragments, d_Fragments, d_FragsForPos, NumFrag, NumFrag);


				// Step 2: Single Switch Toggling
				imp += single_switch_toggling_thread(
#if __CUDA__ == 0
					gridDim, blockDim,
#endif
					haplo_sumD, haplo_seq, DFrag, block,
					sh_bestSum, sh_bestPos,
					d_AlleleData, d_QualData, d_SubFragments, d_Fragments, d_FragsForPos, NumFrag, NumFrag);

				if (!imp) break;				// if not improved, stop
			}

		}
}


#if __CUDA__ == 0
void Find_BestHaplo(const dim3 gridDim, const dim3 blockDim,
#elif __CUDA__ == 1
__global__ void Find_BestHaplo(
#endif
	double *d_Tog_sumD, char *d_Tog_seq,
	const BlockType *block,
	const uint NumPos, const uint NumBlk, const uint phasing_iter)
{
#if __CUDA__ == 0
	dim3 blockIdx;		// gridDim = gDim_Bxx{ (NumBlk - 1) / bDim.x + 1, 1, 1 };
	dim3 threadIdx;		// blockDim = {cu_bsize, 1, 1};
	for (blockIdx.x = 0; blockIdx.x < gridDim.x; ++blockIdx.x)  // FOR EACH group of haplotype blocks
	for (blockIdx.y = 0; blockIdx.y < gridDim.y; ++blockIdx.y) // 1
	for (blockIdx.z = 0; blockIdx.z < gridDim.z; ++blockIdx.z) // 1
		for (threadIdx.x = 0; threadIdx.x < blockDim.x; ++threadIdx.x) // For EACH haplotype block in a thread-group
#endif
		{
			uint blk_id = blockIdx.x * blockDim.x + threadIdx.x;
			if (blk_id >= NumBlk) return;

			uint min_phid = 0;
			for (uint i = 1; i < phasing_iter; ++i)
				if (d_Tog_sumD[min_phid * NumBlk + blk_id] > d_Tog_sumD[i * NumBlk + blk_id])
					min_phid = i;

			d_Tog_sumD[blk_id] = d_Tog_sumD[min_phid * NumBlk + blk_id];	// copy to 0th phasing iteration
			memcpy(d_Tog_seq + block[blk_id].start_pos,
				d_Tog_seq + NumPos * min_phid + block[blk_id].start_pos,
				block[blk_id].length * sizeof(char));
		}
}


////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////

//
// haplotype phasing procedure for a block
//

void haplotype_phasing_iter(double *h_BestHaplo_sumD, char *h_BestHaplo_seq,
	const char *d_AlleleData, const double *d_QualData,
	const SubFragType *d_SubFragments, const FragType *d_Fragments,	const BlockType *d_blocks,
	const FragsForPosType *d_FragsForPos,
	IndvDType *d_Population, char *d_Pop_seq, uint *d_GAcnt,
	double *d_Tog_sumD, char *d_Tog_seq, DFragType *d_DFrag, curandState *d_randstates,
	const uint NumPos, const uint NumFrag, const uint NumBlk, const uint MaxBlkLen, const uint phasing_iter)
{

	///////////////////////////////////////////
	// Step 1: Genetic algorithm (EDA)
	///////////////////////////////////////////

	GA(d_Population, d_Pop_seq, d_GAcnt, d_randstates,
		d_Tog_sumD, d_Tog_seq,
		d_AlleleData, d_QualData, d_SubFragments, d_Fragments, d_blocks,
		NumPos, NumBlk, MaxBlkLen, phasing_iter);

	///////////////////////////////////////////
	// Step 2: Toggling heuristic
	///////////////////////////////////////////

	uint cu_bsize = CU_MAX_BSIZE;

	dim3 bDim = { cu_bsize, 1 , 1 };
	dim3 gDim_PBx = { phasing_iter, NumBlk, 1 };

#if __CUDA__ == 0
	Toggling(gDim_PBx, bDim,
#elif __CUDA__ == 1
	Toggling <<< gDim_PBx, bDim >>> (
#endif
		d_Tog_sumD, d_Tog_seq, d_DFrag,
		d_AlleleData, d_QualData, d_SubFragments, d_Fragments, d_blocks, d_FragsForPos,
		NumPos, NumFrag, NumBlk, phasing_iter);

	cudaCheckSync("Toggling");

	///////////////////////////////////////////
	// Step 3: Select Best Haplotype
	///////////////////////////////////////////

	dim3 gDim_Bxx = { (NumBlk - 1) / bDim.x + 1, 1, 1 };

#if __CUDA__ == 0
	Find_BestHaplo(gDim_Bxx, bDim,
#elif __CUDA__ == 1
	Find_BestHaplo <<< gDim_Bxx, bDim >>> (
#endif
		d_Tog_sumD, d_Tog_seq, d_blocks, NumPos, NumBlk, phasing_iter);


	// memory copy h_BestHaplo <- d_Tog
#if __CUDA__ == 0
	memcpy(h_BestHaplo_sumD, d_Tog_sumD, NumBlk * sizeof(double));
	memcpy(h_BestHaplo_seq, d_Tog_seq, NumPos * sizeof(char));
#elif __CUDA__ == 1
	cudaError_t cudaStatus;
	cudaStatus = cudaMemcpy(h_BestHaplo_sumD, d_Tog_sumD, NumBlk * sizeof(double), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "[phasing-Haplo_sumD] cudaMemcpy failed!");
	}
	cudaStatus = cudaMemcpy(h_BestHaplo_seq, d_Tog_seq, NumPos * sizeof(char), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "[phasing-Haplo_seq] cudaMemcpy failed!");
	}
#endif

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
	uint pos;

	for (uint i = 0; i < block.length; i++)
		Covered[i] = 0;

	startFrag = block.start_frag;
	endFrag = block.end_frag;
	uint blk_offset = block.start_pos;

	// finding uncovered positions
	for (uint i = startFrag; i <= endFrag; i++) 	// for each fragment
		for (uint j = 0; j < Fragments[i].num_subFrag; j++) { // for each subfragment

			uint subfragidx = Fragments[i].subFrag0 + j;
			uint offset_in_block = SubFragments[subfragidx].start_pos - blk_offset;

			for (uint k = 0; k < SubFragments[subfragidx].length; k++) {	// for each position
				pos = k + offset_in_block;
				Covered[pos] = 1;
			} // end_for k (each position)
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
	char *&AlleleData, double *&QualData, vector <SubFragType> *SubFragments, FragType *&Fragments,
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
	QualData = new double[fragDataSize / 2];
	uint offset = 0;		// offset in AlleleData & QualData;
	uint subfrag_idx = 0;	// index in SubFragments vector

	Fragments = new FragType[*NumFrag];

	SubFragments->clear();

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
			uint e_pos = s_pos + len - 1;

			SubFragments->push_back({ offset, s_pos, e_pos, len });
			++subfrag_idx;

			offset += len;	//for next allele sequence
		} // end_for j

		  // storing fragment metadata 
		uint first_subfrag = Fragments[i].subFrag0;
		uint last_subfrag = Fragments[i].subFrag0 + Fragments[i].num_subFrag - 1;
		Fragments[i].start_pos = (*SubFragments)[first_subfrag].start_pos;
		Fragments[i].end_pos = (*SubFragments)[last_subfrag].end_pos;
		Fragments[i].length = Fragments[i].end_pos - Fragments[i].start_pos + 1;

		p = strtok(NULL, "\n");

		// if fragment contains only one site, exclude the fragment in phasing
		if (Fragments[i].length == 1) {
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
			for (k = 0; k < (*SubFragments)[first_subfrag + j].length; k++) {
				uint qual_j = (uint)(*(p++));
				qual_j -= 33;
				QualData[start_offset + k] = pow(10.0, ((-1 * (double)qual_j) / 10.0));
			}

			//QualData[start_offset + k] = -1.0;	 // delimiter 

		} // end_for j

	} // end i

	*MatrixDataSize = offset;
	*NumSubFrag = (uint)SubFragments->size();

	//	sort(Fragments, Fragments + *NumFrag, compare_frag_pos);	// sorting with starting positions

	delete[] FileData;

#ifdef __DEBUG__
	// recording fragment data in file for checking if fragment data is well loaded
	ofstream fragmentOutFile;
	fragmentOutFile.open("ZfragmentOut.txt");

	for (uint i = 0; i < *NumFrag; i++) {
		fragmentOutFile << "start: " << Fragments[i].start_pos
			<< " end: " << Fragments[i].end_pos
			<< " length: " << Fragments[i].length
			<< " num_subFrag: " << Fragments[i].num_subFrag << endl;

		uint first_subfrag = Fragments[i].subFrag0;

		for (uint j = 0; j < Fragments[i].num_subFrag; j++) {
			fragmentOutFile << "    start: " << (*SubFragments)[first_subfrag + j].start_pos
				<< " end: " << (*SubFragments)[first_subfrag + j].end_pos
				<< " length: " << (*SubFragments)[first_subfrag + j].length
				//				<< " offset:" << SubFragments[first_subfrag + j].Moffset
				<< " str:" << AlleleData + (*SubFragments)[first_subfrag + j].Moffset
				<< " qual:" << QualData + (*SubFragments)[first_subfrag + j].Moffset << endl;
		}
		fragmentOutFile << "==============================================================" << endl;
	}
	fragmentOutFile.close();
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
	const FragType *Fragments, const uint NumPos, const uint NumFrag )
{

	// 1. build an array of fragment numbers locating at each position

	FragsForPos = new FragsForPosType[NumPos];

	//for (int i = 0; i < NumPos; i++)
	//FragsForPos[i] = { -1,-1 };

	int cutStart = 0;
	for (int i = 0; i < (int)NumFrag; i++) 		// storing starting fragment #
		for (; cutStart <= (int)Fragments[i].end_pos; cutStart++)
			//if (FragsForPos[cutStart].start_frag == -1)
			FragsForPos[cutStart].start_frag = i;

	int cutEnd = NumPos - 1;
	for (int i = NumFrag - 1; i >= 0; i--) 	// storing ending fragment #
		for (; cutEnd >= (int)Fragments[i].start_pos; cutEnd--)
			//if (FragsForPos[cutEnd].end_frag == -1)
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
	uint blkStartPos, blkLen;
	Blocks->clear();

	*MaxBlkLen = 0;

	uint i;
	for (i = 1; i < NumFrag; i++) {
		if (Fragments[i].start_pos > blkEndPos) {	// if frag[i] is NOT overlapped with the previous frags
			blkStartPos = Fragments[blkStartFrag].start_pos;
			blkLen = blkEndPos - blkStartPos + 1;
			Blocks->push_back({ blkStartFrag, i - 1, blkStartPos, blkLen }); // stroing current block info
			blkStartFrag = i;						// set the new starting fragment

			if (*MaxBlkLen < blkLen)	*MaxBlkLen = blkLen;
		}
		if( blkEndPos < Fragments[i].end_pos)
			blkEndPos = Fragments[i].end_pos; // update end position
	}

	blkStartPos = Fragments[blkStartFrag].start_pos;
	blkLen = blkEndPos - blkStartPos + 1;
	Blocks->push_back({ blkStartFrag, i - 1, blkStartPos, blkLen }); // for the last block
	if (*MaxBlkLen < blkLen)	*MaxBlkLen = blkLen;

	*NumBlk = (uint)Blocks->size();

#ifdef __DEBUG__
	// recording fragment numbers in each block
	ofstream blockFile;
	blockFile.open("ZBlockPos.txt");
	for (i = 0; i < Blocks->size(); i++) {
		blockFile << "[" << i + 1 << "] " << (*Blocks)[i].start_frag << " " << (*Blocks)[i].end_frag
			<< " " << (*Blocks)[i].start_pos << " " << (*Blocks)[i].length << endl;

	}
	blockFile.close();
#endif
}

#ifdef __DEBUG__
// recording fragment data in matrix form
void debug_build_matrix_map_file(
	char * AlleleData, SubFragType *SubFragments, FragType *Fragments, BlockType *Blocks,
	const uint NumPos, const uint NumBlk, const uint MaxBlkLen)
{
	ofstream fragmentForm("ZFragmentForm.SORTED");

	for (uint i = 0; i<NumBlk; i++) {

		fragmentForm << "block # : " << i + 1 << endl;

		for (uint j = Blocks[i].start_frag; j <= Blocks[i].end_frag; j++) {

			uint blk_offset = Blocks[i].start_pos;

			char * frag = new char[(Blocks[i].length + 1) * 2];
			for (uint k = 0; k<Blocks[i].length; k++)
				frag[2 * k] = ' ', frag[2 * k + 1] = '-';
			frag[Blocks[i].length * 2] = '\0';

			for (uint k = 0; k < Fragments[j].num_subFrag; k++) {
				uint subfragidx = Fragments[j].subFrag0 + k;
				uint idx = SubFragments[subfragidx].start_pos - blk_offset;
				char *subfrag_str = AlleleData + SubFragments[subfragidx].Moffset;
				for (uint l = 0; l < SubFragments[subfragidx].length; l++) {
					frag[2 * (idx + l) + 1] = subfrag_str[l];
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

	char *&d_AlleleData, double *&d_QualData, SubFragType *&d_SubFragments, FragType *&d_Fragments,
	BlockType *&d_Blocks, FragsForPosType *&d_FragsForPos,
	IndvDType *&d_Population, char *&d_Pop_seq, uint *&d_GAcnt,
	double *&d_Tog_sumD, char *&d_Tog_seq, DFragType *&d_DFrag, curandState *&d_randstates,
	char *h_AlleleData, double *h_QualData, SubFragType *h_SubFragments, FragType *h_Fragments,
	BlockType *h_Blocks, FragsForPosType *h_FragsForPos,
	uint MatrixDataSize, uint NumPos, uint NumSubFrag, uint NumFrag,
	uint NumBlk, uint MaxBlkLen, uint phasing_iter)

{

	// Memory allocation for host
	h_BestHaplo_sumD = new double[phasing_iter * NumBlk]; // [ph_id][blk_id]
	h_BestHaplo_seq = new char[phasing_iter * NumPos];   // [ph_id][pos_id]

	h_Covered = new char[MaxBlkLen + 1];
	h_BestHaplo2.seq = new char[(MaxBlkLen + 1)];
	h_BestHaplo2.seq2 = new char[(MaxBlkLen + 1)];




#if __CUDA__ == 0

	d_AlleleData = h_AlleleData;
	d_QualData = h_QualData;
	d_SubFragments = h_SubFragments;
	d_Fragments = h_Fragments;
	d_Blocks = h_Blocks;
	d_FragsForPos = h_FragsForPos;

	d_Tog_sumD = new double[phasing_iter * NumBlk];		// [ph_id][blk_id]
	d_Tog_seq = new char[phasing_iter * NumPos];		// [ph_id][pos_id]
	d_Population = new IndvDType[phasing_iter * NumBlk * POPSIZE];	//[ph_id][blk_id][POP_id]
	d_Pop_seq = new char[phasing_iter * NumPos * POPSIZE];		// [ph_idx][pos_idx][POP_idx]

	d_GAcnt = new uint[phasing_iter * NumPos];		// [ph_id][pos_id]
	d_DFrag = new DFragType[phasing_iter * NumFrag];	// [ph_id][frag_id]

	d_randstates = new curandState[phasing_iter * NumBlk];

#elif __CUDA__ == 1

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

	cudaStatus = cudaMalloc((void**)&d_QualData, MatrixDataSize * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "[QualData] cudaMalloc failed!");
	}
	cudaStatus = cudaMemcpy(d_QualData, h_QualData, MatrixDataSize * sizeof(double), cudaMemcpyHostToDevice);
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

	cudaStatus = cudaMalloc((void**)&d_Tog_sumD, phasing_iter * NumBlk * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "[Haplo_sumD]cudaMalloc failed!");
	}

	cudaStatus = cudaMalloc((void**)&d_Tog_seq, phasing_iter * NumPos * sizeof(char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "[Haplo_seq]cudaMalloc failed!");
	}

	cudaStatus = cudaMalloc((void**)&d_Population, phasing_iter * NumBlk * POPSIZE * sizeof(IndvDType));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "[PopSumD]cudaMalloc failed!");
	}

	cudaStatus = cudaMalloc((void**)&d_Pop_seq, phasing_iter * POPSIZE * NumPos * sizeof(char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "[Pop_seq]cudaMalloc failed!");
	}

	cudaStatus = cudaMalloc((void**)&d_GAcnt, phasing_iter * NumPos * sizeof(uint));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "[GAcnt]cudaMalloc failed!");
	}

	cudaStatus = cudaMalloc((void**)&d_DFrag, phasing_iter * NumFrag * sizeof(DFragType));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "[DFrag]cudaMalloc failed!");
	}

	cudaStatus = cudaMalloc((void**)&d_randstates, phasing_iter * NumBlk * sizeof(curandState));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "[RandStates]cudaMalloc failed!");
	}

#endif
}


void deallocate(char *h_AlleleData, double *h_QualData, FragType *h_Fragments, FragsForPosType *h_FragsForPos,
	double *h_BestHaplo_sumD, char *h_BestHaplo_seq, char *h_Covered, HaploType h_BestHaplo2,
	char *d_AlleleData, double *d_QualData, SubFragType *d_SubFragments, FragType *d_Fragments,
	BlockType *d_Blocks, FragsForPosType *d_FragsForPos,
	IndvDType *d_Population, char *d_Pop_seq,	uint *d_GAcnt,
	double *d_Tog_sumD, char *d_Tog_seq, DFragType *d_DFrag, curandState *d_randstates)
{
	// memory deallocation for host
	delete[] h_AlleleData, h_QualData, h_Fragments, h_FragsForPos;

	delete[] h_BestHaplo_sumD, h_BestHaplo_seq;

	delete[] h_BestHaplo2.seq, h_BestHaplo2.seq2;

	delete[] h_Covered;


	// memory deallocation for device
#if __CUDA__ == 0

	delete[] d_Tog_sumD;
	delete[] d_Tog_seq;
	delete[] d_Population;
	delete[] d_Pop_seq;

	delete[] d_GAcnt;
	delete[] d_DFrag;

	delete[] d_randstates;

#elif __CUDA__ == 1

	cudaFree(d_AlleleData);
	cudaFree(d_QualData);

	cudaFree(d_SubFragments);
	cudaFree(d_Fragments);
	cudaFree(d_Blocks);
	cudaFree(d_FragsForPos);

	cudaFree(d_Tog_sumD);
	cudaFree(d_Tog_seq);
	cudaFree(d_Population);
	cudaFree(d_Pop_seq);

	cudaFree(d_GAcnt);
	cudaFree(d_DFrag);

	cudaFree(d_randstates);
#endif

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
	double *h_QualData, *d_QualData;				// matrix quality data

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

	IndvDType *d_Population;		// haplotype sumD in population of GA
	char *d_Pop_seq;					// haplotype sequence in population of GA
	uint *d_GAcnt;					// array for storing the frequency of 1's for each position in GA

	double *d_Tog_sumD;				// haplotype sumD used in toggling
	char *d_Tog_seq;					// haplotype sequence used in toggling
	DFragType *d_DFrag;				// array for storing weighted MEC for each frament

	curandState *d_randstates = NULL;		// random number states used in GA

	// Variables for Size

	uint MatrixDataSize;				// matrix data size
	uint NumPos;						// number of positions
	uint NumSubFrag;					// total number of subfragments
	uint NumFrag;					// number of fragments
	uint NumBlk;						// number of blocks
	uint MaxBlkLen;					// maximum block length

	////////////////////////////////////////////////////////////////////////////

	// For execution time measurement
	std::chrono::system_clock::time_point start = std::chrono::system_clock::now();

	// load matrix data in input file into data structures
	load_matrixFile(matrixFileName,
		h_AlleleData, h_QualData, &h_SubFragments, h_Fragments,
		&MatrixDataSize, &NumPos, &NumSubFrag, &NumFrag);

	// build auxilary data structures
	build_aux_struct(&h_Blocks, h_FragsForPos, &NumBlk, &MaxBlkLen, h_Fragments, NumPos, NumFrag);

#ifdef __DEBUG__
	debug_build_matrix_map_file(h_AlleleData, &h_SubFragments[0], h_Fragments, &h_Blocks[0],
		NumPos, NumBlk, MaxBlkLen);
#endif

	// Memory allocation & memory copy from host to device
	allocate(h_BestHaplo_sumD, h_BestHaplo_seq, h_Covered, h_BestHaplo2,
		d_AlleleData, d_QualData, d_SubFragments, d_Fragments, d_Blocks, d_FragsForPos,
		d_Population, d_Pop_seq, d_GAcnt, d_Tog_sumD, d_Tog_seq, d_DFrag, d_randstates,

		h_AlleleData, h_QualData, &h_SubFragments[0], h_Fragments, &h_Blocks[0], h_FragsForPos,
		MatrixDataSize, NumPos, NumSubFrag, NumFrag, NumBlk, MaxBlkLen, phasing_iter);


	// haplotype phasing procedure running in device
	haplotype_phasing_iter(h_BestHaplo_sumD, h_BestHaplo_seq,
		d_AlleleData, d_QualData, d_SubFragments, d_Fragments, d_Blocks, d_FragsForPos,
		d_Population, d_Pop_seq, d_GAcnt, d_Tog_sumD, d_Tog_seq, d_DFrag, d_randstates,
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
		totalReads += h_Blocks[i].end_frag - h_Blocks[i].start_frag + 1;
		totalWMEC += h_BestHaplo2.sumD;
		totalMEC += mec;

		// ---------------------
		// 1. Printing Block Header
		// ---------------------
		phasingResultFile << "Block Number: " << i + 1;
		phasingResultFile << "  Block Length: " << h_Blocks[i].length;
		phasingResultFile << "  Phased Length: " << phased;
		phasingResultFile << "  Number of Reads: " << h_Blocks[i].end_frag - h_Blocks[i].start_frag + 1;
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
		d_AlleleData, d_QualData, d_SubFragments, d_Fragments, d_Blocks, d_FragsForPos,
		d_Population, d_Pop_seq, d_GAcnt, d_Tog_sumD, d_Tog_seq, d_DFrag, d_randstates);

	std::chrono::system_clock::time_point end = std::chrono::system_clock::now();
	std::chrono::duration<double> DefaultSec = end - start;

//	std::cout << "   " << DefaultSec.count();
//	std::cerr << "   " << DefaultSec.count();

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

}

////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////

//
// main function
// 1st argument : input file name
// 2nd argument : output filne name

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



	procedure(matrixFileName, outputFileName, phasing_iter);		//// main procedure

#if __CUDA__ == 1
//	cout << endl << "-- Run by CUDA!!! -- cuda block size : " << CU_MAX_BSIZE << endl << endl;
#endif

	return 0;
}

