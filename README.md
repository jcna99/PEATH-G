# PEATH/G: Fast Single-Individual Haplotyping method using GPGPU

PEATH/G is a GPGPU version of PEATH, a novel SIH algorithm based on the estimation of distribution algorithm (EDA).
It implementes the method proposed in:
```
J.C. Na et al., PEATH/G: Fast Single-Individual Haplotyping method using GPGPU, submitted to TBC/BIOINFO 2018.
```

## System Environment 

We have tested for compiling and running the code in conventional desktop computer as follows:
Hardware: Intel Core i5, 16GB RAM, and NVIDIA GeForce GTX950 graphics card.
OS: Windows 10 64-bits
IDE: Visual Studio 2017 integrated with CUDA toolkit 9.2

## Compiling PEATH/G

After downloading .cu code, complie the source code using NVCC compiler

```
nvcc -o PEATH-G PEATH-G.cu
```

## Running PEATH/G

To run PEATH/G, use the following command:

```
./PEATH-G <input_file> <output_file> (param)
```

<input_file> is an input matrix for sequence reads and
<output_file> contains phased haplotype.

(param) is an optional parameter for time/accuracy tradeoff which is a positive integer (default: 50).

```
ex) ./PEATH-G chr1.matrix.SORTED chr1.haplo
```

