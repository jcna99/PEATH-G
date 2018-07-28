# PEATH/G: Fast Single-Individual Haplotyping method using GPGPU

PEATH/G is a GPGPU version of PEATH, a novel SIH algorithm based on the estimation of distribution algorithm (EDA).
It implementes the method proposed in:
```
J.C. Na et al., PEATH/G: Fast Single-Individual Haplotyping method using GPGPU, submitted to TBC/BIOINFO 2018.
```

## Compiling PEATH/G

After downloading .cu code, complie the source code using NVCC compiler

```
nvcc -gencode=arch=compute_30,code=\"sm_30,compute_30\" --use-local-env -ccbin "C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Tools\MSVC\14.13.26128\bin\HostX86\x64" -x cu  -I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.2\include" -I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.2\include"     --keep-dir x64\Release -maxrregcount=0  --machine 64 --compile -cudart static     -DWIN32 -DWIN64 -D_CRT_SECURE_NO_WARNINGS -DNDEBUG -D_CONSOLE -D_MBCS -Xcompiler "/EHsc /W3 /nologo /O2 /FS /Zi  /MD " -o x64\Release\PEATH-cuda.cu.obj PEATH-cuda.cu
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
