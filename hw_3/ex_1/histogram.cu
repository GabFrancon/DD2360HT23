
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <random>
#include <cuda_runtime.h>

#define NUM_BINS 4096

__global__ void histogramKernel(unsigned int* _input, unsigned int* _bins, unsigned int _numElements, unsigned int _numBins)
{
    //@@ Insert code below to compute histogram of input using shared memory and atomics

}

__global__ void convertKernel(unsigned int* _bins, unsigned int _numBins)
{
    //@@ Insert code below to clean up bins that saturate at 127

}


int main(int _argc, char** _argv)
{

    int inputLength;
    unsigned int *hostInput;
    unsigned int *hostBins;
    unsigned int *resultRef;
    unsigned int *deviceInput;
    unsigned int *deviceBins;

    //@@ Insert code below to read in inputLength from args

    printf("The input length is %d\n", inputLength);


    //@@ Insert code below to allocate Host memory for input and output


    //@@ Insert code below to initialize hostInput to random numbers whose values range from 0 to (NUM_BINS - 1)


    //@@ Insert code below to create reference result in CPU


    //@@ Insert code below to allocate GPU memory here


    //@@ Insert code to Copy memory to the GPU here


    //@@ Insert code to initialize GPU results


    //@@ Initialize the grid and block dimensions here


    //@@ Launch the GPU Kernel here


    //@@ Initialize the second grid and block dimensions here


    //@@ Launch the second GPU Kernel here


    //@@ Copy the GPU memory back to the CPU here


    //@@ Insert code below to compare the output with the reference


    //@@ Free the GPU memory here


    //@@ Free the CPU memory here


    return EXIT_SUCCESS;
}

