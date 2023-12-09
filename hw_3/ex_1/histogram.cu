/*
 * Histograming using CUDA
 *
 * This program demonstrates parallel array histograming on the GPU
 * using CUDA.
 *
 * Compilation: $ nvcc histogram.cu -o histogram
 *
 * Execution: $ ./histogram.exe <array_length>
 *
 * Parameters: <array_length> - Length of the integer array to be histogramed.
 *
 * Profiling with Nvidia Nsight:
 *   1. Compile the code with profiling information:
 *      $ nvcc -lineinfo histogram.cu -o histogram
 *
 *   2. Run the executable with Nvidia Nsight profiling:
 *      $ ncu -o histogram_profile -f ./histogram.exe <array_length>
 *
 *   3. Analyze the profiling results using Nvidia Nsight Compute:
 *      $ ncu-ui ./histogram_profile.ncu-rep
 *
 * Note: CUDA toolkit must be installed and configured for compilation.
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <random>
#include <cuda_runtime.h>

#define NUM_BINS 4096
#define HISTOGRAM_BLOCK_SIZE 256
#define SATURATE_BLOCK_SIZE 256

double start, stop;

/// @brief Starts the timer.
void startTimer()
{
    start = (double)clock();
    start = start / CLOCKS_PER_SEC;
}

/// @brief Stops the timer and print the elapsed time.
void stopTimer(const char* message)
{
    stop = (double)clock();
    stop = stop / CLOCKS_PER_SEC;

    double elapsedTime = (stop - start) * 1.0e3;
    printf("%s: %.6f ms\n", message, elapsedTime);
}

/// @brief Naive CUDA kernel to compute array histogram.
__global__ void naiveHistogramKernel(unsigned int* _input, unsigned int* _bins, unsigned int _numElements, unsigned int _numBins)
{
    int globalThreadIdx = blockDim.x * blockIdx.x + threadIdx.x;
    int globalThreadCount = blockDim.x * gridDim.x;

    // The for loop ensures the array is fully processed, even if 
    // the number of allocated threads is inferior to _numElements.
    for (int i = globalThreadIdx; i < _numElements;  i+= globalThreadCount)
    {
        unsigned int binValue = _input[i];
        atomicAdd(&_bins[binValue], 1);
    }
}

/// @brief CUDA kernel to saturate the histogram bins.
__global__ void saturateKernel(unsigned int* _bins, unsigned int _numBins)
{
    int globalThreadIdx = blockDim.x * blockIdx.x + threadIdx.x;
    int globalThreadCount = blockDim.x * gridDim.x;

    // The for loop ensures all histogram bins are saturated, even if 
    // the number of allocated threads is inferior to _numBins.
    for (int i = globalThreadIdx; i < _numBins; i += globalThreadCount)
    {
        if (_bins[i] > 127)
        {
            _bins[i] = 127;
        }
    }
}

/// @brief Entry point of the program.
int main(int _argc, char** _argv)
{
    if (_argc != 2)
    {
        fprintf(stderr, "Incorrect input, usage is: ./histogram.exe <array length>\n");
        exit(EXIT_FAILURE);
    }

    // Retrieves array length from the cmd line.
    unsigned int arrayLength = atoi(_argv[1]);
    const int sizeofInput = arrayLength * sizeof(unsigned int);
    const int sizeofBins = NUM_BINS * sizeof(unsigned int);

    unsigned int* hostInput = (unsigned int*)malloc(sizeofInput);
    unsigned int* hostBins  = (unsigned int*)malloc(sizeofBins);
    unsigned int* resultRef = (unsigned int*)malloc(sizeofBins);

    // Init result ref with 0
    memset(resultRef, 0, sizeofBins);

    // Fills input array with random integers in range [0, NUM_BINS - 1], and
    // pre-compute histogram to use as a reference when validating GPU result.
    for (int i = 0; i < arrayLength; ++i)
    {
        unsigned int randValue = rand() % NUM_BINS;
        hostInput[i] = randValue;

        if (resultRef[randValue] < 127)
        {
            resultRef[randValue]++;
        }
    }

    unsigned int* deviceInput;
    unsigned int* deviceBins;

    // Allocates GPU memory.
    cudaMalloc((void**)&deviceInput, sizeofInput);
    cudaMalloc((void**)&deviceBins, sizeofBins);

    // Copies array input to the GPU and initializes bins output to 0.
    cudaMemcpy(deviceInput, hostInput, sizeofInput, cudaMemcpyHostToDevice);
    cudaMemset(deviceBins, 0, sizeofBins);

    // Computes 1D thread grid dimensions adapted to the size of the input array.
    const int histBlockSize = HISTOGRAM_BLOCK_SIZE;
    const int histGridSize = (arrayLength + histBlockSize - 1) / histBlockSize;

    // Profiling scope: histogram kernel
    startTimer();
    {
        // Runs histogram GPU Kernel.
        naiveHistogramKernel<<<histGridSize, histBlockSize>>>(deviceInput, deviceBins, arrayLength, NUM_BINS);
        cudaDeviceSynchronize();

        cudaError_t cudaError = cudaGetLastError();
        if (cudaError != cudaSuccess)
        {
            fprintf(stderr, "CUDA error for histogram kernel: %s\n", cudaGetErrorString(cudaError));
            exit(EXIT_FAILURE);
        }
    }
    stopTimer("Histogram Kernel Time");

    // Computes 1D thread grid dimensions adapated to the number of bins.
    const int saturateBlockSize = SATURATE_BLOCK_SIZE;
    const int saturateGridSize = (NUM_BINS + saturateBlockSize - 1) / saturateBlockSize;

    // Profiling scope: saturate kernel
    startTimer();
    {
        // Runs saturate GPU Kernel.
        saturateKernel<<<saturateGridSize, saturateBlockSize>>>(deviceBins, NUM_BINS);
        cudaDeviceSynchronize();

        cudaError_t cudaError = cudaGetLastError();
        if (cudaError != cudaSuccess)
        {
            fprintf(stderr, "CUDA error for saturate kernel: %s\n", cudaGetErrorString(cudaError));
            exit(EXIT_FAILURE);
        }
    }
    stopTimer("Saturate Kernel Time");

    // Copies the GPU memory back to the CPU.
    cudaMemcpy(hostBins, deviceBins, sizeofBins, cudaMemcpyDeviceToHost);

    // Compares result with the reference.
    for (int i = 0; i < NUM_BINS; ++i)
    {
        if (hostBins[i] != resultRef[i])
        {
            fprintf(stderr, "Result mismatch for integer %d: %u != %u\n", i, hostBins[i], resultRef[i]);
            break;
        }
    }

    // Deallocates GPU memory.
    cudaFree(deviceInput);
    cudaFree(deviceBins);

    // Deallocates CPU memory.
    free(hostInput);
    free(hostBins);
    free(resultRef);

    return EXIT_SUCCESS;
}
