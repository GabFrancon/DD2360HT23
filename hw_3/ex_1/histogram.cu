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
#define THREADS_PER_BLOCK 256
#define HISTOGRAM_MAX_VALUE 127

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

/// @brief Naive CUDA kernel to compute histogram of given _elements.
__global__ void naiveHistogramKernel(unsigned int* _elements, unsigned int* _histogram, unsigned int _numElements)
{
    int globalThreadIdx = blockDim.x * blockIdx.x + threadIdx.x;
    int globalThreadCount = blockDim.x * gridDim.x;

    // The for loop ensures the array is fully processed, even if 
    // the number of allocated threads is inferior to _numElements.
    for (int i = globalThreadIdx; i < _numElements;  i+= globalThreadCount)
    {
        unsigned int binValue = _elements[i];
        atomicAdd(&_histogram[binValue], 1);
    }
}

/// @brief Naive CUDA kernel to saturate _histogram with HISTOGRAM_MAX_VALUE.
__global__ void naiveSaturateKernel(unsigned int* _histogram)
{
    int globalThreadIdx = blockDim.x * blockIdx.x + threadIdx.x;
    int globalThreadCount = blockDim.x * gridDim.x;

    // The for loop ensures all histogram bins are saturated, even if 
    // the number of allocated threads is inferior to NUM_BINS.
    for (int i = globalThreadIdx; i < NUM_BINS; i += globalThreadCount)
    {
        _histogram[i] = max(_histogram[i], HISTOGRAM_MAX_VALUE);
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
    const int sizeofHistogram = NUM_BINS * sizeof(unsigned int);

    unsigned int* hostInput = (unsigned int*)malloc(sizeofInput);
    unsigned int* hostHistogram = (unsigned int*)malloc(sizeofHistogram);
    unsigned int* refHistogram = (unsigned int*)malloc(sizeofHistogram);

    // Initializes reference histogram with 0.
    memset(refHistogram, 0, sizeofHistogram);

    // Fills input array with random integers in range [0, NUM_BINS - 1], and
    // pre-compute histogram to use as a reference when validating GPU result.
    for (int i = 0; i < arrayLength; ++i)
    {
        unsigned int randValue = rand() % NUM_BINS;
        hostInput[i] = randValue;

        if (refHistogram[randValue] < HISTOGRAM_MAX_VALUE)
        {
            refHistogram[randValue]++;
        }
    }

    unsigned int* deviceInput;
    unsigned int* deviceHistogram;

    // Allocates GPU memory.
    cudaMalloc((void**)&deviceInput, sizeofInput);
    cudaMalloc((void**)&deviceHistogram, sizeofHistogram);

    // Copies array input to the GPU and initializes bins output to 0.
    cudaMemcpy(deviceInput, hostInput, sizeofInput, cudaMemcpyHostToDevice);
    cudaMemset(deviceHistogram, 0, sizeofHistogram);

    // Adapts 1D thread grid dimensions to the size of the input array.
    const int histogramBlockSize = THREADS_PER_BLOCK;
    const int histogramGridSize = (arrayLength + histogramBlockSize - 1) / histogramBlockSize;

    // Profiling scope: histogram kernel
    startTimer();
    {
        // Runs histogram GPU Kernel.
        naiveHistogramKernel<<<histogramGridSize, histogramBlockSize>>>(deviceInput, deviceHistogram, arrayLength);
        cudaDeviceSynchronize();

        cudaError_t cudaError = cudaGetLastError();
        if (cudaError != cudaSuccess)
        {
            fprintf(stderr, "CUDA error for histogram kernel: %s\n", cudaGetErrorString(cudaError));
            exit(EXIT_FAILURE);
        }
    }
    stopTimer("Histogram Kernel Time");

    // Adapts 1D thread grid dimensions to the number of bins.
    const int saturateBlockSize = THREADS_PER_BLOCK;
    const int saturateGridSize = (NUM_BINS + saturateBlockSize - 1) / saturateBlockSize;

    // Profiling scope: saturate kernel
    startTimer();
    {
        // Runs saturate GPU Kernel.
        naiveSaturateKernel<<<saturateGridSize, saturateBlockSize>>>(deviceHistogram);
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
    cudaMemcpy(hostHistogram, deviceHistogram, sizeofHistogram, cudaMemcpyDeviceToHost);

    // Compares result with the reference.
    for (int i = 0; i < NUM_BINS; ++i)
    {
        if (hostHistogram[i] != refHistogram[i])
        {
            fprintf(stderr, "Result mismatch for integer %d: %u != %u\n", i, hostHistogram[i], refHistogram[i]);
            break;
        }
    }

    // Deallocates GPU memory.
    cudaFree(deviceInput);
    cudaFree(deviceHistogram);

    // Deallocates CPU memory.
    free(hostInput);
    free(hostHistogram);
    free(refHistogram);

    return EXIT_SUCCESS;
}
