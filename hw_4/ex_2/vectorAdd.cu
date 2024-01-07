/*
 * Vector Addition using CUDA
 *
 * This program demonstrates parallel vector addition on the GPU
 * using CUDA. It creates two random vectors on the CPU, adds them
 * element-wise on the GPU, and stores the result in a third vector.
 * The program includes timing code to measure the execution time of
 * the GPU kernel. The CPU then performs the same vector addition and
 * compares the GPU result with the CPU result to ensure correctness.
 * The program is parameterized by the length of the vectors, specified
 * as a command-line argument.
 *
 * Compilation: $ nvcc vectorAdd.cu -o vectorAdd
 *
 * Execution: $ ./vectorAdd.exe <vector_length>
 *
 * Parameters: <vector_length> - Length of the vectors for vector addition.
 *
 * Profiling with Nvidia Nsight:
 *   1. Compile the code with profiling information:
 *      $ nvcc -lineinfo vectorAdd.cu -o vectorAdd
 *
 *   2. Run the executable with Nvidia Nsight profiling:
 *      $ ncu -o vectorAdd_profile -f ./vectorAdd.exe <vector_length>
 *
 *   3. Analyze the profiling results using Nvidia Nsight Compute:
 *      $ ncu-ui ./vectorAdd_profile.ncu-rep
 *
 * Note: CUDA toolkit must be installed and configured for compilation.
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#define DataType double
#define THREADS_PER_BLOCK 1024
#define NUM_SEGMENTS 4

double start, stop;

/// @brief Starts the timer.
void startTimer()
{
    start = (double)clock();
    start = start / CLOCKS_PER_SEC;
}

/// @brief Stops the timer and print the elapsed time.
void stopTimer(const char* _message)
{
    stop = (double)clock();
    stop = stop / CLOCKS_PER_SEC;

    double elapsedTime = (stop - start) * 1.0e3;
    printf("%s: %.6f ms\n", _message, elapsedTime);
}

/// @brief Calculates the global index for the current thread and performs 
/// element-wise addition of input vectors, storing the result in _out memory.
__global__ void vecAdd(const DataType* _vecA, const DataType* _vecB, DataType* _out, int _len)
{
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (globalIdx < _len)
    {
        _out[globalIdx] = _vecA[globalIdx] + _vecB[globalIdx];
    }
}

/// @brief Entry point of the program. 
int main(int _argc, char** _argv)
{
    if (_argc != 2)
    {
        fprintf(stderr, "Incorrect input, usage is: ./vectorAdd.exe <vector length>\n");
        exit(EXIT_FAILURE);
    }

    // Retrieves vector length from the cmd line.
    int vectorLength = atoi(_argv[1]);
    const int sizeofVec = vectorLength * sizeof(DataType);

    DataType* hostInput1 = (DataType*)malloc(sizeofVec);
    DataType* hostInput2 = (DataType*)malloc(sizeofVec);
    DataType* hostOutput = (DataType*)malloc(sizeofVec);
    DataType* resultRef  = (DataType*)malloc(sizeofVec);

    // Fills input vectors with random numbers, and pre-compute
    // addition to use as a reference when validating GPU results.
    for (int i = 0; i < vectorLength; ++i)
    {
        hostInput1[i] = rand() / (DataType)RAND_MAX;
        hostInput2[i] = rand() / (DataType)RAND_MAX;
        resultRef[i] = hostInput1[i] + hostInput2[i];
    }

    DataType* deviceInput1;
    DataType* deviceInput2;
    DataType* deviceOutput;

    // Allocates GPU memory.
    cudaMalloc((void**)&deviceInput1, sizeofVec);
    cudaMalloc((void**)&deviceInput2, sizeofVec);
    cudaMalloc((void**)&deviceOutput, sizeofVec);

    // Creates CUDA streams
    cudaStream_t streams[NUM_SEGMENTS];
    for (int i = 0; i < NUM_SEGMENTS; ++i)
    {
        cudaStreamCreate(&streams[i]);
    }

    // Profiling scope: memory copy from host to device
    startTimer();
    {
        // Copies memory to GPU using streams
        for (int i = 0; i < NUM_SEGMENTS; ++i)
        {
            int offset = i * (vectorLength / NUM_SEGMENTS);
            int size = (i == NUM_SEGMENTS - 1) ? (vectorLength - offset) : (vectorLength / NUM_SEGMENTS);

            cudaMemcpyAsync(deviceInput1 + offset, hostInput1 + offset, size * sizeof(DataType), cudaMemcpyHostToDevice, streams[i]);
            cudaMemcpyAsync(deviceInput2 + offset, hostInput2 + offset, size * sizeof(DataType), cudaMemcpyHostToDevice, streams[i]);
        }

        // Synchronizes all streams to ensure completion of data transfers
        for (int i = 0; i < NUM_SEGMENTS; ++i)
        {
            cudaStreamSynchronize(streams[i]);
        }
    }
    stopTimer("Asynchronous Copy from Host to Device Time");

    // Computes the 1D thread grid dimensions.
    const int blockSize = THREADS_PER_BLOCK;
    const int gridSize = (vectorLength + blockSize - 1) / blockSize;

    // Profiling Scope: CUDA kernel
    startTimer();
    {
        // Run the GPU Kernel using streams
        for (int i = 0; i < NUM_SEGMENTS; ++i)
        {
            int offset = i * (vectorLength / NUM_SEGMENTS);
            int size = (i == NUM_SEGMENTS - 1) ? (vectorLength - offset) : (vectorLength / NUM_SEGMENTS);

            vecAdd<<<gridSize, blockSize, 0, streams[i]>>>(deviceInput1 + offset, deviceInput2 + offset, deviceOutput + offset, size);
        }

        // Synchronize all streams to ensure completion of kernel executions
        for (int i = 0; i < NUM_SEGMENTS; ++i)
        {
            cudaStreamSynchronize(streams[i]);
        }        
    }
    stopTimer("CUDA Kernel Time");

    // Profiling Scope: Data copy from device to host
    startTimer();
    {
        // Copy GPU memory back to CPU using streams
        for (int i = 0; i < NUM_SEGMENTS; ++i)
        {
            int offset = i * (vectorLength / NUM_SEGMENTS);
            int size = (i == NUM_SEGMENTS - 1) ? (vectorLength - offset) : (vectorLength / NUM_SEGMENTS);

            cudaMemcpyAsync(hostOutput + offset, deviceOutput + offset, size * sizeof(DataType), cudaMemcpyDeviceToHost, streams[i]);
        }

        // Synchronize all streams to ensure completion of data transfers
        for (int i = 0; i < NUM_SEGMENTS; ++i)
        {
            cudaStreamSynchronize(streams[i]);
        }
    }
    stopTimer("Asynchronous Copy from Device to Host Time");

    // Compares result with the reference.
    const double epsilon = 1e-5;

    for (int i = 0; i < vectorLength; ++i)
    {
        if (fabs(hostOutput[i] - resultRef[i]) > epsilon)
        {
            fprintf(stderr, "Result mismatch found at element %d: %f != %f\n", i, hostOutput[i], resultRef[i]);
            break;
        }
    }

    // Deallocates CUDA streams
    for (int i = 0; i < NUM_SEGMENTS; ++i)
    {
        cudaStreamDestroy(streams[i]);
    }

    // Deallocates GPU memory
    cudaFree(deviceInput1);
    cudaFree(deviceInput2);
    cudaFree(deviceOutput);

    // Deallocates CPU memory
    free(hostInput1);
    free(hostInput2);
    free(hostOutput);
    free(resultRef);

    return EXIT_SUCCESS;
}