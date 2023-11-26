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

#define DataType double
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
    int vectorLength;
    if (_argc != 2)
    {
        fprintf(stderr, "Incorrect input, usage is: %s <vector length>\n", _argv[0]);
        exit(EXIT_FAILURE);
    }

    // Retrieves vector length from the cmd line.
    vectorLength = atoi(_argv[1]);
    const int bytesCount = vectorLength * sizeof(DataType);

    DataType* hostInput1 = (DataType*)malloc(bytesCount);
    DataType* hostInput2 = (DataType*)malloc(bytesCount);
    DataType* hostOutput = (DataType*)malloc(bytesCount);
    DataType* resultRef  = (DataType*)malloc(bytesCount);

    // Fills input vectors with random numbers.
    for (int i = 0; i < vectorLength; ++i)
    {
        hostInput1[i] = rand() / (DataType)RAND_MAX;
        hostInput2[i] = rand() / (DataType)RAND_MAX;
        resultRef[i] = hostInput1[i] + hostInput2[i];
    }

    DataType* deviceInput1;
    DataType* deviceInput2;
    DataType* deviceOutput;

    // Allocatse GPU memory.
    cudaMalloc((void**)&deviceInput1, bytesCount);
    cudaMalloc((void**)&deviceInput2, bytesCount);
    cudaMalloc((void**)&deviceOutput, bytesCount);

    // Profiling scope: Data copy from host to device
    startTimer();
    {
        // Copies memory to the GPU.
        cudaMemcpy(deviceInput1, hostInput1, bytesCount, cudaMemcpyHostToDevice);
        cudaMemcpy(deviceInput2, hostInput2, bytesCount, cudaMemcpyHostToDevice);
    }
    stopTimer("Data Copy from Host to Device Time");

    // Computes the 1D thread grid dimensions.
    const int blockSize = 1024;
    const int gridSize = (vectorLength + blockSize - 1) / blockSize;

    // Profiling Scope: CUDA kernel
    startTimer();
    {
        // Runs the GPU Kernel.
        vecAdd<<<gridSize, blockSize>>>(deviceInput1, deviceInput2, deviceOutput, vectorLength);
        cudaDeviceSynchronize();
    }
    stopTimer("CUDA Kernel Time");

    // Profiling Scope: Data copy from device to host
    startTimer();
    {
        // Copies the GPU memory back to CPU.
        cudaMemcpy(hostOutput, deviceOutput, bytesCount, cudaMemcpyDeviceToHost);
    }
    stopTimer("Data Copy from Device to Host Time");

    // Compares result with the reference.
    for (int i = 0; i < vectorLength; ++i)
    {
        if (fabs(hostOutput[i] - resultRef[i]) > 1e-5)
        {
            fprintf(stderr, "Result mismatch found at element %d: %f != %f\n", i, hostOutput[i], resultRef[i]);
            break;
        }
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