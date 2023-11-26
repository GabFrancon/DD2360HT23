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
 * Compilation: $ nvcc -lineinfo vectorAdd.cu -o vectorAdd
 *
 * Execution: $ ./vectorAdd <vector_length>
 *
 * Parameters: <vector_length> - Length of the vectors for vector addition.
 *
 * Profiling with Nvidia Nsight:
 *   1. Compile the code with profiling information:
 *      $ nvcc -lineinfo vectorAdd.cu -o vectorAdd
 *
 *   2. Run the executable with Nvidia Nsight profiling:
 *      $ nsys profile -o vectorAdd_profile ./vectorAdd <vector_length>
 *
 *   3. Analyze the profiling results using Nvidia Nsight UI:
 *      $ nsys ui vectorAdd_profile
 *
 * Note: CUDA toolkit must be installed and configured for compilation.
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define DataType double
cudaEvent_t start, stop;

/// @brief Calculates the global index for the current thread and performs 
/// element-wise addition of input vectors, storing the result in _out memory.
__global__ void vecAdd(const DataType* _elementA, const DataType* _elementB, DataType* _out, int _len)
{
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (globalIdx < _len)
    {
        _out[globalIdx] = _elementA[globalIdx] + _elementB[globalIdx];
    }
}

/// @brief Starts a timer using CUDA events.
void startTimer()
{
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
}

/// @brief Stops the timer and prints the time elapsed since startTimer() was last called.
void stopTimer()
{
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("GPU Vector Addition Time: %.6f ms\n", elapsedTime);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
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

    // Copies memory to the GPU.
    cudaMemcpy(deviceInput1, hostInput1, bytesCount, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceInput2, hostInput2, bytesCount, cudaMemcpyHostToDevice);

    // Computes the 1D thread grid dimensions.
    const int blockSize = 256;
    const int gridSize = (vectorLength + blockSize - 1) / blockSize;

    // Runs the GPU Kernel.
    startTimer();
    vecAdd<<<gridSize, blockSize>>>(deviceInput1, deviceInput2, deviceOutput, vectorLength);
    cudaDeviceSynchronize();
    stopTimer();

    // Copies the GPU memory back to CPU.
    cudaMemcpy(hostOutput, deviceOutput, bytesCount, cudaMemcpyDeviceToHost);

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
