/*
 * Matrix Multiplication using CUDA
 *
 * This program demonstrates parallel matrix multiplication on the GPU
 * using CUDA. It generates two random matrices on the CPU, multiplies them
 * using a GPU kernel, and stores the result in a third matrix. The program
 * includes timing code to measure the execution time of the GPU kernel.
 * The CPU performs the same matrix multiplication, and the results are
 * compared to ensure correctness. The program is parameterized by the
 * dimensions of the matrices, specified as command-line arguments.
 *
 * Compilation: $ nvcc matrixMultiply.cu -o matrixMultiply
 *
 * Execution: $ ./matrixMultiply.exe <numARows> <numAColumns> <numBRows> <numBColumns>
 *
 * Parameters:
 *   <numARows> - Number of rows in matrix A
 *   <numAColumns> - Number of columns in matrix A
 *   <numBRows> - Number of rows in matrix B
 *   <numBColumns> - Number of columns in matrix B
 *
 * Profiling with Nvidia Nsight:
 *   1. Compile the code with profiling information:
 *      $ nvcc -lineinfo matrixMultiply.cu -o matrixMultiply
 *
 *   2. Run the executable with Nvidia Nsight profiling:
 *      $ ncu -o matrixMultiply_profile -f ./matrixMultiply.exe <numARows> <numAColumns> <numBRows> <numBColumns>
 *
 *   3. Analyze the profiling results using Nvidia Nsight Compute:
 *      $ ncu-ui ./matrixMultiply_profile.ncu-rep
 *
 * Note: CUDA toolkit must be installed and configured for compilation.
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

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

/// @brief Computes C = A * B using the General Matrix-Matrix Multiplication (GEMM) algorithm.
/// Each thread is responsible for computing a single element of the resulting matrix C.
/// The matrices A, B, and C are stored in row-major order in device memory.
__global__ void gemm(const DataType* _A, const DataType* _B, DataType* _C, int _numARows, int _numAColumns, int _numBRows, int _numBColumns)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < _numARows && col < _numBColumns)
    {
        DataType sum = 0.0;

        for (int k = 0; k < _numAColumns; ++k)
        {
            sum += _A[row * _numAColumns + k] * _B[k * _numBColumns + col];
        }

        _C[row * _numBColumns + col] = sum;
    }
}

/// @brief Entry point of the program. 
int main(int _argc, char** _argv)
{
    DataType *hostA, *hostB, *hostC, *resultRef;
    DataType *deviceA, *deviceB, *deviceC;
    int numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns;

    if (_argc != 5)
    {
        fprintf(stderr, "Incorrect input, usage is: ./matrixMultiply.exe <numARows> <numAColumns> <numBRows> <numBColumns>\n");
        exit(EXIT_FAILURE);
    }

    // Input matrix dimensions
    numARows = atoi(_argv[1]);
    numAColumns = atoi(_argv[2]);
    numBRows = atoi(_argv[3]);
    numBColumns = atoi(_argv[4]);

    if (numAColumns != numBRows)
    {
        fprintf(stderr, "Error: Incompatible matrix dimensions for multiplication.\n");
        exit(EXIT_FAILURE);
    }

    // Output matrix dimensions
    numCRows = numARows;
    numCColumns = numBColumns;

    // Matrice memory sizes
    int sizeofA = numARows * numAColumns * sizeof(DataType);
    int sizeofB = numBRows * numBColumns * sizeof(DataType);
    int sizeofC = numCRows * numCColumns * sizeof(DataType);

    // Allocates Host memory for input and output
    hostA = (DataType *)malloc(sizeofA);
    hostB = (DataType *)malloc(sizeofB);
    hostC = (DataType *)malloc(sizeofC);
    resultRef = (DataType *)malloc(sizeofC);

    // Initializes hostA and hostB to random numbers
    for (int i = 0; i < numARows * numAColumns; ++i)
    {
        hostA[i] = static_cast<DataType>(rand()) / RAND_MAX;
    }

    for (int i = 0; i < numBRows * numBColumns; ++i)
    {
        hostB[i] = static_cast<DataType>(rand()) / RAND_MAX;
    }

    // Initialize resultRef with zeros
    for (int i = 0; i < numCRows * numCColumns; ++i)
    {
        resultRef[i] = 0.0;
    }

    // Compute the reference result using CPU
    for (int i = 0; i < numARows; ++i)
    {
        for (int j = 0; j < numBColumns; ++j)
        {
            for (int k = 0; k < numAColumns; ++k)
            {
                resultRef[i * numBColumns + j] += hostA[i * numAColumns + k] * hostB[k * numBColumns + j];
            }
        }
    }

    // Allocates GPU memory
    cudaMalloc((void **)&deviceA, sizeofA);
    cudaMalloc((void **)&deviceB, sizeofB);
    cudaMalloc((void **)&deviceC, sizeofC);

    // Profiling scope: Data copy from host to device
    startTimer();
    {
        // Copies memory to the GPU
        cudaMemcpy(deviceA, hostA, sizeofA, cudaMemcpyHostToDevice);
        cudaMemcpy(deviceB, hostB, sizeofB, cudaMemcpyHostToDevice);
    }
    stopTimer("Data Copy from Host to Device Time");

    // Computes the grid and block dimensions
    dim3 blockDim(8, 64);
    dim3 gridDim((numCColumns + blockDim.x - 1) / blockDim.x, (numCRows + blockDim.y - 1) / blockDim.y);

    // Profiling Scope: CUDA kernel
    startTimer();
    {
        // Runs the GPU Kernel
        gemm<<<gridDim, blockDim>>>(deviceA, deviceB, deviceC, numARows, numAColumns, numBRows, numBColumns);
    }
    stopTimer("CUDA Kernel Time");

    // Profiling Scope: Data copy from device to host
    startTimer();
    {
        // Copies the GPU memory back to the CPU
        cudaMemcpy(hostC, deviceC, sizeofC, cudaMemcpyDeviceToHost);
    }
    stopTimer("Data Copy from Device to Host Time");

    // Compare the output with the reference
    const double epsilon = 1e-5;

    for (int i = 0; i < numCRows; ++i)
    {
        for (int j = 0; j < numCColumns; ++j)
        {
            int index = i * numCColumns + j;
            if (fabs(hostC[index] - resultRef[index]) > epsilon)
            {
                fprintf(stderr, "Result mismatch found at element (%d, %d): %f != %f\n", i, j, hostC[index], resultRef[index]);
                break;
            }
        }
    }

    // Free the GPU memory
    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);

    // Free the CPU memory
    free(hostA);
    free(hostB);
    free(hostC);
    free(resultRef);

    return EXIT_SUCCESS;
}
