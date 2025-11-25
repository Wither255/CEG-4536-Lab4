#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

// Naive matrix transposition 
__global__ void transposeKernel(int *out, const int *in, int rows, int cols)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (idx < cols && idy < rows) {
        out[idx * rows + idy] = in[idy * cols + idx];
    }
}

// Naive reduction kernel 
__global__ void reductionKernel(int *data, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    for (int stride = 1; stride < size; stride *= 2) {
        if (idx + stride < size && idx % (stride * 2) == 0) {
            data[idx] += data[idx + stride];
        }
        __syncthreads();
    }
}

cudaError_t naiveTranspose(int *h_out, const int *h_in, int rows, int cols);
cudaError_t naiveReduction(int *h_result, const int *h_in, int size);

int main()
{

    printf("=== Naive Matrix Transposition ===\n");
    
    const int rows = 512;
    const int cols = 512;
    const int matrixSize = rows * cols;
    
    // Initialize input matrix (row-major)
    int h_matrix[matrixSize];
    for (int i = 0; i < matrixSize; i++) {
        h_matrix[i] = i + 1;
    }
    
    printf("Original Matrix (%d x %d, row-major):\n", rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%3d ", h_matrix[i * cols + j]);
        }
        printf("\n");
    }
    
    int h_transposed[matrixSize] = { 0 };
    cudaError_t cudaStatus = naiveTranspose(h_transposed, h_matrix, rows, cols);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "naiveTranspose failed!");
        return 1;
    }
    
    printf("\nTransposed Matrix (%d x %d, row-major):\n", cols, rows);
    for (int i = 0; i < cols; i++) {
        for (int j = 0; j < rows; j++) {
            printf("%3d ", h_transposed[i * rows + j]);
        }
        printf("\n");
    }

    printf("\n== Naive Reduction (Sum) ===\n");
    
    const int arraySize = 512;
    int h_array[arraySize];
    int h_sum = 0;
    
    for (int i = 0; i < arraySize; i++) {
        h_array[i] = i + 1;
        h_sum += h_array[i];
    }
    
    printf("Original Array: ");
    for (int i = 0; i < arraySize; i++) {
        printf("%d ", h_array[i]);
    }
    printf("\n");
    printf("Expected Sum: %d\n", h_sum);
    
    int h_result[1] = { 0 };
    cudaStatus = naiveReduction(h_result, h_array, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "naiveReduction failed!");
        return 1;
    }
    
    printf("GPU Computed Sum: %d\n", h_result[0]);
    
    // Cleanup
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

// Helper function for naive matrix transposition
cudaError_t naiveTranspose(int *h_out, const int *h_in, int rows, int cols)
{
    int *dev_in = 0;
    int *dev_out = 0;
    cudaError_t cudaStatus;
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Choose GPU
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!");
        goto Error;
    }
    
    int matrixSize = rows * cols;
    
    // Allocate GPU memory
    cudaStatus = cudaMalloc((void**)&dev_in, matrixSize * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc for dev_in failed!");
        goto Error;
    }
    
    cudaStatus = cudaMalloc((void**)&dev_out, matrixSize * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc for dev_out failed!");
        goto Error;
    }
    
    // Copy input to device
    cudaStatus = cudaMemcpy(dev_in, h_in, matrixSize * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy to device failed!");
        goto Error;
    }
    
    // Configure grid and block dimensions
    dim3 blockSize(512, 512);
    dim3 gridSize((cols + blockSize.x - 1) / blockSize.x, 
                  (rows + blockSize.y - 1) / blockSize.y);
    
    // Record start time
    cudaEventRecord(start);
    
    // Launch transpose kernel
    transposeKernel<<<gridSize, blockSize>>>(dev_out, dev_in, rows, cols);
    
    // Check for launch errors
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "transposeKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // Synchronize
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize failed!");
        goto Error;
    }
    
    // Record stop time
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Matrix Transpose Execution Time: %.4f ms\n", milliseconds);
    
    // Copy result back to host
    cudaStatus = cudaMemcpy(h_out, dev_out, matrixSize * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy to host failed!");
        goto Error;
    }

Error:
    cudaFree(dev_in);
    cudaFree(dev_out);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return cudaStatus;
}

// Helper function for naive reduction
cudaError_t naiveReduction(int *h_result, const int *h_in, int size)
{
    int *dev_data = 0;
    cudaError_t cudaStatus;
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Choose GPU
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!");
        goto Error;
    }
    
    // Allocate GPU memory
    cudaStatus = cudaMalloc((void**)&dev_data, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    
    // Copy input to device
    cudaStatus = cudaMemcpy(dev_data, h_in, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy to device failed!");
        goto Error;
    }
    
    // Configure grid: use a single block with enough threads
    int blockSize = 262144;
    int gridSize = (size + blockSize - 1) / blockSize;
    
    // Record start time
    cudaEventRecord(start);
    
    // Launch reduction kernel
    reductionKernel<<<gridSize, blockSize>>>(dev_data, size);
    
    // Check for launch errors
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "reductionKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // Synchronize
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize failed!");
        goto Error;
    }
    
    // Record stop time
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Reduction Execution Time: %.4f ms\n", milliseconds);
    
    // Copy result back to host (only first element contains the sum)
    cudaStatus = cudaMemcpy(h_result, dev_data, sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy to host failed!");
        goto Error;
    }

Error:
    cudaFree(dev_data);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return cudaStatus;
}
