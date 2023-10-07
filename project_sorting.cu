
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>


#define BLKCOUNT 32
#define MYBLKDIM 64
#define GRAN 32

const int threadsCount = (BLKCOUNT * MYBLKDIM);

__device__ int getIDofThread() {
    return threadIdx.x + blockIdx.x * blockDim.x;
}

__device__ void gpu_bottomUpMerge(int* source, int* dest, unsigned long start, unsigned long middle, unsigned long end) {
    unsigned long i = start;
    unsigned long j = middle;
    for (unsigned long k = start; k < end; k++) {
        if (i < middle && (j >= end || source[i] < source[j])) {
            dest[k] = source[i];
            i++;
        }
        else {
            dest[k] = source[j];
            j++;
        }
    }
}

__device__ void gpu_mergesort_device(int* source, int* dest, unsigned long N, unsigned long width, unsigned long slices, unsigned long start) {
    unsigned long middle, end;

    for (unsigned long slice = 0; slice < slices; slice++) {
        if (start >= N)
            break;

        middle = min(start + (width >> 1), N);
        end = min(start + width, N);
        gpu_bottomUpMerge(source, dest, start, middle, end);
        start += width;
    }
    __syncthreads();

}

__global__ void gpu_mergesort(int* source, int* dest, unsigned long N, unsigned long width, unsigned long slices) {
    int idx = getIDofThread();
    unsigned long start = width * idx * slices,
        middle,
        end;

    for (unsigned long slice = 0; slice < slices; slice++) {
        if (start >= N)
            break;

        middle = min(start + (width >> 1), N);
        end = min(start + width, N);
        gpu_bottomUpMerge(source, dest, start, middle, end);
        start += width;
    }
    __syncthreads();

}

__global__ void blocksMerge(int* deviceArr, int* deviceResult, unsigned long N) {
    unsigned long index = getIDofThread();
    const unsigned long blockMemoryUsage = MYBLKDIM * GRAN;

    int* A = deviceArr;
    int* B = deviceResult;
    for (unsigned long width = blockMemoryUsage * 2; width < N * 2; width <<= 1) {
        unsigned long threadsSlices = N / (threadsCount * width) + 1;
        unsigned long start = index * width * threadsSlices;
        gpu_mergesort_device(A, B, N, width, threadsSlices, start);


        // Switch the input / output arrays instead of copying them around
        A = A == deviceArr ? deviceResult : deviceArr;
        B = B == deviceArr ? deviceResult : deviceArr;
    }
    deviceResult = A;
}

__global__ void mergesortLocalMemKernel(int* deviceArr, int* deviceResult, unsigned long N) {
    const unsigned long t = threadIdx.x;
    unsigned long index = getIDofThread();
    const unsigned long blockMemoryUsage = MYBLKDIM * GRAN;
    const unsigned long allThreadsMemoryUsage = blockMemoryUsage * BLKCOUNT;

    __shared__ int localArr[blockMemoryUsage];
    __shared__ int localArrSwp[blockMemoryUsage];
    int* A;
    int* B;
    unsigned long blocksSlices = (N % (allThreadsMemoryUsage) == 0) ? (N / (allThreadsMemoryUsage)) : (N / (allThreadsMemoryUsage)+1);

    for (unsigned long blkSlice = 0; blkSlice < blocksSlices; blkSlice++) {
        unsigned long passedElemetsInDevice = (blkSlice * allThreadsMemoryUsage);
        for (unsigned long i = 0; i < GRAN; i++) {
            unsigned long deviceIndex = passedElemetsInDevice + (index * GRAN) + i;
            localArr[(t * GRAN) + i] = INT_MAX;
            if (deviceIndex < N) {
                localArr[(t * GRAN) + i] = deviceArr[deviceIndex];
            }
        }
        __syncthreads();
        A = localArr;
        B = localArrSwp;
        for (unsigned long width = 2; width < (blockMemoryUsage * 2); width <<= 1) {
            // if width < GRAN we use all threads, else each thread has 1 segment
            unsigned long threadsSlices = (width < GRAN) ? GRAN / width : 1;
            unsigned long start = width * threadIdx.x * threadsSlices;
            gpu_mergesort_device(A, B, blockMemoryUsage, width, threadsSlices, start);


            // Switch the input / output arrays instead of copying them around
            A = A == localArr ? localArrSwp : localArr;
            B = B == localArr ? localArrSwp : localArr;
        }

        __syncthreads();

        for (unsigned long i = 0; i < GRAN; i++) {
            unsigned long deviceIndex = passedElemetsInDevice + (index * GRAN) + i;
            if (deviceIndex < N) {
                deviceResult[deviceIndex] = A[(t * GRAN) + i];
            }
        }
    }
}

void mergesortLocalMem(int* data, unsigned long N) {
    int* deviceArr;
    int* deviceResult;

    // Actually allocate the two arrays
    cudaMalloc((void**)&deviceArr, N * sizeof(int));
    cudaMalloc((void**)&deviceResult, N * sizeof(int));

    // Copy from our input list unsigned longo the first array
    cudaMemcpy(deviceArr, data, N * sizeof(unsigned long), cudaMemcpyHostToDevice);

    //
    // Slice up the list and give pieces of it to each thread, letting the pieces grow
    // bigger and bigger until the whole list is sorted
    //
    mergesortLocalMemKernel << <BLKCOUNT, MYBLKDIM >> > (deviceArr, deviceResult, N);
    cudaDeviceSynchronize();
    int* temp = deviceArr;
    deviceArr = deviceResult;
    deviceResult = temp;

    int* A = deviceArr;
    int* B = deviceResult;
    //cudaMemcpy(data, deviceArr, N * sizeof(unsigned long), cudaMemcpyDeviceToHost);
    for (unsigned long width = 2; width < (N << 1); width <<= 1) {
        unsigned long slices = N / ((threadsCount)*width) + 1;

        // Actually call the kernel
        gpu_mergesort << <BLKCOUNT, MYBLKDIM >> > (A, B, N, width, slices);
        cudaDeviceSynchronize();

        // Switch the input / output arrays instead of copying them around
        A = A == deviceArr ? deviceResult : deviceArr;
        B = B == deviceArr ? deviceResult : deviceArr;
    }

    cudaMemcpy(data, deviceResult, N * sizeof(unsigned long), cudaMemcpyDeviceToHost);


    // Free the GPU memory
    cudaFree(deviceArr);
    cudaFree(deviceResult);
}


// No shared memory used

void mergesort(int* data, unsigned long N) {

    //
    // Allocate two arrays on the GPU
    // we switch back and forth between them during the sort
    //
    int* D_data;
    int* D_swp;

    // Actually allocate the two arrays
    cudaMalloc((void**)&D_data, N * sizeof(int));
    cudaMalloc((void**)&D_swp, N * sizeof(int));

    // Copy from our input list unsigned longo the first array
    cudaMemcpy(D_data, data, N * sizeof(unsigned long), cudaMemcpyHostToDevice);

    int* A = D_data;
    int* B = D_swp;


    //
    // Slice up the list and give pieces of it to each thread, letting the pieces grow
    // bigger and bigger until the whole list is sorted
    //
    for (unsigned long width = 2; width < (N << 1); width <<= 1) {
        unsigned long slices = N / ((threadsCount)*width) + 1;

        // Actually call the kernel
        gpu_mergesort << <BLKCOUNT, MYBLKDIM >> > (A, B, N, width, slices);
        cudaDeviceSynchronize();

        // Switch the input / output arrays instead of copying them around
        A = A == D_data ? D_swp : D_data;
        B = B == D_data ? D_swp : D_data;
    }

    cudaMemcpy(data, A, N * sizeof(unsigned long), cudaMemcpyDeviceToHost);


    // Free the GPU memory
    cudaFree(A);
    cudaFree(B);

}

void vec_init(int* arr, unsigned long N) {
    srand(time(NULL));
    unsigned long i;
    for (i = 0; i < N; i++) {
        arr[i] = rand() % N + 1;
    }

}


int main(int argc, char* argv[]) {
    char* arg1 = argv[1];
    const unsigned long N = atoi(arg1);

    int* hostArray;
    hostArray = (int*)malloc(N * sizeof(int));
    int* hostArrayCopy = (int*)malloc(N * sizeof(int));
    vec_init(hostArray, N);
    for (unsigned long i = 0; i < N; i++) {
        hostArrayCopy[i] = hostArray[i];
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);


    cudaEventRecord(start);
    mergesortLocalMem(hostArrayCopy, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("%f,", milliseconds);

    cudaEventRecord(start);

    mergesort(hostArray, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("%f\n", milliseconds);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}
