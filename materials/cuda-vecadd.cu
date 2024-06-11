/****************************************************************************
 *
 * cuda-vecadd3.cu - Sum two arrays with CUDA, using threads and blocks
 *
 * Based on the examples from the CUDA toolkit documentation
 * http://docs.nvidia.com/cuda/cuda-c-programming-guide/
 *
 * Last updated in 2017 by Moreno Marzolla <moreno.marzolla(at)unibo.it>
 *
 * ---------------------------------------------------------------------------
 *
 * Compile with:
 * nvcc cuda-vecadd3.cu -o cuda-vecadd3
 *
 * Run with:
 * ./cuda-vecadd3
 *
 ****************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N (2048*2048)
#define BLKDIM 1024

double gettime( void )
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts );
    return (ts.tv_sec + (double)ts.tv_nsec / 1e9);
}

__global__ void add( int *a, int *b, int *c, int n )
{
    for (int index=0; index<n; index++) {
        c[index] = a[index] + b[index];
    }
}

__global__ void addWithBlocks( int *a, int *b, int *c, int n )
{
    int index = blockIdx.x;
    if ( index < n ) {
        c[index] = a[index] + b[index];
    }
}

__global__ void addWithThreadsPartitions( int *a, int *b, int *c, int n )
{
    int index_init = threadIdx.x * ( (n+blockDim.x-1) / blockDim.x );
    int index_end = index_init + ( (n+blockDim.x-1) / blockDim.x );
    if (index_end>n) index_end=n;
    for (int index = index_init; index < index_end; index++) {
        c[index] = a[index] + b[index];
    }
}

__global__ void addWithThreadsSteps( int *a, int *b, int *c, int n )
{
    for (int index=threadIdx.x; index < n; index += blockDim.x) {
			c[index] = a[index] + b[index];	
    }
}
__global__ void addWithBlocksAndThreads( int *a, int *b, int *c, int n )
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if ( index < n ) {
        c[index] = a[index] + b[index];
    }
}

void vec_init( int *a, int n )
{
    int i;
    for (i=0; i<n; i++) {
        a[i] = i;
    }
}

int main( void ) 
{
    int *a, *b, *c;	          /* host copies of a, b, c */ 
    int *d_a, *d_b, *d_c;	  /* device copies of a, b, c */
    int i;
    const size_t size = N*sizeof(int);
    double tstart,tend;


    /* Allocate space for device copies of a, b, c */
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    /* Allocate space for host copies of a, b, c */
    a = (int*)malloc(size); 
    b = (int*)malloc(size); 
    c = (int*)malloc(size);

    /* Sequential implementation */
    vec_init(a, N);
    vec_init(b, N);
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
    
    tstart=gettime();
    add<<<1, 1>>>(d_a, d_b, d_c, N);
    cudaDeviceSynchronize();
    tend=gettime();
    
    printf("Elapsed time in seconds for sequential implementation: %f\n", tend-tstart);

    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    for (i=0; i<N; i++) {
        if ( c[i] != a[i] + b[i] ) {
            fprintf(stderr, "Error at index %d: a[%d]=%d, b[%d]=%d, c[%d]=%d\n",
                    i, i, a[i], i, b[i], i, c[i]);
            break;
        }
    }
    if (i == N) {
        printf("Check OK\n");
    }
    
    /* Parallel implementation with only blocks */
    vec_init(a, N);
    vec_init(b, N);
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
    
    tstart=gettime();
    addWithBlocks<<<N, 1>>>(d_a, d_b, d_c, N);
    cudaDeviceSynchronize();
    tend=gettime();
    
    printf("Elapsed time in seconds for block parallel implementation: %f\n", tend-tstart);

    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    for (i=0; i<N; i++) {
        if ( c[i] != a[i] + b[i] ) {
            fprintf(stderr, "Error at index %d: a[%d]=%d, b[%d]=%d, c[%d]=%d\n",
                    i, i, a[i], i, b[i], i, c[i]);
            break;
        }
    }
    if (i == N) {
        printf("Check OK\n");
    }
    
    /* Parallel implementation with only threads (partitions) */
    vec_init(a, N);
    vec_init(b, N);
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
    
    tstart=gettime();
    addWithThreadsPartitions<<<1, BLKDIM>>>(d_a, d_b, d_c, N);
    cudaDeviceSynchronize();
    tend=gettime();
    
    printf("Elapsed time in seconds for thread parallel implementation (partitions): %f\n", tend-tstart);

    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    for (i=0; i<N; i++) {
        if ( c[i] != a[i] + b[i] ) {
            fprintf(stderr, "Error at index %d: a[%d]=%d, b[%d]=%d, c[%d]=%d\n",
                    i, i, a[i], i, b[i], i, c[i]);
            break;
        }
    }
    if (i == N) {
        printf("Check OK\n");
    }
    
    /* Parallel implementation with only threads (parallel steps) */
    vec_init(a, N);
    vec_init(b, N);
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
    
    tstart=gettime();
    addWithThreadsSteps<<<1, BLKDIM>>>(d_a, d_b, d_c, N);
    cudaDeviceSynchronize();
    tend=gettime();
    
    printf("Elapsed time in seconds for thread parallel implementation (steps): %f\n", tend-tstart);

    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    for (i=0; i<N; i++) {
        if ( c[i] != a[i] + b[i] ) {
            fprintf(stderr, "Error at index %d: a[%d]=%d, b[%d]=%d, c[%d]=%d\n",
                    i, i, a[i], i, b[i], i, c[i]);
            break;
        }
    }
    if (i == N) {
        printf("Check OK\n");
    }
    
    /* Parallel implementation with blocks and threads */
    vec_init(a, N);
    vec_init(b, N);
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
    
    tstart=gettime();
    addWithBlocksAndThreads<<<(N + BLKDIM-1)/BLKDIM, BLKDIM>>>(d_a, d_b, d_c, N);
    cudaDeviceSynchronize();
    tend=gettime();
    
    printf("Elapsed time in seconds for block and thread parallel implementation: %f\n", tend-tstart);

    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    for (i=0; i<N; i++) {
        if ( c[i] != a[i] + b[i] ) {
            fprintf(stderr, "Error at index %d: a[%d]=%d, b[%d]=%d, c[%d]=%d\n",
                    i, i, a[i], i, b[i], i, c[i]);
            break;
        }
    }
    if (i == N) {
        printf("Check OK\n");
    }
    
    /* Cleanup */
    free(a); free(b); free(c);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    return EXIT_SUCCESS;
    
}
