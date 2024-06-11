/****************************************************************************
 *
 * cuda-matmul.cu - Dense matrix-matrix multiplication with CUDA
 *
 * Last modified in 2018 by Moreno Marzolla <moreno.marzolla(at)unibo.it>
 *
 * To the extent possible under law, the author(s) have dedicated all 
 * copyright and related and neighboring rights to this software to the 
 * public domain worldwide. This software is distributed without any warranty.
 *
 * You should have received a copy of the CC0 Public Domain Dedication
 * along with this software. If not, see 
 * <http://creativecommons.org/publicdomain/zero/1.0/>. 
 *
 * ---------------------------------------------------------------------------
 *
 * Dense matrix-matrix multiplication kernel with CUDA. Two versions
 * of the kernel are provided: one that does not use shared memory,
 * and one that does. A third version uses shared memory and does not
 * require that the matrix size is multiple of BLKDIM.
 *
 * Compile with:
 * nvcc cuda-matmul.cu -o cuda-matmul -lm
 *
 * Run with:
 * ./cuda-matmul [N]
 *
 ****************************************************************************/

#include <stdio.h>
#include <stdlib.h>     /* for malloc() */
#include <math.h>       /* for fabsf()  */
#include <string.h>     /* for bzero()  */

#define BLKDIM 32

double gettime( void )
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts );
    return (ts.tv_sec + (double)ts.tv_nsec / 1e9);
}

/* Compute r = p * q, for square nxn matrices p, q, r; this version
   does not use threads and use one block for each value to 
   be computed. */
__global__ void matmulBlock( const float *p, const float *q, float *r, int n )
{
    const int i = blockIdx.y;
    const int j = blockIdx.x;
    int k;
    float val = 0.0;
    if ( i < n && j < n ) {
        for (k=0; k<n; k++) {
            val += p[i*n + k] * q[k*n + j];
        }
        r[i*n + j] = val;
    }
}

/* Compute r = p * q, for square nxn matrices p, q, r; this version
   does not use shared memory. This kernel does not require that n is
   a multiple of BLKDIM */
__global__ void matmul( const float *p, const float *q, float *r, int n )
{
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    int k;
    float val = 0.0;
    if ( i < n && j < n ) {
        for (k=0; k<n; k++) {
            val += p[i*n + k] * q[k*n + j];
        }
        r[i*n + j] = val;
    }
}
/* Compute r = p * q, for square n x n matrices p, q, r; this version
   uses shared memory. This kernel requires that n is a multiple of
   BLKDIM */
__global__ void matmulb( const float *p, const float *q, float *r, int n )
{
    __shared__ float local_p[BLKDIM][BLKDIM];
    __shared__ float local_q[BLKDIM][BLKDIM];
    const int bx = blockIdx.x, by = blockIdx.y;
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int i = by * BLKDIM + ty;
    const int j = bx * BLKDIM + tx;
    float v = 0.0; int m, k;
    for (m = 0; m < n; m += BLKDIM) { /* loop over tiles */
        local_p[ty][tx] = p[i*n + (m + tx)];
        local_q[ty][tx] = q[(m + ty)*n + j];
        __syncthreads();
        for (k = 0; k < BLKDIM; k++) { /* loop within tile */
            v += local_p[ty][k] * local_q[k][tx];
        }
        __syncthreads();
    }    
    r[i*n + j] = v; /* write back to global memory */
}

__device__ int cuda_min(int a, int b)
{
    return (a < b ? a : b);
}

/* Same as above, but does not require that n is a multiple of
   BLKDIM. To do so, it fills shared buffers so that values outside
   the matrices are treated as zeros. */
__global__ void matmulb_generic( const float *p, const float *q, float *r, int n )
{
    __shared__ float local_p[BLKDIM][BLKDIM];
    __shared__ float local_q[BLKDIM][BLKDIM];
    const int bx = blockIdx.x, by = blockIdx.y;
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int i = by * BLKDIM + ty;
    const int j = bx * BLKDIM + tx;
    float v = 0.0; int m, k;
    for (m = 0; m < n; m += BLKDIM) { /* loop over tiles */
        local_p[ty][tx] = local_q[ty][tx] = 0;
        if (i<n && m+tx<n) 
            local_p[ty][tx] = p[i*n + (m + tx)];
        if (j<n && m+ty<n) 
            local_q[ty][tx] = q[(m + ty)*n + j];

        __syncthreads();

        for (k = 0; k < BLKDIM; k++) { /* loop within tile */
            v += local_p[ty][k] * local_q[k][tx];
        }

        __syncthreads();
    }
    if (i<n && j<n) 
        r[i*n + j] = v; /* write result to global memory */
}


/* Initialize square matrix q */
void mat_init( float *q, int n )
{
    int i;
    for (i=0; i<n*n; i++) {
        q[i] = 1.0;
    }
}

int check_result( const float *r, int n ) 
{
    /* Check result */
    int i, j;
    for (i=0; i<n; i++) {
        for (j=0; j<n; j++) {
            if (fabsf(r[i*n+j] - n) > 1e-5) {
                printf("Check failed: r[%d][%d] = %f, expected %f\n", i, j, r[i*n+j], (float)n);
                return 0;
            }
        }
    }
    printf("Check OK\n");
    return 1;
}

int main( int argc, char* argv[] ) 
{
    float *p, *q, *r;	          /* host copies of p, q, r */ 
    float *d_p, *d_q, *d_r;	  /* device copies of p, q, r */
    int N = 512;
    double tstart, tstop, tnoshared, tshared;

    if ( argc > 1 ) {
        N = atoi(argv[1]);
    }

    dim3 onlyBlock(N,N);
    dim3 onlyThread(N,N);
    dim3 block(BLKDIM, BLKDIM);
    dim3 grid((N+BLKDIM-1)/BLKDIM, (N+BLKDIM-1)/BLKDIM);
    const size_t size = N*N*sizeof(float);

    /* Allocate space for device copies of p, q, r */
    cudaMalloc((void **)&d_p, size);
    cudaMalloc((void **)&d_q, size);
    cudaMalloc((void **)&d_r, size);

    /* Allocate space for host copies of p, q, r */
    p = (float*)malloc(size); mat_init(p, N);
    q = (float*)malloc(size); mat_init(q, N);
    r = (float*)malloc(size);

    /* Copy inputs to device */
    cudaMemcpy(d_p, p, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_q, q, size, cudaMemcpyHostToDevice);

    printf("Matrix-Matrix multiplication (%dx%d)\n", N, N);

    /**
     ** Matrix-matrix multiply WITH only blocks
     **/
    printf("Only blocks:\t\t");
    tstart = gettime();
    matmulBlock<<<onlyBlock,1>>>(d_p, d_q, d_r, N);
    cudaDeviceSynchronize();
    tstop = gettime();
    tnoshared = tstop - tstart;
    printf("%fs\n", tnoshared);
    /* Copy result back to host and check correctness */
    bzero(r, size); /* erase destination buffer, just in case... */
    cudaMemcpy(r, d_r, size, cudaMemcpyDeviceToHost);
    check_result(r, N);

    /* zero out r and d_r, to ensure that we don't read old results */
    cudaMemset(d_r, 0, size);
    bzero(r, size);
   
    /**
     ** Matrix-matrix multiply WITHOUT shared memory
     **/
    printf("No shared memory:\t");
    tstart = gettime();
    matmul<<<grid, block>>>(d_p, d_q, d_r, N);
    cudaDeviceSynchronize();
    tstop = gettime();
    tnoshared = tstop - tstart;
    printf("%fs\n", tnoshared);
    /* Copy result back to host and check correctness */
    bzero(r, size); /* erase destination buffer, just in case... */
    cudaMemcpy(r, d_r, size, cudaMemcpyDeviceToHost);
    check_result(r, N);

    /* zero out r and d_r, to ensure that we don't read old results */
    cudaMemset(d_r, 0, size);
    bzero(r, size);

    /**
     ** Matrix-matrix multiply WITH shared memory
     **/
    printf("Shared memory:\t\t");
    tstart = gettime();
    matmulb_generic<<<grid, block>>>(d_p, d_q, d_r, N);
    cudaDeviceSynchronize();
    tstop = gettime();
    tshared = tstop - tstart;
    printf("%fs\n", tshared);
    /* Copy result back to host and check correctness */
    cudaMemcpy(r, d_r, size, cudaMemcpyDeviceToHost);
    check_result(r, N);

    /* Cleanup */
    free(p); free(q); free(r);
    cudaFree(d_p); cudaFree(d_q); cudaFree(d_r);
    return EXIT_SUCCESS;
}
