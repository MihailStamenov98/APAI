/****************************************************************************
 *
 * firstLecture.cu - Sequential vs Parallel solution to the problem
 * of summing the elements in an array
 *
 * ---------------------------------------------------------------------------
 *
 * Compile with:
 * nvcc arraySum.cu -o arraySum
 *
 * Run with:
 * ./arraySum 10000000 50 1024
 *
 * (where the parameters are the array length, and the number of 
 *  CUBA blocks and threads)
 ****************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

double gettime( void )
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts );
    return (ts.tv_sec + (double)ts.tv_nsec / 1e9);
}

__global__ void sumSequential( int *a, int *s, int l )
{
    int i;
    *s = 0;
    for (i=0; i<l; i++)
    	*s += a[i];
}

__global__ void sumWithMutex( int *a, int *s, int n, int l )
{
    int i,end,index = threadIdx.x + blockIdx.x * blockDim.x;
    end = (n*(index+1) < l) ? n*(index+1) : l;
    for (i=n*index; i<end; i++)
    	atomicAdd(s, a[i]);
}

__global__ void sumWithLocalCounterAndMutex( int *a, int *s, int n, int l )
{
    int i,end,index = threadIdx.x + blockIdx.x * blockDim.x;
    int mySum = 0;
    end = (n*(index+1) < l) ? n*(index+1) : l;
    for (i=n*index; i<end; i++)
    	mySum += a[i];
    __syncthreads(); 
    atomicAdd(s, mySum);
}

__global__ void sumWithOnlyLocalCounters( int *a, int *s, int n, int l )
{
    int i,end,index = threadIdx.x + blockIdx.x * blockDim.x;
    int mysum = 0;
    end = (n*(index+1) < l) ? n*(index+1) : l;
    for (i=n*index; i<end; i++)
    	mysum += a[i];
    s[index] = mysum; 
}

__global__ void sumReduction( int *a, int *s, int l )
{
    __shared__ int temp[1024];
    const int lindex = threadIdx.x;
    const int bindex = blockIdx.x;
    const int gindex = threadIdx.x + blockIdx.x * blockDim.x;
    int bsize = blockDim.x / 2;
    temp[lindex] = (gindex < l) ? a[gindex] : 0;
    __syncthreads(); 
    while ( bsize > 0 ) {
        if ( lindex < bsize ) {
            temp[lindex] += temp[lindex + bsize];
        }
        bsize = bsize / 2;
        __syncthreads(); 
    }
    if ( lindex == 0  ) {
        s[bindex] = temp[0];
    }
}

void vec_init( int *a, int n )
{
	int i;
	srand((unsigned int)time(NULL));    
    for (i=0; i<n; i++)
    	a[i] = rand()%2;
}

int main(int argc, char *argv[]) 
{
    int *a, *s, *sred;	    	/* host copies of a, s, sred */ 
    int result;	    			/* host counter */ 
    int *d_a, *d_s, *d_sred;	/* device copies of a, s, sred */
    int *d_result;				/* device counter */
    int i, tot;
    double tstart, tend, elapsed;

    char *arg1 = argv[1];
    char *arg2 = argv[2];
    char *arg3 = argv[3];
    int size = atoi(arg1);
    int NUMBLK = atoi(arg2);
    int BLKDIM = atoi(arg3);

    int totThreads = (NUMBLK*BLKDIM);
    int sizeThread = (size+totThreads-1)/totThreads;

    int totBlocks = (size+BLKDIM-1) / BLKDIM;
    
        
    /* Allocate space for device copies of a, s */
    cudaMalloc((void **)&d_a, size*sizeof(int));
    cudaMalloc((void **)&d_s, totThreads*sizeof(int));
    cudaMalloc((void **)&d_sred, totBlocks*sizeof(int));

    /* Allocate space for device counter */
    cudaMalloc((void **)&d_result, sizeof(int));


    /* Allocate space for host copies of a, b, c */
    a = (int*)malloc(size*sizeof(int)); vec_init(a, size);
    s = (int*)malloc(totThreads*sizeof(int));
    sred = (int*)malloc(totBlocks*sizeof(int));

    /* Compute sum sequentially on the host */
    tot = 0;
    for (i=0; i<size; i++) 
    	tot += a[i];

    /* Copy inputs to device */
    cudaMemcpy(d_a, a, size*sizeof(int), cudaMemcpyHostToDevice);
    
    printf("*** SEQUENTIAL WITH COUNTER\n");
    tstart=gettime();
    sumSequential<<<1, 1>>>(d_a, d_result, size);
    cudaMemcpy(&result, d_result, sizeof(int), cudaMemcpyDeviceToHost);
    tend=gettime();
    elapsed = tend - tstart;
    printf("Sum = %ld\n", result);
    printf("Elapsed time %lf\n\n", elapsed);
    
    printf("*** PARALLEL WITH MUTEX ON COUNTER\n");
    tstart=gettime();
    result = 0;
    cudaMemcpy(d_result, &result, sizeof(int), cudaMemcpyHostToDevice);
    sumWithMutex<<<NUMBLK, BLKDIM>>>(d_a, d_result, sizeThread, size);
    cudaMemcpy(&result, d_result, sizeof(int), cudaMemcpyDeviceToHost);
    tend=gettime();
    elapsed = tend - tstart;
    printf("Sum = %ld\n", result);
    printf("Elapsed time %lf\n\n", elapsed);

    printf("*** PARALLEL WITH LOCAL COUNTERS AND MUTEX ON GLOBAL COUNTER\n");
    tstart=gettime();
    result = 0;
    cudaMemcpy(d_result, &result, sizeof(int), cudaMemcpyHostToDevice);
    sumWithLocalCounterAndMutex<<<NUMBLK, BLKDIM>>>(d_a, d_result, sizeThread, size);
    cudaMemcpy(&result, d_result, sizeof(int), cudaMemcpyDeviceToHost);
    tend=gettime();
    elapsed = tend - tstart;
    printf("Sum = %ld\n", result);
    printf("Elapsed time %lf\n\n", elapsed);

    printf("*** PARALLEL WITH NO MUTEX\n");
    tstart=gettime();
    sumWithOnlyLocalCounters<<<NUMBLK, BLKDIM>>>(d_a, d_s, sizeThread, size);
    cudaMemcpy(s, d_s, totThreads*sizeof(int), cudaMemcpyDeviceToHost);
    result = 0;
    for (i=0; i<totThreads; i++) 
    	result += s[i];
    tend=gettime();
    elapsed = tend - tstart;
    printf("Sum = %ld\n", result);
    printf("Elapsed time %lf\n\n", elapsed);
    
    if (BLKDIM == 1024 || BLKDIM == 512 || BLKDIM == 256 || 
    	BLKDIM == 128 || BLKDIM == 64 || BLKDIM == 32 || BLKDIM == 16 ||
    	BLKDIM == 8 || BLKDIM == 4 || BLKDIM == 2)
    {
    	printf("*** PARALLEL BY MEANS OF REDUCTION\n");
        tstart=gettime();
        sumReduction<<<totBlocks, BLKDIM>>>(d_a, d_sred, size);
        cudaMemcpy(sred, d_sred, totBlocks*sizeof(int), cudaMemcpyDeviceToHost);
        result = 0;
        for (i=0; i<totBlocks; i++) 
        	result += sred[i];
        tend=gettime();
        elapsed = tend - tstart;
        printf("Sum = %ld\n", result);
        printf("Elapsed time %lf\n\n", elapsed);        
    }

    /* Cleanup */
    free(a); free(s); free(sred);
    cudaFree(d_a); cudaFree(d_s); cudaFree(d_sred); cudaFree(d_result);
    return EXIT_SUCCESS;
}
