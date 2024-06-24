/****************************************************************************
 *
 * omp-parallelfor.c 
 *
 * Compile with:
 * gcc -fopenmp omp-parallelfor.c -o omp-parallelfor
 *
 * Run with:
 * OMP_NUM_THREADS=8 ./omp-parallelfor
 *
 ****************************************************************************/
#include <stdio.h>
#include <omp.h>

int main( void )
{
	/*
	int i, incr=2;
	#pragma omp parallel num_threads(5)
	{
		// Try to change index type, loop check, add break..
		#pragma omp for 
		for(i=0;i<20;i=i+incr) {
			printf("Work %d assigned to thread %d\n", i, omp_get_thread_num());
		}
    }
    */
    double factor, sum;
    int k;
    
    factor = 1.0;
    sum = 0.0;
    #pragma omp parallel for 
    for (k=0; k<10000; k++) {
    	sum += factor/(2*k + 1);
    	#pragma omp atomic
    	factor = 0-factor;
    }
    printf("PI = %f\n", 4.0 * sum);

    factor = 1.0;
    sum = 0.0;
    #pragma omp parallel for private(factor) reduction(+:sum)
    for (k=0; k<10000; k++) {
    	if ( k % 2 == 0 ) {
    		factor = 1.0;
    	} else {
    		factor = -1.0;
    	}
    	sum += factor/(2*k + 1);
    }

    printf("PI = %f\n", 4.0 * sum);

    return 0;
    
    
}
