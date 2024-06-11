/****************************************************************************
*
* omp-fibonacci-tasks.c - Compute Fibonacci numbers with OpenMP tasks
*
* Last updated in 2017 by Moreno Marzolla <moreno.marzolla(at)unibo.it>
*
* To the extent possible under law, the author(s) have dedicated all 
* copyright and related and neighboring rights to this software to the 
* public domain worldwide. This software is distributed without any warranty.
*
* You should have received a copy of the CC0 Public Domain Dedication
* along with this software. If not, see 
* <http://creativecommons.org/publicdomain/zero/1.0/>. 
*
* --------------------------------------------------------------------------
*
* Compile with:
* gcc -std=c99 -Wall -Wpedantic -fopenmp omp-fibonacci-tasks.c -o omp-fibonacci-tasks
*
* Run with:
* OMP_NUM_THREADS=4 ./omp-fibonacci-tasks 10
*
****************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

/* Compute the n-th Fibonacci number using OpenMP tasks.  This
algorithm is based on the inefficient recursive version that
performs O(2^n) calls. */
int fib( int n )
{
    int n1, n2;
    printf("Task fib(%d) assigned to thread %d\n", n, omp_get_thread_num());
    if (n < 2) {
        return 1;
    } else {
    	#pragma omp task shared(n1)
        n1 = fib(n-1);
        #pragma omp task shared(n2)
        n2 = fib(n-2);
        /* Wait for the two tasks above to complete */
        #pragma omp taskwait
        return n1 + n2;
    }
}

int main( int argc, char* argv[] )
{
    int n = 5, res;
    float tstart,elapsed;
    if ( argc == 2 ) {
        n = atoi(argv[1]);
    }
    /* Create a thread pool */
    tstart = omp_get_wtime();
    #pragma omp parallel
    {
        /* Only the master invokes the recursive algorithms (otherwise
        all threads in the pool would start the recursion) */
        #pragma omp master
        res = fib(n);
    }
    elapsed = omp_get_wtime() - tstart;
    printf("fib(%d)=%d\n", n, res);
    printf("Elapsed time %f\n", elapsed);
    return EXIT_SUCCESS;
}