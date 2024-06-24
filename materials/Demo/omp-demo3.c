/****************************************************************************
 *
 * omp-demo3.c - Demo with OpenMP
 *
 * Last updated in in 2017 by Moreno Marzolla <moreno.marzolla(at)unibo.it>
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
 * gcc -fopenmp omp-demo3.c -o omp-demo3 -lgomp
 *
 * Run with:
 * ./omp-demo3
 *
 ****************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void say_hello( void )
{
    int my_rank = omp_get_thread_num();
    int thread_count = omp_get_num_threads();
    printf("Hello from thread %d of %d\n", my_rank, thread_count);
}

int main( int argc, char* argv[] )
{
    /* Request 4 threads, no matter what */
    omp_set_num_threads(4);

#pragma omp parallel
    say_hello();

    return 0;
}
