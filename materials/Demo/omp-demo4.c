/****************************************************************************
 *
 * omp-demo4.c - Demo with OpenMP
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
 * gcc -fopenmp omp-demo4.c -o omp-demo4 -lgomp
 *
 * Run with:
 * ./omp-demo4
 *
 ****************************************************************************/
#include <stdio.h>
#include <omp.h>

int main( int argc, char* argv[] )
{
    printf("Before parallel region: threads=%d, max_threads=%d\n",
           omp_get_num_threads(), omp_get_max_threads());
#pragma omp parallel
    {
        printf("Inside parallel region: threads=%d, max_threads=%d\n",
               omp_get_num_threads(), omp_get_max_threads());
    }

    return 0;
}
