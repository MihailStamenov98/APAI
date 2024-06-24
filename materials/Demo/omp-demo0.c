/****************************************************************************
 *
 * omp-demo0.c - "Hello world" with OpenMP
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
 * gcc -fopenmp omp-demo0.c -o omp-demo0
 *
 * Run with:
 * OMP_NUM_THREADS=8 ./omp-demo0
 *
 ****************************************************************************/
#include <stdio.h>

int main( void )
{
#pragma omp parallel
    {
        printf("Hello, world!\n");
    }
    
    return 0;
}
