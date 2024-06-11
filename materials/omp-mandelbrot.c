/****************************************************************************
*
* omp-mandelbrot.c - displays the Mandelbrot set
*
* Last updated in 2019 by Moreno Marzolla <moreno.marzolla(at)unibo.it>
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
* This program computes and display the Mandelbrot set. This program
* requires the gfx library from
* http://www.nd.edu/~dthain/courses/cse20211/fall2011/gfx (the
* library should be already included in the archive containing this
* source file)
*
* Compile with
* gcc -std=c99 -Wall -Wpedantic -fopenmp omp-mandelbrot.c gfx.c -o omp-mandelbrot -lX11
*
* Run with:
* OMP_NUM_THREADS=4 ./omp-mandelbrot
*
* If you enable the "runtime" scheduling clause, you can select the
* scheduling type at runtime, e.g.,
*
* OMP_NUM_THREADS=4 OMP_SCHEDULE="static,64" ./omp-mandelbrot
*
* At the end, click the left mouse button to close the graphical window.
*
****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "gfx.h"

const int MAXIT = 10000;
//const int XSIZE = 800, YSIZE = 600; 
const int XSIZE = 1024, YSIZE = 768;

typedef struct {
    int r, g, b;
} pixel_t;

/* color gradient from https://stackoverflow.com/questions/16500656/which-color-gradient-is-used-to-color-mandelbrot-in-wikipedia */
const pixel_t colors[] = {
    {66, 30, 15}, /* r, g, b */
    {25, 7, 26},
    {9, 1, 47},
    {4, 4, 73},
    {0, 7, 100},
    {12, 44, 138},    
    {24, 82, 177},
    {57, 125, 209},
    {134, 181, 229},
    {211, 236, 248},
    {241, 233, 191},
    {248, 201, 95},
    {255, 170, 0},
    {204, 128, 0},
    {153, 87, 0},
{106, 52, 3} };
const int NCOLORS = sizeof(colors)/sizeof(colors[0]);

/*
* Iterate the recurrence:
*
* z_0 = 0;
* z_{n+1} = z_n^2 + (cx + i*cy);
*
* Returns the first n such that ||z_n|| > 2, or |MAXIT|
*/
int iterate( float cx, float cy )
{
    float x = 0.0f, y = 0.0f, xnew, ynew;
    int it;
    for ( it = 0; (it < MAXIT) && (x*x + y*y <= 2.0*2.0); it++ ) {
        xnew = x*x - y*y + cx;
        ynew = 2.0*x*y + cy;
        x = xnew;
        y = ynew;
    }
    return it;
}

/*
* Draw a pixel at window coordinates (x, y) with the appropriate
* color; (0,0) is the upper left corner of the window, y grows
* downward.
*/
void drawpixel( int x, int y )
{
    const float cx = -2.5 + 3.5 * (float)(x) / (XSIZE - 1);
    const float cy = 1 - 2.0 * (float)(y) / (YSIZE - 1);
    const int v = iterate(cx, cy);
    /* only one thread can access the display at any time */
    #pragma omp critical 
    {
        if (v < MAXIT) {
            gfx_color( colors[v % NCOLORS].r,
            	colors[v % NCOLORS].g,
            	colors[v % NCOLORS].b );            
        } else {
            gfx_color( 0, 0, 0 );                        
        }
        gfx_point( x, y );
    }
}

int main( int argc, char *argv[] )
{
    int x, y;
    double tstart, elapsed;
    
    gfx_open(XSIZE, YSIZE, "Mandelbrot Set");
    tstart = omp_get_wtime();
    #pragma omp parallel for private(x) schedule(static,1)    
    for ( y = 0; y < YSIZE; y++ ) {
    	for ( x = 0; x < XSIZE; x++ ) {
    		drawpixel( x, y );
    	}
    }
    elapsed = omp_get_wtime() - tstart;
    printf("Elapsed time %f\n", elapsed);
    printf("Click to finish\n");
    gfx_wait();
    return EXIT_SUCCESS;
}