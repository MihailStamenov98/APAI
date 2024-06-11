/****************************************************************************
 *
 * omp-GameOfLife.c - Implementation of the Game of Life
 *
 * Written in 2017 by Moreno Marzolla <moreno.marzolla(at)unibo.it>
 * Last updated on 2019-10-01
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
 * gcc -fopenmp -std=c99 -Wall -Wpedantic omp-GameOfLife.c -o omp-GameOfLife
 *
 * Run with:
 * ./omp-GameOfLife 100
 *
 * To create an animated gif with ImageMagick:
 * magick -delay 10 gol*.pbm gol.gif
 *
 ****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <time.h> /* for time() */
#include <omp.h>

/* grid size (excluding ghost cells) */
#define SIZE 256
//#define SIZE 10000

int cur = 0; /* index of current grid (must be 0 or 1) */
unsigned char grid[2][SIZE+2][SIZE+2];

/* some useful constants; starting and ending rows/columns of the domain */
const int ISTART = 1;
const int IEND   = SIZE; 
const int JSTART = 1;
const int JEND   = SIZE;

/*
     JSTART         JEND  
     |              |
     v              v
  +-+----------------+-+
  |\|\\\\\\\\\\\\\\\\|\|
  +-+----------------+-+
  |\|                |\| <- ISTART
  |\|                |\|
  |\|                |\|
  |\|                |\|
  |\|                |\|
  |\|                |\| <- IEND
  +-+----------------+-+
  |\|\\\\\\\\\\\\\\\\|\| 
  +-+----------------+-+

 */

/* copy the sides of current grid to the ghost cells. This function
   uses the global variables cur and grid. grid[cur] is modified.*/
void copy_sides( void )
{
    int i, j;
    /* copy top and bottom (one should better use memcpy() ) */
    for (j=JSTART; j<JEND+1; j++) {
        grid[cur][ISTART-1][j] = grid[cur][IEND  ][j];
        grid[cur][IEND+1  ][j] = grid[cur][ISTART][j];
    }
    /* copy left and right */
    for (i=ISTART; i<IEND+1; i++) {
        grid[cur][i][JSTART-1] = grid[cur][i][JEND  ];
        grid[cur][i][JEND+1  ] = grid[cur][i][JSTART];
    }
    /* copy corners */
    grid[cur][ISTART-1][JSTART-1] = grid[cur][IEND  ][JEND  ];
    grid[cur][ISTART-1][JEND+1  ] = grid[cur][IEND  ][JSTART];
    grid[cur][IEND+1  ][JSTART-1] = grid[cur][ISTART][JEND  ];
    grid[cur][IEND+1  ][JEND+1  ] = grid[cur][ISTART][JSTART];
}

/* Compute the next grid given the current configuration; this
   function uses the global variables grid and cur; updates are
   written to the (1-cur) grid. */
void step( void )
{
    int i, j, next = 1 - cur;
    #pragma omp for collapse(2)
      for (i=ISTART; i<IEND+1; i++) {
        for (j=JSTART; j<JEND+1; j++) {
            /* count live neighbors of cell (i,j) */
            int nbors = 
                grid[cur][i-1][j-1] + grid[cur][i-1][j] + grid[cur][i-1][j+1] + 
                grid[cur][i  ][j-1] +                     grid[cur][i  ][j+1] + 
                grid[cur][i+1][j-1] + grid[cur][i+1][j] + grid[cur][i+1][j+1];
            /* apply rules of the game of life to cell (i, j) */
            if ( grid[cur][i][j] && (nbors < 2 || nbors > 3) ) {
                grid[next][i][j] = 0;
            } else {
                if ( !grid[cur][i][j] && (nbors == 3) ) {
                    grid[next][i][j] = 1;
                } else {
                    grid[next][i][j] = grid[cur][i][j];
                }
            }
        }
    }
}

/* Initialize the current grid grid[cur] with alive cells with density
   p. This function uses the global variables cur and grid. grid[cur]
   is modified. */
void init( float p )
{
    int i, j;
    for (i=ISTART; i<IEND+1; i++) {
        for (j=JSTART; j<JEND+1; j++) {
            grid[cur][i][j] = (((float)rand())/RAND_MAX < p);
        }
    }
}

/* Write grid[cur] to file fname in pbm (portable bitmap) format. This
   function uses the global variables cur and grid (neither is
   modified). */
void write_pbm( const char* fname )
{
    int i, j;
    FILE *f = fopen(fname, "w");
    if (!f) { 
        printf("Cannot open %s for writing\n", fname);
        abort();
    }
    fprintf(f, "P1\n");
    fprintf(f, "# produced by game-of-life.c\n");
    fprintf(f, "%d %d\n", SIZE, SIZE);
    for (i=ISTART; i<IEND+1; i++) {
        for (j=JSTART; j<JEND+1; j++) {
            fprintf(f, "%d ", grid[cur][i][j]);
        }
        fprintf(f, "\n");
    }
    fclose(f);
}

#define BUFSIZE 128

int main( int argc, char* argv[] )
{
    int s, nsteps = 1000;
    char fname[BUFSIZE];
    double tstart, elapsed;

    srand(time(NULL)); /* init RNG */
    if ( argc > 2 ) {
        fprintf(stderr, "Usage: %s [nsteps]\n", argv[0]);
        return EXIT_FAILURE;
    }
    if ( argc == 2 ) {
        nsteps = atoi(argv[1]);
    }
    cur = 0;
    init(0.3);
    cur = 1;
    tstart = omp_get_wtime();
   
    #pragma omp parallel private(s)
    for (s=0; s<nsteps; s++) {
    	#pragma omp single
    	{
    		snprintf(fname, BUFSIZE, "gol%04d.pbm", s);
    		write_pbm(fname);
    		copy_sides();
    		cur = 1 - cur;
    	}
    	step();
    }
    
    elapsed = omp_get_wtime() - tstart;
    printf("Elapsed time %f\n", elapsed);
    return EXIT_SUCCESS;
}