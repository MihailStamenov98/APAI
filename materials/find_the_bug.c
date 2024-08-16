#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>

int main(int argc, char *argv[])
{
    int n = 5, m = 6;
    int *b = malloc(n * sizeof(int));
    int *d = malloc(n * sizeof(int));
    int *c = malloc(m * sizeof(int));
    int **a = malloc(n * sizeof(int));

    for (int i = 0; i < n; i++)
    {
        b[i] = rand();
        d[i] = rand();
        a[i] = malloc(m * sizeof(int));
        for (int j = 0; j < m; ++j)
        {
            a[i][j] = rand();
        }
    }
    for (int j = 0; j < m; ++j)
    {
        c[j] = rand();
    }
    float tstart, elapsed;
    if (argc == 2)
    {
        n = atoi(argv[1]);
    }
    /* Create a thread pool */
    tstart = omp_get_wtime();
    int i, j;
    int temp = 0;
#pragma omp parallel for private(temp)
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < m; j++)
        {
            temp = b[i] * c[j];
            printf("i = %d, j=%d, &i = %d, &j = %d,  thread = %d\n", i, j, &i, &j, omp_get_thread_num());
            a[i][j] = temp * temp + d[i];
        }
    }
    elapsed = omp_get_wtime() - tstart;
    printf("Elapsed time %f\n", elapsed);
    return EXIT_SUCCESS;
}