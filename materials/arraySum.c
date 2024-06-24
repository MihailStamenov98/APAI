/**************************
Sequential and several parallel solutions (some are WRONG)
to the problem of summing the elements in an array.

Compile with a c compiler supporting -fopenmp, e.g.

gcc -fopenmp arraySum.c -o arraySum

Execute with an appropriate number of threads, e.g.

OMP_NUM_THREADS=6 ./arraySum 2000000

(the passed parameter indicates the array length)
**************************/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

long global_sum1 = 0;
long global_sum2 = 0;
long global_sum3 = 0;
long global_sum4 = 0;
long global_sum5 = 0;
long global_sum6 = 0;
long global_sum7 = 0;
long global_sum8 = 0;
long global_sum9 = 0;

// sequential version 
void sum1(int* A, int n)
{
	int i,j;
	for (i=0; i<n; i++) {
		global_sum1 += A[i];
	}
}

// parallel but WRONG: concurrent updates of the counter
void sum2(int* A, int n)
{
	#pragma omp parallel
	{	
		int my_i,j;
		int my_block_len = n/omp_get_num_threads();    
		int my_start = omp_get_thread_num() * my_block_len; 
		int my_end = my_start + my_block_len;   
		for (my_i=my_start; my_i<my_end; my_i++) {
			global_sum2 += A[my_i];
		}
	}
}

// parallel with mutex on counter but WRONG partitioning (if num_threads is not a divisor of size)
void sum3(int* A, int n)
{
	#pragma omp parallel
	{
		int my_i,j;
		int my_block_len = n/omp_get_num_threads();    
		int my_start = omp_get_thread_num() * my_block_len; 
		int my_end = my_start + my_block_len;   
		for (my_i=my_start; my_i<my_end; my_i++) {
			#pragma omp atomic
			global_sum3 += A[my_i];
		}
	}
}

// parallel with mutex on counter 
void sum4(int* A, int n)
{
	#pragma omp parallel
	{
		int my_i,j;
		int my_start = n * omp_get_thread_num() / omp_get_num_threads();    
		int my_end = n * (omp_get_thread_num()+1) / omp_get_num_threads(); 
		for (my_i=my_start; my_i<my_end; my_i++) {
			#pragma omp atomic
			global_sum4 += A[my_i];
		}
	}
}

// parallel with local counters and mutex on the global counter
void sum5(int* A, int n)
{
	#pragma omp parallel
	{
		int my_i,j;
		int my_start = n * omp_get_thread_num() / omp_get_num_threads();    
		int my_end = n * (omp_get_thread_num()+1) / omp_get_num_threads();
		long my_sum = 0;
		for (my_i=my_start; my_i<my_end; my_i++) {
			my_sum += A[my_i];
		}
		#pragma omp atomic
		global_sum5 += my_sum;
	}
}

// parallel without mutex - WRONG without barrier synchronization
void sum6(int* A, int n)
{
	int i,j;
	int num_threads = omp_get_num_threads();
	int thread_num = omp_get_thread_num();
	long psum[num_threads];
	#pragma omp parallel
	{
		int my_i;
		int my_start = n * thread_num / num_threads;    
		int my_end = n * (thread_num+1) / num_threads;
		long my_sum = 0;
		for (my_i=my_start; my_i<my_end; my_i++) {
			my_sum += A[my_i];
		}
		psum[thread_num]=my_sum;
		if(thread_num==0)
			for (i=0; i<num_threads; i++)
			global_sum6 += psum[i];
	}
}

// parallel without mutex and with barrier synchronization
void sum7(int* A, int n)
{
	int i;
	int num_threads = omp_get_num_threads();
	int thread_num = omp_get_thread_num();
	long psum[num_threads];
	#pragma omp parallel
	{
		int my_i,j;
		int my_start = n * thread_num / num_threads;    
		int my_end = n * (thread_num+1) / num_threads;
		long my_sum = 0;
		for (my_i=my_start; my_i<my_end; my_i++) {
			my_sum += A[my_i];
		}
		psum[thread_num]=my_sum;
	}
	for (i=0; i<num_threads; i++)
		global_sum7 += psum[i];
	
}

// parallel with reduction on local sums
void sum8(int* A, int n)
{
	#pragma omp parallel reduction(+:global_sum8)
	{
		int my_i,j;
		int my_start = n * omp_get_thread_num() / omp_get_num_threads();    
		int my_end = n * (omp_get_thread_num()+1) / omp_get_num_threads();
		long my_sum = 0;
		for (my_i=my_start; my_i<my_end; my_i++) {
			my_sum += A[my_i];
		}
		global_sum8 += my_sum;
	}
}

// parallel with global reduction
void sum9(int* A, int n)
{
	int i,j;
	#pragma omp parallel for reduction(+:global_sum9)
	for (i=0; i<n; i++) {
		global_sum9 += A[i];
	}
}

int main(int argc, char *argv[])
{
	printf("*** HERE\n");

	char *a = argv[1];
    int size = atoi(a);
    int *A = malloc(size * sizeof(int));
    if (A == NULL) {
        printf("Memory allocation failed\n");
        return 1;
    }
    
    srand((unsigned int)time(NULL));    
    for (int i = 0; i < size; i++) {
        A[i] = rand() % 2;
    }
    
    printf("*** SEQUENTIAL WITH COUNTER\n");
    sum1(A, size);
    printf("Sum = %ld\n",global_sum1);
    
    printf("*** PARALLEL WITH CONCURRENT UPDATES OF COUNTER\n");
    sum2(A, size);
    printf("Sum = %ld\n",global_sum2);
    
    printf("*** PARALLEL WITH MUTEX ON COUNTER (BUT WRONG PARTITIONING)\n");
    sum3(A, size);
    printf("Sum = %ld\n",global_sum3);
    
    printf("*** PARALLEL WITH MUTEX ON COUNTER\n");
    sum4(A, size);
    printf("Sum = %ld\n",global_sum4);
    
    printf("*** PARALLEL WITH LOCAL COUNTERS AND MUTEX ON GLOBAL COUNTER\n");
    sum5(A, size);
    printf("Sum = %ld\n",global_sum5);
    
    printf("*** PARALLEL WITH NO MUTEX (BUT WITHOUT BARRIER SYNCHRONIZATION)\n");
    sum6(A, size);
    printf("Sum = %ld\n",global_sum6);
    
    printf("*** PARALLEL WITH NO MUTEX\n");
    sum7(A, size);
    printf("Sum = %ld\n",global_sum7);
    
    printf("*** PARALLEL REDUCTION OF LOCAL SUMS\n");
    sum8(A, size);
    printf("Sum = %ld\n",global_sum8);
    
    printf("*** PARALLEL BY MEANS OF REDUCTION\n");
    sum9(A, size);
    printf("Sum = %ld\n",global_sum9);
    
    return 0;
}