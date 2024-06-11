#include <stdio.h>
#include <omp.h>
#include "bellman_ford.h"
#include <cuda_runtime.h>

void evaluate_performance(void (*algorithm)(int, int, Edge*, int), int vertices, int edges, Edge *edgeList, int source, const char *name) {
    double start_time, end_time;
    
    start_time = omp_get_wtime();
    algorithm(vertices, edges, edgeList, source);
    end_time = omp_get_wtime();
    
    printf("%s execution time: %f seconds\n", name, end_time - start_time);
}

void evaluate_performance_cuda(void (*algorithm)(int, int, Edge*, int), int vertices, int edges, Edge *edgeList, int source, const char *name) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    algorithm(vertices, edges, edgeList, source);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("%s execution time: %f milliseconds\n", name, milliseconds);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    int vertices = 5;
    int edges = 8;
    Edge edgeList[] = {
        {0, 1, -1}, {0, 2, 4}, {1, 2, 3}, {1, 3, 2}, {1, 4, 2}, {3, 2, 5}, {3, 1, 1}, {4, 3, -3}
    };

    printf("Evaluating Bellman-Ford Parallel Basic...\n");
    evaluate_performance(bellman_ford_parallel_basic, vertices, edges, edgeList, 0, "Basic");

    printf("\nEvaluating Bellman-Ford Parallel Dynamic...\n");
    evaluate_performance(bellman_ford_parallel_dynamic, vertices, edges, edgeList, 0, "Dynamic");

    printf("\nEvaluating Bellman-Ford Parallel Queue...\n");
    evaluate_performance(bellman_ford_parallel_queue, vertices, edges, edgeList, 0, "Queue");

    printf("\nEvaluating Bellman-Ford CUDA...\n");
    evaluate_performance_cuda(bellman_ford_cuda, vertices, edges, edgeList, 0, "CUDA");

    return 0;
}
