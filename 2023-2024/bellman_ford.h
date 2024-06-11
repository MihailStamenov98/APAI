#ifndef BELLMAN_FORD_H
#define BELLMAN_FORD_H

typedef struct {
    int src, dest, weight;
} Edge;

// OpenMP Implementations
void bellman_ford_parallel_basic(int vertices, int edges, Edge *edgeList, int source);
void bellman_ford_parallel_dynamic(int vertices, int edges, Edge *edgeList, int source);
void bellman_ford_parallel_queue(int vertices, int edges, Edge *edgeList, int source);

// CUDA Implementation
void bellman_ford_cuda(int vertices, int edges, Edge *edgeList, int source);

#endif // BELLMAN_FORD_H
