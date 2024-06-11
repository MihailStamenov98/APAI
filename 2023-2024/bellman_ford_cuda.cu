#include "bellman_ford.h"
#include <cuda.h>
#include <stdio.h>
#include <limits.h>

#define INF INT_MAX

__global__ void relaxEdges(Edge *edgeList, int *dist, int edges) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < edges) {
        int u = edgeList[idx].src;
        int v = edgeList[idx].dest;
        int weight = edgeList[idx].weight;
        if (dist[u] != INF && dist[u] + weight < dist[v]) {
            dist[v] = dist[u] + weight;
        }
    }
}

void bellman_ford_cuda(int vertices, int edges, Edge *edgeList, int source) {
    int *dist = (int *)malloc(vertices * sizeof(int));
    for (int i = 0; i < vertices; i++) {
        dist[i] = INF;
    }
    dist[source] = 0;

    Edge *d_edgeList;
    int *d_dist;

    cudaMalloc(&d_edgeList, edges * sizeof(Edge));
    cudaMalloc(&d_dist, vertices * sizeof(int));

    cudaMemcpy(d_edgeList, edgeList, edges * sizeof(Edge), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dist, dist, vertices * sizeof(int), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (edges + blockSize - 1) / blockSize;

    for (int i = 1; i <= vertices - 1; i++) {
        relaxEdges<<<numBlocks, blockSize>>>(d_edgeList, d_dist, edges);
        cudaDeviceSynchronize();
    }

    cudaMemcpy(dist, d_dist, vertices * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < vertices; i++) {
        printf("Vertex %d: %d\n", i, dist[i]);
    }

    cudaFree(d_edgeList);
    cudaFree(d_dist);
    free(dist);
}
