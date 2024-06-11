#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <limits.h>
#include "bellman_ford.h"

#define INF INT_MAX

typedef struct {
    int src, dest, weight;
} Edge;

void bellman_ford_parallel_basic(int vertices, int edges, Edge *edgeList, int source) {
    int *dist = (int *)malloc(vertices * sizeof(int));
    for (int i = 0; i < vertices; i++) {
        dist[i] = INF;
    }
    dist[source] = 0;

    for (int i = 1; i <= vertices - 1; i++) {
        #pragma omp parallel for
        for (int j = 0; j < edges; j++) {
            int u = edgeList[j].src;
            int v = edgeList[j].dest;
            int weight = edgeList[j].weight;
            if (dist[u] != INF && dist[u] + weight < dist[v]) {
                dist[v] = dist[u] + weight;
            }
        }
    }

    for (int i = 0; i < vertices; i++) {
        printf("Vertex %d: %d\n", i, dist[i]);
    }

    free(dist);
}

int main() {
    int vertices = 5;
    int edges = 8;
    Edge edgeList[] = {
        {0, 1, -1}, {0, 2, 4}, {1, 2, 3}, {1, 3, 2}, {1, 4, 2}, {3, 2, 5}, {3, 1, 1}, {4, 3, -3}
    };
    
    bellman_ford_parallel_basic(vertices, edges, edgeList, 0);
    return 0;
}
