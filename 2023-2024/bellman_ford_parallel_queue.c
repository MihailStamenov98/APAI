#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <limits.h>
#include <stdbool.h>
#include "bellman_ford.h"

#define INF INT_MAX

typedef struct {
    int src, dest, weight;
} Edge;

void bellman_ford_parallel_queue(int vertices, int edges, Edge *edgeList, int source) {
    int *dist = (int *)malloc(vertices * sizeof(int));
    bool *updated = (bool *)malloc(vertices * sizeof(bool));
    for (int i = 0; i < vertices; i++) {
        dist[i] = INF;
        updated[i] = false;
    }
    dist[source] = 0;
    updated[source] = true;

    bool change = true;
    for (int i = 1; i <= vertices - 1 && change; i++) {
        change = false;
        #pragma omp parallel for
        for (int j = 0; j < edges; j++) {
            int u = edgeList[j].src;
            int v = edgeList[j].dest;
            int weight = edgeList[j].weight;
            if (updated[u] && dist[u] + weight < dist[v]) {
                #pragma omp critical
                {
                    dist[v] = dist[u] + weight;
                    updated[v] = true;
                    change = true;
                }
            }
        }
        #pragma omp parallel for
        for (int k = 0; k < vertices; k++) {
            updated[k] = false;
        }
    }

    for (int i = 0; i < vertices; i++) {
        printf("Vertex %d: %d\n", i, dist[i]);
    }

    free(dist);
    free(updated);
}

int main() {
    int vertices = 5;
    int edges = 8;
    Edge edgeList[] = {
        {0, 1, -1}, {0, 2, 4}, {1, 2, 3}, {1, 3, 2}, {1, 4, 2}, {3, 2, 5}, {3, 1, 1}, {4, 3, -3}
    };
    
    bellman_ford_parallel_queue(vertices, edges, edgeList, 0);
    return 0;
}
