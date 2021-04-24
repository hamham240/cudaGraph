#include "../../cudaGraph/algos/cudaBFS.hpp"

namespace cudaGraph
{
    __global__ void
    BFS_initVectors(int* d_cost, int* d_frontier, int* d_visited, int vertexCount, int srcVertex)
    {
        int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

        if (tid < vertexCount)
        {
            d_visited[tid] = 0;

            if (tid != srcVertex)
            {
                d_cost[tid] = -1;
                d_frontier[tid] = 0;
            }
            else
            {
                d_cost[tid] = 0;
                d_frontier[tid] = 1;
            }
        }
    }

    __global__ void
    BFS(int* d_startIndices, int* d_endIndices, int* d_edges, int* d_frontier, int* d_visited,
        int* d_cost, int vertexCount, int edgeCount, int* stillSearching)
    {
        int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

        if (tid < vertexCount && d_frontier[tid])
        {
            d_frontier[tid] = 0;
            d_visited[tid] = 1;

            int startIndex = d_startIndices[tid];
            int endIndex = d_endIndices[tid];

            #pragma unroll
            for (int i = startIndex; i < endIndex; i++)
            {
                int neighbor = d_edges[i];

                if (d_visited[neighbor] == 0)
                {
                    d_cost[neighbor] = d_cost[tid] + 1;

                    d_frontier[neighbor] = 1;

                    *stillSearching = 1;
                }
            }
        }
    }

    std::vector<int> launchBFS(Graph &g, int srcVertex)
    {
        int vertexCount = g.vertices.size();
        int edgeCount = g.edges.size();

        int sizeOfVertices = vertexCount * sizeof(int);
        
        int block = 1024;
        int grid = (vertexCount/ 1024) + 1;

        int* d_frontier = NULL;
        int* d_visited = NULL;
        int* d_cost = NULL;
        int* stillSearching;

        checkError(cudaMalloc(&d_frontier, sizeOfVertices));
        checkError(cudaMalloc(&d_visited, sizeOfVertices));
        checkError(cudaMalloc(&d_cost, sizeOfVertices));
        checkError(cudaMallocHost((void**) &stillSearching, sizeof(int)));

        BFS_initVectors<<<grid, block>>>(d_cost, d_frontier, d_visited, vertexCount, srcVertex);
        checkError(cudaDeviceSynchronize());

        do
        {
            *stillSearching = 0;
            BFS<<<grid, block>>>(g.d_startIndices, g.d_endIndices, g.d_edges, d_frontier, d_visited,
                d_cost, vertexCount, edgeCount, stillSearching);
            checkError(cudaDeviceSynchronize());
        }
        while (*stillSearching);

        std::vector<int> cost(vertexCount);
        
        checkError(cudaMemcpy(cost.data(), d_cost, sizeOfVertices, cudaMemcpyDeviceToHost));

        checkError(cudaFree(d_frontier));
        checkError(cudaFree(d_visited));
        checkError(cudaFree(d_cost));

        return cost;
    }

    float launchTimedBFS(Graph &g, int srcVertex)
    {
        cudaEvent_t start, stop;
        float time;

        int vertexCount = g.vertices.size();
        int edgeCount = g.edges.size();

        int sizeOfVertices = vertexCount * sizeof(int);
        
        int block = 1024;
        int grid = (vertexCount/ 1024) + 1;

        int* d_frontier = NULL;
        int* d_visited = NULL;
        int* d_cost = NULL;
        int* stillSearching;

        checkError(cudaEventCreate(&start));
        checkError(cudaEventCreate(&stop));

        cudaEventRecord(start);

        checkError(cudaMalloc(&d_frontier, sizeOfVertices));
        checkError(cudaMalloc(&d_visited, sizeOfVertices));
        checkError(cudaMalloc(&d_cost, sizeOfVertices));
        checkError(cudaMallocHost((void**) &stillSearching, sizeof(int)));

        BFS_initVectors<<<grid, block>>>(d_cost, d_frontier, d_visited, vertexCount, srcVertex);
        checkError(cudaDeviceSynchronize());

        do
        {
            *stillSearching = 0;
            BFS<<<grid, block>>>(g.d_startIndices, g.d_endIndices, g.d_edges, d_frontier, d_visited,
                d_cost, vertexCount, edgeCount, stillSearching);
            checkError(cudaDeviceSynchronize());
        }
        while (*stillSearching);

        std::vector<int> cost(vertexCount);
        
        checkError(cudaMemcpy(cost.data(), d_cost, sizeOfVertices, cudaMemcpyDeviceToHost));

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        cudaEventElapsedTime(&time, start, stop);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        checkError(cudaFree(d_frontier));
        checkError(cudaFree(d_visited));
        checkError(cudaFree(d_cost));

        return time;
    }
}