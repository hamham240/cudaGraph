#include "../../cudaGraph/algos/cudaAPSP.hpp"

namespace cudaGraph
{
    std::vector<std::vector<int>> launchAPSP(Graph &g)
    {
        int vertexCount = g.vertices.size();
        int edgeCount = g.edges.size();

        int sizeOfVertices = vertexCount * sizeof(int);
        
        int block = 1024;
        int grid = (vertexCount/ 1024) + 1;

        int* d_cost = NULL;

        checkError(cudaMalloc(&d_cost, sizeOfVertices));

        std::vector<std::vector<int>> costs(vertexCount);

        for (int i = 0; i < vertexCount; i++)
        {
            SSSP_initVectors<<<grid, block>>>(d_cost, vertexCount, i);
            cudaDeviceSynchronize();
    
            for (int j = 0; j < vertexCount - 1; j++)
            {
                bellmanFord<<<grid, block>>>(g.d_startIndices, g.d_endIndices, g.d_edges, g.d_weights, d_cost, vertexCount);
                cudaDeviceSynchronize();
            }
    
            std::vector<int> currentVertexCost(vertexCount);
    
            checkError(cudaMemcpy(currentVertexCost.data(), d_cost, sizeOfVertices, cudaMemcpyDeviceToHost));

            costs[i] = currentVertexCost;
        }

        checkError(cudaFree(d_cost));

        return costs;
    }

    float launchTimedAPSP(Graph &g)
    {
        cudaEvent_t start, stop;
        float time;

        int vertexCount = g.vertices.size();
        int edgeCount = g.edges.size();

        int sizeOfVertices = vertexCount * sizeof(int);
        
        int block = 1024;
        int grid = (vertexCount/ 1024) + 1;

        int* d_cost = NULL;

        checkError(cudaEventCreate(&start));
        checkError(cudaEventCreate(&stop));

        cudaEventRecord(start);

        checkError(cudaMalloc(&d_cost, sizeOfVertices));

        std::vector<std::vector<int>> costs(vertexCount);

        for (int i = 0; i < vertexCount; i++)
        {
            SSSP_initVectors<<<grid, block>>>(d_cost, vertexCount, i);
            cudaDeviceSynchronize();
    
            for (int j = 0; j < vertexCount - 1; j++)
            {
                bellmanFord<<<grid, block>>>(g.d_startIndices, g.d_endIndices, g.d_edges, g.d_weights, d_cost, vertexCount);
                cudaDeviceSynchronize();
            }
    
            std::vector<int> currentVertexCost(vertexCount);
    
            checkError(cudaMemcpy(currentVertexCost.data(), d_cost, sizeOfVertices, cudaMemcpyDeviceToHost));

            costs[i] = currentVertexCost;
        }

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        cudaEventElapsedTime(&time, start, stop);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        checkError(cudaFree(d_cost));

        return time;
    }
}