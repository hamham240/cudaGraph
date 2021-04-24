#include "../../cudaGraph/algos/cudaSSSP.cuh"

namespace cudaGraph
{
    __global__ void
    SSSP_initVectors(int* d_cost, int vertexCount, int srcVertex)
    {
        int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

        if (tid < vertexCount)
        {
            if (tid != srcVertex)
            {
                d_cost[tid] = INT32_MAX;
            }
            else
            {
                d_cost[tid] = 0;
            }
            
        }
    }

    __global__ void
    bellmanFord(int* d_startIndices, int* d_endIndices, int* d_edges, int* d_weights, int* d_cost, int vertexCount)
    {
        int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

        if (tid < vertexCount)
        {
            int startIndex = d_startIndices[tid];
            int endIndex = d_endIndices[tid];

            #pragma unroll
            for (int i = startIndex; i < endIndex; i++)
            {
                int neighbor = d_edges[i];
                int weight = d_weights[i];

                if (d_cost[tid] != INT32_MAX && d_cost[tid] + weight < d_cost[neighbor])
                {
                    d_cost[neighbor] = d_cost[tid] + weight;
                }
            }
        }
    }

    std::vector<int> launchSSSP(Graph &g, int srcVertex)
    {
        int vertexCount = g.vertices.size();
        int edgeCount = g.edges.size();

        int sizeOfVertices = vertexCount * sizeof(int);
        
        int block = 1024;
        int grid = (vertexCount/ 1024) + 1;

        int* d_cost = NULL;

        cudaMalloc(&d_cost, sizeOfVertices);

        SSSP_initVectors<<<grid, block>>>(d_cost, vertexCount, srcVertex);
        cudaDeviceSynchronize();

        for (int i = 0; i < vertexCount - 1; i++)
        {
            bellmanFord<<<grid, block>>>(g.d_startIndices, g.d_endIndices, g.d_edges, g.d_weights, d_cost, vertexCount);
            cudaDeviceSynchronize();
        }

        std::vector<int> cost(vertexCount);

        cudaMemcpy(cost.data(), d_cost, sizeOfVertices, cudaMemcpyDeviceToHost);

        cudaFree(d_cost);

        return cost;
    }

    float launchTimedSSSP(Graph &g, int srcVertex)
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

        cudaMalloc(&d_cost, sizeOfVertices);

        SSSP_initVectors<<<grid, block>>>(d_cost, vertexCount, srcVertex);
        cudaDeviceSynchronize();

        for (int i = 0; i < vertexCount - 1; i++)
        {
            bellmanFord<<<grid, block>>>(g.d_startIndices, g.d_endIndices, g.d_edges, g.d_weights, d_cost, vertexCount);
            cudaDeviceSynchronize();
        }

        std::vector<int> cost(vertexCount);

        cudaMemcpy(cost.data(), d_cost, sizeOfVertices, cudaMemcpyDeviceToHost);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        cudaEventElapsedTime(&time, start, stop);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        cudaFree(d_cost);

        return time;
    }
}
