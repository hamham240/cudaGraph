#include "../../cudaGraph/algos/cudaCD.hpp"

namespace cudaGraph
{
    __global__ void
    computeInDegree(int* d_startIndices, int* d_endIndices, int* d_edges,
                    int* d_inDegree, int vertexCount)
    {
        int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

        if (tid < vertexCount)
        {
            int startIndex = d_startIndices[tid];
            int endIndex = d_endIndices[tid];

            for (int i = startIndex; i < endIndex; i++)
            {
                int neighbor = d_edges[i];
                atomicAdd(&d_inDegree[neighbor], 1);
            }
        }
    }

    __global__ void
    loadQueue(int* d_inDegree, int* d_currentQueue, int* nextQueueSize, int vertexCount)
    {
        int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

        if (tid < vertexCount && d_inDegree[tid] == 0)
        {
            int queuePosition = atomicAdd(nextQueueSize, 1);
            d_currentQueue[queuePosition] = tid;
        }
    }

    __global__ void
    clearQueue(int* d_startIndices, int* d_endIndices, int* d_edges,
                int* d_inDegree, int* d_currentQueue, int* d_nextQueue,
                int* visitedCount,  int* nextQueueSize, int currentQueueSize)
    {
        int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

        if (tid < currentQueueSize)
        {
            atomicAdd(visitedCount, 1);

            int startIndex = d_startIndices[d_currentQueue[tid]];
            int endIndex = d_endIndices[d_currentQueue[tid]];

            for (int i = startIndex; i < endIndex; i++)
            {
                int neighbor = d_edges[i];
                atomicSub(&d_inDegree[neighbor], 1);
                if (d_inDegree[neighbor] == 0)
                {
                    int queuePosition = atomicAdd(nextQueueSize, 1);
                    d_nextQueue[queuePosition] = neighbor;
                }
            }
        }
    }

    bool launchHasCycle(Graph &g)
    {
        int vertexCount = g.vertices.size();
        int edgeCount = g.edges.size();

        int sizeOfVertices = vertexCount * sizeof(int);
        
        int block = 1024;
        int grid = (vertexCount / 1024) + 1;

        int* d_inDegree = NULL;
        int* d_currentQueue = NULL;
        int* d_nextQueue = NULL;
        int* visitedCount;
        int* nextQueueSize;
        int currentQueueSize;

        checkError(cudaMalloc(&d_inDegree, sizeOfVertices));
        checkError(cudaMalloc(&d_currentQueue, sizeOfVertices));
        checkError(cudaMalloc(&d_nextQueue, sizeOfVertices));
        checkError(cudaMallocHost((void**) &nextQueueSize, sizeof(int)));
        checkError(cudaMallocHost((void**) &visitedCount, sizeof(int)));

        computeInDegree<<<grid, block>>>(g.d_startIndices, g.d_endIndices, g.d_edges, d_inDegree, vertexCount);
        cudaDeviceSynchronize();

        loadQueue<<<grid, block>>>(d_inDegree, d_currentQueue, nextQueueSize, vertexCount);
        cudaDeviceSynchronize();

        currentQueueSize = *nextQueueSize;
        *nextQueueSize = 0;
        *visitedCount = 0;

        while (currentQueueSize > 0)
        {
            grid = (currentQueueSize / 1024) + 1;
            clearQueue<<<grid, block>>>(g.d_startIndices, g.d_endIndices, g.d_edges,
                d_inDegree, d_currentQueue, d_nextQueue,
                visitedCount,  nextQueueSize, currentQueueSize);
            cudaDeviceSynchronize();
            
            currentQueueSize = *nextQueueSize;
            *nextQueueSize = 0;

            std::swap(d_currentQueue, d_nextQueue);
        }

        checkError(cudaFree(d_inDegree));
        checkError(cudaFree(d_currentQueue));
        checkError(cudaFree(d_nextQueue));

        if (*visitedCount != vertexCount)
        {
            return true;
        }
        else
        {
            return false;
        }
    }

    float launchTimedHasCycle(Graph &g)
    {
        cudaEvent_t start, stop;
        float time;

        int vertexCount = g.vertices.size();
        int edgeCount = g.edges.size();

        int sizeOfVertices = vertexCount * sizeof(int);
        
        int block = 1024;
        int grid = (vertexCount / 1024) + 1;

        int* d_inDegree = NULL;
        int* d_currentQueue = NULL;
        int* d_nextQueue = NULL;
        int* visitedCount;
        int* nextQueueSize;
        int currentQueueSize;

        checkError(cudaEventCreate(&start));
        checkError(cudaEventCreate(&stop));

        cudaEventRecord(start);

        checkError(cudaMalloc(&d_inDegree, sizeOfVertices));
        checkError(cudaMalloc(&d_currentQueue, sizeOfVertices));
        checkError(cudaMalloc(&d_nextQueue, sizeOfVertices));
        checkError(cudaMallocHost((void**) &nextQueueSize, sizeof(int)));
        checkError(cudaMallocHost((void**) &visitedCount, sizeof(int)));

        computeInDegree<<<grid, block>>>(g.d_startIndices, g.d_endIndices, g.d_edges, d_inDegree, vertexCount);
        cudaDeviceSynchronize();

        loadQueue<<<grid, block>>>(d_inDegree, d_currentQueue, nextQueueSize, vertexCount);
        cudaDeviceSynchronize();

        currentQueueSize = *nextQueueSize;
        *nextQueueSize = 0;
        *visitedCount = 0;

        while (currentQueueSize > 0)
        {
            grid = (currentQueueSize / 1024) + 1;
            clearQueue<<<grid, block>>>(g.d_startIndices, g.d_endIndices, g.d_edges,
                d_inDegree, d_currentQueue, d_nextQueue,
                visitedCount,  nextQueueSize, currentQueueSize);
            cudaDeviceSynchronize();
            
            currentQueueSize = *nextQueueSize;
            *nextQueueSize = 0;

            std::swap(d_currentQueue, d_nextQueue);
        }

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        cudaEventElapsedTime(&time, start, stop);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        checkError(cudaFree(d_inDegree));
        checkError(cudaFree(d_currentQueue));
        checkError(cudaFree(d_nextQueue));

        return time;
    }
}