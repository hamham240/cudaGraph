#ifndef CUDA_GRAPH_LIB_CUDASSSP_HPP
#define CUDA_GRAPH_LIB_CUDASSSP_HPP

#include <cuda_runtime.h>
#include "../graph/graph.hpp"
#include "../cudaUtils/checkError.hpp"

namespace cudaGraph
{
        std::vector<int> launchSSSP(Graph &g, int srcVertex);

        float launchTimedSSSP(Graph &g, int srcVertex);

        __global__ void
        SSSP_initVectors(int* d_cost, int vertexCount, int srcVertex);

        __global__ void
        bellmanFord(int* d_startIndices, int* d_endIndices, int* d_edges, int* d_weights, int* d_cost, int vertexCount);
}

#endif