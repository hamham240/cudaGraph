#ifndef CUDA_GRAPH_LIB_CUDABFS_HPP
#define CUDA_GRAPH_LIB_CUDABFS_HPP

#include <cuda_runtime.h>
#include "../graph/graph.hpp"
#include "../cudaUtils/checkError.hpp"

namespace cudaGraph
{
    std::vector<int> launchBFS(Graph &g, int srcVertex);
    float launchTimedBFS(Graph &g, int srcVertex);
}

#endif