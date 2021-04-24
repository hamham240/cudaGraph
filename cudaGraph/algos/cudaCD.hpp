#ifndef CUDA_GRAPH_LIB_CUDACD_HPP
#define CUDA_GRAPH_LIB_CUDACD_HPP

#include <cuda_runtime.h>
#include "../graph/graph.hpp"
#include "../cudaUtils/checkError.hpp"

namespace cudaGraph
{
    bool launchHasCycle(Graph &g);

    float launchTimedHasCycle(Graph &g);
}

#endif