#ifndef CUDA_GRAPH_LIB_CUDAAPSP_HPP
#define CUDA_GRAPH_LIB_CUDAAPSP_HPP

#include <cuda_runtime.h>
#include "../graph/graph.hpp"
#include "../cudaUtils/checkError.hpp"
#include "cudaSSSP.cuh"

namespace cudaGraph
{
    std::vector<std::vector<int>> launchAPSP(Graph &g);
    float launchTimedAPSP(Graph &g);
}

#endif