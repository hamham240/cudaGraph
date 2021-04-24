#ifndef CUDA_GRAPH_LIB_SERIALCD_HPP
#define CUDA_GRAPH_LIB_SERIALCD_HPP

#include "../graph/graph.hpp"
#include <ctime>
#include <queue>

namespace cudaGraph
{
    bool serialCD(Graph &g);
    float serialTimedCD(Graph &g);
}

#endif