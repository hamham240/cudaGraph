#ifndef CUDA_GRAPH_LIB_SERIALBFS_HPP
#define CUDA_GRAPH_LIB_SERIALBFS_HPP

#include <ctime>
#include <queue>
#include "../graph/graph.hpp"

namespace cudaGraph
{
    std::vector<int> serialBFS(Graph &g, int srcVertex);
    float serialTimedBFS(Graph &g, int srcVertex);
}

#endif