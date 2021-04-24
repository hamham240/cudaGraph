#ifndef CUDA_GRAPH_LIB_SERIALSSSP_HPP
#define CUDA_GRAPH_LIB_SERIALSSSP_HPP

#include "../graph/graph.hpp"
#include <ctime>

namespace cudaGraph
{
    std::vector<int> serialSSSP(Graph &g, int srcVertex);
    float serialTimedSSSP(Graph &g, int srcVertex);
}

#endif