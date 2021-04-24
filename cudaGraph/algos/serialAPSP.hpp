#ifndef CUDA_GRAPH_LIB_SERIALAPSP_HPP
#define CUDA_GRAPH_LIB_SERIALAPSP_HPP

#include <ctime>
#include <queue>
#include "../graph/graph.hpp"
#include "serialSSSP.hpp"

namespace cudaGraph
{
    std::vector<std::vector<int>> serialAPSP(Graph &g);
    float serialTimedAPSP(Graph &g);
}

#endif