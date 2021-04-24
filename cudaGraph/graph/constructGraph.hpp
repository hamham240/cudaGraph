#ifndef CUDA_GRAPH_LIB_CONSTRUCTGRAPH_HPP
#define CUDA_GRAPH_LIB_CONSTRUCTGRAPH_HPP

#include <vector>
#include <map>

namespace cudaGraph
{
    Graph constructGraph(std::vector<int> sources, std::vector<int> destinations, std::vector<int> weights);
    Graph constructGraph(std::vector<int> sources, std::vector<int> destinations);
}

#endif