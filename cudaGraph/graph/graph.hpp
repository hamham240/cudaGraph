#ifndef CUDA_GRAPH_LIB_GRAPH_HPP
#define CUDA_GRAPH_LIB_GRAPH_HPP

#include <vector>
#include <iostream>
#include <fstream>

namespace cudaGraph
{
    struct Graph 
    {
        std::vector<int> vertices;
        std::vector<int> startIndices;
        std::vector<int> endIndices;
        std::vector<int> edges;
        std::vector<int> weights;
        int* d_startIndices;
        int* d_endIndices;
        int* d_edges;
        int* d_weights;
    };

    void graphToCSV(Graph &g);
}

#endif