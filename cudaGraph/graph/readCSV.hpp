#ifndef CUDA_GRAPH_LIB_READCSV_HPP
#define CUDA_GRAPH_LIB_READCSV_HPP

#include <string>
#include "../rapidcsv.h"

namespace cudaGraph
{
    Graph readWeightedCSV(std::string fileName);
    Graph readUnweightedCSV(std::string fileName);
}

#endif