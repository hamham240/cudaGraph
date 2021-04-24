#ifndef CUDA_GRAPH_LIB_CHECKERROR_HPP
#define CUDA_GRAPH_LIB_CHECKERROR_HPP

#include <cuda_runtime.h>
#include <string>

namespace cudaGraph
{
    void checkError(cudaError_t err);
}


#endif