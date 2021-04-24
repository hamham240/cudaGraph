#include "../../cudaGraph/cudaUtils/checkError.hpp"
#include <string>

namespace cudaGraph
{
    void checkError(cudaError_t err)
    {
        if (err != cudaSuccess)
        {
            printf("CUDA Runtime Error: %s\n", cudaGetErrorString(err));
            exit(-1);
        }
    }
}
