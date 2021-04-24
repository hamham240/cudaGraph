#include "../../cudaGraph/graph/cudaGraphTerminate.hpp"
#include "../../cudaGraph/cudaUtils/checkError.hpp"

namespace cudaGraph
{
    void cudaGraphTerminate(Graph &g)
    {
        checkError(cudaFree(g.d_startIndices));
        checkError(cudaFree(g.d_endIndices));
        checkError(cudaFree(g.d_edges));

        if (g.weights.size() > 0)
        {
            checkError(cudaFree(g.d_weights));
        }
    }
}