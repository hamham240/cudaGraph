#include "../../cudaGraph/algos/serialAPSP.hpp"

namespace cudaGraph
{
    std::vector<std::vector<int>> serialAPSP(Graph &g)
    {
        int vertexCount = g.vertices.size();
        int edgeCount = g.edges.size();

        std::vector<std::vector<int>> costs(vertexCount);

        for (int i = 0; i < vertexCount; i++)
        {
            costs[i] = serialSSSP(g, i);
        }

        return costs;
    }

    float serialTimedAPSP(Graph &g)
    {
        std::clock_t start;
        double duration;

        start = std::clock();

        serialAPSP(g);

        duration = (std::clock() - start) / (double) CLOCKS_PER_SEC;

        return 1000*duration;
    }
}