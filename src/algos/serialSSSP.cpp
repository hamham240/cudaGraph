#include "../../cudaGraph/algos/serialSSSP.hpp"

namespace cudaGraph
{
    std::vector<int> serialSSSP(Graph &g, int srcVertex)
    {
        int vertexCount = g.vertices.size();
        int edgeCount = g.edges.size();

        std::vector<int> costs(vertexCount);

        for (int i = 0; i < vertexCount; i++)
        {
            costs[i] = INT32_MAX;
        }

        costs[srcVertex] = 0;

        for (int i = 0; i < vertexCount - 1; i++) 
        {
            for (int v = 0; v < vertexCount; v++)
            {
                for (int e = g.startIndices[v]; e < g.endIndices[v]; e++)
                {
                    int u = g.edges[e];
                    int weight = g.weights[e];

                    if (costs[v] != INT32_MAX && costs[v] + weight < costs[u])
                    {
                        costs[u] = costs[v] + weight;
                    }
                }
            }
        }

        return costs;
    }

    float serialTimedSSSP(Graph &g, int srcVertex)
    {
        std::clock_t start;
        double duration;

        start = std::clock();

        serialSSSP(g, srcVertex);

        duration = (std::clock() - start) / (double) CLOCKS_PER_SEC;

        return 1000*duration;
    }
}