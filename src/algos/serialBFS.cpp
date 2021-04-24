#include "../../cudaGraph/algos/serialBFS.hpp"

namespace 
{
    std::vector<int> createCostVec(int numVertices, int srcVertex)
    {
        std::vector<int> cost;

        for (int i = 0; i < numVertices; i++)
        {
            if (i != srcVertex)
            {
                cost.push_back(-1);
            }
            else
            {
                cost.push_back(0);
            }
        }

        return cost;
    }
}

namespace cudaGraph
{
    std::vector<int> serialBFS(Graph &g, int srcVertex)
    {
        std::vector<int> cost = createCostVec(g.vertices.size(), srcVertex);
        std::vector<bool> visited(g.vertices.size());

        std::queue<int> q;

        q.push(srcVertex);
        visited[srcVertex] = true;

        while (!q.empty())
        {
            int currSize = q.size();

            for (int i = 0; i < currSize; i++)
            {
                int currVertex = q.front();
                q.pop();

                int startIndex = g.startIndices[currVertex];
                int endIndex = g.endIndices[currVertex];

                for (int j = startIndex; j < endIndex; j++)
                {
                    int neighbor = g.edges[j];

                    if (!visited[neighbor])
                    {
                        cost[neighbor] = cost[currVertex] + 1;
                        visited[neighbor] = true;
                        q.push(neighbor);
                    }
                }
            }
        }

        return cost;
    }

    float serialTimedBFS(Graph &g, int srcVertex)
    {
        std::clock_t start;
        double duration;

        start = std::clock();

        serialBFS(g, srcVertex);

        duration = (std::clock() - start) / (double) CLOCKS_PER_SEC;

        return 1000*duration;
    }
}
