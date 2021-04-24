#include "../../cudaGraph/algos/serialCD.hpp"

namespace cudaGraph
{
    bool serialCD(Graph &g)
    {
        int vertexCount = g.vertices.size();
        int edgeCount = g.edges.size();

        std::vector<int> in_degree(vertexCount, 0);

        for (int i = 0; i < vertexCount; i++)
        {
            for (int j = g.startIndices[i]; j < g.endIndices[i]; j++)
            {
                int neighbor = g.edges[j];
                in_degree[neighbor]++;
            }
        }

        std::queue<int> q;
        for (int i = 0; i < vertexCount; i++)
        {
            if (in_degree[i] == 0)
            {
                q.push(i);
            }
        }
            
        int count = 0;

        while (!q.empty())
        {
            int currentVertex = q.front();
            q.pop();

            for (int i = g.startIndices[currentVertex]; i < g.endIndices[currentVertex]; i++)
            {
                int neighbor = g.edges[i];

                if (--in_degree[neighbor] == 0)
                {
                    q.push(neighbor);
                }
            }
            count++;
        }

        if (count != vertexCount)
        {
            return true;
        }
        else
        {
            return false;
        }
    }

    float serialTimedCD(Graph &g)
    {
        std::clock_t start;
        double duration;

        start = std::clock();

        serialCD(g);

        duration = (std::clock() - start) / (double) CLOCKS_PER_SEC;

        return 1000*duration;
    }
}