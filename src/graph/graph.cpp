#include "../../cudaGraph/graph/graph.hpp"

namespace cudaGraph
{
    void graphToCSV(Graph &g)
    {
        std::vector<int> sources;
        std::vector<int> destinations;
        std::vector<int> weights;

        for (int i = 0; i < g.vertices.size(); i++)
        {
            int startIndex = g.startIndices[i];
            int endIndex = g.endIndices[i];

            for(int j = startIndex; j < endIndex; j++)
            {
                sources.push_back(g.vertices[i]);
                destinations.push_back(g.vertices[g.edges[j]]);

                if (g.weights.size() > 0)
                {
                    weights.push_back(g.weights[j]);
                }
            }
        }

        std::ofstream myfile;
        myfile.open("output.csv");

        for(int i = 0; i < sources.size(); i++)
        {
            if (g.weights.size() > 0)
            {
                myfile << std::to_string(sources[i]) << "," << std::to_string(destinations[i]) << "," << std::to_string(weights[i]) << "\n";
            }
            else
            {
                myfile << std::to_string(sources[i]) << "," << std::to_string(destinations[i]) << "\n";
            }
        }
    }
}