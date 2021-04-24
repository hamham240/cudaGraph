#include "../../cudaGraph/graph/graph.hpp"
#include "../../cudaGraph/graph/constructGraph.hpp"
#include <cstdio>

namespace cudaGraph
{
    Graph constructGraph(std::vector<int> sources, std::vector<int> destinations, std::vector<int> wts)
    {
        Graph g;
        std::map<int, int> indicesToVertices;
        std::map<int, int> verticesToIndices;
        std::map<int, std::vector<int>> srcToDests;
        int indexCounter = 0;

        for (int i = 0; i < sources.size(); i++)
        {
            if (verticesToIndices.count(sources[i]) == 0)
            {
                indicesToVertices.insert(std::pair<int, int>(indexCounter, sources[i]));
                verticesToIndices.insert(std::pair<int, int>(sources[i], indexCounter++));
            }

            if (verticesToIndices.count(destinations[i]) == 0)
            {
                indicesToVertices.insert(std::pair<int, int>(indexCounter, destinations[i]));
                verticesToIndices.insert(std::pair<int, int>(destinations[i], indexCounter++));
            }

            if (srcToDests.count(sources[i]) == 0)
            {
                std::vector<int> temp;
                temp.push_back(destinations[i]);
                temp.push_back(wts[i]);

                srcToDests.insert(std::pair<int, std::vector<int>>(sources[i], temp));
            }
            else
            {
                srcToDests[sources[i]].push_back(destinations[i]);
                srcToDests[sources[i]].push_back(wts[i]);
            }

            if (srcToDests.count(destinations[i]) == 0)
            {
                std::vector<int> temp;
                srcToDests[destinations[i]] = temp;
            }
        }

        std::vector<int> vertices(indicesToVertices.size());
        std::vector<int> startIndices(indicesToVertices.size());
        std::vector<int> endIndices(indicesToVertices.size());
        std::vector<int> edges;
        std::vector<int> weights;

        for (auto const& entry : indicesToVertices)
        {
            vertices[entry.first] = entry.second;

            if (srcToDests[entry.second].size() > 0)
            {
                startIndices[entry.first] = edges.size();

                for (int i = 0; i < srcToDests[entry.second].size(); i+=2)
                {
                    edges.push_back(verticesToIndices[srcToDests[entry.second][i]]);
                    weights.push_back(srcToDests[entry.second][i+1]);
                }

                endIndices[entry.first] = edges.size();
            }
            else 
            {
                startIndices[entry.first] = destinations.size();
                endIndices[entry.first] = destinations.size();
            }
            
        }

        g.vertices = vertices;
        g.startIndices = startIndices;
        g.endIndices = endIndices;
        g.edges = edges;
        g.weights = weights;

        return g;
    }

    Graph constructGraph(std::vector<int> sources, std::vector<int> destinations)
    {
        Graph g;
        std::map<int, int> indicesToVertices;
        std::map<int, int> verticesToIndices;
        std::map<int, std::vector<int>> srcToDests;
        int indexCounter = 0;

        for (int i = 0; i < sources.size(); i++)
        {
            if (verticesToIndices.count(sources[i]) == 0)
            {
                indicesToVertices.insert(std::pair<int, int>(indexCounter, sources[i]));
                verticesToIndices.insert(std::pair<int, int>(sources[i], indexCounter++));
            }

            if (verticesToIndices.count(destinations[i]) == 0)
            {
                indicesToVertices.insert(std::pair<int, int>(indexCounter, destinations[i]));
                verticesToIndices.insert(std::pair<int, int>(destinations[i], indexCounter++));
            }

            if (srcToDests.count(sources[i]) == 0)
            {
                std::vector<int> temp;
                temp.push_back(destinations[i]);

                srcToDests.insert(std::pair<int, std::vector<int>>(sources[i], temp));
            }
            else
            {
                srcToDests[sources[i]].push_back(destinations[i]);
            }

            if (srcToDests.count(destinations[i]) == 0)
            {
                std::vector<int> temp;
                srcToDests[destinations[i]] = temp;
            }
        }

        std::vector<int> vertices(indicesToVertices.size());
        std::vector<int> startIndices(indicesToVertices.size());
        std::vector<int> endIndices(indicesToVertices.size());
        std::vector<int> edges;

        for (auto const& entry : indicesToVertices)
        {
            vertices[entry.first] = entry.second;

            if (srcToDests[entry.second].size() > 0)
            {
                startIndices[entry.first] = edges.size();

                for (int i = 0; i < srcToDests[entry.second].size(); i++)
                {
                    edges.push_back(verticesToIndices[srcToDests[entry.second][i]]);
                }
                endIndices[entry.first] = edges.size();
            }
            else 
            {
                startIndices[entry.first] = destinations.size();
                endIndices[entry.first] = destinations.size();
            }  
        }

        g.vertices = vertices;
        g.startIndices = startIndices;
        g.endIndices = endIndices;
        g.edges = edges;

        if (sources.size() != g.edges.size())
        {
            printf("NO. OF EDGES DOES NOT MATCH CSV\n");
        }

        return g;
    }
}