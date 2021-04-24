#include "../../cudaGraph/graph/cudaGraphInit.hpp"
#include "../../cudaGraph/cudaUtils/checkError.hpp"

namespace cudaGraph
{
    void cudaGraphInit(Graph &g)
    {
        int vertexCount = g.vertices.size();
        int edgeCount = g.edges.size();

        int sizeOfVertices = vertexCount * sizeof(int);
        int sizeOfEdges = edgeCount * sizeof(int);

        g.d_startIndices = NULL;
        g.d_endIndices = NULL;
        g.d_edges = NULL;
        g.d_weights = NULL;

        checkError(cudaMalloc(&g.d_startIndices, sizeOfVertices));
        checkError(cudaMalloc(&g.d_endIndices, sizeOfVertices));
        checkError(cudaMalloc(&g.d_edges, sizeOfEdges));

        if (g.weights.size() > 0)
        {
            checkError(cudaMalloc(&g.d_weights, sizeOfEdges));
        }

        checkError(cudaMemcpy(g.d_startIndices, g.startIndices.data(), sizeOfVertices, cudaMemcpyHostToDevice));
        checkError(cudaMemcpy(g.d_endIndices, g.endIndices.data(), sizeOfVertices, cudaMemcpyHostToDevice));
        checkError(cudaMemcpy(g.d_edges, g.edges.data(), sizeOfEdges, cudaMemcpyHostToDevice));

        if (g.weights.size() > 0)
        {
            checkError(cudaMemcpy(g.d_weights, g.weights.data(), sizeOfEdges, cudaMemcpyHostToDevice));
        }
    }
}