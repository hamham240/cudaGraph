#include "../../cudaGraph/graph/graph.hpp"
#include "../../cudaGraph/graph/readCSV.hpp"
#include "../../cudaGraph/graph/constructGraph.hpp"

namespace {
    template<typename elementType>
    bool contains(std::vector<elementType> vec, elementType target)
    {
        bool exists = false;

        for (elementType ele : vec)
        {
            if (ele == target)
            {
                exists = true;
                break;
            }
        }

        return exists;
    }
}

namespace cudaGraph
{
    Graph readUnweightedCSV(std::string fileName)
    {
        rapidcsv::Document doc(fileName, rapidcsv::LabelParams(-1, -1));

        std::vector<int> source = doc.GetColumn<int>(0);
        std::vector<int> dest = doc.GetColumn<int>(1);

        return constructGraph(source, dest);
    }

    Graph readWeightedCSV(std::string fileName)
    {
        rapidcsv::Document doc(fileName, rapidcsv::LabelParams(-1, -1));

        std::vector<int> source = doc.GetColumn<int>(0);
        std::vector<int> dest = doc.GetColumn<int>(1);
        std::vector<int> weights = doc.GetColumn<int>(2);

        return constructGraph(source, dest, weights);
    }
}
