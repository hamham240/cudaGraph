What is Cuda Graph Library?
===========================
Cuda Graph Library is a CUDA library that offers a set of parallel graph algorithms.

Which Algorithms are Currently Supported?
=========================================
* Breadth-First Search
* Cycle Detection
* Single Source Shortest Path
* All Pairs Shortest Path

Software Requirements
=====================
* Linux OS
* CUDA Toolkit v11.2+
* CMake v3.0+

Hardware Requirements
=====================
* NVIDIA GPU with compute capability of 3 or higher

Installation
============
Run these set of commands in your CLI in the order that they are presented:

```bash
$ git clone https://github.com/hamham240/cudaGraph.git

$ cd cudaGraph

$ cmake .

$ cmake --build .

$ sudo make install
```

Uninstallation
==============
```bash
$ cd cudaGraph

$ sudo make uninstall
```

Example
=======
```cpp
	#include "cudaGraph/cudaGraph.h"
	
	namespace cgl = cudaGraph;

	int main()
	{
		cgl::Graph g = cgl::readWeightedCSV("example.csv");

		cgl::cudaGraphInit(g);

		std::vector<int> bfsOutput = cgl::launchBFS(g, 0);

		cgl::cudaGraphTerminate(g);
	}
```

Sources
=======
* rapidCSV Header-Only Library
	* https://github.com/d99kris/rapidcsv
	
* Parallel BFS Algorithm
	* Harish, Pawan, and P. J. Narayanan. 
	  Accelerating Large Graph Algorithms on the GPU Using CUDA. 2007.

* Parallel Cycle Detection Algorithm
	* [GeeksforGeeks Detect Cycles in a Directed Graph using BFS](https://www.geeksforgeeks.org/detect-cycle-in-a-directed-graph-using-bfs/#:~:text=Steps%20involved%20in%20detecting%20cycle,of%20visited%20nodes%20as%200.&text=Step%2D3%3A%20Remove%20a%20vertex,of%20visited%20nodes%20by%201.)

* Parallel Bellman-Ford Algorithm
	* [GeeksforGeeks Bellman-Ford Algorithm](https://www.geeksforgeeks.org/bellman-ford-algorithm-dp-23/)
