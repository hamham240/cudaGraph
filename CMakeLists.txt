cmake_minimum_required(VERSION 3.0)

#declaring project
project(cudaGraph)

#enabling CUDA
enable_language(CUDA)

#library headers
set(cudaGraph_HEADERS
	cudaGraph/cudaGraph.h
	cudaGraph/rapidcsv.h
	cudaGraph/algos/cudaBFS.hpp
	cudaGraph/algos/cudaDFS.hpp
	cudaGraph/algos/cudaSSSP.cuh
	cudaGraph/algos/cudaAPSP.hpp
	cudaGraph/algos/cudaCD.hpp
	cudaGraph/algos/serialBFS.hpp
	cudaGraph/algos/serialDFS.hpp
	cudaGraph/algos/serialSSSP.hpp
	cudaGraph/algos/serialAPSP.hpp
	cudaGraph/algos/serialCD.hpp
	cudaGraph/graph/graph.hpp
	cudaGraph/graph/constructGraph.hpp
	cudaGraph/graph/readCSV.hpp
	cudaGraph/graph/cudaGraphInit.hpp
	cudaGraph/graph/cudaGraphTerminate.hpp
	cudaGraph/cudaUtils/checkError.hpp
	)

#library sources
set(cudaGraph_SOURCES
	src/algos/cudaBFS.cu
	src/algos/cudaDFS.cu
	src/algos/cudaSSSP.cu
	src/algos/cudaAPSP.cu
	src/algos/cudaCD.cu
	src/algos/serialBFS.cpp
	src/algos/serialDFS.cpp
	src/algos/serialSSSP.cpp
	src/algos/serialAPSP.cpp
	src/algos/serialCD.cpp
	src/graph/graph.cpp
	src/graph/constructGraph.cpp
	src/graph/readCSV.cpp
	src/graph/cudaGraphInit.cpp
	src/graph/cudaGraphTerminate.cpp
	src/cudaUtils/checkError.cpp
	)

#building and linking
add_library(cudaGraph SHARED ${cudaGraph_HEADERS} ${cudaGraph_SOURCES})

#installation
install(TARGETS cudaGraph
		LIBRARY DESTINATION lib)
install(DIRECTORY cudaGraph/ DESTINATION include/cudaGraph)

#uninstallation
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/uninstall.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/uninstall.cmake"
    IMMEDIATE @ONLY
)

add_custom_target(uninstall
    "${CMAKE_COMMAND}" -P "${CMAKE_CURRENT_BINARY_DIR}/uninstall.cmake"
)

#finding and including CUDA
find_package(CUDA  REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")