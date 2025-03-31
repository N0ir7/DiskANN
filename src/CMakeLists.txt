# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

set(CMAKE_CXX_STANDARD 17)

if(MSVC)
	add_subdirectory(dll)
else()
	#file(GLOB CPP_SOURCES *.cpp)
	set(CPP_SOURCES ann_exception.cpp aux_utils.cpp index.cpp
        linux_aligned_file_reader.cpp math_utils.cpp memory_mapper.cpp
        partition_and_pq.cpp pq_flash_index.cpp logger.cpp distance.cpp
		 utils.cpp v2/graph_delta.cpp v2/index_merger.cpp 
        v2/merge_insert.cpp
			lsm/merger/util/disk_index_merger.cpp lsm/merger/util/source_disk_index_merger.cpp lsm/merger/util/index_data_iterator.cpp lsm/merger/util/multi_index_iterator.cpp lsm/level/disk_index_proxy.cpp lsm/level/multi_disk_index_proxy.cpp lsm/level/mem_index_proxy.cpp lsm/level/multi_mem_index_proxy.cpp lsm/merger/level0_merger.cpp lsm/merger/leveln_merger.cpp lsm/merger/mem_flusher.cpp lsm/merger/mem_to_level1_merger.cpp lsm/lsm_index.cpp lsm/tag_deleter.cpp)
	add_library(${PROJECT_NAME} ${CPP_SOURCES})
	add_library(${PROJECT_NAME}_s STATIC ${CPP_SOURCES})
endif()
install()
