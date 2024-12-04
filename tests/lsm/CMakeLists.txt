# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

set(CMAKE_CXX_STANDARD 14)

# add_executable(build_lsm_index lsm/build_lsm_index.cpp)
# if(MSVC)
# 	target_link_options(build_lsm_index PRIVATE /MACHINE:x64 /DEBUG:FULL "/INCLUDE:_tcmalloc")
# 	target_link_libraries(build_lsm_index debug ${CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG}/nsg_dll.lib ${PROJECT_SOURCE_DIR}/dependencies/windows/tcmalloc/libtcmalloc_minimal.lib)
# 	target_link_libraries(build_lsm_index optimized ${CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE}/nsg_dll.lib ${PROJECT_SOURCE_DIR}/dependencies/windows/tcmalloc/libtcmalloc_minimal.lib)
# else() 
# 	target_link_libraries(build_lsm_index ${PROJECT_NAME} -ltcmalloc aio)
# endif()

# add_executable(search_lsm_index lsm/search_lsm_index.cpp
#     ${PROJECT_SOURCE_DIR}/src/aux_utils.cpp )
# if(MSVC)
# 	target_link_options(search_lsm_index PRIVATE /MACHINE:x64 /DEBUG:FULL)
# 	target_link_libraries(search_lsm_index debug ${CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG}/diskann_dll.lib)
# 	target_link_libraries(search_lsm_index optimized ${CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE}/diskann_dll.lib)
# else() 
# 	target_link_libraries(search_lsm_index ${PROJECT_NAME} aio -ltcmalloc)
# endif()

