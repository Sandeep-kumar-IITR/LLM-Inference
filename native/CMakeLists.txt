cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(llm_engine_native LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(Torch REQUIRED)

set(SOURCES
    cuda/bindings.cpp
    cuda/custom_ops.cpp
    cuda/custom_kernels.cu
    cpp/scheduler_utils.cpp
)

add_library(llm_engine_native SHARED ${SOURCES})

target_include_directories(llm_engine_native PRIVATE
    ${TORCH_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp
)

target_link_libraries(llm_engine_native PRIVATE ${TORCH_LIBRARIES})

set_target_properties(llm_engine_native PROPERTIES
    PREFIX ""
    CUDA_SEPARABLE_COMPILATION ON
)

if(MSVC)
    target_compile_options(llm_engine_native PRIVATE /O2 /std:c++17)
else()
    target_compile_options(llm_engine_native PRIVATE -O3 -std=c++17)
endif()
