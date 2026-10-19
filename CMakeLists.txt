cmake_minimum_required(VERSION 3.18)
project(flashattn_tile_native LANGUAGES CXX CUDA)

# Optional manual build. The Python package normally builds this through
# torch.utils.cpp_extension.load in flashattn_tile/native.py, which picks up the
# installed PyTorch toolchain automatically. This CMake path is for users who
# prefer a standalone shared library.

find_package(Torch REQUIRED)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

add_library(flashattn_native SHARED
    csrc/paged_decode.cu
    csrc/bindings.cpp)

target_include_directories(flashattn_native PRIVATE ${TORCH_INCLUDE_DIRS})
target_link_libraries(flashattn_native PRIVATE ${TORCH_LIBRARIES})
target_compile_options(flashattn_native PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)

set_target_properties(flashattn_native PROPERTIES
    PREFIX ""
    CUDA_SEPARABLE_COMPILATION ON)
