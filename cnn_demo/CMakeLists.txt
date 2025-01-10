cmake_minimum_required(VERSION 3.10)
project(cnn_wasm)

set(CMAKE_CXX_STANDARD 17)

# add compilation flags for emscripten
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} \
    -O3 \
    -s WASM=1 \
    -s ALLOW_MEMORY_GROWTH=1 \
    -s EXPORTED_RUNTIME_METHODS=['ccall','cwrap'] \
    -s EXPORTED_FUNCTIONS=['_malloc','_free'] \
    -s NO_DISABLE_EXCEPTION_CATCHING=1 \
    --preload-file ${CMAKE_SOURCE_DIR}/flatmodel_large.bin@/flatmodel_large.bin \
    -lembind")

add_executable(cnn_wasm wasm_bindings.cpp)