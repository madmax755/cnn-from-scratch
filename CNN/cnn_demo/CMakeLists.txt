cmake_minimum_required(VERSION 3.10)
project(cnn_wasm)

set(CMAKE_CXX_STANDARD 17)

# add compilation flags for emscripten
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} \
    -s WASM=1 \
    -s ALLOW_MEMORY_GROWTH=1 \
    -s EXPORTED_RUNTIME_METHODS=['ccall','cwrap'] \
    -s EXPORTED_FUNCTIONS=['_malloc','_free'] \
    -s NO_DISABLE_EXCEPTION_CATCHING=1 \
    --preload-file ${CMAKE_SOURCE_DIR}/float16.32.64.200.100.bin@/float16.32.64.200.100.bin \
    -lembind")

add_executable(cnn_wasm wasm_bindings.cpp)