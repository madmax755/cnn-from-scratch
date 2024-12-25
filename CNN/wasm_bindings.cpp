#include <emscripten/bind.h>
#include <emscripten/val.h>
#include "cnn.cpp"

using namespace emscripten;

// global network instance
static std::unique_ptr<NeuralNetwork> g_network;

// load the model and store in global instance
bool loadModel(const std::string& path) {
    try {
        std::cout << "attempting to load model from: " << path << std::endl;
        g_network = std::make_unique<NeuralNetwork>(NeuralNetwork::load_model(path));
        std::cout << "model loaded successfully" << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "error loading model: " << e.what() << std::endl;
        return false;
    }
}

// wrapper function to handle vector conversion
std::vector<double> predict(const std::vector<double>& input) {
    if (!g_network) {
        throw std::runtime_error("model not loaded - call loadModel first");
    }
    return g_network->predict_digit(input);
}

EMSCRIPTEN_BINDINGS(cnn_module) {
    register_vector<double>("VectorDouble");
    
    function("predict", &predict);
    function("loadModel", &loadModel);
}