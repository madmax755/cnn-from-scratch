#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

// ---------------------------------- ACTIVATION FUNCTIONS -------------------------------------------

// sigmoid activation function
double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }

// sigmoid derivative
double sigmoid_derivative(double x) {
    double s = sigmoid(x);
    return s * (1.0 - s);
}

// relu activation function
double relu(double x) { return std::max(x, 0.0); }

// relu derivative
double relu_derivative(double x) { return (x > 0) ? x : 0.0; }

// read binary file into a vector
std::vector<unsigned char> read_file(const std::string &path) {
    std::ifstream file(path, std::ios::in | std::ios::binary);

    if (file) {
        std::vector<unsigned char> buffer(std::istreambuf_iterator<char>(file), {});

        // for (unsigned char byte : buffer) {
        //     std::cout << static_cast<int>(byte) << " ";
        // }
        // std::cout << "\n";

        return buffer;
    } else {
        std::cout << "Error reading file " << path << "\n";

        return std::vector<unsigned char>();  // return an empty vector
    }
}

class Tensor3D {
   public:
    std::vector<std::vector<std::vector<double>>> data;
    size_t height, width, depth;

    Tensor3D() : height(0), width(0), depth(0) {}

    // for compatability with old matrix code
    Tensor3D(size_t rows, size_t cols) : depth(1), height(rows), width(cols) {
        data.resize(depth, std::vector<std::vector<double>>(height, std::vector<double>(width, 0.0)));
    }

    Tensor3D(size_t depth, size_t height, size_t width) : height(height), width(width), depth(depth) {
        data.resize(depth, std::vector<std::vector<double>>(height, std::vector<double>(width, 0.0)));
    }

    // compute dot product with a kernel centered at specific position - the argument must be the kernel
    double dot_with_kernel_at_postion(const Tensor3D &kernel, size_t start_x, size_t start_y) const {
        double sum = 0.0;

        // to facilitate start_x and start_y being the centre postion
        int kernel_width_offset = (kernel.width - 1) / 2;
        int kernel_height_offset = (kernel.height - 1) / 2;

        // check if the proceeding loop will be out of range for the input kernel
        if (abs(start_x) < kernel_width_offset or abs(start_x - data[0][0].size()) < kernel_width_offset or
            abs(start_y) < kernel_height_offset or abs(start_y - data[0].size()) < kernel_height_offset) {
            throw std::runtime_error("Cannot compute dot product at this postition - index would be out of range in convolution");
        }

        // iterate through all channels and kernel positions
        for (size_t d = 0; d < depth; d++) {
            for (size_t kh = 0; kh < kernel.height; kh++) {
                for (size_t kw = 0; kw < kernel.width; kw++) {
                    sum +=
                        data[d][start_y + kh - kernel_height_offset][start_x + kw - kernel_width_offset] * kernel.data[d][kh][kw];
                }
            }
        }
        return sum;
    }

    // return a new tensor with the width and height axis padded by 'amount'.
    static Tensor3D pad(const Tensor3D &input, int amount = 1) {
        Tensor3D output(input.depth, input.height + amount, input.width + amount);
        for (int depth_index = 0; depth_index < output.depth; ++depth_index) {
            for (int height_index = amount; height_index < output.height - amount; ++height_index) {
                for (int width_index = amount; width_index < output.width - amount; ++width_index) {
                    output.data[depth_index][height_index][width_index] =
                        input.data[depth_index][height_index - amount][width_index - amount];
                }
            }
        }
        return output;
    }

    // initialization methods
    void he_initialise() {
        std::random_device rd;
        std::mt19937 gen(rd());
        double std_dev = std::sqrt(2.0 / (height * width * depth));
        std::normal_distribution<> dis(0, std_dev);

        for (auto &channel : data) {
            for (auto &row : channel) {
                for (auto &val : row) {
                    val = dis(gen);
                }
            }
        }
    }

    void xavier_initialise() {
        std::random_device rd;
        std::mt19937 gen(rd());
        double limit = std::sqrt(6.0 / (height * width * depth));
        std::uniform_real_distribution<> dis(-limit, limit);

        for (auto &channel : data) {
            for (auto &row : channel) {
                for (auto &val : row) {
                    val = dis(gen);
                }
            }
        }
    }

    void uniform_initialise() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0, 1);

        for (auto &channel : data) {
            for (auto &row : channel) {
                for (auto &val : row) {
                    val = dis(gen);
                }
            }
        }
    }

    void zero_initialise() {
        for (auto &channel : data) {
            for (auto &row : channel) {
                for (auto &val : row) {
                    val = 0.0;
                }
            }
        }
    }

    // operators

    /**
     * @brief Overloads the multiplication operator for matrix multiplication on each depth slice.
     * @param other The tensor to multiply with.
     * @return The resulting tensor after multiplication.
     */
    Tensor3D operator*(const Tensor3D &other) const {
        // check dimensions match for matrix multiplication at each depth
        if (width != other.height) {
            throw std::invalid_argument("tensor dimensions don't match for multiplication: (" + std::to_string(height) + "x" +
                                        std::to_string(width) + "x" + std::to_string(depth) + ") * (" +
                                        std::to_string(other.height) + "x" + std::to_string(other.width) + "x" +
                                        std::to_string(other.depth) + ")");
        }
        if (depth != other.depth) {
            throw std::invalid_argument("tensor depths must match for multiplication");
        }

        // result will have dimensions: (this.height x other.width x depth)
        Tensor3D result(depth, height, other.width);

        // perform matrix multiplication for each depth slice
        for (size_t d = 0; d < depth; d++) {
            // cache-friendly loop order (k before j)
            for (size_t i = 0; i < height; i++) {
                for (size_t k = 0; k < width; k++) {
                    for (size_t j = 0; j < other.width; j++) {
                        result.data[d][i][j] += data[d][i][k] * other.data[d][k][j];
                    }
                }
            }
        }

        return result;
    }

    Tensor3D operator+(const Tensor3D &other) const {
        if (height != other.height || width != other.width || depth != other.depth) {
            throw std::invalid_argument("tensor dimensions don't match for addition");
        }

        Tensor3D result(depth, height, width);
        for (size_t d = 0; d < depth; d++) {
            for (size_t i = 0; i < height; i++) {
                for (size_t j = 0; j < width; j++) {
                    result.data[d][i][j] = data[d][i][j] + other.data[d][i][j];
                }
            }
        }
        return result;
    }

    Tensor3D operator-(const Tensor3D &other) const {
        if (height != other.height || width != other.width || depth != other.depth) {
            throw std::invalid_argument("tensor dimensions don't match for subtraction");
        }

        Tensor3D result(depth, height, width);
        for (size_t d = 0; d < depth; d++) {
            for (size_t i = 0; i < height; i++) {
                for (size_t j = 0; j < width; j++) {
                    result.data[d][i][j] = data[d][i][j] - other.data[d][i][j];
                }
            }
        }
        return result;
    }

    Tensor3D operator*(double scalar) const {
        Tensor3D result(depth, height, width);
        for (size_t d = 0; d < depth; d++) {
            for (size_t i = 0; i < height; i++) {
                for (size_t j = 0; j < width; j++) {
                    result.data[d][i][j] = data[d][i][j] * scalar;
                }
            }
        }
        return result;
    }

    Tensor3D hadamard(const Tensor3D &other) const {
        if (height != other.height || width != other.width || depth != other.depth) {
            throw std::invalid_argument("tensor dimensions don't match for Hadamard product");
        }

        Tensor3D result(depth, height, width);
        for (size_t d = 0; d < depth; d++) {
            for (size_t i = 0; i < height; i++) {
                for (size_t j = 0; j < width; j++) {
                    result.data[d][i][j] = data[d][i][j] * other.data[d][i][j];
                }
            }
        }
        return result;
    }

    Tensor3D apply(double (*func)(double)) const {
        Tensor3D result(depth, height, width);
        for (size_t d = 0; d < depth; d++) {
            for (size_t i = 0; i < height; i++) {
                for (size_t j = 0; j < width; j++) {
                    result.data[d][i][j] = func(data[d][i][j]);
                }
            }
        }
        return result;
    }

    template <typename Func>
    Tensor3D apply(Func func) const {
        Tensor3D result(depth, height, width);
        for (size_t d = 0; d < depth; d++) {
            for (size_t i = 0; i < height; i++) {
                for (size_t j = 0; j < width; j++) {
                    result.data[d][i][j] = func(data[d][i][j]);
                }
            }
        }
        return result;
    }

    Tensor3D transpose() const {
        Tensor3D result(depth, height, width);
        for (size_t d = 0; d < depth; d++) {
            for (size_t i = 0; i < height; i++) {
                for (size_t j = 0; j < width; j++) {
                    result.data[d][j][i] = data[d][i][j];
                }
            }
        }
        return result;
    }

    // todo add a reshape method for conv layer -> nn layer translation

    // softmax across width dimension for back-compatibility with old matrix class
    Tensor3D softmax() const {
        Tensor3D result(depth, height, width);

        for (size_t d = 0; d < depth; d++) {
            for (size_t w = 0; w < width; w++) {  // equivalent to "columns" in matrix
                // find max
                double max_val = -std::numeric_limits<double>::infinity();
                for (size_t h = 0; h < height; h++) {  // equivalent to "rows" in matrix
                    max_val = std::max(max_val, data[d][h][w]);
                }

                // compute exp and sum
                double sum = 0.0;
                for (size_t h = 0; h < height; h++) {
                    result.data[d][h][w] = std::exp(data[d][h][w] - max_val);
                    sum += result.data[d][h][w];
                }

                // normalize
                for (size_t h = 0; h < height; h++) {
                    result.data[d][h][w] /= sum;
                }
            }
        }
        return result;
    }

    Tensor3D flatten() const {
        // create tensor of shape (1, depth*height*width, 1)
        Tensor3D result(1, depth * height * width, 1);

        // copy values sequentially
        size_t idx = 0;
        for (size_t d = 0; d < depth; d++) {
            for (size_t h = 0; h < height; h++) {
                for (size_t w = 0; w < width; w++) {
                    result.data[0][idx][0] = data[d][h][w];
                    idx++;
                }
            }
        }

        return result;
    }
};

class Layer {
   public:
    virtual ~Layer() = default;

    // pure virtual class - requires implementation in derived objects.
    virtual Tensor3D feedforward(const Tensor3D &input) = 0;
};

class DenseLayer : public Layer {
   public:
    Tensor3D weights;
    Tensor3D bias;
    std::string activation_function;

    /**
     * @brief Constructs a DenseLayer object with specified input size, output size, and activation function.
     * @param input_size The number of input neurons.
     * @param output_size The number of output neurons.
     * @param activation_function The activation function to use (default: "sigmoid").
     */
    DenseLayer(size_t input_size, size_t output_size, std::string activation_function = "relu")
        : weights(output_size, input_size), bias(output_size, 1), activation_function(activation_function) {
        if (activation_function == "sigmoid") {
            weights.xavier_initialise();
        } else if (activation_function == "relu") {
            weights.he_initialise();
        } else {
            weights.uniform_initialise();
        }
    }

    /**
     * @brief Performs feedforward operation for this layer.
     * @param input The input matrix.
     * @return The output matrix after applying the layer's transformation.
     */
    Tensor3D feedforward(const Tensor3D &input) override {
        if (weights.width != input.height) {
            throw std::runtime_error(
                "Input vector dimension is not appropriate for the weight matrix dimension, check input dimensions against layer "
                "dimensions");
        }
        Tensor3D z = weights * input + bias;
        Tensor3D output(z.height, z.width);
        if (activation_function == "sigmoid") {
            output = z.apply(sigmoid);
        } else if (activation_function == "relu") {
            output = z.apply(relu);
        } else if (activation_function == "softmax") {
            output = z.softmax();
        } else {
            output = z;
        }
        return output;
    }

    /**
     * @brief Performs feedforward operation for this layer and returns both output and pre-activation.
     * @param input The input matrix.
     * @return A vector containing the output matrix and pre-activation matrix.
     */
    std::vector<Tensor3D> feedforward_backprop(const Tensor3D &input) const {
        Tensor3D z = weights * input + bias;
        Tensor3D output(z.height, z.width);
        if (activation_function == "sigmoid") {
            output = z.apply(sigmoid);
        } else if (activation_function == "relu") {
            output = z.apply(relu);
        } else if (activation_function == "softmax") {
            output = z.softmax();
        } else {
            throw std::runtime_error("no activation function found for layer");
        }

        return {output, z};
    }
};

class ConvolutionLayer : public Layer {
   public:
    std::vector<Tensor3D> weights;
    std::vector<double> bias;
    int channels_in;
    int out_channels;
    int kernel_size;
    int stride;
    std::string mode;

    ConvolutionLayer(int channels_in, int out_channels, int kernel_size, int stride = 1, std::string mode = "same")
        : channels_in(channels_in), out_channels(out_channels), kernel_size(kernel_size), stride(stride), mode(mode) {
        // creates a list of kernel tensors - one for each output channel
        weights.reserve(out_channels);
        for (int i = 0; i < out_channels; i++) {
            weights.emplace_back(channels_in, kernel_size, kernel_size);
            weights.back().he_initialise();
        }

        // creates a list of bias values - one for each output channel
        bias.reserve(out_channels);
        for (int i = 0; i < out_channels; i++) {
            bias.push_back(0.0);
        }
    }

    /**
     * @brief Performs feedforward operation for this layer.
     * @param input The input tensor usually (x, y, RGB channel).
     * @return The output tensor after applying the layer's transformation (x, y, output_feature_map)
     */
    Tensor3D feedforward(const Tensor3D &input) override {
        if (input.depth != channels_in) {
            throw std::runtime_error("Input tensor depth does not match the number of input channels for this layer");
        }
        if (input.height == 0 or input.width == 0) {
            throw std::runtime_error("Input tensor has zero height or width");
        }

        Tensor3D output(weights.size(), input.height, input.width);
        int pad_amount = 0;

        if (mode == "same") {
            Tensor3D padded_input = Tensor3D::pad(input);
            pad_amount = (kernel_size - 1) / 2;

            for (int feature_map_index = 0; feature_map_index < weights.size(); ++feature_map_index) {
                for (int y = pad_amount; y < input.height - pad_amount; ++y) {
                    for (int x = pad_amount; x < input.width - pad_amount; ++x) {
                        double z = input.dot_with_kernel_at_postion(weights[feature_map_index], x, y) + bias[feature_map_index];
                        output.data[feature_map_index][y - pad_amount][x - pad_amount] = relu(z);
                        // todo replace in the future with chosen activation function set by user
                    }
                }
            }

        } else {
            throw std::runtime_error("mode not specified or handled correctly");
        }

        return output;
    }
};

class PoolingLayer : public Layer {
   public:
    int kernel_size;
    int stride;
    std::string mode;

    PoolingLayer(int kernel_size = 2, int stride = -1, std::string mode = "max")
        : kernel_size(kernel_size), stride(stride == -1 ? kernel_size : stride), mode(mode) {
        if (stride > kernel_size) {
            throw std::runtime_error("stride should not be greater than kernel size");
        }
    }

    Tensor3D feedforward(const Tensor3D &input) override {
        // calculate output dimensions including partial window pooling
        int new_height = std::ceil((input.height - kernel_size) / stride + 1);
        int new_width = std::ceil((input.width - kernel_size) / stride + 1);
        Tensor3D output(input.depth, new_height, new_width);

        if (mode == "max") {
            for (int d = 0; d < input.depth; ++d) {
                for (int y = 0; y < new_height; ++y) {
                    for (int x = 0; x < new_width; ++x) {
                        // calculate window boundaries
                        int start_y = y * stride;
                        int start_x = x * stride;
                        int end_y = std::min(start_y + kernel_size, static_cast<int>(input.height));
                        int end_x = std::min(start_x + kernel_size, static_cast<int>(input.width));

                        // find maximum in this window
                        double max_val = -std::numeric_limits<double>::infinity();
                        for (int wy = start_y; wy < end_y; ++wy) {
                            for (int wx = start_x; wx < end_x; ++wx) {
                                max_val = std::max(max_val, input.data[d][wy][wx]);
                            }
                        }
                        output.data[d][y][x] = max_val;
                    }
                }
            }
        } else {
            throw std::runtime_error("mode not specified or handled correctly");
        }

        return output;
    }
};

class Loss {
   public:
    virtual ~Loss() = default;

    // Compute the loss value
    virtual double compute(const Tensor3D &predicted, const Tensor3D &target) const = 0;

    // Compute the derivative of the loss with respect to the predicted values
    virtual Tensor3D derivative(const Tensor3D &predicted, const Tensor3D &target) const = 0;
};

class CrossEntropyLoss : public Loss {
   public:
    double compute(const Tensor3D &predicted, const Tensor3D &target) const override {
        double loss = 0.0;
        for (size_t i = 0; i < predicted.height; ++i) {
            for (size_t j = 0; j < predicted.width; ++j) {
                // Add small epsilon to avoid log(0)
                loss -= target.data[0][i][j] * std::log(predicted.data[0][i][j] + 1e-10);
            }
        }
        return loss / predicted.width;  // Average over batch
    }

    Tensor3D derivative(const Tensor3D &predicted, const Tensor3D &target) const override {
        // For cross entropy with softmax, the derivative simplifies to (predicted - target)
        return predicted - target;
    }
};

class MSELoss : public Loss {
   public:
    double compute(const Tensor3D &predicted, const Tensor3D &target) const override {
        double loss = 0.0;
        for (size_t i = 0; i < predicted.height; ++i) {
            for (size_t j = 0; j < predicted.width; ++j) {
                double diff = predicted.data[0][i][j] - target.data[0][i][j];
                loss += diff * diff;
            }
        }
        return loss / (2.0 * predicted.width);  // Average over batch and divide by 2
    }

    Tensor3D derivative(const Tensor3D &predicted, const Tensor3D &target) const override {
        return (predicted - target) * (1.0 / predicted.width);
    }
};

// -----------------------------------------------------------------------------------------------------
// ---------------------------------------- OPTIMISERS -------------------------------------------------

// base Optimiser class
class Optimiser {
   public:
    /**
     * @brief Computes and applies updates to the network layers based on gradients.
     * @param layers The layers of the neural network to update.
     * @param gradients The gradients used for updating the layers.
     */
    virtual void compute_and_apply_updates(std::vector<DenseLayer> &layers,
                                           const std::vector<std::vector<Tensor3D>> &gradients) = 0;

    /**
     * @brief Virtual destructor for the Optimiser class.
     */
    virtual ~Optimiser() = default;

    struct GradientResult {
        std::vector<std::vector<Tensor3D>>
            gradients;                  // list of layers, each layer has a list of weight and bias gradient matrices
        Tensor3D input_layer_gradient;  // gradient of the input layer - for more general use as parts of bigger architectures
        Tensor3D output;                // output of the network
    };

    /**
     * @brief Calculates gradients for a single example.
     * @param layers The layers of the neural network.
     * @param input The input matrix.
     * @param target The target matrix.
     * @return A GradientResult struct containing gradients and the output of the network.
     */
    virtual GradientResult calculate_gradient(const std::vector<DenseLayer> &layers, const Tensor3D &input,
                                              const Tensor3D &target, const Loss &loss) {
        // forward pass
        std::vector<Tensor3D> activations = {input};
        std::vector<Tensor3D> preactivations = {input};

        for (const auto &layer : layers) {
            auto results = layer.feedforward_backprop(activations.back());
            activations.push_back(results[0]);
            preactivations.push_back(results[1]);
        }

        // backward pass
        int num_layers = layers.size();
        std::vector<Tensor3D> deltas;
        deltas.reserve(num_layers);

        // output layer error (δ^L = ∇_a C ⊙ σ'(z^L))
        Tensor3D output_delta = loss.derivative(activations.back(), target);
        if (layers.back().activation_function == "sigmoid") {
            output_delta = output_delta.hadamard(preactivations.back().apply(sigmoid_derivative));
        } else if (layers.back().activation_function == "relu") {
            output_delta = output_delta.hadamard(preactivations.back().apply(relu_derivative));
        } else if (layers.back().activation_function == "softmax" or layers.back().activation_function == "none") {
            // for softmax and none, the delta is already correct (assuming cross-entropy loss)
        } else {
            throw std::runtime_error("Unsupported activation function");
        }
        deltas.push_back(output_delta);

        // hidden layer errors (δ^l = ((w^(l+1))^T δ^(l+1)) ⊙ σ'(z^l))
        for (int l = num_layers - 2; l >= 0; --l) {
            Tensor3D delta = (layers[l + 1].weights.transpose() * deltas.back());
            if (layers[l].activation_function == "sigmoid") {
                delta = delta.hadamard(preactivations[l + 1].apply(sigmoid_derivative));
            } else if (layers[l].activation_function == "relu") {
                delta = delta.hadamard(preactivations[l + 1].apply(relu_derivative));
            } else if (layers[l].activation_function == "none") {
                // delta = delta
            } else {
                throw std::runtime_error("Unsupported activation function");
            }
            deltas.push_back(delta);
        }

        // reverse deltas to match layer order
        std::reverse(deltas.begin(), deltas.end());

        // calculate gradients
        std::vector<std::vector<Tensor3D>> gradients;
        for (int l = 0; l < num_layers; ++l) {
            Tensor3D weight_gradient = deltas[l] * activations[l].transpose();
            gradients.push_back({weight_gradient, deltas[l]});
        }

        // return a GradientResult struct for purposes of tracking loss
        return {gradients, deltas.front(), activations.back()};
    }

    /**
     * @brief Averages gradients from multiple examples.
     * @param batch_gradients A vector of gradients from multiple examples.
     * @return The averaged gradients.
     */
    std::vector<std::vector<Tensor3D>> average_gradients(const std::vector<std::vector<std::vector<Tensor3D>>> &batch_gradients) {
        std::vector<std::vector<Tensor3D>> avg_gradients;
        size_t num_layers = batch_gradients[0].size();
        size_t batch_size = batch_gradients.size();

        for (size_t l = 0; l < num_layers; ++l) {
            Tensor3D avg_weight_grad(batch_gradients[0][l][0].height, batch_gradients[0][l][0].width);
            Tensor3D avg_bias_grad(batch_gradients[0][l][1].height, batch_gradients[0][l][1].width);

            for (const auto &example_gradients : batch_gradients) {
                avg_weight_grad = avg_weight_grad + example_gradients[l][0];
                avg_bias_grad = avg_bias_grad + example_gradients[l][1];
            }

            avg_weight_grad = avg_weight_grad * (1.0 / batch_size);
            avg_bias_grad = avg_bias_grad * (1.0 / batch_size);

            avg_gradients.push_back({avg_weight_grad, avg_bias_grad});
        }

        return avg_gradients;
    }
};

class SGDOptimiser : public Optimiser {
   private:
    double learning_rate;
    std::vector<std::vector<Tensor3D>> velocity;

   public:
    /**
     * @brief Constructs an SGDOptimiser object with the specified learning rate.
     * @param lr The learning rate (default: 0.1).
     */
    SGDOptimiser(double lr = 0.1) : learning_rate(lr) {}

    /**
     * @brief initialises the velocity vectors for SGD optimization.
     * @param layers The layers of the neural network.
     */
    void initialise_velocity(const std::vector<DenseLayer> &layers) {
        velocity.clear();
        for (const auto &layer : layers) {
            velocity.push_back(
                {Tensor3D(layer.weights.height, layer.weights.width), Tensor3D(layer.bias.height, layer.bias.width)});
        }
    }

    /**
     * @brief Computes and applies updates using Stochastic Gradient Descent.
     * @param layers The layers of the neural network to update.
     * @param gradients The gradients used for updating the layers.
     */
    void compute_and_apply_updates(std::vector<DenseLayer> &layers,
                                   const std::vector<std::vector<Tensor3D>> &gradients) override {
        if (velocity.empty()) {
            initialise_velocity(layers);
        }

        // compute and apply updates
        for (size_t l = 0; l < layers.size(); ++l) {
            for (int i = 0; i < 2; ++i) {  // 0 for weights, 1 for biases
                // compute adjustment
                velocity[l][i] = gradients[l][i] * -learning_rate;
            }
            // apply adjustment
            layers[l].weights = layers[l].weights + velocity[l][0];
            layers[l].bias = layers[l].bias + velocity[l][1];
        }
    }
};

class SGDMomentumOptimiser : public Optimiser {
   private:
    double learning_rate;
    double momentum;
    std::vector<std::vector<Tensor3D>> velocity;

   public:
    /**
     * @brief Constructs an SGDMomentumOptimiser object with the specified learning rate and momentum.
     * @param lr The learning rate (default: 0.1).
     * @param mom The momentum coefficient (default: 0.9).
     */
    SGDMomentumOptimiser(double lr = 0.1, double mom = 0.9) : learning_rate(lr), momentum(mom) {}

    /**
     * @brief initialises the velocity vectors for SGD with Momentum optimization.
     * @param layers The layers of the neural network.
     */
    void initialise_velocity(const std::vector<DenseLayer> &layers) {
        velocity.clear();
        for (const auto &layer : layers) {
            velocity.push_back(
                {Tensor3D(layer.weights.height, layer.weights.width), Tensor3D(layer.bias.height, layer.bias.width)});
        }
    }

    /**
     * @brief Computes and applies updates using Stochastic Gradient Descent with Momentum.
     * @param layers The layers of the neural network to update.
     * @param gradients The gradients used for updating the layers.
     */
    void compute_and_apply_updates(std::vector<DenseLayer> &layers,
                                   const std::vector<std::vector<Tensor3D>> &gradients) override {
        if (velocity.empty()) {
            initialise_velocity(layers);
        }

        // compute updates
        for (size_t l = 0; l < layers.size(); ++l) {
            for (int i = 0; i < 2; ++i) {  // 0 for weights, 1 for biases
                // compute adjustments
                velocity[l][i] = (velocity[l][i] * momentum) - (gradients[l][i] * learning_rate);
            }
            // apply adjustments
            layers[l].weights = layers[l].weights + velocity[l][0];
            layers[l].bias = layers[l].bias + velocity[l][1];
        }
    }
};

class NesterovMomentumOptimiser : public Optimiser {
   private:
    double learning_rate;
    double momentum;
    std::vector<std::vector<Tensor3D>> velocity;

   public:
    /**
     * @brief Constructs a NesterovMomentumOptimiser object with the specified learning rate and momentum.
     * @param lr The learning rate (default: 0.1).
     * @param mom The momentum coefficient (default: 0.9).
     */
    NesterovMomentumOptimiser(double lr = 0.1, double mom = 0.9) : learning_rate(lr), momentum(mom) {}

    /**
     * @brief initialises the velocity vectors for Nesterov Momentum optimization.
     * @param layers The layers of the neural network.
     */
    void initialise_velocity(const std::vector<DenseLayer> &layers) {
        velocity.clear();
        for (const auto &layer : layers) {
            velocity.push_back(
                {Tensor3D(layer.weights.height, layer.weights.width), Tensor3D(layer.bias.height, layer.bias.width)});
        }
    }

    /**
     * @brief Calculates gradients using Nesterov momentum.
     * @param layers The layers of the neural network.
     * @param input The input matrix.
     * @param target The target matrix.
     * @return A GradientResult struct containing gradients and the output of the network.
     */
    GradientResult calculate_gradient(const std::vector<DenseLayer> &layers, const Tensor3D &input, const Tensor3D &target,
                                      const Loss &loss) override {
        // get lookahead position
        std::vector<DenseLayer> tmp_layers = layers;
        if (velocity.empty()) {
            initialise_velocity(tmp_layers);
        } else {
            for (int l = 0; l < tmp_layers.size(); l++) {
                tmp_layers[l].weights = tmp_layers[l].weights + (velocity[l][0] * momentum);
                tmp_layers[l].bias = tmp_layers[l].bias + (velocity[l][1] * momentum);
            }
        }

        // compute gradient at lookahead position
        GradientResult gradients = Optimiser::calculate_gradient(tmp_layers, input, target, loss);

        // this returns a GradientResult struct for the lookahead position (not totally correct for loss tracking but not bad)
        return gradients;
    }

    /**
     * @brief Computes and applies updates using Nesterov Momentum.
     * @param layers The layers of the neural network to update.
     * @param gradients The gradients used for updating the layers.
     */
    void compute_and_apply_updates(std::vector<DenseLayer> &layers,
                                   const std::vector<std::vector<Tensor3D>> &gradients) override {
        if (velocity.empty()) {
            initialise_velocity(layers);
        }

        // compute and apply updates
        for (size_t l = 0; l < layers.size(); ++l) {
            for (int i = 0; i < 2; ++i) {  // 0 for weights, 1 for biases
                // compute adjustments
                velocity[l][i] = (velocity[l][i] * momentum) - (gradients[l][i] * learning_rate);
            }
            // apply updates
            layers[l].weights = layers[l].weights + velocity[l][0];
            layers[l].bias = layers[l].bias + velocity[l][1];
        }
    }
};

class AdamOptimiser : public Optimiser {
   private:
    double learning_rate;
    double beta1;
    double beta2;
    double epsilon;
    int t;                                 // timestep
    std::vector<std::vector<Tensor3D>> m;  // first moment
    std::vector<std::vector<Tensor3D>> v;  // second moment

   public:
    /**
     * @brief Constructs an AdamOptimiser object with the specified parameters.
     * @param lr The learning rate (default: 0.001).
     * @param b1 The beta1 parameter (default: 0.9).
     * @param b2 The beta2 parameter (default: 0.999).
     * @param eps The epsilon parameter for numerical stability (default: 1e-8).
     */
    AdamOptimiser(double lr = 0.001, double b1 = 0.9, double b2 = 0.999, double eps = 1e-8)
        : learning_rate(lr), beta1(b1), beta2(b2), epsilon(eps), t(0) {}

    /**
     * @brief initialises the first and second moment vectors for Adam optimization.
     * @param layers The layers of the neural network.
     */
    void initialise_moments(const std::vector<DenseLayer> &layers) {
        m.clear();
        v.clear();
        m.reserve(layers.size());
        v.reserve(layers.size());
        for (const auto &layer : layers) {
            m.push_back({Tensor3D(layer.weights.height, layer.weights.width), Tensor3D(layer.bias.height, layer.bias.width)});
            v.push_back({Tensor3D(layer.weights.height, layer.weights.width), Tensor3D(layer.bias.height, layer.bias.width)});
        }
    }

    /**
     * @brief Computes and applies updates using the Adam optimization algorithm.
     * @param layers The layers of the neural network to update.
     * @param gradients The gradients used for updating the layers.
     */
    void compute_and_apply_updates(std::vector<DenseLayer> &layers,
                                   const std::vector<std::vector<Tensor3D>> &gradients) override {
        if (m.empty() or v.empty()) {
            initialise_moments(layers);
        }

        t++;  // increment timestep

        for (size_t l = 0; l < layers.size(); ++l) {
            for (int i = 0; i < 2; ++i) {  // 0 for weights, 1 for biases
                // update biased first moment estimate
                m[l][i] = m[l][i] * beta1 + gradients[l][i] * (1.0 - beta1);

                // update biased second raw moment estimate
                v[l][i] = v[l][i] * beta2 + gradients[l][i].hadamard(gradients[l][i]) * (1.0 - beta2);

                // compute bias-corrected first moment estimate
                Tensor3D m_hat = m[l][i] * (1.0 / (1.0 - std::pow(beta1, t)));

                // compute bias-corrected second raw moment estimate
                Tensor3D v_hat = v[l][i] * (1.0 / (1.0 - std::pow(beta2, t)));

                // compute the update
                Tensor3D update = m_hat.hadamard(v_hat.apply([this](double x) { return 1.0 / (std::sqrt(x) + epsilon); }));

                // apply the update
                if (i == 0) {
                    layers[l].weights = layers[l].weights - update * learning_rate;
                } else {
                    layers[l].bias = layers[l].bias - update * learning_rate;
                }
            }
        }
    }
};

class AdamWOptimiser : public Optimiser {
   private:
    double learning_rate;
    double beta1;
    double beta2;
    double epsilon;
    double weight_decay;
    int t;                                 // timestep
    std::vector<std::vector<Tensor3D>> m;  // first moment
    std::vector<std::vector<Tensor3D>> v;  // second moment

   public:
    /**
     * @brief Constructs an AdamWOptimiser object with the specified parameters.
     * @param lr The learning rate (default: 0.001).
     * @param b1 The beta1 parameter (default: 0.9).
     * @param b2 The beta2 parameter (default: 0.999).
     * @param eps The epsilon parameter for numerical stability (default: 1e-8).
     * @param wd The weight decay parameter (default: 0.01).
     */
    AdamWOptimiser(double lr = 0.001, double b1 = 0.9, double b2 = 0.999, double eps = 1e-8, double wd = 0.01)
        : learning_rate(lr), beta1(b1), beta2(b2), epsilon(eps), weight_decay(wd), t(0) {}

    /**
     * @brief initialises the first and second moment vectors for AdamW optimization.
     * @param layers The layers of the neural network.
     */
    void initialise_moments(const std::vector<DenseLayer> &layers) {
        m.clear();
        v.clear();
        m.reserve(layers.size());
        v.reserve(layers.size());
        for (const auto &layer : layers) {
            m.push_back({Tensor3D(layer.weights.height, layer.weights.width), Tensor3D(layer.bias.height, layer.bias.width)});
            v.push_back({Tensor3D(layer.weights.height, layer.weights.width), Tensor3D(layer.bias.height, layer.bias.width)});
        }
    }

    /**
     * @brief Computes and applies updates using the AdamW optimization algorithm.
     * @param layers The layers of the neural network to update.
     * @param gradients The gradients used for updating the layers.
     */
    void compute_and_apply_updates(std::vector<DenseLayer> &layers,
                                   const std::vector<std::vector<Tensor3D>> &gradients) override {
        if (m.empty() || v.empty()) {
            initialise_moments(layers);
        }

        t++;  // increment timestep

        for (size_t l = 0; l < layers.size(); ++l) {
            for (int i = 0; i < 2; ++i) {  // 0 for weights, 1 for biases
                // update biased first moment estimate
                m[l][i] = m[l][i] * beta1 + gradients[l][i] * (1.0 - beta1);

                // update biased second raw moment estimate
                v[l][i] = v[l][i] * beta2 + gradients[l][i].hadamard(gradients[l][i]) * (1.0 - beta2);

                // compute bias-corrected first moment estimate
                Tensor3D m_hat = m[l][i] * (1.0 / (1.0 - std::pow(beta1, t)));

                // compute bias-corrected second raw moment estimate
                Tensor3D v_hat = v[l][i] * (1.0 / (1.0 - std::pow(beta2, t)));

                // compute the Adam update
                Tensor3D update = m_hat.hadamard(v_hat.apply([this](double x) { return 1.0 / (std::sqrt(x) + epsilon); }));

                // apply the update
                if (i == 0) {  // for weights
                    // apply weight decay
                    layers[l].weights = layers[l].weights * (1.0 - learning_rate * weight_decay);
                    // apply Adam update
                    layers[l].weights = layers[l].weights - (update * learning_rate);
                } else {  // for biases
                    // biases typically don't use weight decay
                    layers[l].bias = layers[l].bias - update * learning_rate;
                }
            }
        }
    }
};

// -----------------------------------------------------------------------------------------------------

class NeuralNetwork {
   private:
    struct LayerSpec {
        enum Type { CONV, POOL, DENSE } type;

        // conv parameters
        int in_channels = 0;
        int out_channels = 0;
        int kernel_size = 0;
        int stride = 1;
        std::string mode = "same";

        // pool parameters
        int pool_size = 0;
        int pool_stride = 0;
        std::string pool_mode = "max";

        // dense parameters
        std::string activation = "relu";
        size_t output_size = 0;

        static LayerSpec Conv(int out_channels, int kernel_size, int stride = 1, std::string mode = "same") {
            LayerSpec spec;
            spec.type = CONV;
            spec.out_channels = out_channels;
            spec.kernel_size = kernel_size;
            spec.stride = stride;
            spec.mode = mode;
            return spec;
        }

        static LayerSpec Pool(int pool_size = 2, int stride = -1, std::string mode = "max") {
            LayerSpec spec;
            spec.type = POOL;
            spec.pool_size = pool_size;
            spec.pool_stride = (stride == -1) ? pool_size : stride;
            spec.pool_mode = mode;
            return spec;
        }

        static LayerSpec Dense(size_t output_size, std::string activation = "relu") {
            LayerSpec spec;
            spec.type = DENSE;
            spec.output_size = output_size;
            spec.activation = activation;
            return spec;
        }
    };

    struct LayerDimensions {
        size_t height;
        size_t width;
        size_t depth;
    };

    void create_layers(const Tensor3D &input) {
        // stores the dimensions of the previous layer
        LayerDimensions dims = {input.height, input.width, input.depth};

        for (auto &spec : layer_specs) {
            switch (spec.type) {
                case LayerSpec::CONV: {
                    spec.in_channels = dims.depth;
                    layers.push_back(std::make_unique<ConvolutionLayer>(spec.in_channels, spec.out_channels, spec.kernel_size,
                                                                        spec.stride, spec.mode));
                    
                    // change dims.width and dims.height here if ever put more modes than same

                    dims.depth = spec.out_channels;
                    break;
                }
                case LayerSpec::POOL: {
                    layers.push_back(std::make_unique<PoolingLayer>(spec.pool_size, spec.pool_stride, spec.pool_mode));

                    // update 'previous' layer dimensions with formula for pooling layer output dimensions
                    dims.height = std::ceil((dims.height - spec.pool_size) / static_cast<double>(spec.pool_stride) + 1);
                    dims.width = std::ceil((dims.width - spec.pool_size) / static_cast<double>(spec.pool_stride) + 1);
                    break;
                }
                case LayerSpec::DENSE: {
                    int total_inputs = dims.height * dims.width * dims.depth;
                    layers.push_back(std::make_unique<DenseLayer>(total_inputs, spec.output_size, spec.activation));
                    dims = {1, spec.output_size, 1};  // flatten
                    break;
                }
            }
        }
        layers_created = true;
    }

   public:
    std::vector<LayerSpec> layer_specs;
    std::vector<std::unique_ptr<Layer>> layers;
    std::unique_ptr<Optimiser> optimiser;
    std::unique_ptr<Loss> loss;
    bool layers_created = false;

    struct EvaluationMetrics {
        double loss;
        double accuracy;
        double precision;
        double recall;
        double f1_score;

        // Add this operator overload for printing
        friend std::ostream &operator<<(std::ostream &os, const EvaluationMetrics &metrics) {
            os << "----------------\n"
               << "Loss: " << metrics.loss << "\n"
               << "Accuracy: " << metrics.accuracy << "\n"
               << "Precision: " << metrics.precision << "\n"
               << "Recall: " << metrics.recall << "\n"
               << "F1 Score: " << metrics.f1_score << "\n"
               << "----------------";
            return os;
        }
    };

    // default constructor
    NeuralNetwork() {}

    // user-facing funcs to add layers (just adds their specs to layer_specs to be created with correct input dimensions later)
    void add_conv_layer(int out_channels, int kernel_size, int stride = 1, std::string mode = "same") {
        layer_specs.push_back(LayerSpec::Conv(out_channels, kernel_size, stride, mode));
    }

    void add_pool_layer(int pool_size = 2, int stride = -1, std::string mode = "max") {
        layer_specs.push_back(LayerSpec::Pool(pool_size, stride, mode));
    }

    void add_dense_layer(int output_size, std::string activation = "relu") {
        layer_specs.push_back(LayerSpec::Dense(output_size, activation));
    }

    /**
     * @brief Sets the optimiser for the neural network.
     * @param new_optimiser A unique pointer to the new Optimiser object.
     */
    void set_optimiser(std::unique_ptr<Optimiser> new_optimiser) { optimiser = std::move(new_optimiser); }

    /**
     * @brief Sets the loss function for the neural network.
     * @param new_loss A unique pointer to the new Loss object.
     */
    void set_loss(std::unique_ptr<Loss> new_loss) { loss = std::move(new_loss); }

    /**
     * @brief Performs feedforward operation through all layers of the network.
     * @param input The input matrix.
     * @return The output matrix after passing through all layers.
     */
    Tensor3D feedforward(const Tensor3D &input) {
        Tensor3D current = input;
        if (!layers_created) {
            // calculate input dimensions for each layer and create layer objects with these dimensions
            create_layers(input);
        }

        for (size_t i = 0; i < layers.size(); i++) {
            // attempt to cast base type pointer to derived type pointer (returns nullptr if not possible)
            auto *current_conv = dynamic_cast<ConvolutionLayer *>(layers[i].get());
            auto *current_dense = dynamic_cast<DenseLayer *>(layers[i].get());
            auto *current_pooling = dynamic_cast<PoolingLayer *>(layers[i].get());

            // get next layer type (if it exists)
            ConvolutionLayer *next_conv = nullptr;
            DenseLayer *next_dense = nullptr;
            PoolingLayer *next_pooling = nullptr;
            if (i + 1 < layers.size()) {
                next_conv = dynamic_cast<ConvolutionLayer *>(layers[i + 1].get());
                next_dense = dynamic_cast<DenseLayer *>(layers[i + 1].get());
                next_pooling = dynamic_cast<PoolingLayer *>(layers[i + 1].get());
            }

            // handle transitions
            if (current_conv) {
                current = current_conv->feedforward(current);

                // if next layer is dense, we need to flatten to (1, N, 1) where N is total elements
                if (next_dense) {
                    current = current.flatten();
                }
            } else if (current_pooling) {
                current = current_pooling->feedforward(current);

                // if next layer is dense, we need to flatten to (1, N, 1) where N is total elements
                if (next_dense) {
                    current = current.flatten();
                }
            } else if (current_dense) {
                if (next_pooling or next_conv) {
                    throw std::runtime_error("dense layer cannot be followed by a pooling or convolution layer");
                }
                current = current_dense->feedforward(current);
            } else {
                throw std::runtime_error("unknown layer type encountered");
            }
        }
        return current;
    }
};

// todo:
// - dynamic input dimension calculation on layer addition to mitigate need to manually calculate input dimensions for each
//   layer especially with partial window pooling

// runner code
int main() {
    int input_width = 28;
    int input_height = 28;
    int input_channels = 3;
    int kernel_size = 3;

    // test usage
    NeuralNetwork nn;
    nn.add_conv_layer(5, kernel_size);
    nn.add_pool_layer();
    nn.add_conv_layer(10, kernel_size);
    nn.add_pool_layer();
    nn.add_dense_layer(20);
    nn.add_dense_layer(10);
    nn.add_dense_layer(1, "none");

    // test feedforward
    Tensor3D input(input_channels, input_height, input_width);
    Tensor3D output = nn.feedforward(input);
    std::cout << "done" << std::endl;
}
