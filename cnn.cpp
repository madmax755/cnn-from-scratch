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
float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

// sigmoid derivative
float sigmoid_derivative(float x) {
    float s = sigmoid(x);
    return s * (1.0f - s);
}

// relu activation function
float relu(float x) { return std::max(x, 0.0f); }

// relu derivative
float relu_derivative(float x) { return (x > 0.0f) ? 1.0f : 0.0f; }

// read binary file into a vector
std::vector<unsigned char> read_file(const std::string &path) {
    std::ifstream file(path, std::ios::in | std::ios::binary);

    if (file) {
        std::vector<unsigned char> buffer(std::istreambuf_iterator<char>(file), {});
        return buffer;
    } else {
        std::cout << "Error reading file " << path << "\n";
        return std::vector<unsigned char>();  // return an empty vector
    }
}

class Tensor3D {
   public:
    std::vector<std::vector<std::vector<float>>> data;
    size_t height, width, depth;

    Tensor3D() : height(0), width(0), depth(0) {}

    // for compatability with old Tensor3D code
    Tensor3D(size_t rows, size_t cols) : depth(1), height(rows), width(cols) {
        data.resize(depth, std::vector<std::vector<float>>(height, std::vector<float>(width, 0.0f)));
    }

    Tensor3D(size_t depth, size_t height, size_t width) : height(height), width(width), depth(depth) {
        data.resize(depth, std::vector<std::vector<float>>(height, std::vector<float>(width, 0.0f)));
    }

    // compute dot product with a kernel centered at specific position - the argument must be the kernel
    float dot_with_kernel_at_postion(const Tensor3D &kernel, size_t start_x, size_t start_y) const {
        float sum = 0.0;

        // to facilitate start_x and start_y being the centre postion
        int kernel_width_offset = (kernel.width - 1) / 2;
        int kernel_height_offset = (kernel.height - 1) / 2;

        // check if the proceeding loop will be out of range for the input kernel
        if (std::abs(static_cast<int>(start_x)) < kernel_width_offset or 
            std::abs(static_cast<int>(start_x - data[0][0].size())) < kernel_width_offset or
            std::abs(static_cast<int>(start_y)) < kernel_height_offset or 
            std::abs(static_cast<int>(start_y - data[0].size())) < kernel_height_offset) {
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
        Tensor3D output(input.depth, input.height + 2 * amount, input.width + 2 * amount);
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
        float std_dev = std::sqrt(2.0f / (height * width * depth));
        std::normal_distribution<float> dis(0.0f, std_dev);

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
        float limit = std::sqrt(6.0f / (height * width * depth));
        std::uniform_real_distribution<float> dis(-limit, limit);

        for (auto &channel : data) {
            for (auto &row : channel) {
                for (auto &val : row) {
                    val = dis(gen);
                }
            }
        }
    }

    void uniform_initialise(float lower_bound = 0.0f, float upper_bound = 1.0f) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(lower_bound, upper_bound);

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
                    val = 0.0f;
                }
            }
        }
    }

    // operators

    /**
     * @brief Overloads the multiplication operator for Tensor3D multiplication on each depth slice.
     * @param other The tensor to multiply with.
     * @return The resulting tensor after multiplication.
     */
    Tensor3D operator*(const Tensor3D &other) const {
        // check dimensions match for Tensor3D multiplication at each depth
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

        // perform Tensor3D multiplication for each depth slice
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

    Tensor3D operator+(const float &other) const {
        Tensor3D result(depth, height, width);
        for (size_t d = 0; d < depth; d++) {
            for (size_t i = 0; i < height; i++) {
                for (size_t j = 0; j < width; j++) {
                    result.data[d][i][j] = data[d][i][j] + other;
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

    Tensor3D operator*(float scalar) const {
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

    Tensor3D apply(float (*func)(float)) const {
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
        Tensor3D result(depth, width, height);
        for (size_t d = 0; d < depth; d++) {
            for (size_t i = 0; i < height; i++) {
                for (size_t j = 0; j < width; j++) {
                    result.data[d][j][i] = data[d][i][j];
                }
            }
        }
        return result;
    }

    // softmax across height dimension for back-compatibility with old Tensor3D class
    Tensor3D softmax() const {
        Tensor3D result(depth, height, width);

        for (size_t d = 0; d < depth; d++) {
            // find max across height (class scores)
            float max_val = -std::numeric_limits<float>::infinity();
            for (size_t h = 0; h < height; h++) {
                max_val = std::max(max_val, data[d][h][0]);
            }

            // compute exp and sum across height
            float sum = 0.0f;
            for (size_t h = 0; h < height; h++) {
                result.data[d][h][0] = std::exp(data[d][h][0] - max_val);
                sum += result.data[d][h][0];
            }

            // normalize across height
            for (size_t h = 0; h < height; h++) {
                result.data[d][h][0] /= sum;
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

    Tensor3D unflatten(size_t new_depth, size_t new_height, size_t new_width) const {
        // check if dimensions match
        if (depth != 1 || width != 1 || height != new_depth * new_height * new_width) {
            throw std::runtime_error("cannot unflatten tensor - dimensions don't match. Expected flattened tensor of height " +
                                     std::to_string(new_depth * new_height * new_width) + " but got height " +
                                     std::to_string(height));
        }

        Tensor3D result(new_depth, new_height, new_width);
        size_t idx = 0;

        // copy values back to 3D structure
        for (size_t d = 0; d < new_depth; d++) {
            for (size_t h = 0; h < new_height; h++) {
                for (size_t w = 0; w < new_width; w++) {
                    result.data[d][h][w] = data[0][idx][0];
                    idx++;
                }
            }
        }

        return result;
    }

    static Tensor3D Conv(const Tensor3D &input, const Tensor3D &kernel) {
        // check dimensions
        if (input.depth != kernel.depth) {
            throw std::runtime_error("input and kernel must have same depth for convolution");
        }

        // perform full convolution (no padding)
        Tensor3D output(1, input.height - kernel.height + 1, input.width - kernel.width + 1);

        // for each position in the output
        for (int y = 0; y < output.height; ++y) {
            for (int x = 0; x < output.width; ++x) {
                float sum = 0.0f;

                // sum over all channels and kernel positions
                for (int d = 0; d < input.depth; ++d) {
                    for (int ky = 0; ky < kernel.height; ++ky) {
                        for (int kx = 0; kx < kernel.width; ++kx) {
                            sum += input.data[d][y + ky][x + kx] * kernel.data[d][ky][kx];
                        }
                    }
                }
                output.data[0][y][x] = sum;
            }
        }
        return output;
    }

    Tensor3D rotate_180() const {
        Tensor3D result(depth, height, width);
        for (size_t d = 0; d < depth; d++) {
            for (size_t h = 0; h < height; h++) {
                for (size_t w = 0; w < width; w++) {
                    result.data[d][height - 1 - h][width - 1 - w] = data[d][h][w];
                }
            }
        }
        return result;
    }

    Tensor3D get_depth_slice(size_t depth_index) const {
        if (depth_index >= depth) {
            throw std::runtime_error("depth_index out of range in get_depth_slice");
        }

        Tensor3D slice(1, height, width);
        slice.data[0] = data[depth_index];
        return slice;
    }

    void set_depth_slice(size_t depth_index, const Tensor3D &slice) {
        if (depth_index >= depth) {
            throw std::runtime_error("depth_index out of range in set_depth_slice");
        }
        if (slice.depth != 1 || slice.height != height || slice.width != width) {
            throw std::runtime_error("slice dimensions don't match in set_depth_slice");
        }

        data[depth_index] = slice.data[0];
    }

    friend std::ostream &operator<<(std::ostream &os, const Tensor3D &tensor) {
        os << "Tensor3D(" << tensor.depth << ", " << tensor.height << ", " << tensor.width << ")\n";

        for (size_t d = 0; d < tensor.depth; ++d) {
            os << "Depth " << d << ":\n";
            for (size_t h = 0; h < tensor.height; ++h) {
                os << "[";
                for (size_t w = 0; w < tensor.width; ++w) {
                    os << std::fixed << std::setprecision(4) << tensor.data[d][h][w];
                    if (w < tensor.width - 1) os << ", ";
                }
                os << "]\n";
            }
            if (d < tensor.depth - 1) os << "\n";
        }
        return os;
    }
};

struct BackwardReturn {
    Tensor3D input_error;
    std::vector<Tensor3D> weight_grads;
    std::vector<Tensor3D> bias_grads;
};

// ---------------------------------- LAYER CLASSES -------------------------------------------
class Layer {
   public:
    std::vector<Tensor3D> weights;
    std::vector<Tensor3D> biases;
    virtual ~Layer() = default;

    // pure virtual class - requires implementation in derived objects.
    virtual Tensor3D forward(const Tensor3D &input) = 0;
    virtual BackwardReturn backward(const Tensor3D &d_output) = 0;
};

class DenseLayer : public Layer {
   public:
    std::string activation_function;
    Tensor3D input;
    Tensor3D z;

    /**
     * @brief Constructs a DenseLayer object with specified input size, output size, and activation function.
     * @param input_size The number of input neurons.
     * @param output_size The number of output neurons.
     * @param activation_function The activation function to use (default: "sigmoid").
     */
    DenseLayer(size_t input_size, size_t output_size, std::string activation_function = "relu")
        : activation_function(activation_function) {
        weights.emplace_back(output_size, input_size);
        biases.emplace_back(output_size, 1);

        if (activation_function == "sigmoid") {
            weights[0].xavier_initialise();
        } else if (activation_function == "relu") {
            weights[0].he_initialise();
        } else if (activation_function == "softmax") {
            weights[0].uniform_initialise(0, 1);
        } else {
            weights[0].uniform_initialise();
        }
    }

    // returns post-activation - stores input and preactivation for backprop
    Tensor3D forward(const Tensor3D &input) override {
        if (weights[0].width != input.height) {
            throw std::runtime_error("Input vector dimension is not appropriate for the weight Tensor3D dimension");
        }

        this->input = input;
        z = weights[0] * input + biases[0];

        Tensor3D a;

        if (activation_function == "sigmoid") {
            a = z.apply(sigmoid);
        } else if (activation_function == "relu") {
            a = z.apply(relu);
        } else if (activation_function == "softmax") {
            a = z.softmax();
        } else {
            a = z;
        }

        return a;
    }

    BackwardReturn backward(const Tensor3D &d_output) override{
        Tensor3D d_activation;
        Tensor3D d_z;

        if (activation_function == "sigmoid") {
            d_activation = z.apply(sigmoid_derivative);
            d_z = d_output.hadamard(d_activation);
        } else if (activation_function == "relu") {
            d_activation = z.apply(relu_derivative);
            d_z = d_output.hadamard(d_activation);
        } else if (activation_function == "softmax" or activation_function == "none") {
            d_z = d_output;  // combination of softmax and cross entropy loss means d_z is just predicted - target i.e. what is
                             // passed in as d_output
        } else {
            throw std::runtime_error("Unsupported activation function");
        }

        Tensor3D d_input = weights[0].transpose() * d_z;
        std::vector<Tensor3D> d_weights = {d_z * input.transpose()};
        std::vector<Tensor3D> d_biases = {d_z};

        return {d_input, d_weights, d_biases};
    }
};

class ConvolutionLayer : public Layer {
   public:
    int channels_in;
    int out_channels;
    int kernel_size;
    std::string mode;

    Tensor3D input;
    Tensor3D z;

    ConvolutionLayer(int channels_in, int out_channels, int kernel_size, std::string mode = "same")
        : channels_in(channels_in), out_channels(out_channels), kernel_size(kernel_size), mode(mode) {
        // creates a list of kernel tensors - one for each output channel
        weights.reserve(out_channels);
        for (int i = 0; i < out_channels; i++) {
            weights.emplace_back(channels_in, kernel_size, kernel_size);
            weights.back().he_initialise();
        }

        // initialise biases to vector of 1x1x1 tensors with all elements set to 0
        biases.resize(out_channels, Tensor3D(1, 1, 1));
    }

    // returns post-activation - stores input and preactivation for backprop
    Tensor3D forward(const Tensor3D &input) override {
        if (input.depth != channels_in) {
            throw std::runtime_error("Input tensor depth does not match the number of input channels for this layer");
        }
        if (input.height == 0 or input.width == 0) {
            throw std::runtime_error("Input tensor has zero height or width");
        }

        this->input = input;
        Tensor3D a(weights.size(), input.height, input.width);
        Tensor3D tmp_z(
            weights.size(), input.height,
            input.width);  // actual z attribute is not initialised to correct dimensions so use this and then copy to z
        int pad_amount = 0;

        if (mode == "same") {
            Tensor3D padded_input = Tensor3D::pad(input, (kernel_size - 1) / 2);

            for (int feature_map_index = 0; feature_map_index < weights.size(); ++feature_map_index) {
                Tensor3D preactivation =
                    Tensor3D::Conv(padded_input, weights[feature_map_index]) + biases[feature_map_index].data[0][0][0];

                // store the preactivation for this feature map
                tmp_z.set_depth_slice(feature_map_index, preactivation);

                // apply the activation function and store the result for this feature map
                a.data[feature_map_index] = preactivation.apply(relu).data[0];
            }

            // copy tmp_z to z attribute
            z = tmp_z;
            return a;

        } else {
            throw std::runtime_error("mode not specified or handled correctly");
        }
    }

    // fixme later assumes 'same' padding
    BackwardReturn backward(const Tensor3D &d_output) override {
        Tensor3D d_z = d_output.hadamard(z.apply(relu_derivative));

        // initialise gradient tensors
        std::vector<Tensor3D> d_weights;
        std::vector<Tensor3D> d_bias;
        d_weights.reserve(weights.size());
        d_bias.reserve(weights.size());

        // compute d_input
        Tensor3D d_input(input.depth, input.height, input.width);
        d_input.zero_initialise();

        // pad d_output for full convolution
        int pad_amount = (kernel_size - 1) / 2;
        Tensor3D padded_d_z = Tensor3D::pad(d_z, pad_amount);

        for (int in_ch = 0; in_ch < input.depth; in_ch++) {
            // sum contributions from all output channels
            for (int k = 0; k < weights.size(); k++) {
                // extract relevant delta channel and kernel slice
                Tensor3D relevant_d_z_slice = padded_d_z.get_depth_slice(k);
                Tensor3D relevant_kernel_slice = weights[k].get_depth_slice(in_ch).rotate_180();

                // perform full convolution
                Tensor3D d_input_contribution = Tensor3D::Conv(relevant_d_z_slice, relevant_kernel_slice);

                // add the contribution to d_input
                for (int y = 0; y < d_input.height; y++) {
                    for (int x = 0; x < d_input.width; x++) {
                        d_input.data[in_ch][y][x] += d_input_contribution.data[0][y][x];
                    }
                }
            }
        }

        // compute weight gradients

        Tensor3D padded_input = Tensor3D::pad(input, pad_amount);

        for (int k = 0; k < weights.size(); k++) {
            Tensor3D d_weight(input.depth, kernel_size, kernel_size);
            for (int in_ch = 0; in_ch < input.depth; in_ch++) {
                // extract relevant input and gradient channels
                Tensor3D padded_input_channel = padded_input.get_depth_slice(in_ch);
                Tensor3D d_z_channel = d_z.get_depth_slice(k);

                // compute gradient for this input-output channel pair
                Tensor3D channel_grad = Tensor3D::Conv(padded_input_channel, d_z_channel);
                d_weight.set_depth_slice(in_ch, channel_grad);
            }
            d_weights.push_back(d_weight);

            // compute bias gradient (sum of d_output)
            float d_bias_val = 0.0f;
            for (int y = 0; y < d_output.height; y++) {
                for (int x = 0; x < d_output.width; x++) {
                    d_bias_val += d_z.data[k][y][x];
                }
            }
            Tensor3D bias_grad(1, 1, 1);
            bias_grad.data[0][0][0] = d_bias_val;
            d_bias.push_back(bias_grad);
        }

        return {d_input, d_weights, d_bias};
    }
};

class PoolingLayer : public Layer {
   public:
    int kernel_size;
    int stride;
    std::string mode;
    Tensor3D input;                                                            // store input for backward pass
    std::vector<std::vector<std::vector<std::pair<int, int>>>> max_positions;  // stores positions of max values
    size_t output_depth, output_height, output_width;                          // store output dimensions for unflattening

    PoolingLayer(int kernel_size = 2, int stride = -1, std::string mode = "max")
        : kernel_size(kernel_size), stride(stride == -1 ? kernel_size : stride), mode(mode) {
        if (stride > kernel_size) {
            throw std::runtime_error("stride should not be greater than kernel size");
        }
    }

    Tensor3D forward(const Tensor3D &input) override {
        this->input = input;  // store input for backward pass
        // calculate output dimensions including partial window pooling
        output_height = std::ceil((input.height - kernel_size) / stride + 1);
        output_width = std::ceil((input.width - kernel_size) / stride + 1);
        output_depth = input.depth;
        Tensor3D output(output_depth, output_height, output_width);

        // initialise max_positions storage
        max_positions.resize(input.depth);
        for (auto &channel : max_positions) {
            channel.resize(output_height, std::vector<std::pair<int, int>>(output_width));
        }

        if (mode == "max") {
            for (int d = 0; d < input.depth; ++d) {
                for (int y = 0; y < output_height; ++y) {
                    for (int x = 0; x < output_width; ++x) {
                        // calculate window boundaries
                        int start_y = y * stride;
                        int start_x = x * stride;
                        int end_y = std::min(start_y + kernel_size, static_cast<int>(input.height));
                        int end_x = std::min(start_x + kernel_size, static_cast<int>(input.width));

                        // find maximum in this window
                        float max_val = -std::numeric_limits<float>::infinity();
                        int max_y = -1, max_x = -1;
                        for (int wy = start_y; wy < end_y; ++wy) {
                            for (int wx = start_x; wx < end_x; ++wx) {
                                if (input.data[d][wy][wx] > max_val) {
                                    max_val = input.data[d][wy][wx];
                                    max_y = wy;
                                    max_x = wx;
                                }
                            }
                        }
                        output.data[d][y][x] = max_val;
                        max_positions[d][y][x] = {max_y, max_x};  // store position of maximum
                    }
                }
            }
        } else {
            throw std::runtime_error("mode not specified or handled correctly");
        }

        return output;
    }

    BackwardReturn backward(const Tensor3D &d_output) override {
        // if the gradient is coming from a dense layer, we need to unflatten it first
        Tensor3D d_output_unflattened = d_output;
        if (d_output.depth == 1 && d_output.width == 1) {
            d_output_unflattened = d_output.unflatten(output_depth, output_height, output_width);
        }

        // initialise d_input with zeros
        Tensor3D d_input(input.depth, input.height, input.width);
        d_input.zero_initialise();

        if (mode == "max") {
            // for max pooling, propagate gradient only to the position where maximum was found
            // d_output should have same dimensions as the output from forward pass
            if (d_output_unflattened.depth != max_positions.size() || d_output_unflattened.height != max_positions[0].size() ||
                d_output_unflattened.width != max_positions[0][0].size()) {
                throw std::runtime_error(
                    "d_output dimensions do not match stored max_positions dimensions in pooling layer backward pass");
            }

            for (size_t d = 0; d < d_output_unflattened.depth; ++d) {
                for (size_t y = 0; y < d_output_unflattened.height; ++y) {
                    for (size_t x = 0; x < d_output_unflattened.width; ++x) {
                        // get the position where the maximum was found
                        auto [max_y, max_x] = max_positions[d][y][x];
                        // propagate gradient to that position
                        d_input.data[d][max_y][max_x] += d_output_unflattened.data[d][y][x];
                    }
                }
            }
        }

        // pooling layers have no weights or biases to update
        std::vector<Tensor3D> empty_weight_grads;
        std::vector<Tensor3D> empty_bias_grads;

        return {d_input, empty_weight_grads, empty_bias_grads};
    }
};

// ---------------------------------- LOSS FUNCTIONS -------------------------------------------
class Loss {
   public:
    virtual ~Loss() = default;

    // Compute the loss value
    virtual float compute(const Tensor3D &predicted, const Tensor3D &target) const = 0;

    // Compute the derivative of the loss with respect to the predicted values
    virtual Tensor3D derivative(const Tensor3D &predicted, const Tensor3D &target) const = 0;
}; 

class CrossEntropyLoss : public Loss {
   public:
    float compute(const Tensor3D &predicted, const Tensor3D &target) const override {
        float loss = 0.0f;
        for (size_t i = 0; i < predicted.height; ++i) {
            for (size_t j = 0; j < predicted.width; ++j) {
                // Add small epsilon to avoid log(0)
                loss -= target.data[0][i][j] * std::log(predicted.data[0][i][j] + 1e-10f);
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
    float compute(const Tensor3D &predicted, const Tensor3D &target) const override {
        float loss = 0.0f;
        for (size_t i = 0; i < predicted.height; ++i) {
            for (size_t j = 0; j < predicted.width; ++j) {
                float diff = predicted.data[0][i][j] - target.data[0][i][j];
                loss += diff * diff;
            }
        }
        return loss / (2.0f * predicted.width);  // Average over batch and divide by 2
    }

    Tensor3D derivative(const Tensor3D &predicted, const Tensor3D &target) const override {
        return (predicted - target) * (1.0 / predicted.width);
    }
};

// ---------------------------------- OPTIMISERS -------------------------------------------
class Optimiser {
   public:
    virtual ~Optimiser() = default;

    virtual void compute_and_apply_updates(
        const std::vector<std::unique_ptr<Layer>> &layers,
        const std::vector<std::pair<std::vector<Tensor3D>, std::vector<Tensor3D>>> &gradients) = 0;
};

class SGDOptimiser : public Optimiser {
   private:
    float learning_rate;

   public:
    SGDOptimiser(float lr = 0.1f) : learning_rate(lr) {}

    void compute_and_apply_updates(
        const std::vector<std::unique_ptr<Layer>> &layers,
        const std::vector<std::pair<std::vector<Tensor3D>, std::vector<Tensor3D>>> &gradients) override {
        // gradients is a list of pairs
        // each pair corresponds to a layer and contains firstly a list of weight gradients and secondly a list of bias gradients

        for (int layer_index = 0; layer_index < layers.size(); layer_index++) {
            auto [weight_gradients, bias_gradients] = gradients[layer_index];

            for (int weight_index = 0; weight_index < weight_gradients.size(); weight_index++) {
                // update the corresponding layer's corresponding weight Tensor
                (layers[layer_index])->weights[weight_index] =
                    (layers[layer_index])->weights[weight_index] - (weight_gradients[weight_index] * learning_rate);
            }

            for (int bias_index = 0; bias_index < bias_gradients.size(); bias_index++) {
                // update the corresponding layer's corresponding bias Tensor
                (layers[layer_index])->biases[bias_index] =
                    (layers[layer_index])->biases[bias_index] - (bias_gradients[bias_index] * learning_rate);
            }
        }
    }
};

class AdamWOptimiser : public Optimiser {
   private:
    float learning_rate;
    float beta1;
    float beta2;
    float epsilon;
    float weight_decay;
    int t;                                                                   // timestep
    std::vector<std::pair<std::vector<Tensor3D>, std::vector<Tensor3D>>> m;  // first moment
    std::vector<std::pair<std::vector<Tensor3D>, std::vector<Tensor3D>>> v;  // second moment

    // list of layers
    // each element is a pair of vectors
    // first vector is weight gradients
    // second vector is bias gradients

   public:
    /**
     * @brief Constructs an AdamWOptimiser object with the specified parameters.
     * @param lr The learning rate (default: 0.001).
     * @param b1 The beta1 parameter (default: 0.9).
     * @param b2 The beta2 parameter (default: 0.999).
     * @param eps The epsilon parameter for numerical stability (default: 1e-8).
     * @param wd The weight decay parameter (default: 0.01).
     */
    AdamWOptimiser(float lr = 0.001f, float b1 = 0.9f, float b2 = 0.999f, float eps = 1e-8f, float wd = 0.01f)
        : learning_rate(lr), beta1(b1), beta2(b2), epsilon(eps), weight_decay(wd), t(0) {}

    /**
     * @brief Initializes the first and second moment vectors for AdamW optimization.
     * @param layers Vector of unique pointers to layers of the neural network.
     */
    void initialize_moments(const std::vector<std::unique_ptr<Layer>> &layers) {
        m.clear();
        v.clear();

        // iterate through each layer
        for (const auto &layer : layers) {
            std::vector<Tensor3D> layer_weight_m, layer_weight_v;
            std::vector<Tensor3D> layer_bias_m, layer_bias_v;

            // initialize moments for weights
            for (const auto &weight : layer->weights) {
                Tensor3D zero_tensor(weight.depth, weight.height, weight.width);
                layer_weight_m.push_back(zero_tensor);
                layer_weight_v.push_back(zero_tensor);
            }

            // initialize moments for biases
            for (const auto &bias : layer->biases) {
                Tensor3D zero_tensor(bias.depth, bias.height, bias.width);
                layer_bias_m.push_back(zero_tensor);
                layer_bias_v.push_back(zero_tensor);
            }

            // store the moments for this layer
            m.push_back({layer_weight_m, layer_bias_m});
            v.push_back({layer_weight_v, layer_bias_v});
        }
    }

    /**
     * @brief Computes and applies updates using the AdamW optimization algorithm.
     * @param layers The layers of the neural network to update.
     * @param gradients The gradients used for updating the layers.
     */
    void compute_and_apply_updates(
        const std::vector<std::unique_ptr<Layer>> &layers,
        const std::vector<std::pair<std::vector<Tensor3D>, std::vector<Tensor3D>>> &gradients) override {
        if (m.empty() or v.empty()) {
            initialize_moments(layers);
        }

        t++;  // increment timestep

        for (size_t layer_index = 0; layer_index < layers.size(); ++layer_index) {
            // calculate updates for each weight parameter
            for (int param_no = 0; param_no < m[layer_index].first.size(); ++param_no) {
                // update biased first moment estimate
                m[layer_index].first[param_no] =
                    m[layer_index].first[param_no] * beta1 + gradients[layer_index].first[param_no] * (1.0 - beta1);

                // update biased second raw moment estimate
                v[layer_index].first[param_no] =
                    v[layer_index].first[param_no] * beta2 +
                    gradients[layer_index].first[param_no].hadamard(gradients[layer_index].first[param_no]) * (1.0 - beta2);

                // compute bias-corrected first moment estimate
                Tensor3D m_hat = m[layer_index].first[param_no] * (1.0 / (1.0 - std::pow(beta1, t)));

                // compute bias-corrected second raw moment estimate
                Tensor3D v_hat = v[layer_index].first[param_no] * (1.0 / (1.0 - std::pow(beta2, t)));

                // compute the Adam update
                Tensor3D update = m_hat.hadamard(v_hat.apply([this](float x) { return 1.0f / (std::sqrt(x) + epsilon); }));

                // apply weight decay
                layers[layer_index]->weights[param_no] =
                    layers[layer_index]->weights[param_no] * (1.0 - learning_rate * weight_decay);
                // apply Adam update
                layers[layer_index]->weights[param_no] = layers[layer_index]->weights[param_no] - (update * learning_rate);
            }

            // calculate updates for each bias parameter
            for (int param_no = 0; param_no < m[layer_index].second.size(); ++param_no) {
                // update biased first moment estimate
                m[layer_index].second[param_no] =
                    m[layer_index].second[param_no] * beta1 + gradients[layer_index].second[param_no] * (1.0 - beta1);

                // update biased second raw moment estimate
                v[layer_index].second[param_no] =
                    v[layer_index].second[param_no] * beta2 +
                    gradients[layer_index].second[param_no].hadamard(gradients[layer_index].second[param_no]) * (1.0 - beta2);

                // compute bias-corrected first moment estimate
                Tensor3D m_hat = m[layer_index].second[param_no] * (1.0 / (1.0 - std::pow(beta1, t)));

                // compute bias-corrected second raw moment estimate
                Tensor3D v_hat = v[layer_index].second[param_no] * (1.0 / (1.0 - std::pow(beta2, t)));

                // compute the Adam update
                Tensor3D update = m_hat.hadamard(v_hat.apply([this](float x) { return 1.0f / (std::sqrt(x) + epsilon); }));

                // no weight decay for biases

                // apply Adam update
                layers[layer_index]->biases[param_no] = layers[layer_index]->biases[param_no] - (update * learning_rate);
            }
        }
    }
};

// ---------------------------------- NEURAL NETWORK -------------------------------------------
class NeuralNetwork {
   private:
    struct LayerSpec {
        enum Type { CONV, POOL, DENSE } type;

        // conv parameters
        int in_channels = 0;
        int out_channels = 0;
        int kernel_size = 0;
        std::string mode = "same";

        // pool parameters
        int pool_size = 0;
        int pool_stride = 0;
        std::string pool_mode = "max";

        // dense parameters
        std::string activation = "relu";
        size_t output_size = 0;

        static LayerSpec Conv(int out_channels, int kernel_size, std::string mode = "same") {
            LayerSpec spec;
            spec.type = CONV;
            spec.out_channels = out_channels;
            spec.kernel_size = kernel_size;
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
                    layers.push_back(
                        std::make_unique<ConvolutionLayer>(spec.in_channels, spec.out_channels, spec.kernel_size, spec.mode));

                    // todo change dims.width and dims.height here if ever put more modes than same

                    dims.depth = spec.out_channels;
                    break;
                }
                case LayerSpec::POOL: {
                    layers.push_back(std::make_unique<PoolingLayer>(spec.pool_size, spec.pool_stride, spec.pool_mode));

                    // update 'previous' layer dimensions with formula for pooling layer output dimensions
                    dims.height = std::floor((dims.height - spec.pool_size) / static_cast<float>(spec.pool_stride) + 1);
                    dims.width = std::floor((dims.width - spec.pool_size) / static_cast<float>(spec.pool_stride) + 1);
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

    // helper functions for saving/loading Tensor3D
    static void save_tensor(std::ofstream& file, const Tensor3D& tensor) {
        uint32_t depth = static_cast<uint32_t>(tensor.depth);
        uint32_t height = static_cast<uint32_t>(tensor.height);
        uint32_t width = static_cast<uint32_t>(tensor.width);
        
        file.write(reinterpret_cast<const char*>(&depth), sizeof(depth));
        file.write(reinterpret_cast<const char*>(&height), sizeof(height));
        file.write(reinterpret_cast<const char*>(&width), sizeof(width));
        
        for (const auto& channel : tensor.data) {
            for (const auto& row : channel) {
                file.write(reinterpret_cast<const char*>(row.data()), row.size() * sizeof(float));
            }
        }
    }

    static void load_tensor(std::ifstream& file, Tensor3D& tensor) {
        uint32_t depth, height, width;
        file.read(reinterpret_cast<char*>(&depth), sizeof(depth));
        file.read(reinterpret_cast<char*>(&height), sizeof(height));
        file.read(reinterpret_cast<char*>(&width), sizeof(width));
        
        tensor = Tensor3D(depth, height, width);
        
        for (auto& channel : tensor.data) {
            for (auto& row : channel) {
                file.read(reinterpret_cast<char*>(row.data()), row.size() * sizeof(float));
            }
        }
    }


   public:

    struct EvalMetrics {
        float accuracy;
        float precision;
        float recall;
        float f1_score;

        friend std::ostream &operator<<(std::ostream &os, const EvalMetrics &metrics) {
            os << "accuracy: " << metrics.accuracy << ", precision: " << metrics.precision << ", recall: " << metrics.recall
               << ", f1_score: " << metrics.f1_score;
            return os;
        }
    };


    std::vector<LayerSpec> layer_specs;
    std::vector<std::unique_ptr<Layer>> layers;
    std::unique_ptr<Optimiser> optimiser;
    std::unique_ptr<Loss> loss;

    bool layers_created = false;

    // default constructor
    NeuralNetwork() {}

    // user-facing funcs to add layers (just adds their specs to layer_specs to be created with correct input dimensions later)
    void add_conv_layer(int out_channels, int kernel_size, std::string mode = "same") {
        layer_specs.push_back(LayerSpec::Conv(out_channels, kernel_size, mode));
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
    void set_loss(std::unique_ptr<Loss> new_loss) {
        bool layers_empty = layer_specs.empty();
        bool last_layer_is_softmax = layer_specs.back().activation == "softmax";
        bool new_loss_is_cross_entropy = dynamic_cast<CrossEntropyLoss*>(new_loss.get()) != nullptr;

        if (layers_empty) {
            throw std::runtime_error("no layers created yet - set layers first");
        } else if (!last_layer_is_softmax and new_loss_is_cross_entropy){
            throw std::runtime_error("last layer must be softmax for cross entropy loss");
        }

        loss = std::move(new_loss); 
    }

    /**
     * @brief Performs feedforward operation through all layers of the network.
     * @param input The input Tensor3D.
     * @return The output Tensor3D after passing through all layers.
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
                current = current_conv->forward(current);

                // if next layer is dense, we need to flatten to (1, N, 1) where N is total elements
                if (next_dense) {
                    current = current.flatten();
                }
            } else if (current_pooling) {
                current = current_pooling->forward(current);

                // if next layer is dense, we need to flatten to (1, N, 1) where N is total elements
                if (next_dense) {
                    current = current.flatten();
                }
            } else if (current_dense) {
                if (next_pooling or next_conv) {
                    throw std::runtime_error("dense layer cannot be followed by a pooling or convolution layer");
                }
                current = current_dense->forward(current);
            } else {
                throw std::runtime_error("unknown layer type encountered");
            }
        }
        return current;
    }

    /**
     * @brief Performs backpropagation through the network for a single training example
     * @param input The input to the network
     * @param target The target output
     * @return A vector of pairs, each containing weight and bias gradients for a layer
     */
    std::vector<std::pair<std::vector<Tensor3D>, std::vector<Tensor3D>>> calculate_gradients(const Tensor3D &input,
                                                                                             const Tensor3D &target) {
        if (!loss) {
            throw std::runtime_error("loss function not set");
        }

        // perform forward pass to get predictions
        Tensor3D predicted = feedforward(input);

        // store gradients for each layer
        std::vector<std::pair<std::vector<Tensor3D>, std::vector<Tensor3D>>> all_gradients;
        all_gradients.reserve(layers.size());

        // compute initial gradient from loss function
        Tensor3D gradient = loss->derivative(predicted, target);

        // backpropagate through layers in reverse order
        for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
            // compute gradients for current layer
            BackwardReturn layer_grads = (*it)->backward(gradient);

            // store the weight and bias gradients
            all_gradients.push_back({layer_grads.weight_grads, layer_grads.bias_grads});

            // update gradient for next layer
            gradient = layer_grads.input_error;
        }

        // reverse the gradients so they match the order of layers
        std::reverse(all_gradients.begin(), all_gradients.end());

        return all_gradients;
    }

    EvalMetrics evaluate(const std::vector<std::vector<Tensor3D>> &eval_set) {
        // initialise counts for metrics
        int total_samples = 0;
        int correct_predictions = 0;
        std::vector<int> true_positives(10, 0);
        std::vector<int> false_positives(10, 0);
        std::vector<int> false_negatives(10, 0);

        // evaluate each sample
        for (const auto &sample : eval_set) {
            const auto &input = sample[0];
            const auto &target = sample[1];

            // get network prediction
            Tensor3D predicted = feedforward(augment_image(input));

            // find predicted and actual class
            int predicted_class = 0;
            int actual_class = 0;
            float max_pred = predicted.data[0][0][0];
            float max_target = target.data[0][0][0];

            for (int i = 1; i < 10; ++i) {
                if (predicted.data[0][i][0] > max_pred) {
                    max_pred = predicted.data[0][i][0];
                    predicted_class = i;
                }
                if (target.data[0][i][0] > max_target) {
                    max_target = target.data[0][i][0];
                    actual_class = i;
                }
            }

            // update counts
            total_samples++;
            if (predicted_class == actual_class) {
                correct_predictions++;
                true_positives[actual_class]++;
            } else {
                false_positives[predicted_class]++;
                false_negatives[actual_class]++;
            }
        }

        // calculate metrics
        float accuracy = static_cast<float>(correct_predictions) / total_samples;

        // calculate macro-averaged precision, recall and f1
        float total_precision = 0.0f;
        float total_recall = 0.0f;
        float total_f1 = 0.0f;
        int num_classes = 0;

        for (int i = 0; i < 10; ++i) {
            // skip classes with no samples to avoid division by zero
            if (true_positives[i] + false_positives[i] + false_negatives[i] == 0) {
                continue;
            }

            float class_precision = true_positives[i] / static_cast<float>(true_positives[i] + false_positives[i]);
            float class_recall = true_positives[i] / static_cast<float>(true_positives[i] + false_negatives[i]);
            float class_f1 = 2 * (class_precision * class_recall) / (class_precision + class_recall);

            // handle cases where precision + recall = 0
            if (std::isnan(class_f1)) {
                class_f1 = 0.0;
            }

            total_precision += class_precision;
            total_recall += class_recall;
            total_f1 += class_f1;
            num_classes++;
        }

        // compute macro averages
        float macro_precision = total_precision / num_classes;
        float macro_recall = total_recall / num_classes;
        float macro_f1 = total_f1 / num_classes;

        return {accuracy, macro_precision, macro_recall, macro_f1};
    }

    void train(std::vector<std::vector<Tensor3D>> &training_set,
               const std::vector<std::vector<Tensor3D>> &eval_set,
               const int num_epochs,
               const int batch_size) {

        const int batches_per_epoch = training_set.size() / batch_size;
        // training loop
        for (int epoch = 0; epoch < num_epochs; ++epoch) {
            // shuffle training data at start of each epoch
            std::shuffle(training_set.begin(), training_set.end(), std::random_device());

            float epoch_loss = 0.0f;

            // batch loop
            for (int batch = 0; batch < batches_per_epoch; ++batch) {
                std::vector<std::pair<std::vector<Tensor3D>, std::vector<Tensor3D>>> batch_gradients;
                float batch_loss = 0.0f;

                // accumulate gradients over batch
                for (int i = 0; i < batch_size; ++i) {
                    int idx = batch * batch_size + i;
                    auto &input = training_set[idx][0];
                    auto &target = training_set[idx][1];

                    // apply augmentation to input
                    auto augmented_input = augment_image(input);
                    
                    // use augmented input for training
                    auto gradients = calculate_gradients(augmented_input, target);

                    // initialise batch_gradients if first sample
                    if (i == 0) {
                        batch_gradients = gradients;
                    } else {
                        // add gradients element-wise
                        for (size_t layer = 0; layer < gradients.size(); ++layer) {
                            for (size_t w = 0; w < gradients[layer].first.size(); ++w) {
                                batch_gradients[layer].first[w] = batch_gradients[layer].first[w] + gradients[layer].first[w];
                            }
                            for (size_t b = 0; b < gradients[layer].second.size(); ++b) {
                                batch_gradients[layer].second[b] = batch_gradients[layer].second[b] + gradients[layer].second[b];
                            }
                        }
                    }

                    // accumulate loss
                    batch_loss += loss->compute(feedforward(input), target);
                }

                // average gradients and loss over batch
                for (auto &layer_grads : batch_gradients) {
                    for (auto &w_grad : layer_grads.first) {
                        w_grad = w_grad * (1.0 / batch_size);
                    }
                    for (auto &b_grad : layer_grads.second) {
                        b_grad = b_grad * (1.0 / batch_size);
                    }
                }
                batch_loss /= batch_size;
                epoch_loss += batch_loss;

                // apply averaged gradients
                optimiser->compute_and_apply_updates(layers, batch_gradients);

                // print batch progress
                if (batch % 1 == 0) {
                    EvalMetrics metrics = evaluate(eval_set);
                    std::cout << "epoch " << epoch + 1 << ", batch " << batch << "/" << batches_per_epoch << ": " << metrics << std::endl;
                }
            }
        
            std::string model_path = "model" + std::to_string(epoch) + ".bin";
            save_model(model_path);
            
        }
    }

    Tensor3D augment_image(const Tensor3D& input, float offset_range = 5.0f, float scale_range = 0.2f, float angle_range = 20.0f) {
        // create output tensor of same size
        Tensor3D augmented(1, 28, 28);
        
        // initialise random number generation
        std::random_device rd;
        std::mt19937 gen(rd());
        
        // random offset range (-2 to 2 pixels)
        std::uniform_int_distribution<> offset_dist(-offset_range, offset_range);
        int offsetX = offset_dist(gen);
        int offsetY = offset_dist(gen);
        
        // random slight rotation (-15 to 15 degrees)
        std::uniform_real_distribution<> angle_dist(-angle_range, angle_range);
        float angle = angle_dist(gen) * M_PI / 180.0f;
        
        // random slight scaling (0.9 to 1.1)
        std::uniform_real_distribution<> scale_dist(1.0f - scale_range, 1.0f + scale_range);
        float scale = scale_dist(gen);
        
        // centre point for rotation
        float centerX = input.width / 2.0f;
        float centerY = input.height / 2.0f;
        
        // fill output tensor with transformed input
        for (size_t y = 0; y < augmented.height; y++) {
            for (size_t x = 0; x < augmented.width; x++) {
                // work backwards from output to input coordinates to avoid gaps
                
                // translate to origin
                float dx = x - centerX - offsetX;
                float dy = y - centerY - offsetY;
                
                // unscale
                dx /= scale;
                dy /= scale;
                
                // unrotate
                float srcX = dx * cos(-angle) - dy * sin(-angle) + centerX;
                float srcY = dx * sin(-angle) + dy * cos(-angle) + centerY;
                
                // if source pixel is within bounds, copy it
                if (srcX >= 0 && srcX < input.width - 1 && 
                    srcY >= 0 && srcY < input.height - 1) {
                    
                    // bilinear interpolation
                    int x0 = static_cast<int>(srcX);
                    int x1 = x0 + 1;
                    int y0 = static_cast<int>(srcY);
                    int y1 = y0 + 1;
                    
                    float wx1 = srcX - x0;
                    float wx0 = 1 - wx1;
                    float wy1 = srcY - y0;
                    float wy0 = 1 - wy1;
                    
                    augmented.data[0][y][x] = 
                        input.data[0][y0][x0] * wx0 * wy0 +
                        input.data[0][y0][x1] * wx1 * wy0 +
                        input.data[0][y1][x0] * wx0 * wy1 +
                        input.data[0][y1][x1] * wx1 * wy1;
                }
            }
        }
        
        return augmented;
    }

    // save model to file - must be called after training/forward pass
    void save_model(const std::string filename) const {
        std::ofstream file(filename, std::ios::binary);
        if (!file) {
            throw std::runtime_error("unable to open file for writing: " + filename);
        }

        // write number of layers
        uint32_t num_layers = static_cast<uint32_t>(layers.size());
        file.write(reinterpret_cast<const char*>(&num_layers), sizeof(num_layers));

        // write layer specifications
        for (const auto& layer : layers) {
            // write layer type
            uint32_t layer_type;
            if (dynamic_cast<ConvolutionLayer*>(layer.get())) {
                layer_type = 0;
            } else if (dynamic_cast<PoolingLayer*>(layer.get())) {
                layer_type = 1;
            } else if (dynamic_cast<DenseLayer*>(layer.get())) {
                layer_type = 2;
            }
            file.write(reinterpret_cast<const char*>(&layer_type), sizeof(layer_type));

            // write layer-specific parameters
            if (auto conv_layer = dynamic_cast<ConvolutionLayer*>(layer.get())) {
                uint32_t channels_in = static_cast<uint32_t>(conv_layer->channels_in);
                uint32_t out_channels = static_cast<uint32_t>(conv_layer->out_channels);
                uint32_t kernel_size = static_cast<uint32_t>(conv_layer->kernel_size);
                uint32_t mode_length = static_cast<uint32_t>(conv_layer->mode.length());

                file.write(reinterpret_cast<const char*>(&channels_in), sizeof(channels_in));
                file.write(reinterpret_cast<const char*>(&out_channels), sizeof(out_channels));
                file.write(reinterpret_cast<const char*>(&kernel_size), sizeof(kernel_size));
                file.write(reinterpret_cast<const char*>(&mode_length), sizeof(mode_length));
                file.write(conv_layer->mode.c_str(), mode_length);

                // save weights and biases
                for (const auto& weight : conv_layer->weights) {
                    save_tensor(file, weight);
                }
                for (const auto& bias : conv_layer->biases) {
                    save_tensor(file, bias);
                }
            } 
            else if (auto pool_layer = dynamic_cast<PoolingLayer*>(layer.get())) {
                uint32_t kernel_size = static_cast<uint32_t>(pool_layer->kernel_size);
                uint32_t stride = static_cast<uint32_t>(pool_layer->stride);
                uint32_t mode_length = static_cast<uint32_t>(pool_layer->mode.length());

                file.write(reinterpret_cast<const char*>(&kernel_size), sizeof(kernel_size));
                file.write(reinterpret_cast<const char*>(&stride), sizeof(stride));
                file.write(reinterpret_cast<const char*>(&mode_length), sizeof(mode_length));
                file.write(pool_layer->mode.c_str(), mode_length);
            }
            else if (auto dense_layer = dynamic_cast<DenseLayer*>(layer.get())) {
                uint32_t input_size = static_cast<uint32_t>(dense_layer->weights[0].width);
                uint32_t output_size = static_cast<uint32_t>(dense_layer->weights[0].height);
                uint32_t activation_length = static_cast<uint32_t>(dense_layer->activation_function.length());

                file.write(reinterpret_cast<const char*>(&input_size), sizeof(input_size));
                file.write(reinterpret_cast<const char*>(&output_size), sizeof(output_size));
                file.write(reinterpret_cast<const char*>(&activation_length), sizeof(activation_length));
                file.write(dense_layer->activation_function.c_str(), activation_length);

                // save weights and biases
                for (const auto& weight : dense_layer->weights) {
                    save_tensor(file, weight);
                }
                for (const auto& bias : dense_layer->biases) {
                    save_tensor(file, bias);
                }
            }
        }
    }

    static NeuralNetwork load_model(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary);
        if (!file) {
            throw std::runtime_error("unable to open file for reading: " + filename);
        }

        NeuralNetwork nn;

        // read number of layers
        uint32_t num_layers;
        file.read(reinterpret_cast<char*>(&num_layers), sizeof(num_layers));

        // read and create each layer
        for (uint32_t i = 0; i < num_layers; ++i) {
            uint32_t layer_type;
            file.read(reinterpret_cast<char*>(&layer_type), sizeof(layer_type));

            if (layer_type == 0) {  // ConvolutionLayer
                uint32_t channels_in, out_channels, kernel_size, mode_length;
                file.read(reinterpret_cast<char*>(&channels_in), sizeof(channels_in));
                file.read(reinterpret_cast<char*>(&out_channels), sizeof(out_channels));
                file.read(reinterpret_cast<char*>(&kernel_size), sizeof(kernel_size));
                file.read(reinterpret_cast<char*>(&mode_length), sizeof(mode_length));

                std::string mode(mode_length, '\0');
                file.read(&mode[0], mode_length);

                auto layer = std::make_unique<ConvolutionLayer>(channels_in, out_channels, kernel_size, mode);
                
                // load weights and biases
                for (auto& weight : layer->weights) {
                    load_tensor(file, weight);
                }
                for (auto& bias : layer->biases) {
                    load_tensor(file, bias);
                }
                
                nn.layers.push_back(std::move(layer));
            }
            else if (layer_type == 1) {  // PoolingLayer
                uint32_t kernel_size, stride, mode_length;
                file.read(reinterpret_cast<char*>(&kernel_size), sizeof(kernel_size));
                file.read(reinterpret_cast<char*>(&stride), sizeof(stride));
                file.read(reinterpret_cast<char*>(&mode_length), sizeof(mode_length));

                std::string mode(mode_length, '\0');
                file.read(&mode[0], mode_length);

                nn.layers.push_back(std::make_unique<PoolingLayer>(kernel_size, stride, mode));
            }
            else if (layer_type == 2) {  // DenseLayer
                uint32_t input_size, output_size, activation_length;
                file.read(reinterpret_cast<char*>(&input_size), sizeof(input_size));
                file.read(reinterpret_cast<char*>(&output_size), sizeof(output_size));
                file.read(reinterpret_cast<char*>(&activation_length), sizeof(activation_length));

                std::string activation(activation_length, '\0');
                file.read(&activation[0], activation_length);

                auto layer = std::make_unique<DenseLayer>(input_size, output_size, activation);
                
                // load weights and biases
                for (auto& weight : layer->weights) {
                    load_tensor(file, weight);
                }
                for (auto& bias : layer->biases) {
                    load_tensor(file, bias);
                }
                
                nn.layers.push_back(std::move(layer));
            }
        }

        return nn;
    }

    // method for WASM interface for interactive demo
    std::vector<float> predict_digit(const std::vector<float>& input_data) {
        if (input_data.size() != 28 * 28) {
            throw std::runtime_error("input data must be a flat vector of 28x28 pixels");
        }
        // convert flat vector to Tensor3D (assuming MNIST 28x28 input)
        Tensor3D input(1, 28, 28);
        for (int i = 0; i < 28; i++) {
            for (int j = 0; j < 28; j++) {
                input.data[0][i][j] = input_data[i * 28 + j];
            }
        }

        // get prediction
        Tensor3D output = feedforward(input);
        
        // convert output to vector
        std::vector<float> result(10);
        for (int i = 0; i < 10; i++) {
            result[i] = output.data[0][i][0];
        }
        return result;
    }
};

// todo:
// implement other modes than same
// implement different strides in convlayer
// batch normalisation
// dropout


std::pair<std::vector<std::vector<Tensor3D>>, std::vector<std::vector<Tensor3D>>> load_mnist_data(std::string path_to_data) {
    // create training set from binary image data files
    
    std::vector<std::vector<Tensor3D>> training_set;
    training_set.reserve(9000);
    std::vector<std::vector<Tensor3D>> eval_set;
    eval_set.reserve(1000);

    for (int i = 0; i < 10; ++i) {
        std::string file_path = path_to_data + "/data" + std::to_string(i) + ".dat";
        std::vector<unsigned char> full_digit_data = read_file(file_path);

        for (int j = 0; j < 784000; j += 28 * 28) {  // todo make more general with training ratio
            // create the input Tensor3D with shape (1, 28, 28)
            Tensor3D input_data(1, 28, 28);

            // fill the tensor with normalised pixel values
            for (int row = 0; row < 28; ++row) {
                for (int col = 0; col < 28; ++col) {
                    float normalised_pixel = static_cast<float>(full_digit_data[j + row * 28 + col]) / 255.0f;
                    input_data.data[0][row][col] = normalised_pixel;
                }
            }

            // create the label Tensor3D
            Tensor3D label_data(10, 1);
            std::vector<std::vector<float>> data;

            // construct the label Tensor3D with 1.0 in the position of the digit and zeros elsewhere
            for (size_t l = 0; l < i; ++l) {
                data.push_back({0.0});
            }
            data.push_back({1.0});
            for (size_t l = 0; l + i + 1 < 10; ++l) {
                data.push_back({0.0});
            }

            label_data.data[0] = data;

            // push both image and label into appropriate set
            if (j < 705600) {
                training_set.push_back({input_data, label_data});
            } else {
                eval_set.push_back({input_data, label_data});
            }
        }
    }
    return {training_set, eval_set};
}

int main() {
    
    auto [training_set, eval_set] = load_mnist_data("mnist-data");

    // network architecture setup   
    NeuralNetwork nn;
    nn.add_conv_layer(16, 3);
    nn.add_pool_layer();
    nn.add_conv_layer(32, 3);
    nn.add_pool_layer();
    nn.add_dense_layer(100);
    nn.add_dense_layer(10, "softmax");
    nn.set_loss(std::make_unique<CrossEntropyLoss>());
    nn.set_optimiser(std::make_unique<AdamWOptimiser>());

    // training hyperparameters
    const int num_epochs = 20;
    const int batch_size = 100;

    nn.train(training_set, eval_set, num_epochs, batch_size);
    std::cout << "Training complete\n" << std::endl;

    return 0;
}
