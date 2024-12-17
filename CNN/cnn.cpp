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

    // for compatability with old Tensor3D code
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

    Tensor3D operator+(const double &other) const {
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

    // softmax across width dimension for back-compatibility with old Tensor3D class
    Tensor3D softmax() const {
        Tensor3D result(depth, height, width);

        for (size_t d = 0; d < depth; d++) {
            for (size_t w = 0; w < width; w++) {  // equivalent to "columns" in Tensor3D
                // find max
                double max_val = -std::numeric_limits<double>::infinity();
                for (size_t h = 0; h < height; h++) {  // equivalent to "rows" in Tensor3D
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
                double sum = 0.0;

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
};

struct BackwardReturn {
    Tensor3D input_error;
    std::vector<Tensor3D> weight_grads;
    std::vector<Tensor3D> bias_grads;
};

// ---------------------------------- LAYER CLASSES -------------------------------------------
class Layer {
   public:
    virtual ~Layer() = default;

    // pure virtual class - requires implementation in derived objects.
    virtual Tensor3D forward(const Tensor3D &input) = 0;
    virtual BackwardReturn backward(const Tensor3D &d_output) = 0;
};

class DenseLayer : public Layer {
   public:
    std::vector<Tensor3D> weights;  // only a list for compatibility with other layer types - one element
    std::vector<Tensor3D> bias;     // only a list for compatibility with other layer types - one element
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
        bias.emplace_back(output_size, 1);

        if (activation_function == "sigmoid") {
            weights[0].xavier_initialise();
        } else if (activation_function == "relu") {
            weights[0].he_initialise();
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
        z = weights[0] * input + bias[0];
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

    BackwardReturn backward(const Tensor3D &d_output) {
        Tensor3D d_activation;
        if (activation_function == "sigmoid") {
            d_activation = z.apply(sigmoid_derivative);
        } else if (activation_function == "relu") {
            d_activation = z.apply(relu_derivative);
        } else if (activation_function == "softmax" or activation_function == "none") {
            d_activation = d_output;  // derivative already included in loss function
        } else {
            throw std::runtime_error("Unsupported activation function");
        }

        Tensor3D d_z = d_output.hadamard(d_activation);
        Tensor3D d_input = weights[0].transpose() * d_z;
        std::vector<Tensor3D> d_weights = {d_z * input.transpose()};
        std::vector<Tensor3D> d_bias = {d_z};

        return {d_input, d_weights, d_bias};
    }
};

class ConvolutionLayer : public Layer {
   public:
    std::vector<Tensor3D> weights;
    std::vector<double> bias;
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

        // creates a list of bias values - one for each output channel
        bias.reserve(out_channels);
        for (int i = 0; i < out_channels; i++) {
            bias.push_back(0.0);
        }
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
            Tensor3D padded_input = Tensor3D::pad(input);

            for (int feature_map_index = 0; feature_map_index < weights.size(); ++feature_map_index) {
                Tensor3D preactivation = Tensor3D::Conv(padded_input, weights[feature_map_index]) + bias[feature_map_index];

                // store the preactivation for this feature map
                tmp_z.data[feature_map_index] = preactivation.data[0];

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
        Tensor3D padded_d_output = Tensor3D::pad(d_output, pad_amount);

        for (int in_ch = 0; in_ch < input.depth; in_ch++) {
            // sum contributions from all output channels
            for (int k = 0; k < weights.size(); k++) {
                // extract relevant delta channel and kernel slice
                Tensor3D relevant_d_z_slice = padded_d_output.get_depth_slice(k);
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
        for (int k = 0; k < weights.size(); k++) {
            Tensor3D d_weight(input.depth, kernel_size, kernel_size);

            // pad input for full convolution
            Tensor3D padded_input = Tensor3D::pad(input, pad_amount);

            for (int in_ch = 0; in_ch < input.depth; in_ch++) {
                // extract relevant input and gradient channels
                Tensor3D input_channel = padded_input.get_depth_slice(in_ch);
                Tensor3D d_output_channel = d_output.get_depth_slice(k);

                // compute gradient for this input-output channel pair
                Tensor3D channel_grad = Tensor3D::Conv(input_channel, d_output_channel);
                d_weight.set_depth_slice(in_ch, channel_grad);
            }
            d_weights.push_back(d_weight);

            // compute bias gradient (sum of d_output)
            double d_bias_val = 0.0;
            for (int y = 0; y < d_output.height; y++) {
                for (int x = 0; x < d_output.width; x++) {
                    d_bias_val += d_output.data[k][y][x];
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
                        double max_val = -std::numeric_limits<double>::infinity();
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

// ---------------------------------- OPTIMISERS -------------------------------------------

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
    // std::unique_ptr<Optimiser> optimiser;
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
    // void set_optimiser(std::unique_ptr<Optimiser> new_optimiser) { optimiser = std::move(new_optimiser); }

    /**
     * @brief Sets the loss function for the neural network.
     * @param new_loss A unique pointer to the new Loss object.
     */
    void set_loss(std::unique_ptr<Loss> new_loss) { loss = std::move(new_loss); }

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
};

// todo:
// implement other modes than same
// implement different strides in convlayer
// change to float values in Tensor3D

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
    nn.set_loss(std::make_unique<MSELoss>());

    // test feedforward
    Tensor3D input(input_channels, input_height, input_width);
    input.uniform_initialise();
    // Tensor3D output = nn.feedforward(input);
    Tensor3D target(1, 1, 1);
    target.data[0][0][0] = 1;

    auto gradients = nn.calculate_gradients(input, target);

    std::cout << "done" << std::endl;
}
