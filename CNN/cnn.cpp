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

    static Tensor3D Conv(const Tensor3D &input, const Tensor3D &kernel) {
        // check dimensions
        if (input.depth != kernel.depth) {
            throw std::runtime_error("input and kernel must have same depth for convolution");
        }

        int pad_amount = (kernel.height - 1) / 2;
        Tensor3D output(1, input.height - pad_amount * 2, input.width - pad_amount * 2);

        // for each position in the output
        for (int y = pad_amount; y < input.height - pad_amount; ++y) {
            for (int x = pad_amount; x < input.width - pad_amount; ++x) {
                double sum = 0.0;
                
                // sum over all channels and kernel positions
                for (int d = 0; d < input.depth; ++d) {
                    for (int ky = 0; ky < kernel.height; ++ky) {
                        for (int kx = 0; kx < kernel.width; ++kx) {
                            sum += input.data[d][y + ky - pad_amount][x + kx - pad_amount] * 
                                  kernel.data[d][ky][kx];
                        }
                    }
                }
                output.data[0][y - pad_amount][x - pad_amount] = sum;
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
};

struct BackwardReturn {
    Tensor3D input_error;
    std::vector<Tensor3D> weight_grads;
    std::vector<Tensor3D> bias_grads;
};

class Layer {
   public:
    virtual ~Layer() = default;

    // pure virtual class - requires implementation in derived objects.
    virtual Tensor3D forward(const Tensor3D &input) = 0;
};

class DenseLayer : public Layer {
   public:
    Tensor3D weights;
    Tensor3D bias;
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
        : weights(output_size, input_size), bias(output_size, 1), activation_function(activation_function) {
        if (activation_function == "sigmoid") {
            weights.xavier_initialise();
        } else if (activation_function == "relu") {
            weights.he_initialise();
        } else {
            weights.uniform_initialise();
        }
    }

    // returns post-activation - stores input and preactivation for backprop
    Tensor3D forward(const Tensor3D &input) override {
        if (weights.width != input.height) {
            throw std::runtime_error(
                "Input vector dimension is not appropriate for the weight matrix dimension");
        }
        
        this->input = input;
        z = weights * input + bias;
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
        Tensor3D d_input = weights.transpose() * d_z;
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
        Tensor3D tmp_z(weights.size(), input.height, input.width);  // actual z attribute is not initialised to correct dimensions so use this and then copy to z
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

    BackwardReturn backward(const Tensor3D &d_output) {
        // first compute d_z by applying relu derivative
        Tensor3D d_z = d_output.hadamard(z.apply(relu_derivative));

        // initialise gradient tensors
        std::vector<Tensor3D> d_weights;
        std::vector<Tensor3D> d_bias;
        d_weights.reserve(out_channels);
        d_bias.reserve(out_channels);

        // pad d_z for later computation of d_input
        Tensor3D padded_error = Tensor3D::pad(d_z);

        // compute gradients for each output channel
        for (int out_c = 0; out_c < out_channels; out_c++) {
            // extract this output channel's gradient
            Tensor3D channel_d_z(1, d_z.height, d_z.width);
            channel_d_z.data[0] = d_z.data[out_c];
            
            // initialise kernel gradient for this output channel
            Tensor3D d_kernel(channels_in, kernel_size, kernel_size);
            
            // compute gradient for each input channel's kernel
            for (int in_c = 0; in_c < channels_in; in_c++) {
                // extract this input channel
                Tensor3D channel_input(1, input.height, input.width);
                channel_input.data[0] = input.data[in_c];
                
                // compute gradient for this input-output channel pair
                Tensor3D channel_grad = Tensor3D::Conv(channel_input, channel_d_z);
                d_kernel.data[in_c] = channel_grad.data[0];
            }
            d_weights.push_back(d_kernel);
            
            // compute d_bias for this output channel (sum of d_z)
            double channel_d_bias = 0.0;
            for (size_t h = 0; h < d_z.height; h++) {
                for (size_t w = 0; w < d_z.width; w++) {
                    channel_d_bias += d_z.data[out_c][h][w];
                }
            }
            Tensor3D bias_grad(1, 1, 1);
            bias_grad.data[0][0][0] = channel_d_bias;
            d_bias.push_back(bias_grad);
        }

        // compute d_input

        Tensor3D d_input(input.depth, input.height, input.width);
        Tensor3D relevant_d_z_slice(1, d_z.height, d_z.width);
        Tensor3D relevant_kernel_slice(1, weights[0].height, weights[0].width);

        for (int in_ch = 0; in_ch < weights[0].depth; in_ch++) {
            // sum contributions from all output channels
            for (int k = 0; k < weights.size(); k++) {
                
                // extract relevant delta channel to relevant_d_z_slice
                for (int y = 0; y < d_z.height; y++) {
                    for (int x = 0; x < d_z.width; x++) {
                        relevant_d_z_slice.data[0][y][x] = d_z.data[k][y][x];
                    }
                }

                // extract and rotate relevant kernel slice to relevant_kernel_slice
                for (int y = 0; y < weights[0].height; y++) {
                    for (int x = 0; x < weights[0].width; x++) {
                        // rotate 180 degrees by inverting indices
                        relevant_kernel_slice.data[0][y][x] = 
                            weights[k].data[in_ch][weights[0].height-1-y][weights[0].width-1-x];
                    }
                }
                
                // Conv function will handle 'same' padding internally
                Tensor3D d_input_contribution = Tensor3D::Conv(relevant_d_z_slice, relevant_kernel_slice);

                // add the contribution of this kernel to this input channel's d_input layer
                for (int y = 0; y < d_input_contribution.height; y++) {
                    for (int x = 0; x < d_input_contribution.width; x++) {
                        d_input.data[in_ch][y][x] += d_input_contribution.data[0][y][x];
                    }
                }
            }
        }

        return {d_input, d_weights, d_bias};
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

    Tensor3D forward(const Tensor3D &input) override {
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
                    layers.push_back(std::make_unique<ConvolutionLayer>(spec.in_channels, spec.out_channels, spec.kernel_size,
                                                                        spec.mode));
                    
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
};

// todo:
// implement convolutionlayer::backward something like
    // d_kernel = Tensor3D::Conv(input, d_z);        // Gradient w.r.t kernel
    // d_bias = d_z.sum;               // Gradient w.r.t bias
    // Tensor3D rotated_kernel = kernel.rotate_180(); // Rotate kernel 180 degrees
    // Tensor3D d_input = Tensor3D::Conv(d_z.pad(kernel.height / 2), rotated_kernel); // Gradient w.r.t input

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

    // test feedforward
    Tensor3D input(input_channels, input_height, input_width);
    Tensor3D output = nn.feedforward(input);
    std::cout << "done" << std::endl;
}
