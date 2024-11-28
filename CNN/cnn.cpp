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

// matrix class for handling matrix operations
class Matrix {
   public:
    std::vector<std::vector<double>> data;  // 2D vector to store matrix data
    size_t rows, cols;                      // dimensions of the matrix

    // default constructor
    Matrix() : rows(0), cols(0) {}

    /**
     * @brief Constructs a Matrix object with the specified dimensions.
     * @param rows Number of rows in the matrix.
     * @param cols Number of columns in the matrix.
     */
    Matrix(size_t rows, size_t cols) : rows(rows), cols(cols) {
        data.resize(rows,
                    std::vector<double>(cols, 0.0));  // resise the data vector to have 'rows' elements with a vector as the value
    }

    /**
     * @brief Initializes the matrix with random values between -1 and 1.
     */
    void uniform_initialise() {
        std::random_device rd;                            // obtain a random number from hardware
        std::mt19937 gen(rd());                           // seed the generator
        std::uniform_real_distribution<> dis(-1.0, 1.0);  // define the range
        for (auto &row : data) {
            for (auto &elem : row) {
                elem = dis(gen);  // generate random number
            }
        }
    }

    /**
     * @brief Initializes the matrix with zeros.
     */
    void zero_initialise() {
        for (auto &row : data) {
            for (auto &elem : row) {
                elem = 0;  // generate random number
            }
        }
    }

    /**
     * @brief Initializes the matrix using Xavier initialization method.
     * Suitable for sigmoid activation functions.
     */
    void xavier_initialize() {
        std::random_device rd;
        std::mt19937 gen(rd());
        double limit = sqrt(6.0 / (rows + cols));
        std::uniform_real_distribution<> dis(-limit, limit);
        for (auto &row : data) {
            for (auto &elem : row) {
                elem = dis(gen);
            }
        }
    }

    /**
     * @brief Initializes the matrix using He initialization method.
     * Suitable for ReLU activation functions.
     */
    void he_initialise() {
        std::random_device rd;
        std::mt19937 gen(rd());
        double std_dev = sqrt(2.0 / cols);
        std::normal_distribution<> dis(0, std_dev);
        for (auto &row : data) {
            for (auto &elem : row) {
                elem = dis(gen);
            }
        }
    }

    /**
     * @brief Overloads the multiplication operator for matrix multiplication.
     * @param other The matrix to multiply with.
     * @return The resulting matrix after multiplication.
     */
    Matrix operator*(const Matrix &other) const {
        if (cols != other.rows) {
            std::cerr << "Attempted to multiply matrices of incompatible dimensions: "
                      << "(" << rows << "x" << cols << ") * (" << other.rows << "x" << other.cols << ")" << std::endl;
            throw std::invalid_argument("Matrix dimensions don't match for multiplication");
        }
        Matrix result(rows, other.cols);
        // weird loop order (k before j) makes more cache friendly
        for (size_t i = 0; i < rows; i++) {
            for (size_t k = 0; k < cols; k++) {
                for (size_t j = 0; j < other.cols; j++) {
                    result.data[i][j] += data[i][k] * other.data[k][j];
                }
            }
        }
        return result;
    }

    /**
     * @brief Overloads the addition operator for element-wise matrix addition.
     * @param other The matrix to add.
     * @return The resulting matrix after addition.
     */
    Matrix operator+(const Matrix &other) const {
        if (rows != other.rows or cols != other.cols) {
            throw std::invalid_argument("Matrix dimensions don't match for addition");
        }
        Matrix result(rows, cols);
        for (size_t i = 0; i < rows; i++) {
            for (size_t j = 0; j < cols; j++) {
                result.data[i][j] = data[i][j] + other.data[i][j];
            }
        }
        return result;
    }

    /**
     * @brief Overloads the subtraction operator for element-wise matrix subtraction.
     * @param other The matrix to subtract.
     * @return The resulting matrix after subtraction.
     */
    Matrix operator-(const Matrix &other) const {
        if (rows != other.rows or cols != other.cols) {
            throw std::invalid_argument("Matrix dimensions don't match for subtraction");
        }
        Matrix result(rows, cols);
        for (size_t i = 0; i < rows; i++) {
            for (size_t j = 0; j < cols; j++) {
                result.data[i][j] = data[i][j] - other.data[i][j];
            }
        }
        return result;
    }

    /**
     * @brief Overloads the multiplication operator for scalar multiplication.
     * @param scalar The scalar value to multiply with.
     * @return The resulting matrix after scalar multiplication.
     */
    Matrix operator*(double scalar) const {
        Matrix result(rows, cols);
        for (size_t i = 0; i < rows; i++) {
            for (size_t j = 0; j < cols; j++) {
                result.data[i][j] = data[i][j] * scalar;
            }
        }
        return result;
    }

    /**
     * @brief Computes the transpose of the matrix.
     * @return The transposed matrix.
     */
    Matrix transpose() const {
        Matrix result(cols, rows);
        for (size_t i = 0; i < rows; i++) {
            for (size_t j = 0; j < cols; j++) {
                result.data[j][i] = data[i][j];
            }
        }
        return result;
    }

    /**
     * @brief Computes the Hadamard product (element-wise multiplication) of two matrices.
     * @param other The matrix to perform Hadamard product with.
     * @return The resulting matrix after Hadamard product.
     */
    Matrix hadamard(const Matrix &other) const {
        if (rows != other.rows or cols != other.cols) {
            throw std::invalid_argument("Matrix dimensions don't match for Hadamard product");
        }
        Matrix result(rows, cols);
        for (size_t i = 0; i < rows; i++) {
            for (size_t j = 0; j < cols; j++) {
                result.data[i][j] = data[i][j] * other.data[i][j];
            }
        }
        return result;
    }

    /**
     * @brief Applies a function to every element in the matrix.
     * @param func A function pointer to apply to each element.
     * @return The resulting matrix after applying the function.
     */
    Matrix apply(double (*func)(double)) const {
        Matrix result(rows, cols);
        for (size_t i = 0; i < rows; i++) {
            for (size_t j = 0; j < cols; j++) {
                result.data[i][j] = func(data[i][j]);
            }
        }
        return result;
    }

    /**
     * @brief Applies a function to every element in the matrix.
     * @tparam Func The type of the callable object.
     * @param func A callable object to apply to each element.
     * @return The resulting matrix after applying the function.
     */
    template <typename Func>
    Matrix apply(Func func) const {
        Matrix result(rows, cols);
        for (size_t i = 0; i < rows; i++) {
            for (size_t j = 0; j < cols; j++) {
                result.data[i][j] = func(data[i][j]);
            }
        }
        return result;
    }

    /**
     * @brief Applies the softmax function to the matrix.
     * @return The resulting matrix after applying softmax.
     */
    Matrix softmax() const {
        Matrix result(rows, cols);

        for (size_t j = 0; j < cols; ++j) {
            double max_val = -std::numeric_limits<double>::infinity();
            for (size_t i = 0; i < rows; ++i) {
                max_val = std::max(max_val, data[i][j]);
            }

            double sum = 0.0;
            for (size_t i = 0; i < rows; ++i) {
                result.data[i][j] = std::exp(data[i][j] - max_val);
                sum += result.data[i][j];
            }

            for (size_t i = 0; i < rows; ++i) {
                result.data[i][j] /= sum;
            }
        }

        return result;
    }
};

// layer class representing a single dense (fully connected) layer in the neural network
class DenseLayer {
   public:
    Matrix weights;
    Matrix bias;
    std::string activation_function;

    /**
     * @brief Constructs a DenseLayer object with specified input size, output size, and activation function.
     * @param input_size The number of input neurons.
     * @param output_size The number of output neurons.
     * @param activation_function The activation function to use (default: "sigmoid").
     */
    DenseLayer(size_t input_size, size_t output_size, std::string activation_function = "sigmoid")
        : weights(output_size, input_size), bias(output_size, 1), activation_function(activation_function) {
        if (activation_function == "sigmoid") {
            weights.xavier_initialize();
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
    Matrix feedforward(const Matrix &input) {
        Matrix z = weights * input + bias;
        Matrix output(z.rows, z.cols);
        if (activation_function == "sigmoid") {
            output = z.apply(sigmoid);
        } else if (activation_function == "relu") {
            output = z.apply(relu);
        } else if (activation_function == "softmax") {
            output = z.softmax();
        } else {
            throw std::runtime_error("no activation function found for layer");
        }

        return output;
    }

    /**
     * @brief Performs feedforward operation for this layer and returns both output and pre-activation.
     * @param input The input matrix.
     * @return A vector containing the output matrix and pre-activation matrix.
     */
    std::vector<Matrix> feedforward_backprop(const Matrix &input) const {
        Matrix z = weights * input + bias;
        Matrix output(z.rows, z.cols);
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

class Loss {
   public:
    virtual ~Loss() = default;

    // Compute the loss value
    virtual double compute(const Matrix &predicted, const Matrix &target) const = 0;

    // Compute the derivative of the loss with respect to the predicted values
    virtual Matrix derivative(const Matrix &predicted, const Matrix &target) const = 0;
};

class CrossEntropyLoss : public Loss {
   public:
    double compute(const Matrix &predicted, const Matrix &target) const override {
        double loss = 0.0;
        for (size_t i = 0; i < predicted.rows; ++i) {
            for (size_t j = 0; j < predicted.cols; ++j) {
                // Add small epsilon to avoid log(0)
                loss -= target.data[i][j] * std::log(predicted.data[i][j] + 1e-10);
            }
        }
        return loss / predicted.cols;  // Average over batch
    }

    Matrix derivative(const Matrix &predicted, const Matrix &target) const override {
        // For cross entropy with softmax, the derivative simplifies to (predicted - target)
        return predicted - target;
    }
};

class MSELoss : public Loss {
   public:
    double compute(const Matrix &predicted, const Matrix &target) const override {
        double loss = 0.0;
        for (size_t i = 0; i < predicted.rows; ++i) {
            for (size_t j = 0; j < predicted.cols; ++j) {
                double diff = predicted.data[i][j] - target.data[i][j];
                loss += diff * diff;
            }
        }
        return loss / (2.0 * predicted.cols);  // Average over batch and divide by 2
    }

    Matrix derivative(const Matrix &predicted, const Matrix &target) const override {
        return (predicted - target) * (1.0 / predicted.cols);
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
                                           const std::vector<std::vector<Matrix>> &gradients) = 0;

    /**
     * @brief Virtual destructor for the Optimiser class.
     */
    virtual ~Optimiser() = default;

    struct GradientResult {
        std::vector<std::vector<Matrix>> gradients;  // list of layers, each layer has a list of weight and bias gradient matrices
        Matrix input_layer_gradient;  // gradient of the input layer - for more general use as parts of bigger architectures
        Matrix output;                // output of the network
    };

    /**
     * @brief Calculates gradients for a single example.
     * @param layers The layers of the neural network.
     * @param input The input matrix.
     * @param target The target matrix.
     * @return A GradientResult struct containing gradients and the output of the network.
     */
    virtual GradientResult calculate_gradient(const std::vector<DenseLayer> &layers, const Matrix &input, const Matrix &target,
                                              const Loss &loss) {
        // forward pass
        std::vector<Matrix> activations = {input};
        std::vector<Matrix> preactivations = {input};

        for (const auto &layer : layers) {
            auto results = layer.feedforward_backprop(activations.back());
            activations.push_back(results[0]);
            preactivations.push_back(results[1]);
        }

        // backward pass
        int num_layers = layers.size();
        std::vector<Matrix> deltas;
        deltas.reserve(num_layers);

        // output layer error (δ^L = ∇_a C ⊙ σ'(z^L))
        Matrix output_delta = loss.derivative(activations.back(), target);
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
            Matrix delta = (layers[l + 1].weights.transpose() * deltas.back());
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
        std::vector<std::vector<Matrix>> gradients;
        for (int l = 0; l < num_layers; ++l) {
            Matrix weight_gradient = deltas[l] * activations[l].transpose();
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
    std::vector<std::vector<Matrix>> average_gradients(const std::vector<std::vector<std::vector<Matrix>>> &batch_gradients) {
        std::vector<std::vector<Matrix>> avg_gradients;
        size_t num_layers = batch_gradients[0].size();
        size_t batch_size = batch_gradients.size();

        for (size_t l = 0; l < num_layers; ++l) {
            Matrix avg_weight_grad(batch_gradients[0][l][0].rows, batch_gradients[0][l][0].cols);
            Matrix avg_bias_grad(batch_gradients[0][l][1].rows, batch_gradients[0][l][1].cols);

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
    std::vector<std::vector<Matrix>> velocity;

   public:
    /**
     * @brief Constructs an SGDOptimiser object with the specified learning rate.
     * @param lr The learning rate (default: 0.1).
     */
    SGDOptimiser(double lr = 0.1) : learning_rate(lr) {}

    /**
     * @brief Initializes the velocity vectors for SGD optimization.
     * @param layers The layers of the neural network.
     */
    void initialize_velocity(const std::vector<DenseLayer> &layers) {
        velocity.clear();
        for (const auto &layer : layers) {
            velocity.push_back({Matrix(layer.weights.rows, layer.weights.cols), Matrix(layer.bias.rows, layer.bias.cols)});
        }
    }

    /**
     * @brief Computes and applies updates using Stochastic Gradient Descent.
     * @param layers The layers of the neural network to update.
     * @param gradients The gradients used for updating the layers.
     */
    void compute_and_apply_updates(std::vector<DenseLayer> &layers, const std::vector<std::vector<Matrix>> &gradients) override {
        if (velocity.empty()) {
            initialize_velocity(layers);
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
    std::vector<std::vector<Matrix>> velocity;

   public:
    /**
     * @brief Constructs an SGDMomentumOptimiser object with the specified learning rate and momentum.
     * @param lr The learning rate (default: 0.1).
     * @param mom The momentum coefficient (default: 0.9).
     */
    SGDMomentumOptimiser(double lr = 0.1, double mom = 0.9) : learning_rate(lr), momentum(mom) {}

    /**
     * @brief Initializes the velocity vectors for SGD with Momentum optimization.
     * @param layers The layers of the neural network.
     */
    void initialize_velocity(const std::vector<DenseLayer> &layers) {
        velocity.clear();
        for (const auto &layer : layers) {
            velocity.push_back({Matrix(layer.weights.rows, layer.weights.cols), Matrix(layer.bias.rows, layer.bias.cols)});
        }
    }

    /**
     * @brief Computes and applies updates using Stochastic Gradient Descent with Momentum.
     * @param layers The layers of the neural network to update.
     * @param gradients The gradients used for updating the layers.
     */
    void compute_and_apply_updates(std::vector<DenseLayer> &layers, const std::vector<std::vector<Matrix>> &gradients) override {
        if (velocity.empty()) {
            initialize_velocity(layers);
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
    std::vector<std::vector<Matrix>> velocity;

   public:
    /**
     * @brief Constructs a NesterovMomentumOptimiser object with the specified learning rate and momentum.
     * @param lr The learning rate (default: 0.1).
     * @param mom The momentum coefficient (default: 0.9).
     */
    NesterovMomentumOptimiser(double lr = 0.1, double mom = 0.9) : learning_rate(lr), momentum(mom) {}

    /**
     * @brief Initializes the velocity vectors for Nesterov Momentum optimization.
     * @param layers The layers of the neural network.
     */
    void initialize_velocity(const std::vector<DenseLayer> &layers) {
        velocity.clear();
        for (const auto &layer : layers) {
            velocity.push_back({Matrix(layer.weights.rows, layer.weights.cols), Matrix(layer.bias.rows, layer.bias.cols)});
        }
    }

    /**
     * @brief Calculates gradients using Nesterov momentum.
     * @param layers The layers of the neural network.
     * @param input The input matrix.
     * @param target The target matrix.
     * @return A GradientResult struct containing gradients and the output of the network.
     */
    GradientResult calculate_gradient(const std::vector<DenseLayer> &layers, const Matrix &input, const Matrix &target,
                                      const Loss &loss) override {
        // get lookahead position
        std::vector<DenseLayer> tmp_layers = layers;
        if (velocity.empty()) {
            initialize_velocity(tmp_layers);
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
    void compute_and_apply_updates(std::vector<DenseLayer> &layers, const std::vector<std::vector<Matrix>> &gradients) override {
        if (velocity.empty()) {
            initialize_velocity(layers);
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
    int t;                               // timestep
    std::vector<std::vector<Matrix>> m;  // first moment
    std::vector<std::vector<Matrix>> v;  // second moment

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
     * @brief Initializes the first and second moment vectors for Adam optimization.
     * @param layers The layers of the neural network.
     */
    void initialize_moments(const std::vector<DenseLayer> &layers) {
        m.clear();
        v.clear();
        m.reserve(layers.size());
        v.reserve(layers.size());
        for (const auto &layer : layers) {
            m.push_back({Matrix(layer.weights.rows, layer.weights.cols), Matrix(layer.bias.rows, layer.bias.cols)});
            v.push_back({Matrix(layer.weights.rows, layer.weights.cols), Matrix(layer.bias.rows, layer.bias.cols)});
        }
    }

    /**
     * @brief Computes and applies updates using the Adam optimization algorithm.
     * @param layers The layers of the neural network to update.
     * @param gradients The gradients used for updating the layers.
     */
    void compute_and_apply_updates(std::vector<DenseLayer> &layers, const std::vector<std::vector<Matrix>> &gradients) override {
        if (m.empty() or v.empty()) {
            initialize_moments(layers);
        }

        t++;  // increment timestep

        for (size_t l = 0; l < layers.size(); ++l) {
            for (int i = 0; i < 2; ++i) {  // 0 for weights, 1 for biases
                // update biased first moment estimate
                m[l][i] = m[l][i] * beta1 + gradients[l][i] * (1.0 - beta1);

                // update biased second raw moment estimate
                v[l][i] = v[l][i] * beta2 + gradients[l][i].hadamard(gradients[l][i]) * (1.0 - beta2);

                // compute bias-corrected first moment estimate
                Matrix m_hat = m[l][i] * (1.0 / (1.0 - std::pow(beta1, t)));

                // compute bias-corrected second raw moment estimate
                Matrix v_hat = v[l][i] * (1.0 / (1.0 - std::pow(beta2, t)));

                // compute the update
                Matrix update = m_hat.hadamard(v_hat.apply([this](double x) { return 1.0 / (std::sqrt(x) + epsilon); }));

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
    int t;                               // timestep
    std::vector<std::vector<Matrix>> m;  // first moment
    std::vector<std::vector<Matrix>> v;  // second moment

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
     * @brief Initializes the first and second moment vectors for AdamW optimization.
     * @param layers The layers of the neural network.
     */
    void initialize_moments(const std::vector<DenseLayer> &layers) {
        m.clear();
        v.clear();
        m.reserve(layers.size());
        v.reserve(layers.size());
        for (const auto &layer : layers) {
            m.push_back({Matrix(layer.weights.rows, layer.weights.cols), Matrix(layer.bias.rows, layer.bias.cols)});
            v.push_back({Matrix(layer.weights.rows, layer.weights.cols), Matrix(layer.bias.rows, layer.bias.cols)});
        }
    }

    /**
     * @brief Computes and applies updates using the AdamW optimization algorithm.
     * @param layers The layers of the neural network to update.
     * @param gradients The gradients used for updating the layers.
     */
    void compute_and_apply_updates(std::vector<DenseLayer> &layers, const std::vector<std::vector<Matrix>> &gradients) override {
        if (m.empty() || v.empty()) {
            initialize_moments(layers);
        }

        t++;  // increment timestep

        for (size_t l = 0; l < layers.size(); ++l) {
            for (int i = 0; i < 2; ++i) {  // 0 for weights, 1 for biases
                // update biased first moment estimate
                m[l][i] = m[l][i] * beta1 + gradients[l][i] * (1.0 - beta1);

                // update biased second raw moment estimate
                v[l][i] = v[l][i] * beta2 + gradients[l][i].hadamard(gradients[l][i]) * (1.0 - beta2);

                // compute bias-corrected first moment estimate
                Matrix m_hat = m[l][i] * (1.0 / (1.0 - std::pow(beta1, t)));

                // compute bias-corrected second raw moment estimate
                Matrix v_hat = v[l][i] * (1.0 / (1.0 - std::pow(beta2, t)));

                // compute the Adam update
                Matrix update = m_hat.hadamard(v_hat.apply([this](double x) { return 1.0 / (std::sqrt(x) + epsilon); }));

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
   public:
    std::vector<DenseLayer> layers;
    std::unique_ptr<Optimiser> optimiser;
    std::unique_ptr<Loss> loss;
    // have to use a pointer otherwise class cannot be constructed (mutex is not moveable/copyable etc.)
    std::unique_ptr<std::mutex> layers_mutex;
    std::unique_ptr<TrainingHistoryLogger> history_logger;

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

    /**
     * @brief Constructs a NeuralNetwork object with the specified topology and activation functions.
     * @param topology A vector specifying the number of neurons in each layer.
     * @param activation_functions A vector specifying the activation function for each layer (optional).
     */
    NeuralNetwork(const std::vector<int> &topology, const std::vector<std::string> activation_functions = {})
        : layers_mutex(std::make_unique<std::mutex>()) {
        if (topology.empty()) {
            throw std::invalid_argument("Topology cannot be empty");
        }
        for (int size : topology) {
            if (size <= 0) {
                throw std::invalid_argument("DenseLayer size must be positive");
            }
        }
        if ((activation_functions.size() + 1 != topology.size()) and (activation_functions.size() != 0)) {
            throw std::invalid_argument(
                "the size of activations_functions vector must be the same size as no. layers (ex. input)");
        } else if (activation_functions.size() == 0) {
            for (size_t i = 1; i < topology.size(); i++) {
                // do not pass in specific activation function - use the default specified in the layer constructor
                layers.emplace_back(topology[i - 1], topology[i]);
            }
        } else {
            for (size_t i = 1; i < topology.size(); i++) {
                layers.emplace_back(topology[i - 1], topology[i], activation_functions[i - 1]);
            }
        }
    }

    /**
     * @brief Performs feedforward operation through all layers of the network.
     * @param input The input matrix.
     * @return The output matrix after passing through all layers.
     */
    Matrix feedforward(const Matrix &input) {
        Matrix current = input;
        for (auto &layer : layers) {
            current = layer.feedforward(current);
        }
        return current;
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

    void enable_history_logging(const std::string &filename) {
        history_logger = std::make_unique<TrainingHistoryLogger>(filename);
    }

    // train the neural network using optimiser set
    void train_mt_optimiser(const std::vector<std::vector<Matrix>> &training_data,
                            const std::vector<std::vector<Matrix>> &eval_data, int epochs, int batch_size) {
        unsigned int num_threads = std::thread::hardware_concurrency();
        std::vector<std::thread> threads(num_threads);
        std::vector<std::vector<std::vector<std::vector<Matrix>>>> thread_gradients(num_threads);
        std::mutex gradients_mutex;
        int counter = 0;

        // used for tracking loss of each batch
        std::vector<Matrix> batch_outputs;
        std::vector<Matrix> batch_targets;

        // create a vector of indices
        std::vector<size_t> indices(training_data.size());
        std::iota(indices.begin(), indices.end(), 0);

        // random number generator
        std::random_device rd;
        std::mt19937 generator(rd());

        // iterate through epochs
        for (int epoch = 0; epoch < epochs; epoch++) {
            std::shuffle(indices.begin(), indices.end(), generator);

            std::cout << "Epoch " << epoch + 1 << "\n";

            // iterate through batches
            for (size_t batch_start = 0; batch_start < training_data.size(); batch_start += batch_size) {
                // Clear previous batch data
                batch_outputs.clear();
                batch_targets.clear();
                size_t current_batch_size = std::min(batch_size, static_cast<int>(training_data.size() - batch_start));
                batch_outputs.reserve(current_batch_size);
                batch_targets.reserve(current_batch_size);

                // clear previous gradients
                for (auto &grad : thread_gradients) {
                    grad.clear();
                }

                // spawn threads - splits the batch up between the number of threads
                for (unsigned int t = 0; t < num_threads; t++) {
                    threads[t] = std::thread([&, t, current_batch_size, batch_start]() {
                        size_t start = t * current_batch_size / num_threads;
                        size_t end = (t + 1) * current_batch_size / num_threads;

                        std::vector<std::vector<std::vector<Matrix>>> local_gradients;
                        local_gradients.reserve(end - start);

                        // Local vectors to collect outputs and targets
                        std::vector<Matrix> local_outputs;
                        std::vector<Matrix> local_targets;
                        local_outputs.reserve(end - start);
                        local_targets.reserve(end - start);

                        for (size_t i = start; i < end; i++) {
                            size_t index = indices[batch_start + i];
                            if (index < training_data.size()) {
                                const auto &data_pair = training_data[index];
                                const Matrix &input = data_pair[0];
                                const Matrix &target = data_pair[1];
                                auto [gradient, input_layer_gradient, output] =
                                    optimiser->calculate_gradient(layers, input, target, *loss);
                                local_gradients.push_back(gradient);
                                local_outputs.push_back(output);
                                local_targets.push_back(target);
                            }
                        }

                        // Single lock to update shared data
                        {
                            std::lock_guard<std::mutex> lock(gradients_mutex);
                            batch_outputs.insert(batch_outputs.end(), local_outputs.begin(), local_outputs.end());
                            batch_targets.insert(batch_targets.end(), local_targets.begin(), local_targets.end());
                            thread_gradients[t] = std::move(local_gradients);
                        }
                    });
                }

                // join threads
                for (auto &thread : threads) {
                    thread.join();
                }

                // flatten gradients
                std::vector<std::vector<std::vector<Matrix>>> batch_gradients;
                for (const auto &thread_grad : thread_gradients) {
                    batch_gradients.insert(batch_gradients.end(), thread_grad.begin(), thread_grad.end());
                }

                // average gradients
                auto avg_gradient = optimiser->average_gradients(batch_gradients);

                // apply gradients
                optimiser->compute_and_apply_updates(layers, avg_gradient);

                if (history_logger) {
                    // Calculate average loss across batch (fast)
                    double batch_loss = 0.0;
                    for (size_t i = 0; i < batch_outputs.size(); ++i) {
                        batch_loss += loss->compute(batch_outputs[i], batch_targets[i]);
                    }
                    batch_loss /= batch_outputs.size();

                    TrainingMetrics metrics{batch_loss,
                                            0.0,  // These will be updated properly at epoch end
                                            0.0,
                                            0.0,
                                            0.0,
                                            epoch + 1,
                                            static_cast<int>(batch_start / batch_size) + 1,
                                            std::chrono::system_clock::now()};

                    history_logger->log_metrics(metrics);
                }
            }
            auto eval_results = evaluate_nn(eval_data);

            // special metrics for epoch end
            if (history_logger) {
                TrainingMetrics metrics{eval_results.loss,  // current validation loss
                                        eval_results.accuracy,
                                        eval_results.precision,
                                        eval_results.recall,
                                        eval_results.f1_score,
                                        epoch + 1,
                                        -1,  // Special batch number to indicate epoch-end metrics
                                        std::chrono::system_clock::now()};
                history_logger->log_metrics(metrics);
                std::cout << eval_results << std::endl;
            }
        }
    }

    /**
     * @brief Trains the neural network using multi-threaded optimization.
     * @param training_data The training dataset.
     * @param eval_data The evaluation dataset.
     * @param epochs The number of training epochs.
     * @param batch_size The size of each batch for training.
     */
    size_t get_index_of_max_element_in_nx1_matrix(const Matrix &matrix) const {
        size_t index = 0;
        double max_value = matrix.data[0][0];
        for (size_t i = 1; i < matrix.rows; ++i) {
            if (matrix.data[i][0] > max_value) {
                index = i;
                max_value = matrix.data[i][0];
            }
        }
        return index;
    }

    /**
     * @brief Evaluates the neural network on the given test data.
     * @param test_data The test dataset.
     * @return An EvaluationMetrics struct containing accuracy, precision, recall, and F1 score.
     */
    EvaluationMetrics evaluate_nn(const std::vector<std::vector<Matrix>> &test_data) {
        if (test_data.empty() or test_data[0].size() != 2) {
            throw std::invalid_argument(
                "Test data must be a non-empty vector of vectors, each containing an input and a target matrix.");
        }

        int true_positives = 0, false_positives = 0, false_negatives = 0;
        int total_correct = 0;
        size_t total_examples = test_data.size();

        double total_loss = 0.0;
        for (const auto &example : test_data) {
            const Matrix &input = example[0];
            const Matrix &target = example[1];

            Matrix output = this->feedforward(input);
            total_loss += loss->compute(output, target);

            // assuming output and target are nx1 matrices
            size_t predicted_class = get_index_of_max_element_in_nx1_matrix(output);
            size_t actual_class = get_index_of_max_element_in_nx1_matrix(target);

            if (predicted_class == actual_class) {
                total_correct++;
                true_positives++;
            } else {
                false_positives++;
                false_negatives++;
            }
        }

        double average_loss = total_loss / total_examples;

        double accuracy = static_cast<double>(total_correct) / total_examples;

        // avoid division by zero
        double precision = (true_positives + false_positives > 0)
                               ? static_cast<double>(true_positives) / (true_positives + false_positives)
                               : 0.0;
        double recall = (true_positives + false_negatives > 0)
                            ? static_cast<double>(true_positives) / (true_positives + false_negatives)
                            : 0.0;

        double f1_score = (precision + recall > 0) ? 2 * (precision * recall) / (precision + recall) : 0.0;

        return {average_loss, accuracy, precision, recall, f1_score};
    }

    /**
     * @brief Saves the current state of the neural network to a file.
     * @param filename The name of the file to save the model to.
     */
    void save_model(const std::string &filename) const {
        std::ofstream file(filename, std::ios::binary);
        if (!file) {
            throw std::runtime_error("Unable to open file for writing: " + filename);
        }

        uint32_t num_layers = static_cast<uint32_t>(layers.size());
        file.write(reinterpret_cast<const char *>(&num_layers), sizeof(num_layers));

        // first, write all layer information
        for (size_t i = 0; i < layers.size(); ++i) {
            const auto &layer = layers[i];
            uint32_t input_size = static_cast<uint32_t>(layer.weights.cols);
            uint32_t output_size = static_cast<uint32_t>(layer.weights.rows);

            file.write(reinterpret_cast<const char *>(&input_size), sizeof(input_size));
            file.write(reinterpret_cast<const char *>(&output_size), sizeof(output_size));

            uint32_t activation_function_length = static_cast<uint32_t>(layer.activation_function.length());
            file.write(reinterpret_cast<const char *>(&activation_function_length), sizeof(activation_function_length));
            file.write(layer.activation_function.c_str(), activation_function_length);
        }

        // then, write all weights and biases
        for (size_t i = 0; i < layers.size(); ++i) {
            const auto &layer = layers[i];

            for (const auto &row : layer.weights.data) {
                file.write(reinterpret_cast<const char *>(row.data()), row.size() * sizeof(double));
            }

            for (const auto &row : layer.bias.data) {
                file.write(reinterpret_cast<const char *>(row.data()), row.size() * sizeof(double));
            }
        }

        std::cout << "Model saved successfully. File size: " << file.tellp() << " bytes" << std::endl;
    }

    /**
     * @brief Loads a neural network model from a file.
     * @param filename The name of the file to load the model from.
     * @return A NeuralNetwork object initialized with the loaded model.
     */
    static NeuralNetwork load_model(const std::string &filename) {
        std::ifstream file(filename, std::ios::binary);
        if (!file) {
            throw std::runtime_error("Unable to open file for reading: " + filename);
        }

        uint32_t num_layers;
        file.read(reinterpret_cast<char *>(&num_layers), sizeof(num_layers));

        std::vector<int> topology;
        std::vector<std::string> activation_functions;

        // first, read all layer information
        for (uint32_t i = 0; i < num_layers; ++i) {
            uint32_t input_size, output_size;
            file.read(reinterpret_cast<char *>(&input_size), sizeof(input_size));
            file.read(reinterpret_cast<char *>(&output_size), sizeof(output_size));

            if (i == 0) {
                topology.push_back(input_size);
            }
            topology.push_back(output_size);

            uint32_t activation_function_length;
            file.read(reinterpret_cast<char *>(&activation_function_length), sizeof(activation_function_length));

            std::string activation_function(activation_function_length, '\0');
            file.read(&activation_function[0], activation_function_length);

            activation_functions.push_back(activation_function);
        }

        // create the network with the loaded topology and activation functions
        NeuralNetwork nn(topology, activation_functions);

        // then, read all weights and biases
        for (uint32_t i = 0; i < num_layers; ++i) {
            auto &layer = nn.layers[i];

            for (auto &row : layer.weights.data) {
                file.read(reinterpret_cast<char *>(row.data()), row.size() * sizeof(double));
            }

            for (auto &row : layer.bias.data) {
                file.read(reinterpret_cast<char *>(row.data()), row.size() * sizeof(double));
            }
        }

        if (file.peek() != EOF) {
            throw std::runtime_error("Unexpected data at end of file");
        }

        std::cout << "Model loaded successfully" << std::endl;
        return nn;
    }
};

// runner code
int main() {

    // create the neural network
    int input_size = 28 * 28;
    std::vector<int> topology = {input_size, 64, 32, 32, 10};
    std::vector<std::string> activation_functions = {"sigmoid", "sigmoid", "sigmoid", "softmax"};
    NeuralNetwork nn(topology, activation_functions);
    nn.enable_history_logging("mnist_training_metrics.csv");

    int batch_size = 128;
    int epochs = 50;
    // double learning_rate = 0.1;
    // double momentum_coefficient = 0.8;

    // train the neural network
    nn.set_optimiser(std::make_unique<AdamWOptimiser>());
    nn.set_loss(std::make_unique<CrossEntropyLoss>());
    nn.train_mt_optimiser(training_set, eval_set, epochs, batch_size);
    nn.save_model("mnist.model");

    return 0;
}
