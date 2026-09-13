//
// Created by cartercpp on 9/12/26.
//

#include "neural_network.h"
#include <fstream>
#include <stdexcept>
#include <initializer_list>
#include <vector>
#include <utility>
#include <random>
#include <cctype>
#include <cstddef>
#include <cmath>
#include "math_vector.h"
#include "matrix.h"

static double read_next_double(std::istream &input)
{
    while (input)
    {
        const int next = input.peek();

        if (next == std::char_traits<char>::eof())
            break;

        const char c = static_cast<char>(next);

        if (
            std::isdigit(static_cast<unsigned char>(c))
            || c == '-'
            || c == '+'
            || c == '.'
        )
        {
            double value;

            if (!(input >> value))
                throw std::runtime_error{"Failed to parse parameter"};

            return value;
        }

        input.get(); // skip { } , spaces, etc.
    }

    throw std::runtime_error{"Not enough parameters in file"};
}

void neural_network::load_parameters(
    const std::string& weightsFile,
    const std::string& biasesFile
)
{
    std::ifstream weightStream{weightsFile};

    if (!weightStream)
        throw std::runtime_error{
        "Could not open weights file: " + weightsFile
    };

    for (auto& matrix : m_weightMatrices)
        for (std::size_t row = 0; row < matrix.rows(); ++row)
            for (std::size_t column = 0; column < matrix.columns(); ++column)
                matrix[row][column] = read_next_double(weightStream);

    std::ifstream biasStream{biasesFile};

    if (!biasStream)
        throw std::runtime_error{
        "Could not open biases file: " + biasesFile
    };

    for (auto& vector : m_biasVectors)
        for (std::size_t i = 0; i < vector.size(); ++i)
            vector[i] = read_next_double(biasStream);

    zero_deltas();
}

math_vector<double> neural_network::Relu(math_vector<double> vec)
{
    for (std::size_t i = 0; i < vec.size(); ++i)
        vec[i] = (vec[i] > 0) ? vec[i] : 0;

    return vec;
}

math_vector<double> neural_network::ReluDerivative(math_vector<double> vec)
{
    for (std::size_t i = 0; i < vec.size(); ++i)
        vec[i] = vec[i] > 0;

    return vec;
}

double neural_network::Sigmoid(double x)
{
    return 1 / (1 + std::exp(-x));
}

math_vector<double> neural_network::Sigmoid(math_vector<double> vec)
{
    for (std::size_t i = 0; i < vec.size(); ++i)
        vec[i] = Sigmoid(vec[i]);

    return vec;
}

double neural_network::SigmoidDerivative(double x)
{
    return x * (1 - x);
}

math_vector<double> neural_network::SigmoidDerivative(math_vector<double> vec)
{
    for (std::size_t i = 0; i < vec.size(); ++i)
        vec[i] = SigmoidDerivative(vec[i]);

    return vec;
}

neural_network::neural_network(
    std::initializer_list<matrix<double>> weightMatrices,
    std::initializer_list<math_vector<double>> biasVectors,
    std::initializer_list<std::size_t> neuronsPerLayer,
    double learningRate
) : m_weightMatrices{weightMatrices}, m_biasVectors{biasVectors}, m_neuronsPerLayer{neuronsPerLayer},
    m_learningRate{learningRate}
{
    if (weightMatrices.size() != m_neuronsPerLayer.size() - 1)
        throw std::invalid_argument{"The # of weight matrices must = # of layers - 1"};
    else if (biasVectors.size() != m_neuronsPerLayer.size() - 1)
        throw std::invalid_argument{"The # of bias vectors must = # of layers - 1"};

    m_weightDeltas.reserve(m_weightMatrices.size());
    m_biasDeltas.reserve(m_biasVectors.size());

    for (std::size_t i = 0; i < m_weightMatrices.size(); ++i)
    {
        const std::size_t layerSize = m_neuronsPerLayer[i + 1],
                          prevLayerSize = m_neuronsPerLayer[i];

        m_weightDeltas.emplace_back(layerSize, prevLayerSize, 0);
        m_biasDeltas.emplace_back(layerSize, 0);

        const matrix<double>& layerWeights{m_weightMatrices[i]};
        const math_vector<double>& layerBiases{m_biasVectors[i]};

        if ((layerWeights.rows() != layerSize) || (layerWeights.columns() != prevLayerSize))
            throw std::invalid_argument{"The dimensions of one or more of the weight matrices is wrong"};
        else if (layerBiases.size() != layerSize)
            throw std::invalid_argument{"The size of one or more of the bias vectors is wrong"};
    }
}

neural_network::neural_network(
    std::initializer_list<std::size_t> neuronsPerLayer,
    double learningRate
) : m_neuronsPerLayer{neuronsPerLayer}, m_learningRate{learningRate}
{
    m_weightMatrices.reserve(m_neuronsPerLayer.size() - 1);
    m_weightDeltas.reserve(m_neuronsPerLayer.size() - 1);
    m_biasVectors.reserve(m_neuronsPerLayer.size() - 1);
    m_biasDeltas.reserve(m_neuronsPerLayer.size() - 1);

    std::random_device rd;
    std::mt19937 gen(rd());

    for (std::size_t i = 0; i < m_neuronsPerLayer.size() - 1; ++i)
    {
        const std::size_t layerSize = m_neuronsPerLayer[i + 1],
                          prevLayerSize = m_neuronsPerLayer[i];

        matrix<double> weights(layerSize, prevLayerSize, 0);

        std::normal_distribution<double> dist(0, std::sqrt(2.0 / static_cast<double>(prevLayerSize)));
        for (std::size_t row = 0; row < layerSize; ++row)
            for (std::size_t column = 0; column < prevLayerSize; ++column)
                weights[row][column] = dist(gen);

        m_weightMatrices.emplace_back(std::move(weights));
        m_weightDeltas.emplace_back(layerSize, prevLayerSize, 0);
        m_biasVectors.emplace_back(layerSize, 0);
        m_biasDeltas.emplace_back(layerSize, 0);
    }
}

std::vector<math_vector<double>> neural_network::forward(const math_vector<double>& input)
{
    std::vector<math_vector<double>> activations;
    activations.reserve(m_neuronsPerLayer.size());
    activations.push_back(input);

    for (std::size_t i = 0; i < m_weightMatrices.size(); ++i)
    {
        if (i + 1 < m_weightMatrices.size())
            activations.emplace_back(Relu(m_weightMatrices[i] * activations[i] + m_biasVectors[i]));
        else
            activations.emplace_back(Sigmoid(m_weightMatrices[i] * activations[i] + m_biasVectors[i]));
    }

    return activations;
}

math_vector<double> neural_network::backward(
    const std::vector<math_vector<double>>& activations,
    math_vector<double> delta,
    bool outputDeltaIsPreActivation
)
{
    if (activations.size() != m_neuronsPerLayer.size())
        throw std::invalid_argument{"`activations` doesn't fit the architecture of this neural net"};

    for (std::size_t layer = 0; layer < activations.size(); ++layer)
        if (activations[layer].size() != m_neuronsPerLayer[layer])
            throw std::invalid_argument{"`activations` doesn't fit the architecture of this neural net"};

    for (std::size_t i = 0; i < m_weightMatrices.size(); ++i)
    {
        if (i == 0)
        {
            if (!outputDeltaIsPreActivation)
                delta = delta.multiply(SigmoidDerivative(activations[activations.size() - i - 1]));
        }
        else
            delta = delta.multiply(ReluDerivative(activations[activations.size() - i - 1]));

        m_weightDeltas[m_weightDeltas.size() - i - 1]
            += outer_product(delta, activations[activations.size() - i - 2]);

        m_biasDeltas[m_biasDeltas.size() - i - 1] += delta;

        delta = m_weightMatrices[m_weightMatrices.size() - i - 1].transpose() * delta;
    }

    return delta;
}

void neural_network::update_weights()
{
    for (std::size_t i = 0; i < m_weightMatrices.size(); ++i)
    {
        m_weightMatrices[i] -= m_learningRate * m_weightDeltas[i];
        m_biasVectors[i] -= m_learningRate * m_biasDeltas[i];
    }

    zero_deltas();
}

void neural_network::zero_deltas()
{
    for (std::size_t i = 0; i < m_weightMatrices.size(); ++i)
    {
        const std::size_t rows = m_weightDeltas[i].rows(),
                          columns = m_weightDeltas[i].columns();

        for (std::size_t row = 0; row < rows; ++row)
        {
            for (std::size_t column = 0; column < columns; ++column)
                m_weightDeltas[i][row][column] = 0;

            m_biasDeltas[i][row] = 0;
        }
    }
}

const std::vector<matrix<double>>& neural_network::weights() const noexcept
{
    return m_weightMatrices;
}

const std::vector<math_vector<double>>& neural_network::biases() const noexcept
{
    return m_biasVectors;
}