//
// Created by cartercpp on 9/12/26.
//

#ifndef GAN_NEURAL_NETWORK_H
#define GAN_NEURAL_NETWORK_H

#include <string>
#include <initializer_list>
#include <vector>
#include <cstddef>
#include "math_vector.h"
#include "matrix.h"

class neural_network
{
public:

    // CONSTRUCTORS

    explicit neural_network(
        std::initializer_list<matrix<double>>,
        std::initializer_list<math_vector<double>>,
        std::initializer_list<std::size_t>,
        double
    );

    explicit neural_network(
        std::initializer_list<std::size_t>,
        double
    );

    // METHODS

    void load_parameters(const std::string&, const std::string&);

    std::vector<math_vector<double>> forward(const math_vector<double>&);
    math_vector<double> backward(const std::vector<math_vector<double>>&, math_vector<double>, bool);
    void update_weights();
    void zero_deltas();

    const std::vector<matrix<double>>& weights() const noexcept;
    const std::vector<math_vector<double>>& biases() const noexcept;

private:

    static math_vector<double> Relu(math_vector<double>);
    static math_vector<double> ReluDerivative(math_vector<double>);
    static double Sigmoid(double);
    static math_vector<double> Sigmoid(math_vector<double>);
    static double SigmoidDerivative(double);
    static math_vector<double> SigmoidDerivative(math_vector<double>);

    std::vector<matrix<double>> m_weightMatrices,
                                m_weightDeltas;
    std::vector<math_vector<double>> m_biasVectors,
                                     m_biasDeltas;
    std::vector<std::size_t> m_neuronsPerLayer;
    double m_learningRate;
};

#endif //GAN_NEURAL_NETWORK_H