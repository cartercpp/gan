#include <iostream>
#include <print>
#include <format>
#include <fstream>
#include <string>
#include <vector>
#include <random>
#include <numeric>
#include <thread>
#include <stop_token>
#include "neural_network.h"

template <typename ValueType>
struct std::formatter<std::vector<ValueType>>
{
    constexpr auto parse(std::format_parse_context& ctx)
    {
        return ctx.begin();
    }

    auto format(const std::vector<ValueType>& vec, std::format_context& ctx) const
    {
        auto out = ctx.out();

        *out++ = '{';

        for (std::size_t i = 0; i < vec.size(); ++i)
        {
            if (i != 0)
                out = std::format_to(out, ", ");

            out = std::format_to(out, "{}", vec[i]);
        }

        *out++ = '}';

        return out;
    }
};

int main()
{
    // essentials:
    auto generateHeart = [](std::size_t rows, std::size_t columns) {
        constexpr double minX = -16,
                         maxX = 16,
                         minY = -17,
                         maxY = 12;

        std::vector<std::string> output(rows, std::string(columns, ' '));

        for (double t = 0; t <= 2 * std::numbers::pi; t += 0.05)
        {
            const double x = 16 * std::pow(std::sin(t), 3),
                         y = 13 * std::cos(t) - 5 * std::cos(2 * t) - 2 * std::cos(3 * t) - std::cos(4 * t);

            const std::size_t row = rows - 1 - static_cast<std::size_t>((y - minY) / (maxY - minY) * (rows - 1)),
                              column = static_cast<std::size_t>((x - minX) / (maxX - minX) * (columns - 1));

            if (row > 0)
                output[row - 1][column] = 'o';

            if (row + 1 < rows)
                output[row + 1][column] = 'o';

            if (column > 0)
                output[row][column - 1] = 'o';

            if (column + 1 < columns)
                output[row][column + 1] = 'o';

            output[row][column] = 'o';
        }

        return output;
    };

    constexpr std::size_t gridRows = 40,
                          gridColumns = 80,
                          noiseInputSize = 100;

    auto heartToVector = [](const std::vector<std::string>& heart) {
        math_vector<double> vec(gridRows * gridColumns, 0);

        const std::size_t heartRows = heart.size(),
                          heartColumns = heart[0].size();

        std::random_device rd;
        std::uniform_int_distribution<int> rowOffsetDist(0, gridRows - heartRows),
                                           columnOffsetDist(0, gridColumns - heartColumns);

        const std::size_t rowOffset = rowOffsetDist(rd),
                          columnOffset = columnOffsetDist(rd);

        for (std::size_t row = 0; row < heartRows; ++row)
            for (std::size_t column = 0; column < heartColumns; ++column)
            {
                const std::size_t state = (rowOffset + row) * gridColumns + columnOffset + column;
                vec[state] = heart[row][column] == 'o';
            }

        return vec;
    };

    std::random_device rd;
    std::uniform_int_distribution<std::size_t> rowDist(20, gridRows);
    std::uniform_real_distribution<double> noiseDist(-1, 1);

    neural_network generatorNN({noiseInputSize, 256, 512, gridRows * gridColumns}, 0.0005),
                   discriminatorNN({gridRows * gridColumns, 512, 256, 1}, 0.0001);

    generatorNN.load_parameters(
        "/home/cartercpp/Documents/C++/GAN/GeneratorWeights.txt",
        "/home/cartercpp/Documents/C++/GAN/GeneratorBiases.txt"
        );

    discriminatorNN.load_parameters(
        "/home/cartercpp/Documents/C++/GAN/DiscriminatorWeights.txt",
        "/home/cartercpp/Documents/C++/GAN/DiscriminatorBiases.txt"
    );

    // training thread:
    {
        std::jthread thr{[&](std::stop_token st) {
            std::size_t epochs = 0;

            while (!st.stop_requested())
            {
                // sample real data:
                constexpr std::size_t heartRows = gridRows, // could be random, but keeping static for now
                                      heartColumns = gridColumns;

                const math_vector<double> realHeart{heartToVector(generateHeart(heartRows, heartColumns))};

                // generate fake data:
                math_vector<double> noise(noiseInputSize, 0);
                for (std::size_t i = 0; i < noiseInputSize; ++i)
                    noise[i] = noiseDist(rd);

                const math_vector<double> fakeHeart{generatorNN.forward(noise).back()};

                // train discriminator:
                const auto realHeartPrediction = discriminatorNN.forward(realHeart);
                const auto fakeHeartPrediction = discriminatorNN.forward(fakeHeart);

                discriminatorNN.backward(
                    realHeartPrediction,
                    realHeartPrediction.back() - math_vector<double>{0.9},
                    true
                );
                discriminatorNN.backward(
                    fakeHeartPrediction,
                    fakeHeartPrediction.back() - math_vector<double>{0.0},
                    true
                );
                discriminatorNN.update_weights();

                std::println("Epoch {} | real: {:.4f} | fake: {:.4f}",
                            epochs, realHeartPrediction.back()[0], fakeHeartPrediction.back()[0]);

                // generate new fake data:
                for (int iter = 0; iter < 10; ++iter)
                {
                    math_vector<double> newNoise(noiseInputSize, 0);
                    for (std::size_t i = 0; i < noiseInputSize; ++i)
                        newNoise[i] = noiseDist(rd);

                    const std::vector<math_vector<double>> newFakeHeartGeneration{generatorNN.forward(newNoise)};
                    const math_vector<double>& newFakeHeart{newFakeHeartGeneration.back()};

                    const auto newFakeHeartPrediction = discriminatorNN.forward(newFakeHeart);
                    const auto discriminatorGradients = discriminatorNN.backward(
                        newFakeHeartPrediction,
                        newFakeHeartPrediction.back() - math_vector<double>{1.0},
                        true
                    );
                    discriminatorNN.zero_deltas();

                    generatorNN.backward(
                        newFakeHeartGeneration,
                        discriminatorGradients,
                        false
                    );
                    generatorNN.update_weights();
                }

                // save data:
                if ((epochs % 100 == 0) && (epochs > 0))
                {
                    std::ofstream{"/home/cartercpp/Documents/C++/GAN/GeneratorWeights.txt"}
                        << std::format("{}", generatorNN.weights());

                    std::ofstream{"/home/cartercpp/Documents/C++/GAN/GeneratorBiases.txt"}
                        << std::format("{}", generatorNN.biases());

                    std::ofstream{"/home/cartercpp/Documents/C++/GAN/DiscriminatorWeights.txt"}
                        << std::format("{}", discriminatorNN.weights());

                    std::ofstream{"/home/cartercpp/Documents/C++/GAN/DiscriminatorBiases.txt"}
                        << std::format("{}", discriminatorNN.biases());
                }

                ++epochs;
            }
        }};
        std::cin.get();
    }

    std::ofstream{"/home/cartercpp/Documents/C++/GAN/GeneratorWeights.txt"}
        << std::format("{}", generatorNN.weights());

    std::ofstream{"/home/cartercpp/Documents/C++/GAN/GeneratorBiases.txt"}
        << std::format("{}", generatorNN.biases());

    std::ofstream{"/home/cartercpp/Documents/C++/GAN/DiscriminatorWeights.txt"}
        << std::format("{}", discriminatorNN.weights());

    std::ofstream{"/home/cartercpp/Documents/C++/GAN/DiscriminatorBiases.txt"}
        << std::format("{}", discriminatorNN.biases());
}
