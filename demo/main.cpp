#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <numbers>
#include <random>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "neural_network.h"

namespace ansi
{
    constexpr std::string_view reset = "\033[0m";
    constexpr std::string_view bold = "\033[1m";
    constexpr std::string_view dim = "\033[2m";

    constexpr std::string_view green = "\033[92m";
    constexpr std::string_view red = "\033[91m";
    constexpr std::string_view cyan = "\033[96m";
    constexpr std::string_view magenta = "\033[95m";
    constexpr std::string_view gray = "\033[90m";
}

constexpr std::size_t gridRows = 40;
constexpr std::size_t gridColumns = 80;
constexpr std::size_t noiseInputSize = 100;

constexpr std::size_t confidenceBarWidth = 40;


std::vector<std::string> generate_heart(
    std::size_t rows,
    std::size_t columns
)
{
    constexpr double minX = -16;
    constexpr double maxX = 16;
    constexpr double minY = -17;
    constexpr double maxY = 12;

    std::vector<std::string> output(
        rows,
        std::string(columns, ' ')
    );

    for (
        double t = 0;
        t <= 2 * std::numbers::pi;
        t += 0.05
    )
    {
        const double x =
            16 * std::pow(std::sin(t), 3);

        const double y =
            13 * std::cos(t)
            - 5 * std::cos(2 * t)
            - 2 * std::cos(3 * t)
            - std::cos(4 * t);

        const std::size_t row =
            rows - 1 - static_cast<std::size_t>(
                (y - minY)
                / (maxY - minY)
                * (rows - 1)
            );

        const std::size_t column =
            static_cast<std::size_t>(
                (x - minX)
                / (maxX - minX)
                * (columns - 1)
            );

        output[row][column] = 'o';
    }

    return output;
}


math_vector<double> heart_to_vector(
    const std::vector<std::string>& heart,
    std::mt19937& rng
)
{
    math_vector<double> output(
        gridRows * gridColumns,
        0.0
    );

    const std::size_t heartRows = heart.size();
    const std::size_t heartColumns = heart[0].size();

    std::uniform_int_distribution<std::size_t> rowOffsetDist(
        0,
        gridRows - heartRows
    );

    std::uniform_int_distribution<std::size_t> columnOffsetDist(
        0,
        gridColumns - heartColumns
    );

    const std::size_t rowOffset =
        rowOffsetDist(rng);

    const std::size_t columnOffset =
        columnOffsetDist(rng);

    for (std::size_t row = 0; row < heartRows; ++row)
    {
        for (
            std::size_t column = 0;
            column < heartColumns;
            ++column
        )
        {
            const std::size_t state =
                (rowOffset + row) * gridColumns
                + columnOffset
                + column;

            output[state] =
                heart[row][column] == 'o';
        }
    }

    return output;
}


math_vector<double> generate_fake(
    neural_network& generator,
    std::mt19937& rng
)
{
    std::uniform_real_distribution<double> noiseDist(
        -1.0,
        1.0
    );

    math_vector<double> noise(
        noiseInputSize,
        0.0
    );

    for (std::size_t i = 0; i < noise.size(); ++i)
        noise[i] = noiseDist(rng);

    return generator.forward(noise).back();
}


void print_image(
    const math_vector<double>& image,
    std::string_view color
)
{
    std::cout << color;

    for (std::size_t row = 0; row < gridRows; ++row)
    {
        std::cout << "  ";

        for (
            std::size_t column = 0;
            column < gridColumns;
            ++column
        )
        {
            const std::size_t index =
                row * gridColumns + column;

            std::cout << (
                image[index] >= 0.5
                ? "█"
                : " "
            );
        }

        std::cout << '\n';
    }

    std::cout << ansi::reset;
}


void print_confidence_bar(
    double probability,
    std::string_view color
)
{
    probability = std::clamp(
        probability,
        0.0,
        1.0
    );

    const std::size_t filled =
        static_cast<std::size_t>(
            std::round(
                probability * confidenceBarWidth
            )
        );

    std::cout << color;

    for (std::size_t i = 0; i < filled; ++i)
        std::cout << "█";

    std::cout << ansi::gray;

    for (
        std::size_t i = filled;
        i < confidenceBarWidth;
        ++i
    )
    {
        std::cout << "░";
    }

    std::cout << ansi::reset;
}


int main()
{
    neural_network generatorNN(
        {
            noiseInputSize,
            256,
            512,
            gridRows * gridColumns
        },
        0.001
    );

    neural_network discriminatorNN(
        {
            gridRows * gridColumns,
            512,
            256,
            1
        },
        0.001
    );

    generatorNN.load_parameters(
        "/home/cartercpp/Documents/C++/GAN/GeneratorWeights.txt",
        "/home/cartercpp/Documents/C++/GAN/GeneratorBiases.txt"
    );

    discriminatorNN.load_parameters(
        "/home/cartercpp/Documents/C++/GAN/DiscriminatorWeights.txt",
        "/home/cartercpp/Documents/C++/GAN/DiscriminatorBiases.txt"
    );

    std::mt19937 rng{
        std::random_device{}()
    };

    std::uniform_int_distribution<std::size_t> heartRowDist(
        20,
        gridRows
    );

    bool showReal = true;
    std::size_t sample = 1;

    while (true)
    {
        math_vector<double> image(
            gridRows * gridColumns,
            0.0
        );

        bool actuallyReal = false;

        if (showReal)
        {
            const std::size_t heartRows =
                heartRowDist(rng);

            const std::size_t heartColumns =
                static_cast<std::size_t>(
                    heartRows
                    * static_cast<double>(gridColumns)
                    / gridRows
                );

            image = heart_to_vector(
                generate_heart(
                    heartRows,
                    heartColumns
                ),
                rng
            );

            actuallyReal = true;
        }
        else
        {
            image = generate_fake(
                generatorNN,
                rng
            );

            actuallyReal = false;
        }

        const double realProbability =
            std::clamp(
                discriminatorNN
                    .forward(image)
                    .back()[0],
                0.0,
                1.0
            );

        const bool predictedReal =
            realProbability >= 0.5;

        const double confidence =
            predictedReal
            ? realProbability
            : 1.0 - realProbability;

        const bool correct =
            predictedReal == actuallyReal;

        const std::string_view resultColor =
            correct
            ? ansi::green
            : ansi::red;

        const std::string_view imageColor =
            actuallyReal
            ? ansi::cyan
            : ansi::magenta;


        // Clear screen + move cursor home.
        std::cout << "\033[2J\033[H";


        std::cout
            << ansi::gray
            << "╭──────────────────────────────────────────────────────────────────────────────────╮\n"
            << ansi::reset;

        std::cout
            << "  "
            << ansi::bold
            << ansi::cyan
            << "GAN FROM SCRATCH"
            << ansi::reset
            << ansi::gray
            << "   //   HEART DISCRIMINATOR"
            << ansi::reset
            << "   #"
            << sample
            << '\n';

        std::cout
            << ansi::gray
            << "╰──────────────────────────────────────────────────────────────────────────────────╯\n\n"
            << ansi::reset;


        std::cout
            << ansi::gray
            << "INPUT"
            << ansi::reset
            << "   ";

        if (actuallyReal)
        {
            std::cout
                << ansi::cyan
                << "REAL HEART"
                << ansi::reset;
        }
        else
        {
            std::cout
                << ansi::magenta
                << "GENERATOR OUTPUT"
                << ansi::reset;
        }

        std::cout << "\n\n";


        print_image(
            image,
            imageColor
        );


        std::cout
            << "\n"
            << ansi::gray
            << "DISCRIMINATOR OUTPUT\n\n"
            << ansi::reset;


        std::cout << "  ";

        print_confidence_bar(
            confidence,
            resultColor
        );

        std::cout
            << "  "
            << resultColor
            << ansi::bold
            << std::fixed
            << std::setprecision(1)
            << confidence * 100.0
            << "%"
            << ansi::reset
            << '\n';


        std::cout
            << "\n  PREDICTION: "
            << resultColor
            << ansi::bold
            << (
                predictedReal
                ? "REAL HEART"
                : "FAKE"
            )
            << ansi::reset;


        std::cout
            << "\n  ACTUAL:     "
            << (
                actuallyReal
                ? ansi::cyan
                : ansi::magenta
            )
            << ansi::bold
            << (
                actuallyReal
                ? "REAL HEART"
                : "GENERATED"
            )
            << ansi::reset;


        std::cout
            << "\n  RESULT:     "
            << resultColor
            << ansi::bold
            << (
                correct
                ? "CORRECT"
                : "WRONG"
            )
            << ansi::reset
            << "\n\n";


        std::cout
            << ansi::dim
            << "  input -> discriminator -> real / fake"
            << "   •   next sample in 2s"
            << "   •   Ctrl+C to stop"
            << ansi::reset
            << std::flush;


        showReal = !showReal;
        ++sample;

        std::this_thread::sleep_for(
            std::chrono::seconds(2)
        );
    }
}
