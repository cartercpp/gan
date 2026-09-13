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
    constexpr std::string_view reset   = "\033[0m";
    constexpr std::string_view bold    = "\033[1m";
    constexpr std::string_view dim     = "\033[2m";

    constexpr std::string_view green   = "\033[92m";
    constexpr std::string_view red     = "\033[91m";
    constexpr std::string_view cyan    = "\033[96m";
    constexpr std::string_view magenta = "\033[95m";
    constexpr std::string_view gray    = "\033[90m";
}

constexpr std::size_t gridRows = 40;
constexpr std::size_t gridColumns = 80;
constexpr std::size_t noiseInputSize = 100;

constexpr std::size_t confidenceBarWidth = 40;


// ------------------------------------------------------------
// REAL HEART
// ------------------------------------------------------------

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

        // Same thicker heart used during training.

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
}


math_vector<double> heart_to_vector(
    const std::vector<std::string>& heart
)
{
    math_vector<double> output(
        gridRows * gridColumns,
        0.0
    );

    for (std::size_t row = 0; row < gridRows; ++row)
    {
        for (
            std::size_t column = 0;
            column < gridColumns;
            ++column
        )
        {
            const std::size_t index =
                row * gridColumns + column;

            output[index] =
                heart[row][column] == 'o';
        }
    }

    return output;
}


// ------------------------------------------------------------
// GENERATOR
// ------------------------------------------------------------

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


// ------------------------------------------------------------
// VISUALIZATION
// ------------------------------------------------------------

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


void print_probability_bar(
    double probability,
    std::string_view color
)
{
    probability =
        std::clamp(probability, 0.0, 1.0);

    const std::size_t filled =
        static_cast<std::size_t>(
            std::round(
                probability
                * confidenceBarWidth
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


// ------------------------------------------------------------
// MAIN
// ------------------------------------------------------------

int main()
{
    neural_network generatorNN(
        {
            noiseInputSize,
            256,
            512,
            gridRows * gridColumns
        },
        0.0005
    );

    neural_network discriminatorNN(
        {
            gridRows * gridColumns,
            512,
            256,
            1
        },
        0.0001
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


    // This is exactly the real sample used during training.
    const math_vector<double> realHeart{
        heart_to_vector(
            generate_heart(
                gridRows,
                gridColumns
            )
        )
    };


    bool displayReal = true;
    std::size_t sample = 1;


    while (true)
    {
        math_vector<double> image(
            gridRows * gridColumns,
            0.0
        );

        bool actuallyReal;


        // ----------------------------------------------------
        // INPUT
        // ----------------------------------------------------

        if (displayReal)
        {
            image = realHeart;
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


        // ----------------------------------------------------
        // DISCRIMINATOR
        // ----------------------------------------------------

        const double realProbability =
            std::clamp(
                discriminatorNN
                    .forward(image)
                    .back()[0],
                0.0,
                1.0
            );

        const double fakeProbability =
            1.0 - realProbability;

        const bool predictedReal =
            realProbability >= 0.5;

        const bool correct =
            predictedReal == actuallyReal;

        const double confidence =
            predictedReal
                ? realProbability
                : fakeProbability;


        const std::string_view resultColor =
            correct
                ? ansi::green
                : ansi::red;

        const std::string_view imageColor =
            actuallyReal
                ? ansi::cyan
                : ansi::magenta;


        // ----------------------------------------------------
        // DRAW
        // ----------------------------------------------------

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
            << "   //   HEART GAN"
            << ansi::reset
            << "   #"
            << sample
            << '\n';

        std::cout
            << ansi::gray
            << "╰──────────────────────────────────────────────────────────────────────────────────╯\n\n"
            << ansi::reset;


        // ----------------------------------------------------
        // IMAGE
        // ----------------------------------------------------

        std::cout
            << ansi::gray
            << "INPUT"
            << ansi::reset
            << "   ";

        if (actuallyReal)
        {
            std::cout
                << ansi::cyan
                << ansi::bold
                << "REAL TRAINING HEART"
                << ansi::reset;
        }
        else
        {
            std::cout
                << ansi::magenta
                << ansi::bold
                << "GENERATOR OUTPUT"
                << ansi::reset;
        }

        std::cout << "\n\n";


        print_image(
            image,
            imageColor
        );


        // ----------------------------------------------------
        // DISCRIMINATOR OUTPUT
        // ----------------------------------------------------

        std::cout
            << "\n"
            << ansi::gray
            << "DISCRIMINATOR OUTPUT"
            << ansi::reset
            << "\n\n";


        std::cout << "  REAL  ";

        print_probability_bar(
            realProbability,
            ansi::cyan
        );

        std::cout
            << "  "
            << std::fixed
            << std::setprecision(1)
            << realProbability * 100.0
            << "%\n";


        std::cout << "  FAKE  ";

        print_probability_bar(
            fakeProbability,
            ansi::magenta
        );

        std::cout
            << "  "
            << std::fixed
            << std::setprecision(1)
            << fakeProbability * 100.0
            << "%\n";


        // ----------------------------------------------------
        // RESULT
        // ----------------------------------------------------

        std::cout
            << "\n  PREDICTION: "
            << resultColor
            << ansi::bold
            << (
                predictedReal
                    ? "REAL"
                    : "FAKE"
            )
            << ansi::reset;


        std::cout
            << "\n  CONFIDENCE: "
            << resultColor
            << ansi::bold
            << std::fixed
            << std::setprecision(1)
            << confidence * 100.0
            << "%"
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
                    ? "REAL"
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
            << "  noise -> generator -> image -> discriminator -> real / fake"
            << "   •   next sample in 2s"
            << "   •   Ctrl+C to stop"
            << ansi::reset
            << std::flush;


        displayReal = !displayReal;
        ++sample;

        std::this_thread::sleep_for(
            std::chrono::seconds(2)
        );
    }
}
