# gan

A from-scratch generative adversarial network in modern C++. Two feed-forward networks compete over a 40×80 ASCII canvas: a generator that maps 100-dimensional noise to an image, and a discriminator that scores that image as a real or fake heart curve.

No PyTorch, Eigen, or other ML libraries — just the language and the standard library. Part of [ml-from-scratch](https://github.com/cartercpp/ml-from-scratch).

## Layout

```
training/   train the pair; press Enter to stop and dump weights/biases
demo/       load those parameters and alternate real hearts vs generator output
```

Both directories carry the same `neural_network`, `matrix`, and `math_vector` sources. Shared pieces:

- **Generator** — `100 → 256 → 512 → 3200` (40×80 pixels), learning rate `0.001`
- **Discriminator** — `3200 → 512 → 256 → 1`, same learning rate
- Hidden layers use ReLU; the last layer uses sigmoid
- Weights are He-initialized (`N(0, √(2 / fan_in))`); biases start at zero

## What's implemented

**Real data.** A parametric heart curve is rasterized onto a grid of random size and then dropped at a random offset on the 40×80 canvas. That binary image is the discriminator's "real" class.

**Adversarial training** (`training/main.cpp`). Each step:

1. Sample a real heart and a fake image from the generator.
2. Train the discriminator toward `1` on the real sample and `0` on the fake.
3. Sample new noise, run it through the generator, then backprop through the discriminator with a target of `1` ("fool the discriminator"). Those gradients are passed into `generator.backward` so the generator updates against the discriminator's opinion, not a pixel-wise target.
4. Print `real` / `fake` discriminator scores each epoch.

Training runs on a `std::jthread` until you press Enter. Parameters are then written to the four text files the demo loads.

**Demo** (`demo/main.cpp`). Loads saved weights, then every two seconds shows either a real heart or a generator sample, the discriminator's confidence bar, and whether the prediction was correct.

## Build

C++20 or later (training uses `<print>`, `<format>`, and `std::jthread`).

```bash
# train, then press Enter to save parameters
c++ -std=c++20 -O2 training/main.cpp training/neural_network.cpp -o train
./train

# inspect real vs generated samples
c++ -std=c++20 -O2 demo/main.cpp demo/neural_network.cpp -o demo
./demo
```

`training/main.cpp` and `demo/main.cpp` write/read parameter files from a hardcoded path (`/home/cartercpp/Documents/C++/GAN/`). Point those four strings at a directory you own before running.
