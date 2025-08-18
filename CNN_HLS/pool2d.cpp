#include <iostream>
#include "pool2d.hpp"

int main() {
    std::vector<std::vector<float>> image = {
        {1, 2, 3, 0},
        {4, 5, 6, 1},
        {7, 8, 9, 2},
        {1, 2, 3, 4},
    };

    // 2x2 max pooling, stride 2, no padding
    Pool2D pool_max(2, Pool2D::Mode::MAX, 2, 0);
    auto out_max = pool_max.apply(image);

    std::cout << "MaxPool 2x2 s=2:\n";
    for (auto& r : out_max) {
        for (auto v : r) std::cout << v << " ";
        std::cout << "\n";
    }

    // 3x3 average pooling, stride 1, padding 1 (same-like)
    Pool2D pool_avg(3, Pool2D::Mode::AVG, 1, 1, 0.0f);
    auto out_avg = pool_avg.apply(image);

    std::cout << "\nAvgPool 3x3 s=1 p=1:\n";
    for (auto& r : out_avg) {
        for (auto v : r) std::cout << v << " ";
        std::cout << "\n";
    }
    return 0;
}
