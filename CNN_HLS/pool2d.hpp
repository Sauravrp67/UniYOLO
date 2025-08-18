// pool2d.hpp
#pragma once
#include <vector>
#include <stdexcept>
#include <cmath>
#include <algorithm>

class Pool2D {
public:
    enum class Mode { MAX, AVG };

    Pool2D(int kernel_size, Mode mode = Mode::MAX, int stride = 1, int padding = 0, float pad_value = 0.0f)
        : k_(kernel_size), mode_(mode), s_(stride), p_(padding), padv_(pad_value)
    {
        if (k_ < 1) throw std::invalid_argument("kernel_size must be >= 1");
        if (s_ < 1) throw std::invalid_argument("stride must be >= 1");
        if (p_ < 0) throw std::invalid_argument("padding must be >= 0");
    }

    std::vector<std::vector<float>>
    apply(const std::vector<std::vector<float>>& image) const {
        validateRect(image);
        const int H = static_cast<int>(image.size());
        const int W = static_cast<int>(image[0].size());

        if (H < k_ || W < k_) {
            throw std::invalid_argument("Image dimension must be >= kernel dimension");
        }

        const int Hp = H + 2 * p_;
        const int Wp = W + 2 * p_;

        const int out_h = static_cast<int>(std::floor((Hp - k_) / static_cast<float>(s_))) + 1;
        const int out_w = static_cast<int>(std::floor((Wp - k_) / static_cast<float>(s_))) + 1;

        if (out_h <= 0 || out_w <= 0) {
            throw std::invalid_argument("Invalid configuration: empty output (check stride/padding/kernel)");
        }

        std::vector<std::vector<float>> out(out_h, std::vector<float>(out_w, 0.0f));

        for (int oi = 0, i0 = 0; oi < out_h; ++oi, i0 += s_) {
            for (int oj = 0, j0 = 0; oj < out_w; ++oj, j0 += s_) {
                out[oi][oj] = poolWindow(image, H, W, i0, j0);
            }
        }
        return out;
    }

private:
    int k_;          // kernel size (square)
    Mode mode_;      // pooling mode
    int s_;          // stride
    int p_;          // padding
    float padv_;     // pad value

    static void validateRect(const std::vector<std::vector<float>>& img) {
        if (img.empty()) throw std::invalid_argument("image must be non-empty");
        const size_t W = img[0].size();
        if (W == 0) throw std::invalid_argument("image rows must be non-empty");
        for (const auto& row : img) {
            if (row.size() != W) throw std::invalid_argument("image must be rectangular");
        }
    }

    // safe accessor with padding: returns pad value when (r, c) is outside original image
    float getPixel(const std::vector<std::vector<float>>& img, int H, int W, int r, int c) const {
        // r, c are indices in the padded space; map to original by subtracting p_
        const int ri = r - p_;
        const int cj = c - p_;
        if (ri < 0 || cj < 0 || ri >= H || cj >= W) return padv_;
        return img[ri][cj];
    }

    // compute pooled value over k_ x k_ window whose top-left in padded space is (i0, j0)
    float poolWindow(const std::vector<std::vector<float>>& img, int H, int W, int i0, int j0) const {
        if (mode_ == Mode::MAX) {
            float m = getPixel(img, H, W, i0, j0);
            for (int di = 0; di < k_; ++di) {
                for (int dj = 0; dj < k_; ++dj) {
                    float v = getPixel(img, H, W, i0 + di, j0 + dj);
                    if (v > m) m = v;
                }
            }
            return m;
        } else {
            double sum = 0.0; // use double accumulator for stability
            for (int di = 0; di < k_; ++di) {
                for (int dj = 0; dj < k_; ++dj) {
                    sum += static_cast<double>(getPixel(img, H, W, i0 + di, j0 + dj));
                }
            }
            return static_cast<float>(sum / static_cast<double>(k_ * k_));
        }
    }
};
