#ifndef NAIVE
#define NAIVE

#include<vector>
#include <functional>
#include <cstdint>
#include "thresholds.h"
#include <iostream>

int referenceInner(const int nf, const float& accu) {
    int result = -128;
    for (unsigned int i = 0; i < 256; i++) {
        result += std::less<float>()(thresholds[nf * 256 + i], accu);
    }
    return result;
}

std::vector<int8_t> referenceOuter(const std::vector<float>& inp) {
    std::vector<int8_t> ret;
    ret.reserve(inp.size());
    for (int elemindex = 0; elemindex < inp.size(); ++elemindex) {
        int8_t inner = referenceInner(elemindex, inp[elemindex]);
        std::cout << (int)inner << "\n";
        ret.emplace_back(inner);
    }
    return ret;
}

std::vector<int8_t> multithreshold(const std::vector<float>& inp) {
    std::vector<int8_t> ret;
    ret.reserve(inp.size());
    for (int elemindex = 0; elemindex < inp.size(); ++elemindex) {
        int result = -128;
        for (int i = 0; i < 256; ++i) {
            if (thresholds[elemindex * 256 + i] >= inp[elemindex]) {
                result = i + 1;
                break;
            }
        }
        std::cout << (int)result << "\n";
        constexpr int8_t val = 0;
        ret.emplace_back(val + result);
    }
    return ret;
}


#endif // NAIVE
