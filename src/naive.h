#ifndef NAIVE
#define NAIVE

#include <vector>
#include <functional>
#include <cstdint>
#include "thresholds.h"
#include <iostream>

int referenceInner(const int nf, const float& accu) {
    int result = -128;
    for (unsigned int i = 0; i < 255; i++) {
        result += std::less<float>()(thresholds.at(nf * 255 + i), accu);
    }
    return result;
}

template<size_t elemcount>
std::vector<int8_t> referenceOuter(const std::vector<float>& inp) {
    std::vector<int8_t> ret;
    ret.reserve(inp.size());
    for (size_t batchindex = 0; batchindex < inp.size() / elemcount; ++batchindex) {
        for (int elemindex = 0; elemindex < elemcount; ++elemindex) {
            int8_t inner = referenceInner(elemindex, inp[batchindex * elemcount + elemindex]);
            //std::cout << (int)inner << "\n";
            ret.emplace_back(inner);
        }
    }
    return ret;
}

template<size_t elemcount>
std::vector<int8_t> multithreshold(const std::vector<float>& inp) {
    std::vector<int8_t> ret;
    ret.reserve(inp.size());
    for (size_t batchindex = 0; batchindex < inp.size() / elemcount; ++batchindex) {
        for (int elemindex = 0; elemindex < elemcount; ++elemindex) {
            int result = -128;
            for (int i = 0; i < 255; ++i) {
                if (thresholds.at(elemindex * 255 + i) >= inp[batchindex * elemcount + elemindex]) {
                    result += i;
                    break;
                }
            }
            ret.emplace_back(result);
        }
    }
    return ret;
}


#endif // NAIVE
