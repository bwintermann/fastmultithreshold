#ifndef OPTIMIZED
#define OPTIMIZED

#include <vector>
#include <functional>
#include <cstdint>
#include "thresholds.h"
#include <iostream>
#include <algorithm>
#include <omp.h>
#include <bit>

namespace FinnUtils {
    template<typename T>
    inline constexpr T fastLog2(T value)
    {
        return std::bit_width(value) - 1;
    }
}

namespace optimized {

    template<size_t elemcount>
    std::vector<int8_t> multithreshold(const std::vector<float>& inp) {
        std::vector<int8_t> ret;
        ret.reserve(inp.size());
        for (size_t batchindex = 0; batchindex < inp.size() / elemcount; ++batchindex) {
            for (int elemindex = 0; elemindex < elemcount; ++elemindex) {
                int result = -128;
                result += std::distance(thresholds.begin() + elemindex * 255, std::upper_bound(thresholds.begin() + elemindex * 255, thresholds.begin() + (elemindex + 1) * 255, inp[batchindex * elemcount + elemindex]));
                ret.emplace_back(result);
            }
        }
        return ret;
    }

    template<size_t elemcount>
    std::vector<int8_t> multithresholdLE(const std::vector<float>& inp) {
        std::vector<int8_t> ret;
        ret.reserve(inp.size());
        for (int elemindex = 0; elemindex < elemcount; ++elemindex) {
            for (size_t batchindex = 0; batchindex < inp.size() / elemcount; ++batchindex) {
                int result = -128;
                result += std::distance(thresholds.begin() + elemindex * 255, std::upper_bound(thresholds.begin() + elemindex * 255, thresholds.begin() + (elemindex + 1) * 255, inp[batchindex * elemcount + elemindex]));
                ret.emplace_back(result);
            }
        }
        return ret;
    }

    template<size_t elemcount>
    std::vector<int8_t> multithresholdLEMT(const std::vector<float>& inp) {
        std::vector<int8_t> ret(inp.size());
        std::size_t threadcount = std::min({ elemcount ,static_cast<std::size_t>(omp_get_num_procs()), FinnUtils::fastLog2(inp.size() / elemcount) });
        omp_set_num_threads(threadcount);
#pragma omp parallel for
        for (int elemindex = 0; elemindex < elemcount; ++elemindex) {
            for (size_t batchindex = 0; batchindex < inp.size() / elemcount; ++batchindex) {
                int result = -128;
                result += std::distance(thresholds.begin() + elemindex * 255, std::upper_bound(thresholds.begin() + elemindex * 255, thresholds.begin() + (elemindex + 1) * 255, inp[batchindex * elemcount + elemindex]));
                ret[batchindex * elemcount + elemindex] = result;
            }
        }
        return ret;
    }

};

#endif // OPTIMIZED
