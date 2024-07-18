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
#include <limits>

namespace FinnUtils {
    template<typename T>
    inline constexpr T fastLog2(T value)
    {
        return std::bit_width(value) - 1;
    }
}

namespace optimized {

    constexpr float a = (thresholds[254] - thresholds[0]) / 255;

    std::vector<int8_t> multithresholdLinearPerTensor(const std::vector<float>& inp) {
        std::vector<int8_t> ret(inp.size(), -128);
        std::size_t threadcount = std::min({ 24ul ,static_cast<std::size_t>(omp_get_num_procs()), FinnUtils::fastLog2(inp.size() / 24) });
        omp_set_num_threads(threadcount);
#pragma omp for simd
        for (size_t i = 0; i < inp.size(); ++i) {
            const int val = std::clamp(static_cast<int>((inp[i] - thresholds[0]) / a), 0, 254);
            ret[i] += (thresholds[val] < inp[i]) ? val + 1 : val;
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
                result += std::distance(thresholds.begin() + elemindex * 255, std::upper_bound(thresholds.begin() + elemindex * 255, thresholds.begin() + (elemindex + 1) * 255, inp[batchindex * elemcount + elemindex]));
                ret.emplace_back(result);
            }
        }
        return ret;
    }

    template<size_t elemcount>
    std::vector<int8_t> multithresholdLE(const std::vector<float>& inp) {
        std::vector<int8_t> ret(inp.size(), -128);
        if (inp.size() == elemcount) {
            for (size_t batchindex = 0; batchindex < inp.size() / elemcount; ++batchindex) {
                for (int elemindex = 0; elemindex < elemcount; ++elemindex) {
                    ret[batchindex * elemcount + elemindex] += std::distance(thresholds.begin() + elemindex * 255, std::upper_bound(thresholds.begin() + elemindex * 255, thresholds.begin() + (elemindex + 1) * 255, inp[batchindex * elemcount + elemindex]));
                }
            }
        }
        else {
            for (int elemindex = 0; elemindex < elemcount; ++elemindex) {
                float last = std::numeric_limits<float>::lowest();
                std::size_t indexLast = 0;
                for (size_t batchindex = 0; batchindex < inp.size() / elemcount; ++batchindex) {
                    float curr = inp[batchindex * elemcount + elemindex];
                    std::size_t indexCurr = 0;
                    if (curr == last) {
                        indexCurr = indexLast;
                    }
                    else if (curr > last) {
                        // search [last+1, end)
                        indexCurr = std::distance(thresholds.begin() + elemindex * 255 + indexLast, std::upper_bound(thresholds.begin() + elemindex * 255 + indexLast, thresholds.begin() + (elemindex + 1) * 255, curr));
                    }
                    else {
                        // search [begin, last)
                        indexCurr = std::distance(thresholds.begin() + elemindex * 255, std::upper_bound(thresholds.begin() + elemindex * 255, thresholds.begin() + (elemindex + 1) * 255 - (255 - indexLast), curr));
                    }
                    ret[batchindex * elemcount + elemindex] += indexCurr;
                    last = curr;
                    indexLast = indexCurr;
                }
            }
        }
        return ret;
    }

    template<size_t elemcount>
    std::vector<int8_t> multithresholdLEMT(const std::vector<float>& inp) {
        std::vector<int8_t> ret(inp.size(), -128);
        if (inp.size() == elemcount) {
            for (size_t batchindex = 0; batchindex < inp.size() / elemcount; ++batchindex) {
                for (int elemindex = 0; elemindex < elemcount; ++elemindex) {
                    ret[batchindex * elemcount + elemindex] += std::distance(thresholds.begin() + elemindex * 255, std::upper_bound(thresholds.begin() + elemindex * 255, thresholds.begin() + (elemindex + 1) * 255, inp[batchindex * elemcount + elemindex]));
                }
            }
        }
        else {
            std::size_t threadcount = std::min({ elemcount ,static_cast<std::size_t>(omp_get_num_procs()), FinnUtils::fastLog2(inp.size() / elemcount) });
            omp_set_num_threads(threadcount);
#pragma omp parallel for
            for (int elemindex = 0; elemindex < elemcount; ++elemindex) {
                float last = std::numeric_limits<float>::lowest();
                std::size_t indexLast = 0;
                for (size_t batchindex = 0; batchindex < inp.size() / elemcount; ++batchindex) {
                    float curr = inp[batchindex * elemcount + elemindex];
                    std::size_t indexCurr = 0;
                    if (curr == last) {
                        indexCurr = indexLast;
                    }
                    else if (curr > last) {
                        // search [last+1, end)
                        indexCurr = std::distance(thresholds.begin() + elemindex * 255 + indexLast, std::upper_bound(thresholds.begin() + elemindex * 255 + indexLast, thresholds.begin() + (elemindex + 1) * 255, curr));
                    }
                    else {
                        // search [begin, last)
                        indexCurr = std::distance(thresholds.begin() + elemindex * 255, std::upper_bound(thresholds.begin() + elemindex * 255, thresholds.begin() + (elemindex + 1) * 255 - (255 - indexLast), curr));
                    }
                    ret[batchindex * elemcount + elemindex] += indexCurr;
                    last = curr;
                    indexLast = indexCurr;
                }
            }
        }
        return ret;
    }

};

#endif // OPTIMIZED
