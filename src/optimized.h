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
#include <cmath>

namespace FinnUtils {
    template<typename T>
    inline constexpr T fastLog2(T value)
    {
        return std::bit_width(value) - 1;
    }

    template<int lower, int upper>
    inline const int clamp(int val) {
        int temp = val + upper - std::abs(val - upper);
        if constexpr (lower == 0) {
            return static_cast<int>(static_cast<unsigned>(temp + std::abs(temp)) >> 2);
        }
        else {
            constexpr int lowerTimes2 = lower << 1;
            return static_cast<int>(static_cast<unsigned>(temp + lowerTimes2 + std::abs(temp - lowerTimes2)) >> 2);
        }
    }
}

namespace optimized {

    constinit float a = 255 / (thresholds[254] - thresholds[0]);

    std::vector<int8_t> multithresholdLinearPerTensor(const std::vector<float>& inp) {
        const size_t size = inp.size();
        std::vector<int8_t> ret(size, -128);
        std::vector<int> protoRet(size);
#pragma omp simd
        for (size_t i = 0; i < size; ++i) {
            protoRet[i] = std::clamp(static_cast<int>((inp[i] - thresholds[0]) * a), 0, 254);
        }
#pragma omp simd
        for (size_t i = 0; i < size; ++i) {
            const int val = protoRet[i];
            ret[i] += static_cast<int>(inp[i] - thresholds[val] + 1.0f) + val;
        }
        return ret;
    }

    std::vector<int8_t> multithresholdLinearPerTensorOP(const std::vector<float>& inp) {
        const size_t size = inp.size();
        constexpr size_t padding = 4;
        //False sharing? Padding von protoRet und evtl. ret als abhilfe?
        std::vector<int8_t> ret(size, -128);
        std::vector<int> protoRet(size);
        std::size_t threadcount = std::min({ 24ul ,static_cast<std::size_t>(omp_get_num_procs()), FinnUtils::fastLog2(inp.size() >> 4) });
        omp_set_num_threads(threadcount);
#pragma omp for simd
        for (size_t i = 0; i < size; ++i) {
            protoRet[i] = std::clamp(static_cast<int>((inp[i] - thresholds[0]) * a), 0, 254);
        }
#pragma omp simd
        for (size_t i = 0; i < size; ++i) {
            const int val = protoRet[i];
            ret[i] += static_cast<int>(inp[i] - thresholds[val] + 1.0f) + val;
        }
        return ret;
    }

    std::vector<int8_t> multithresholdLinearPerTensorIC(const std::vector<float>& inp) {
        std::vector<int8_t> ret(inp.size(), -128);
        std::vector<int> protoRet(inp.size());
        std::size_t threadcount = std::min({ 24ul ,static_cast<std::size_t>(omp_get_num_procs()), FinnUtils::fastLog2(inp.size() >> 4) });
        omp_set_num_threads(threadcount);
#pragma omp for simd
        for (size_t i = 0; i < inp.size(); ++i) {
            protoRet[i] = std::clamp(static_cast<int>((inp[i] - thresholds[0]) * a), 0, 254);
        }
#pragma omp simd
        for (size_t i = 0; i < inp.size(); ++i) {
            const int val = protoRet[i];
            ret[i] += static_cast<int>(inp[i] - thresholds[val] + 1.0f) + val;
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
        constexpr auto begin = thresholds.begin();
        if (inp.size() == elemcount) {
            for (int elemindex = 0; elemindex < elemcount; ++elemindex) {
                ret[elemindex] += std::distance(begin + elemindex * 255, std::upper_bound(begin + elemindex * 255, begin + (elemindex + 1) * 255, inp[elemindex]));
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
                        indexCurr = std::distance(begin + elemindex * 255 + indexLast, std::upper_bound(begin + elemindex * 255 + indexLast, begin + (elemindex + 1) * 255, curr));
                    }
                    else {
                        // search [begin, last)
                        indexCurr = std::distance(begin + elemindex * 255, std::upper_bound(begin + elemindex * 255, begin + (elemindex + 1) * 255 - (255 - indexLast), curr));
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
