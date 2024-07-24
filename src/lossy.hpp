#include <iostream>
#include <numeric>
#include <algorithm>
#include <type_traits>
#include <array>
#include <cmath>
#include <stdlib.h>
#include <chrono>
#include <vector>
#include <time.h>

namespace lossy {
    /**
     * Get the index in the lookup table from a given floating point value
     * 
     * T: The type that should be emitted from the table (usually uint8_t)
     * F: The type of the non-quantized input
     * Scale: The scale that was also used to generate the table
     */
    template<typename T, typename F, unsigned int Scale>
    constexpr unsigned int _lookup_index(F value, const unsigned int shift) {
        return static_cast<T>(value * Scale + shift);
    }

    
    /**
     * Get the maximum scaled value or, equivalently, the size of the generated table
     */
    constexpr unsigned int _get_max_scale(const unsigned int scale, const float float_range) {
        return static_cast<unsigned int>(scale * float_range) + 1;
    }

    /**
     * Given the minimum value of the flaots calculate the necessary offset
     */
    constexpr unsigned int _get_shift(const unsigned int scale, const float min_float) {
        return static_cast<unsigned int>((min_float < 0) ? -min_float * scale : min_float * scale);
    }

    /**
     * Get the range of the input values
     */
    constexpr float _input_range(const float min, const float max) {
        return max - min;
    }

    /**
     * Internally used function to generate the lookup table
     */
    template<typename T, typename F, std::size_t N, unsigned int Scale, unsigned int Shift, int TableSize>
    constexpr std::array<T, TableSize> _create_lookup_table(const std::array<F, N> float_nums, const float min_float, const float max_float) {
        if (!std::is_constant_evaluated()) {
            std::cout << "Table creation at runtime!\n";
        }
        std::array<T, TableSize> results;
        results.fill(0);
        int i = 0;
        for (auto elem : float_nums) {
            results[lookup_index<unsigned int, float, Scale>(elem, Shift)] = i;
            i++;
        }    
        T current = 0;
        for (int j = 0; j < TableSize; j++) {
            if (results[j] != 0 && results[j] != current) {
                current = results[j];
            } else {
                results[j] = current; 
            }
        }
        return results;
    }

    /**
     * Call this function to create a lookup table
     */
    template<typename T, typename F, std::size_t S, unsigned int Scale>
    constexpr auto create_table(const std::array<F,S> &thresholds) {
        const float max_float = thresholds[S-1];
        const float min_float = thresholds[0];
        constexpr float float_range = _input_range(min_float, max_float);
        constexpr unsigned int max_scaled = _get_max_scale(Scale, float_range);
        constexpr unsigned int shift = _get_shift(Scale, min_float);
        return _create_lookup_table<T, F, S, Scale, shift, max_scaled>(thresholds, min_float, max_float);
    } 

    /**
     * Get the result of a threshold operation with the given table
     */
    template<typename T, unsigned int Scale, unsigned int Shift, unsigned int TableSize>
    constexpr T get_element(float i, const std::array<T, TableSize> &table) {
        return table[static_cast<T>(i * Scale) + Shift];
    }
}