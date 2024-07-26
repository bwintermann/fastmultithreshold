#include <iostream>
#include <execution>
#include <numeric>
#include <algorithm>
#include <type_traits>
#include <array>
#include <cmath>
#include <stdlib.h>
#include <chrono>
#include <vector>
#include <time.h>

/**
 * In this namespace the template parameters usually mean the follwing:
 * 
 * T: Output type (resulting threshold value)
 * F: Input (float) type
 * Scale: How much the floats get multiplied by - i.e. how much precision is conserved. Influences how large the table becomes
 * Shift: How much all values need to be shifted (as large integers), if the lower bound is a negative number. (E.g. -3.0 becomes -30, then Shift needs to be 30 so index 0 comes out)
 * TableSize: The max size of the resulting lookup table. Can be calculated by _get_max_scale()
 */
namespace lossy {
    template<typename T, typename F, unsigned int Scale>
    constexpr unsigned int _lookup_index(F value, const unsigned int shift) {
        return static_cast<T>(value * Scale + shift);
    }

    consteval unsigned int _get_max_scale(const unsigned int scale, const float float_range) {
        return static_cast<unsigned int>(scale * float_range) + 1;
    }

    consteval unsigned int _get_shift(const unsigned int scale, const float min_float) {
        return static_cast<unsigned int>((min_float < 0) ? -min_float * scale : min_float * scale);
    }

    constexpr float _input_range(const float min, const float max) {
        return max - min;
    }

    template<typename T, typename F, std::size_t N, unsigned int Scale, unsigned int Shift, int TableSize>
    constexpr std::array<T, TableSize> _create_lookup_table(const std::array<F, N> float_nums) {
        /**
         * First is a check to see if the table was created at runtime. This can be useful to
         * catch bugs, preventing compile time evaluation
         */
        if (!std::is_constant_evaluated()) {
            std::cout << "Table creation at runtime!\n";
        }

        /**
         * We create a table of the required size. This size is passed as a template parameter and is calculated from the largest index that can possibly be generated from the input data.
         */
        std::array<T, TableSize> results;
        results.fill(0);
        
        /**
         * This is the initial filling of the table. For every threshold we calculate the integer index.
         */
        int i = 0;
        for (auto elem : float_nums) {
            results[_lookup_index<unsigned int, float, Scale>(elem, Shift)] = i;
            i++;
        }    

        /**
         * Finally, the empty spaces between the set indices are being filled. We use the current highest threshold, until a new one is encountered. So in a sequence of 0 1 0 0 5 0 0 7 0 10
         * the result would be 0 1 1 1 5 5 5 7 7 10
         */
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

    template<typename T, unsigned int Scale, unsigned int Shift, unsigned int TableSize>
    constexpr T get_element(float i, const std::array<T, TableSize> &table) {
        return table[static_cast<T>(i * Scale) + Shift];
    }

    /**
     * A class managed approach at runtime instead of compile time. More convenient but slower
     */
    template<typename F, typename T, std::size_t N>
    class LossyThresholdLookup {
        private:
        unsigned int scale;
        unsigned int shift;
        std::vector<T> table;
        float min;
        float max;

        unsigned int index(F input) {
            return static_cast<unsigned int>(input * scale) + shift;
        }

        public:
        LossyThresholdLookup() {}
        LossyThresholdLookup(const std::array<F, N> &thresholds, unsigned int precision_digits) {
            min = *std::min_element(thresholds.begin(), thresholds.end());            
            max = *std::max_element(thresholds.begin(), thresholds.end());            
            scale = static_cast<unsigned int>(std::pow(10, precision_digits));
            shift = static_cast<unsigned int>((min < 0) ? -min * scale : min * scale);
            
            // Reserve enough space
            table = std::vector<T>(scale * (max - min));
            for (unsigned int i = 0; i < thresholds.size(); i++) {
                table[index(thresholds[i])] = i;
            }

            // Fill gaps
            T current = 0;
            for (unsigned int i = 0; i < table.size(); i++) {
                if (table[i] != 0 && table[i] != current) {
                    current = table[i];
                } else {
                    table[i] = current;
                }
            }
        }

        T threshold(F input) {
            if (input > max) {
                return table.back();
            } else if (input < min) {
                return table.front();
            }
            return table[index(input)];
        }

        std::vector<T> thresholds(std::vector<F> &inputs) {
            std::vector<T> r(inputs.size());
            std::transform(
                std::execution::par_unseq, 
                inputs.begin(), 
                inputs.end(), 
                //std::back_inserter(r), 
                r.begin(),
                [this] (F i) {
                    return threshold(i);
                }
            );
            return r;
        }

        void thresholds(std::vector<F> &inputs, std::vector<T> &out) {
            if (out.size() < inputs.size()) {
                out.reserve(inputs.size());
            }
            std::transform(std::execution::par_unseq, inputs.begin(), inputs.end(), out.begin(), [this](F i){return threshold(i);});
        }
    };
}