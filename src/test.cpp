#include <iostream>
#include <vector>
#include "naive.h"
#include <ios>
#include "join.hpp"
#include "optimized.h"
#include "lossy.hpp"

int main() {
    std::vector<float> inputs = { 0.5527185,0.39846906, -0.11766014,  0.19299345, -0.38549745,  0.08441927
,   0.26880047,  0.42681944, -0.10539523, -0.02164167,  0.41527015, -0.09802981
,  -0.07409753, -0.41598308,  0.09711669, -0.11594991, -0.4557323,   0.27337435
,  -0.11517189,  0.37859723, -0.15901394,  0.29185423, -0.344608,    0.08293352 };


    std::vector<float> inputs2 = { 0.5527185,0.39846906, -0.11766014,  0.19299345, -0.38549745,  0.08441927
    ,   0.26880047,  0.42681944, -0.10539523, -0.02164167,  0.41527015, -0.09802981
    ,  -0.07409753, -0.41598308,  0.09711669, -0.11594991, -0.4557323,   0.27337435
    ,  -0.11517189,  0.37859723, -0.15901394,  0.29185423, -0.344608,    0.08293352, 0.5527185,0.39846906, -0.11766014,  0.19299345, -0.38549745,  0.08441927
    ,   0.26880047,  0.42681944, -0.10539523, -0.02164167,  0.41527015, -0.09802981
    ,  -0.07409753, -0.41598308,  0.09711669, -0.11594991, -0.4557323,   0.27337435
    ,  -0.11517189,  0.37859723, -0.15901394,  0.29185423, -0.344608,    0.08293352, 0.5527185,0.39846906, -0.11766014,  0.19299345, -0.38549745,  0.08441927
    ,   0.26880047,  0.42681944, -0.10539523, -0.02164167,  0.41527015, -0.09802981
    ,  -0.07409753, -0.41598308,  0.09711669, -0.11594991, -0.4557323,   0.27337435
    ,  -0.11517189,  0.37859723, -0.15901394,  0.29185423, -0.344608,    0.08293352,0.5527185,0.39846906, -0.11766014,  0.19299345, -0.38549745,  0.08441927
    ,   0.26880047,  0.42681944, -0.10539523, -0.02164167,  0.41527015, -0.09802981
    ,  -0.07409753, -0.41598308,  0.09711669, -0.11594991, -0.4557323,   0.27337435
    ,  -0.11517189,  0.37859723, -0.15901394,  0.29185423, -0.344608,    0.08293352 };


    std::vector<int8_t> expectedResults = { 23,  17,  -5,   8, -16,   4,  11,  18,  -4,  -1,  17,  -4,  -3, -17,
    4,  -5, -19,  11,  -5,  16,  -7,  12, -14,   3 };

    std::vector<int8_t> expectedResults2 = { 23,  17,  -5,   8, -16,   4,  11,  18,  -4,  -1,  17,  -4,  -3, -17,
    4,  -5, -19,  11,  -5,  16,  -7,  12, -14,   3, 23,  17,  -5,   8, -16,   4,  11,  18,  -4,  -1,  17,  -4,  -3, -17,
    4,  -5, -19,  11,  -5,  16,  -7,  12, -14,   3, 23,  17,  -5,   8, -16,   4,  11,  18,  -4,  -1,  17,  -4,  -3, -17,
    4,  -5, -19,  11,  -5,  16,  -7,  12, -14,   3, 23,  17,  -5,   8, -16,   4,  11,  18,  -4,  -1,  17,  -4,  -3, -17,
    4,  -5, -19,  11,  -5,  16,  -7,  12, -14,   3 };


    auto ret = referenceOuter<24>(inputs);

    std::cout << thresholds.size() << "\n";

    auto ret2 = multithreshold<24>(inputs);
    //std::vector<int8_t> ret2 = {};

    std::cout << std::boolalpha << "Reference equal to expected:         " << (ret == expectedResults) << "\n";
    std::cout << std::boolalpha << "Naive equal to expected:             " << (expectedResults == ret2) << "\n";

    std::cout << std::boolalpha << "B4 Naive equal to expected:          " << (expectedResults2 == multithreshold<24>(inputs2)) << "\n";

    auto data = optimized::multithresholdLinearPerTensor(inputs);
    std::cout << std::boolalpha << "Optimized equal to expected:         " << (expectedResults == optimized::multithreshold<24>(inputs)) << "\n";
    std::cout << std::boolalpha << "B4 Optimized equal to expected:      " << (expectedResults2 == optimized::multithreshold<24>(inputs2)) << "\n";

    std::cout << std::boolalpha << "Optimized LE equal to expected:      " << (expectedResults == optimized::multithresholdLE<24>(inputs)) << "\n";
    std::cout << std::boolalpha << "B4 Optimized LE equal to expected:   " << (expectedResults2 == optimized::multithresholdLE<24>(inputs2)) << "\n";

    std::cout << std::boolalpha << "Optimized LEMT equal to expected:    " << (expectedResults == optimized::multithresholdLEMT<24>(inputs)) << "\n";
    std::cout << std::boolalpha << "B4 Optimized LEMT equal to expected: " << (expectedResults2 == optimized::multithresholdLEMT<24>(inputs2)) << "\n";

    std::cout << std::boolalpha << "Optimized LinearPT equal to expected:    " << (expectedResults == optimized::multithresholdLinearPerTensor(inputs)) << "\n";
    std::cout << std::boolalpha << "B4 Optimized LinearPT equal to expected: " << (expectedResults2 == optimized::multithresholdLinearPerTensor(inputs2)) << "\n";

    std::vector<int> out(data.begin(), data.end());
    std::cout << "OptimizedLEMT Out:" << join(out, ",") << "\n";

    std::vector<int> out2(ret2.begin(), ret2.end());
    std::cout << "Naive Out:    " << join(out2, ",") << "\n";

    std::cout << "Clamp Tests:\n";
    std::cout << "Inp: -1 Out: " << FinnUtils::clamp<0,254>(-1) << "\n";
    std::cout << "Inp: 0 Out: " << FinnUtils::clamp<0,254>(0) << "\n";
    std::cout << "Inp: 64 Out: " << FinnUtils::clamp<0,254>(64) << "\n";
    std::cout << "Inp: 254 Out: " << FinnUtils::clamp<0,254>(254) << "\n";
    std::cout << "Inp: 255 Out: " << FinnUtils::clamp<0,254>(255) << "\n";
}