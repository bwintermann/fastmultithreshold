#include <benchmark/benchmark.h>
#include "naive.h"
#include "optimized.h"
#include <random>
#include "lossy.hpp"
#include <span>
#include <immintrin.h>

std::vector<float> base = { 0.5527185,0.39846906, -0.11766014,  0.19299345, -0.38549745,  0.08441927
,   0.26880047,  0.42681944, -0.10539523, -0.02164167,  0.41527015, -0.09802981
,  -0.07409753, -0.41598308,  0.09711669, -0.11594991, -0.4557323,   0.27337435
,  -0.11517189,  0.37859723, -0.15901394,  0.29185423, -0.344608,    0.08293352 };

std::vector<float> inp;
std::vector<float> in;

std::vector<float> getBatchInputs(size_t batchSize) {

  std::random_device rndDevice;
  std::mt19937 mersenneEngine{ rndDevice() };  // Generates random integers

  std::uniform_real_distribution<float> dist{ -4.0, 4.0 };

  auto gen = [&dist, &mersenneEngine]() { return dist(mersenneEngine); };

  std::vector<float> ret(24 * batchSize);

  std::generate(ret.begin(), ret.end(), gen);

  return ret;
}


// ------ FOR CONSTEXPR BENCHS ------
std::size_t threadcount = std::min({ 24ul ,static_cast<std::size_t>(omp_get_num_procs()), FinnUtils::fastLog2(inp.size() >> 4) });


constexpr float max_float = *std::max_element(std::begin(first_thresholds), std::end(first_thresholds));
constexpr float min_float = *std::min_element(std::begin(first_thresholds), std::end(first_thresholds));
constexpr float float_range = max_float - min_float;
constexpr unsigned int scale = 10000;
constexpr unsigned int shift = lossy::_get_shift(scale, min_float);
constexpr unsigned int max_scaled = lossy::_get_max_scale(scale, float_range);

constexpr auto table = lossy::_create_lookup_table<int8_t, float, 255, scale, shift, max_scaled>(first_thresholds);


class LossyFixture : public benchmark::Fixture {
public:
  LossyFixture() {
    Iterations(1000);
  }

  lossy::LossyThresholdLookup<float, int8_t, 255> lu;
  void SetUp(::benchmark::State& state) {
    constexpr auto elements = 255;
    std::array<float, elements> subarray;
    std::copy(thresholds.begin(), thresholds.begin() + elements, subarray.begin());
    lu = lossy::LossyThresholdLookup<float, int8_t, elements>(subarray, 5);
  }
};


__m256i_u get_indices(const float* inputs) {
  const __m256 scales = _mm256_castsi256_ps(_mm256_set_epi32(scale, scale, scale, scale, scale, scale, scale, scale));
  const __m256 shifts = _mm256_castsi256_ps(_mm256_set_epi32(shift, shift, shift, shift, shift, shift, shift, shift)); 
  const __m256 inputs_reg = _mm256_loadu_ps(inputs);
  return _mm256_castps_si256(_mm256_fmadd_ps(inputs_reg, scales, shifts));
}

std::vector<int8_t> lossy_simd_lookup(std::vector<float> &inputs) {
  if (inputs.size() % 8 != 0) {
    throw std::runtime_error("Cannot do simd lookup on non-multiples of 8");
  }
  __m256i_u* indices = (__m256i_u*) malloc(sizeof(__m256i_u) * inputs.size() / 8);
  for (int i = 0; i < inputs.size() / 8; i++) {
    indices[i] = get_indices(inputs.data() + i * 8);
  }

  int8_t last = table[max_scaled - 1];
  int8_t first = table[0];
  std::vector<int8_t> v(inputs.size(), first); // Pre initialize to table[0], this way we can avoid one more comparison
  omp_set_num_threads(threadcount);

#pragma omp parallel for
  for (int index = 0; index < inputs.size(); index++) {
    float i = inputs[index];
    if (i > max_float) {
      v[index] = last;
    } else if (i < min_float) {
      continue;
    } else {
      v[index] = table[((float*) indices)[index]];
    }
  }
  return v;
}



std::vector<int8_t> lossy_constexpr_lookup(std::vector<float> &inputs) {
  int8_t last = table[max_scaled - 1];
  int8_t first = table[0];
  std::vector<int8_t> v(inputs.size(), first); // Pre initialize to table[0], this way we can avoid one more comparison
  omp_set_num_threads(threadcount);

#pragma omp parallel for
  for (int index = 0; index < inputs.size(); index++) {
    float i = inputs[index];
    if (i > max_float) {
      v[index] = last;
    } else if (i < min_float) {
      continue;
    } else {
      v[index] = table[static_cast<unsigned int>(i * scale + shift)];
    }
  }
  return v;
}

std::vector<int8_t> lossy_constexpr_lookup_std(std::vector<float> &inputs) {
  std::vector<int8_t> v(inputs.size(), 0);
  std::transform(
    std::execution::par_unseq,
    inputs.begin(),
    inputs.end(),
    v.begin(),
    [](float i) {
      if (i > max_float) {
        return table[max_scaled-i];
      } else if (i < min_float) {
        return table[0];
      }
      return table[static_cast<unsigned int>(i * scale + shift)];
    }
  );
  return v;
}

std::vector<int8_t> lossy_constexpr_lookup_unparallel(std::vector<float> &inputs) {
  std::vector<int8_t> v(inputs.size(), 0);
  for (int index = 0; index < inputs.size(); index++) {
    if (inputs[index] > max_float) {
      v[index] = table[max_scaled-1];
    } else if (inputs[index] < min_float) {
      v[index] = table[0];
    } else {
      v[index] = table[static_cast<unsigned int>(inputs[index] * scale + shift)];
    }
  }
  return v;
}



//--------------------------------------------------------------------------------

BENCHMARK_F(LossyFixture, BM_lossy1_precision_digits_4)(benchmark::State& state) {
  for (auto _ : state) {
    auto out = lu.thresholds(in);
    benchmark::DoNotOptimize(out);
  }
}

void BM_lossy1_precision_digits_4_constexpr(benchmark::State& state) {
  for (auto _ : state) {
    auto out = lossy_constexpr_lookup_unparallel(in);
    benchmark::DoNotOptimize(out);
  }
}

void BM_referenceB1(benchmark::State& state) {
  for (auto _ : state) {
    auto out = referenceOuter<24>(in);
    benchmark::DoNotOptimize(out);
  }
}

void BM_naiveB1(benchmark::State& state) {
  for (auto _ : state) {
    auto out = multithreshold<24>(in);
    benchmark::DoNotOptimize(out);
  }
}

void BM_optimizedB1(benchmark::State& state) {
  for (auto _ : state) {
    auto out = optimized::multithreshold<24>(in);
    benchmark::DoNotOptimize(out);
  }
}

void BM_optimizedLEB1(benchmark::State& state) {
  for (auto _ : state) {
    auto out = optimized::multithresholdLE<24>(in);
    benchmark::DoNotOptimize(out);
  }
}

void BM_optimizedLinearPTB1(benchmark::State& state) {
  for (auto _ : state) {
    auto out = optimized::multithresholdLinearPerTensor(in);
    benchmark::DoNotOptimize(out);
  }
}

BENCHMARK_F(LossyFixture, BM_lossy4096_precision_digits_4)(benchmark::State& state) {
  for (auto _ : state) {
    auto out = lu.thresholds(inp);
    benchmark::DoNotOptimize(out);
  }
}

void BM_lossy4096_precision_digits_4_constexpr(benchmark::State& state) {
  for (auto _ : state) {
    auto out = lossy_constexpr_lookup(inp);
    benchmark::DoNotOptimize(out);
  }
}

void BM_lossy4096_precision_digits_4_constexpr_simd(benchmark::State& state) {
  for (auto _ : state) {
    auto out = lossy_simd_lookup(inp);
    benchmark::DoNotOptimize(out);
  }
}

void BM_lossy4096_precision_digits_4_constexpr_std(benchmark::State& state) {
  for (auto _ : state) {
    auto out = lossy_constexpr_lookup_std(inp);
    benchmark::DoNotOptimize(out);
  }
}

void BM_lossy4096_precision_digits_4_constexpr_unparallel(benchmark::State& state) {
  for (auto _ : state) {
    auto out = lossy_constexpr_lookup_unparallel(inp);
    benchmark::DoNotOptimize(out);
  }
}

void BM_optimizedLinearPTB4096(benchmark::State& state) {
  for (auto _ : state) {
    auto out = optimized::multithresholdLinearPerTensor(inp);
    benchmark::DoNotOptimize(out);
  }
}

void BM_optimizedLinearPTOPB4096(benchmark::State& state) {
  for (auto _ : state) {
    auto out = optimized::multithresholdLinearPerTensorOP(inp);
    benchmark::DoNotOptimize(out);
  }
}

void BM_optimizedLinearPTOPB1(benchmark::State& state) {
  for (auto _ : state) {
    auto out = optimized::multithresholdLinearPerTensorOP(in);
    benchmark::DoNotOptimize(out);
  }
}

void BM_optimizedLinearPTICB4096(benchmark::State& state) {
  for (auto _ : state) {
    auto out = optimized::multithresholdLinearPerTensorIC(inp);
    benchmark::DoNotOptimize(out);
  }
}

void BM_referenceB4096(benchmark::State& state) {
  for (auto _ : state) {
    auto out = referenceOuter<24>(inp);
    benchmark::DoNotOptimize(out);
  }
}


void BM_optimizedLEMTB1(benchmark::State& state) {
  for (auto _ : state) {
    auto out = optimized::multithresholdLEMT<24>(in);
    benchmark::DoNotOptimize(out);
  }
}


void BM_naiveB4096(benchmark::State& state) {
  for (auto _ : state) {
    auto out = multithreshold<24>(inp);
    benchmark::DoNotOptimize(out);
  }
}

void BM_optimizedB4096(benchmark::State& state) {
  for (auto _ : state) {
    auto out = optimized::multithreshold<24>(inp);
    benchmark::DoNotOptimize(out);
  }
}

void BM_optimizedLEB4096(benchmark::State& state) {
  for (auto _ : state) {
    auto out = optimized::multithresholdLE<24>(inp);
    benchmark::DoNotOptimize(out);
  }
}

void BM_optimizedLEMTB4096(benchmark::State& state) {
  for (auto _ : state) {
    auto out = optimized::multithresholdLEMT<24>(inp);
    benchmark::DoNotOptimize(out);
  }
}

void BM_intclamp(benchmark::State& state) {
  std::vector<int> inp(1000);
  std::iota(inp.begin(), inp.end(), -5);
  for (auto _ : state) {
    for (auto&& elem : inp) {
      auto out = FinnUtils::clamp<0, 254>(elem);
      benchmark::DoNotOptimize(out);
    }
  }
}

void BM_stdclamp(benchmark::State& state) {
  std::vector<int> inp(1000);
  std::iota(inp.begin(), inp.end(), -5);
  for (auto _ : state) {
    for (auto&& elem : inp) {
      auto out = std::clamp(elem, 0, 254);
      benchmark::DoNotOptimize(out);
    }
  }
}

//--------------------------------------------------------------------------------
// clang-format off
BENCHMARK(BM_lossy1_precision_digits_4_constexpr)->Iterations(1000);
BENCHMARK(BM_referenceB1)->Iterations(1000);
BENCHMARK(BM_optimizedLEB1)->Iterations(1000);
BENCHMARK(BM_optimizedLEMTB1)->Iterations(1000);
BENCHMARK(BM_naiveB1)->Iterations(1000);
BENCHMARK(BM_optimizedB1)->Iterations(1000);
BENCHMARK(BM_optimizedLinearPTB1)->Iterations(1000);
BENCHMARK(BM_optimizedLinearPTOPB1)->Iterations(1000);
BENCHMARK(BM_referenceB4096)->Iterations(1000);
BENCHMARK(BM_lossy4096_precision_digits_4_constexpr)->Iterations(1000);
BENCHMARK(BM_lossy4096_precision_digits_4_constexpr_simd)->Iterations(1000);
BENCHMARK(BM_lossy4096_precision_digits_4_constexpr_std)->Iterations(1000);
BENCHMARK(BM_lossy4096_precision_digits_4_constexpr_unparallel)->Iterations(1000);
BENCHMARK(BM_optimizedB4096)->Iterations(1000);
BENCHMARK(BM_naiveB4096)->Iterations(1000);
BENCHMARK(BM_optimizedLEB4096)->Iterations(1000);
BENCHMARK(BM_optimizedLEMTB4096)->Iterations(1000);
BENCHMARK(BM_optimizedLinearPTB4096)->Iterations(1000);
BENCHMARK(BM_optimizedLinearPTOPB4096)->Iterations(1000);
BENCHMARK(BM_optimizedLinearPTICB4096)->Iterations(1000);

BENCHMARK(BM_intclamp)->Iterations(1000);
BENCHMARK(BM_stdclamp)->Iterations(1000);
// clang-format off

//--------------------------------------------------------------------------------


int main(int argc, char** argv) {
  in = getBatchInputs(1);
  inp = getBatchInputs(4096);

  auto v1 = lossy_constexpr_lookup(inp);
  auto v2 = lossy_simd_lookup(inp);
  for (int i = 0; i < v1.size(); i++) {
    if (v1[i] != v2[i]) {
      std::cout << "non equal: " << static_cast<unsigned int>(v1[i]) << " "  << static_cast<unsigned int>(v2[i]) << std::endl;
    }
  }

  ::benchmark::Initialize(&argc, argv);
  ::benchmark::RunSpecifiedBenchmarks();
}
