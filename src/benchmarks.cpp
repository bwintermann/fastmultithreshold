#include <benchmark/benchmark.h>
#include "naive.h"
#include "optimized.h"
#include <random>

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

//--------------------------------------------------------------------------------
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

void BM_optimizedLinearPTB4096(benchmark::State& state) {
  for (auto _ : state) {
    auto out = optimized::multithresholdLinearPerTensor(inp);
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

//--------------------------------------------------------------------------------
// clang-format off
BENCHMARK(BM_referenceB1)->Iterations(1000);
BENCHMARK(BM_optimizedLEB1)->Iterations(1000);
BENCHMARK(BM_optimizedLEMTB1)->Iterations(1000);
BENCHMARK(BM_naiveB1)->Iterations(1000);
BENCHMARK(BM_optimizedB1)->Iterations(1000);
BENCHMARK(BM_optimizedLinearPTB1)->Iterations(1000);
BENCHMARK(BM_referenceB4096)->Iterations(1000);
BENCHMARK(BM_optimizedB4096)->Iterations(1000);
BENCHMARK(BM_naiveB4096)->Iterations(1000);
BENCHMARK(BM_optimizedLEB4096)->Iterations(1000);
BENCHMARK(BM_optimizedLEMTB4096)->Iterations(1000);
BENCHMARK(BM_optimizedLinearPTB4096)->Iterations(1000);
// clang-format off

//--------------------------------------------------------------------------------


int main(int argc, char** argv) {
  in = getBatchInputs(1);
  inp = getBatchInputs(4096);
  ::benchmark::Initialize(&argc, argv);
  ::benchmark::RunSpecifiedBenchmarks();
}
