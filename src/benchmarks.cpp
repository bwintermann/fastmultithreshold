#include <benchmark/benchmark.h>
#include "naive.h"
#include "optimized.h"

std::vector<float> base = { 0.5527185,0.39846906, -0.11766014,  0.19299345, -0.38549745,  0.08441927
,   0.26880047,  0.42681944, -0.10539523, -0.02164167,  0.41527015, -0.09802981
,  -0.07409753, -0.41598308,  0.09711669, -0.11594991, -0.4557323,   0.27337435
,  -0.11517189,  0.37859723, -0.15901394,  0.29185423, -0.344608,    0.08293352 };

std::vector<float> getBatchInputs(size_t batchSize) {
  std::vector<float> ret;
  ret.reserve(base.size() * batchSize);
  for (size_t i = 0; i < batchSize; ++i) {
    std::copy(base.begin(), base.end(), std::back_inserter(ret));
  }
  return ret;
}

//--------------------------------------------------------------------------------
void BM_referenceB1(benchmark::State& state) {
  for (auto _ : state) {
    auto out = referenceOuter<24>(base);
    benchmark::DoNotOptimize(out);
    benchmark::ClobberMemory();
  }
}

void BM_naiveB1(benchmark::State& state) {
  for (auto _ : state) {
    auto out = multithreshold<24>(base);
    benchmark::DoNotOptimize(out);
    benchmark::ClobberMemory();
  }
}

void BM_optimizedB1(benchmark::State& state) {
  for (auto _ : state) {
    auto out = optimized::multithreshold<24>(base);
    benchmark::DoNotOptimize(out);
    benchmark::ClobberMemory();
  }
}

void BM_optimizedLEB1(benchmark::State& state) {
  for (auto _ : state) {
    auto out = optimized::multithresholdLE<24>(base);
    benchmark::DoNotOptimize(out);
    benchmark::ClobberMemory();
  }
}

void BM_optimizedLEMTB1(benchmark::State& state) {
  for (auto _ : state) {
    auto out = optimized::multithresholdLEMT<24>(base);
    benchmark::DoNotOptimize(out);
    benchmark::ClobberMemory();
  }
}

void BM_referenceB4096(benchmark::State& state) {
  auto inp = getBatchInputs(4096);
  for (auto _ : state) {
    auto out = referenceOuter<24>(inp);
    benchmark::DoNotOptimize(out);
    benchmark::ClobberMemory();
  }
}


void BM_naiveB4096(benchmark::State& state) {
  auto inp = getBatchInputs(4096);
  for (auto _ : state) {
    auto out = multithreshold<24>(inp);
    benchmark::DoNotOptimize(out);
    benchmark::ClobberMemory();
  }
}

void BM_optimizedB4096(benchmark::State& state) {
  auto inp = getBatchInputs(4096);
  for (auto _ : state) {
    auto out = optimized::multithreshold<24>(inp);
    benchmark::DoNotOptimize(out);
    benchmark::ClobberMemory();
  }
}

void BM_optimizedLEB4096(benchmark::State& state) {
  auto inp = getBatchInputs(4096);
  for (auto _ : state) {
    auto out = optimized::multithresholdLE<24>(inp);
    benchmark::DoNotOptimize(out);
    benchmark::ClobberMemory();
  }
}

void BM_optimizedLEMTB4096(benchmark::State& state) {
  auto inp = getBatchInputs(4096);
  for (auto _ : state) {
    auto out = optimized::multithresholdLEMT<24>(inp);
    benchmark::DoNotOptimize(out);
    benchmark::ClobberMemory();
  }
}

//--------------------------------------------------------------------------------
// clang-format off
BENCHMARK(BM_referenceB1)->Iterations(1000);
BENCHMARK(BM_optimizedLEB1)->Iterations(1000);
BENCHMARK(BM_optimizedLEMTB1)->Iterations(1000);
BENCHMARK(BM_naiveB1)->Iterations(1000);
BENCHMARK(BM_optimizedB1)->Iterations(1000);
BENCHMARK(BM_referenceB4096)->Iterations(1000);
BENCHMARK(BM_optimizedB4096)->Iterations(1000);
BENCHMARK(BM_naiveB4096)->Iterations(1000);
BENCHMARK(BM_optimizedLEB4096)->Iterations(1000);
BENCHMARK(BM_optimizedLEMTB4096)->Iterations(1000);
// clang-format off

//--------------------------------------------------------------------------------


int main(int argc, char** argv) {
  ::benchmark::Initialize(&argc, argv);
  ::benchmark::RunSpecifiedBenchmarks();
}
