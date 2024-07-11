#include <benchmark/benchmark.h>


//--------------------------------------------------------------------------------
void BM_base(benchmark::State& state) {
  int i;
  for (auto _ : state) {

    benchmark::DoNotOptimize(i);
  }
}

//--------------------------------------------------------------------------------
// clang-format off
BENCHMARK(BM_base);

// clang-format off

//--------------------------------------------------------------------------------


int main(int argc, char** argv) {
  ::benchmark::Initialize(&argc, argv);
  ::benchmark::RunSpecifiedBenchmarks();
}
