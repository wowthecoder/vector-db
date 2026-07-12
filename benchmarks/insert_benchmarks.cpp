#include <benchmark/benchmark.h>

namespace
{

    void BM_CollectionInsert(benchmark::State &state)
    {
        // state.range(0): vectors inserted per iteration
        // state.range(1): vector dimension
        // TODO: Construct a fresh collection and insert pre-generated vectors.
        state.SkipWithError("TODO: implement collection insert benchmark");
    }

}

BENCHMARK(BM_CollectionInsert)
    ->Args({1'000, 128})
    ->Args({10'000, 128})
    ->Unit(benchmark::kMillisecond);

