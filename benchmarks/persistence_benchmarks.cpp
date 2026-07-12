#include <benchmark/benchmark.h>

namespace
{

    void BM_CollectionSave(benchmark::State &state)
    {
        // state.range(0): collection size
        // state.range(1): vector dimension
        // TODO: Populate once and time writes to a temporary file.
        state.SkipWithError("TODO: implement collection save benchmark");
    }

    void BM_CollectionLoad(benchmark::State &state)
    {
        // state.range(0): collection size
        // state.range(1): vector dimension
        // TODO: Save one fixture file, then time repeated Collection::load calls.
        state.SkipWithError("TODO: implement collection load benchmark");
    }

}

BENCHMARK(BM_CollectionSave)
    ->Args({1'000, 128})
    ->Args({10'000, 128})
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_CollectionLoad)
    ->Args({1'000, 128})
    ->Args({10'000, 128})
    ->Unit(benchmark::kMillisecond);

