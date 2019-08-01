#include <benchmark/benchmark.h>
#include <thread_pool/for_loop.hpp>
#include <iostream>
#include <unistd.h>

template <bool is_parallel>
static void SumBenchmark(benchmark::State& state) {
    auto pool = MakeThreadPoolExecutor(state.range(1));
    std::vector<uint32_t> counters(state.range(0), 0);
    uint32_t expected_val{0};

    auto func = [](auto* val) {
        *val *= 11;
        *val %= 127;
    };

    if constexpr (is_parallel) {
        for (size_t i = 0; i < 1; ++i) {
            ParallelFor{0, counters.size(), 1, pool.get()}([&](size_t ix) { func(&counters[ix]); });
	    func(&expected_val);
        }
    }

    while (state.KeepRunning()) {
        if constexpr (is_parallel) {
            ParallelFor{0, counters.size(), 1, pool.get()}([&](size_t ix) { func(&counters[ix]); });
        } else {
            for (auto& elem : counters) {
		func(&elem);
            }
        }
	func(&expected_val);
        benchmark::DoNotOptimize(counters[0]);
    }

    for (auto val : counters) {
        if (val != expected_val) {
            throw std::runtime_error("Something went wrong");
        }
    }

    pool->StartShutdown();
    pool->WaitShutdown();
}

static void CustomArguments(benchmark::internal::Benchmark* b) {
    std::array thread_nums{1,2,4,6,8,12,16};
    for (int thread_num : thread_nums) {
	int last = 1024 * 1024 * 6;
        for (int i = 1024 * 6; i <= last; i *= 16) {
            b->Args({i, thread_num});
        }
        b->Args({last, thread_num});
    }
}

BENCHMARK_TEMPLATE(SumBenchmark, 0)->Ranges({{1024 * 6, 1024 * 1024 * 6}, {1, 1}});
BENCHMARK_TEMPLATE(SumBenchmark, 1)->Apply(CustomArguments);
BENCHMARK_MAIN();
