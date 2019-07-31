#include <main/experiment.hpp>
#include <main/uniform_experiment.hpp>

int main() {
    std::mt19937 rd{1234};
    constexpr size_t kNumPendulums = 10;
    constexpr size_t kNumIterations = 10;
    constexpr size_t kNumParticles = 2048;
    
    AbstractExperiment::Builder builder;
    {
	auto env_params = BuildEnvironment(kNumPendulums, &rd);
        builder.SetEnvironment(std::move(env_params))
            .SetNumIterations(kNumIterations)
            .SetNumParticles(kNumParticles)
            .SetNumPendulums(kNumPendulums)
            .SetRandomDevice(&rd);
    }

    auto uniform_experiment = UniformExperiment::Make(builder);
    uniform_experiment->EstimateQFunc();
    return 0;
}
