#include <curse_dim/experiment.hpp>
#include <curse_dim/uniform_experiment.hpp>
#include <curse_dim/stationary_experiment.hpp>
#include <glog/logging.h>

int main(int, char** argv) {
    google::LogToStderr();
    google::InitGoogleLogging(argv[0]);

    std::mt19937 rd{1234};
    constexpr size_t kNumPendulums = 2;
    constexpr size_t kNumIterations = 2;
    constexpr size_t kNumParticles = 1024;

    AbstractExperiment::Builder builder;
    {
        auto env_params = BuildEnvironment(kNumPendulums, &rd);
        builder.SetEnvironment(std::move(env_params))
            .SetNumIterations(kNumIterations)
            .SetNumParticles(kNumParticles)
            .SetNumPendulums(kNumPendulums)
            .SetRandomDevice(&rd);
    }

    FloatT uniform_experiment_score, stationary_experiment_score;
    {
        auto uniform_experiment = UniformExperiment::Make(builder);
        uniform_experiment_score = uniform_experiment->Score();
    }
    {
        auto stationary_experiment = StationaryExperiment::Make(builder);
        stationary_experiment_score = stationary_experiment->Score();
    }
    LOG(INFO) << "Uniform experiment score: " << uniform_experiment_score;
    LOG(INFO) << "Stationary experiment score: " << stationary_experiment_score;
    return 0;
}
