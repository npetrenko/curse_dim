#include <include/kernel.hpp>
#include <include/agent_policy.hpp>
#include <include/bellman_operators/uniform_operator.hpp>
#include <main/pendulum.hpp>

int main() {
    std::mt19937 rd{1234};
    constexpr size_t kNumPendulums = 10;
    ActionConditionedKernel ac_kernel{Pendulum::Kernel<-1>{kNumPendulums, &rd},
                                      Pendulum::Kernel<0>{kNumPendulums, &rd},
                                      Pendulum::Kernel<1>{kNumPendulums, &rd}};
    EnvParams env_params{ac_kernel, Pendulum::RewardFunc{}, 0.95};

    UniformBellmanOperator::Builder builder;
    builder.SetEnvParams(env_params).SetGamma(0.95).SetNumParticles(1024).SetRandomDevice(&rd);
    return 0;
}
