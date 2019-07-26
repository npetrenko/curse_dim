#include <include/kernel.hpp>
#include <include/agent_policy.hpp>
#include <include/bellman_operators/uniform_operator.hpp>
#include <main/pendulum.hpp>

int main() {
    std::mt19937 rd{1234};
    const size_t kNumPendulums = 10;
    const size_t kNumParticles = 1024;
    ActionConditionedKernel ac_kernel{Pendulum::Kernel<-1>{kNumPendulums, &rd},
                                      Pendulum::Kernel<0>{kNumPendulums, &rd},
                                      Pendulum::Kernel<1>{kNumPendulums, &rd}};
    EnvParams env_params{ac_kernel, Pendulum::RewardFunc{}, 0.95};
    UniformBellmanOperator bellman_op{env_params, kNumParticles, 1., &rd};
    bellman_op.MakeIteration();
    return 0;
}
