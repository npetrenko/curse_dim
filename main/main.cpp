#include <src/kernel.hpp>
#include <src/agent_policy.hpp>
#include <main/pendulum.hpp>

class Policy : public AbstractAgentPolicy<Policy> {
};

int main() {
    std::mt19937 rd{1234};
    const size_t kNumPendulums = 10;
    Policy pol;
    MDPKernel kernel{ActionConditionedKernel{Pendulum::Kernel<-1>{kNumPendulums, &rd},
                                             Pendulum::Kernel<0>{kNumPendulums, &rd},
                                             Pendulum::Kernel<1>{kNumPendulums, &rd}},
                     &pol};
    return 0;
}
