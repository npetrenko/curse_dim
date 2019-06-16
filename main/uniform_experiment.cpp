#include <src/types.hpp>
#include <main/pendulum.hpp>
#include <src/bellman_operators/uniform_operator.hpp>

void RunUniformExperiment(size_t num_particles, size_t num_iterations, size_t num_pendulums) {
    std::mt19937 rd{1234};
    ActionConditionedKernel action_conditioned_kernel{Pendulum::Kernel<1>{num_pendulums, &rd},
                                                      Pendulum::Kernel<0>{num_pendulums, &rd},
                                                      Pendulum::Kernel<-1>{num_pendulums, &rd}};

    EnvParams env_params{action_conditioned_kernel, Pendulum::RewardFunc{}, 0.95};

    UniformBellmanOperator bellman_op{env_params, 2048, 1., &rd};
    for (size_t i = 0; i < num_iterations; ++i) {
        bellman_op.MakeIteration();
    }

    QFuncEstForGreedy qfunc_est{env_params, std::move(bellman_op.GetQFunc()),
                                // Correction for importance sampling
                                [](auto) { return 2.; }};
    GreedyPolicy policy{qfunc_est};
    MDPKernel mdp_kernel{action_conditioned_kernel, &policy};

    for (FloatT init : std::array{0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.}) {
        std::cout << "\n///////////////////////////////////////////"
                  << "\n";
        Particle state{ConstantInitializer(init, 1)};
        for (int i = 0; i < 10; ++i) {
            std::cout << state << " " << qfunc_est.ValueAtPoint(state) << "\n";
            mdp_kernel.Evolve(state, &state);
        }
        ASSERT_TRUE(state[0]);
    }
}
