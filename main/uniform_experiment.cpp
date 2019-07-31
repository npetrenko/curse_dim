#include <main/pendulum.hpp>
#include <main/experiment.hpp>
#include <include/bellman_operators/uniform_operator.hpp>

class UniformExperimentImpl : public AbstractExperiment {
public:
    UniformExperimentImpl(AbstractExperiment::Builder builder)
        : AbstractExperiment(std::move(builder).Build()) {
    }

    std::unique_ptr<IQFuncEstimate> EstimateQFunc() override {
        UniformBellmanOperatorPtr bellman_op;
        {
            UniformBellmanOperator::Builder builder;
            builder.SetInitRadius(M_PI)
                .SetEnvParams(GetEnvParams())
                .SetNumParticles(GetNumParticles())
                .SetRandomDevice(GetRandomDevice());
            bellman_op = std::move(builder).Build();
        }
        for (size_t i = 0; i < GetNumIterations(); ++i) {
            bellman_op->MakeIteration();
        }

        auto importance_function = [=](size_t) { return 1 / pow(2 * M_PI, GetNumPendulums() * 2); };
        QFuncEstForGreedy qfunc_est(GetEnvParams(), std::move(*bellman_op).GetQFunc(),
                                    importance_function);
        return std::make_unique<std::remove_reference_t<decltype(qfunc_est)>>(std::move(qfunc_est));
    }
};

namespace UniformExperiment {
std::unique_ptr<AbstractExperiment> Make(AbstractExperiment::Builder builder) {
    return std::make_unique<UniformExperimentImpl>(std::move(builder));
}
}  // namespace UniformExperiment
