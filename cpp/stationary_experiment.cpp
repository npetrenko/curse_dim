#include <curse_dim/pendulum.hpp>
#include <curse_dim/experiment.hpp>
#include <bellman/bellman_operators/stationary_operator.hpp>
#include <glog/logging.h>

class StationaryExperimentImpl : public AbstractExperiment {
public:
    StationaryExperimentImpl(AbstractExperiment::Builder builder)
        : AbstractExperiment(std::move(builder).Build()) {
    }

    std::unique_ptr<IQFuncEstimate> EstimateQFunc() override {
        LOG(INFO) << "Initializing uniform bellman op";
        StationaryBellmanOperatorPtr bellman_op;
        {
            StationaryBellmanOperator::Builder builder;
            builder.SetInitRadius(M_PI)
                .SetEnvParams(GetEnvParams())
                .SetNumParticles(GetNumParticles())
                .SetRandomDevice(GetRandomDevice())
                .SetNumBurninIter(1)
                .SetDensityRatioThreshold(100.)
                .SetInvariantDensityThreshold(1e-3);
            bellman_op = std::move(builder).Build();
        }
        for (size_t i = 0; i < GetNumIterations(); ++i) {
            LOG(INFO) << "Started iter " << i;
            bellman_op->MakeIteration();
        }

        PrevSampleReweighingHelper rew_helper{&bellman_op->GetSamplingDistribution(), std::nullopt};
        QFuncEstForGreedy qfunc_est{GetEnvParams(), std::move(*bellman_op).GetQFunc(), rew_helper};
        return std::make_unique<std::remove_reference_t<decltype(qfunc_est)>>(std::move(qfunc_est));
    }
};

namespace StationaryExperiment {
std::unique_ptr<AbstractExperiment> Make(AbstractExperiment::Builder builder) {
    return std::make_unique<StationaryExperimentImpl>(std::move(builder));
}
}  // namespace StationaryExperiment
