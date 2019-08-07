#include <curse_dim/pendulum.hpp>
#include <curse_dim/experiment.hpp>
#include <bellman/bellman_operators/stationary_operator.hpp>
#include <glog/logging.h>

class StationaryExperimentImpl : public AbstractExperiment {
public:
    StationaryExperimentImpl(AbstractExperiment::Builder builder)
        : AbstractExperiment(std::move(builder).Build()) {
    }

    std::string GetName() const override {
        return "StationaryExperiment";
    }

public:
    void MakeIterationImpl() override {
        last_qfunc_est_.reset(nullptr);
        MaybeInitializeBellmanOp();
        bellman_op_->MakeIteration();
    }

    IQFuncEstimate* EstimateQFuncImpl() override {
        if (!bellman_op_) {
            throw ExperimentNotRunException();
        }
        if (last_qfunc_est_) {
            return last_qfunc_est_.get();
        }

        auto rew_helper = [sampling_distr =
                               bellman_op_->GetSamplingDistribution()](size_t sample_index) {
            PrevSampleReweighingHelper impl(&sampling_distr, std::nullopt);
            return impl(sample_index);
        };

        last_qfunc_est_ = std::unique_ptr<IQFuncEstimate>(
            new QFuncEstForGreedy(GetEnvParams(), bellman_op_->GetQFunc(), std::move(rew_helper)));
        return last_qfunc_est_.get();
    }

    void MaybeInitializeBellmanOp() {
        if (bellman_op_) {
            return;
        }

        VLOG(3) << "Initializing stationary bellman op";
        StationaryBellmanOperator::Builder builder;
        builder.SetInitRadius(M_PI)
            .SetEnvParams(GetEnvParams())
            .SetNumParticles(GetNumParticles())
            .SetRandomDevice(GetRandomDevice())
            .SetNumBurninIter(1)
            .SetDensityRatioThreshold(100.)
            .SetInvariantDensityThreshold(1e-3);
        bellman_op_ = std::move(builder).Build();
    }

    StationaryBellmanOperatorPtr bellman_op_{nullptr};
    std::unique_ptr<IQFuncEstimate> last_qfunc_est_{nullptr};
};

namespace StationaryExperiment {
std::unique_ptr<AbstractExperiment> Make(AbstractExperiment::Builder builder) {
    return std::make_unique<StationaryExperimentImpl>(std::move(builder));
}
}  // namespace StationaryExperiment
