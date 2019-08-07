#include <curse_dim/pendulum.hpp>
#include <curse_dim/experiment.hpp>
#include <bellman/bellman_operators/uniform_operator.hpp>
#include <glog/logging.h>

class UniformExperimentImpl final : public AbstractExperiment {
public:
    UniformExperimentImpl(AbstractExperiment::Builder builder)
        : AbstractExperiment(std::move(builder).Build()) {
    }

    std::string GetName() const override {
	return "UniformExperiment";
    }

private:
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

        auto importance_function = [=](size_t) { return 1 / pow(2 * M_PI, GetNumPendulums() * 2); };
        last_qfunc_est_ = std::unique_ptr<IQFuncEstimate>(
            new QFuncEstForGreedy(GetEnvParams(), bellman_op_->GetQFunc(), importance_function));
        return last_qfunc_est_.get();
    }

    void MaybeInitializeBellmanOp() {
        if (bellman_op_) {
            return;
        }
	VLOG(3) << "Initializing uniform bellman op";
        UniformBellmanOperator::Builder builder;
        builder.SetInitRadius(M_PI)
            .SetEnvParams(GetEnvParams())
            .SetNumParticles(GetNumParticles())
            .SetRandomDevice(GetRandomDevice());
        bellman_op_ = std::move(builder).Build();
    }

    UniformBellmanOperatorPtr bellman_op_{nullptr};
    std::unique_ptr<IQFuncEstimate> last_qfunc_est_{nullptr};
};

namespace UniformExperiment {
std::unique_ptr<AbstractExperiment> Make(AbstractExperiment::Builder builder) {
    return std::make_unique<UniformExperimentImpl>(std::move(builder));
}
}  // namespace UniformExperiment
