#include <curse_dim/pendulum.hpp>
#include <curse_dim/random_experiment.hpp>
#include <glog/logging.h>

class RandomQFuncEstimate final
    : public EnableClone<RandomQFuncEstimate, InheritFrom<IQFuncEstimate>> {
private:
    static constexpr size_t kMaxStateSize = 1024;

public:
    RandomQFuncEstimate(::NumActions num_actions) : num_actions_(num_actions) {
    }

    FloatT ValueAtPoint(TypeErasedParticleRef state, size_t action) const override {
        auto randomize = [action](FloatT val) {
            val *= 1e6 * action + 1.5 * action;
            val -= static_cast<int64_t>(val);
            return val;
        };

        FloatT result = 0;
        for (const FloatT val : state) {
            result += randomize(val);
        }

        return result;
    }

    ConstMemoryView ValueAtIndex(size_t) const override {
        throw NotImplementedError();
    }

    MemoryView ValueAtIndex(size_t) override {
        throw NotImplementedError();
    }

    size_t NumActions() const override {
        return num_actions_;
    }

private:
    size_t num_actions_;
};

class RandomExperimentImpl final : public AbstractExperiment {
public:
    RandomExperimentImpl(AbstractExperiment::Builder builder)
        : AbstractExperiment(std::move(builder).Build()) {
        qfunc_est_ = std::make_unique<RandomQFuncEstimate>(
            ::NumActions(GetEnvParams().ac_kernel->GetNumActions()));
    }

    std::string GetName() const override {
        return "RandomExperiment";
    }

private:
    IQFuncEstimate* EstimateQFuncImpl() override {
        return qfunc_est_.get();
    }

    using ScoringResultT = std::pair<DurT, FloatT>;

    std::unique_ptr<IQFuncEstimate> qfunc_est_;
    std::optional<ScoringResultT> prev_score_;

protected:
    ScoringResultT RunScoring() override {
        if (!prev_score_) {
            VLOG(3) << "Creating score result for RandomExperiment";
            prev_score_ = AbstractExperiment::RunScoring();
            return *prev_score_;
        }
        VLOG(3) << "Reusing prev scoring results";
        return *prev_score_;
    }

    void MakeIterationImpl() override {
    }
};

namespace RandomExperiment {
std::unique_ptr<AbstractExperiment> Make(AbstractExperiment::Builder builder) {
    return std::make_unique<RandomExperimentImpl>(std::move(builder));
}
}  // namespace RandomExperiment
