#pragma once

#include "../particle.hpp"
#include "../cloneable.hpp"
#include "../bellman.hpp"
#include "environment.hpp"

#include <optional>

class DiscreteQFuncEst final : public EnableClone<DiscreteQFuncEst, InheritFrom<IQFuncEstimate>> {
public:
    DiscreteQFuncEst() = default;

    DiscreteQFuncEst(size_t num_particles, size_t num_actions);

    void SetZero();

    template <class RandomDistT>
    void SetRandom(std::mt19937* rd, RandomDistT distr) {
        for (auto& val : values_) {
            val = distr(*rd);
        }
    }

    template <class ClusterT>
    void SetParticleCluster(ClusterT&& other) {
        particle_cluster_ = std::forward<ClusterT>(other);
    }

    inline const ParticleCluster& GetParticleCluster() const {
        return particle_cluster_.value();
    }

    inline ParticleCluster& GetParticleCluster() {
        return particle_cluster_.value();
    }

    inline size_t NumActions() const override {
        return num_actions_;
    }

    virtual FloatT ValueAtPoint(TypeErasedParticleRef, size_t) const override;

    inline ConstMemoryView ValueAtIndex(size_t state_ix) const override {
        return {&values_[state_ix * num_actions_], num_actions_};
    }

    inline MemoryView ValueAtIndex(size_t state_ix) override {
        return {&values_[state_ix * num_actions_], num_actions_};
    }

    friend std::ostream& operator<<(std::ostream&, const DiscreteQFuncEst&);

protected:
    std::vector<FloatT> values_;
    size_t num_actions_;
    std::optional<ParticleCluster> particle_cluster_;
};

template <class ImportanceFuncT>
class QFuncEstForGreedy
    : public EnableClone<QFuncEstForGreedy<ImportanceFuncT>, InheritFrom<IQFuncEstimate>> {

public:
    QFuncEstForGreedy() = default;

    QFuncEstForGreedy(EnvParams env_params, DiscreteQFuncEst dqf, ImportanceFuncT importance_func)
        : env_params_{std::move(env_params)},
          discrete_est_{std::move(dqf)},
          importance_func_{std::move(importance_func)} {
    }

    size_t NumActions() const override {
        return discrete_est_.NumActions();
    }

    template <class IF>
    friend std::ostream& operator<<(std::ostream&, const QFuncEstForGreedy<IF>&);

    Particle<ParticleStorage> ValueAtPoint(TypeErasedParticleRef state) const {
        Particle<ParticleStorage> result{
            EmptyInitializer<ParticleStorage>{ParticleDim{NumActions()}}};
        for (size_t i = 0; i < NumActions(); ++i) {
            result[i] = ValueAtPoint(state, i);
        }
        return result;
    }

    ConstMemoryView ValueAtIndex(size_t index) const override {
        return discrete_est_.ValueAtIndex(index);
    }

    MemoryView ValueAtIndex(size_t index) override {
        return discrete_est_.ValueAtIndex(index);
    }

    FloatT ValueAtPoint(TypeErasedParticleRef state, size_t action) const override {
        const ParticleCluster& cluster = discrete_est_.GetParticleCluster();
        GreedyPolicy greedy_policy(*this);

        FloatT result = env_params_.reward_function(state, action);
        FloatT weight_sum = 0;
        for (size_t next_state_index = 0; next_state_index < cluster.size(); ++next_state_index) {
            const Particle<MemoryView>& next_state = cluster[next_state_index];
            FloatT weight =
                env_params_.ac_kernel->GetTransDensityConditionally(state, next_state, action) *
                importance_func_(next_state_index);
            weight_sum += weight;

            size_t next_state_reaction = greedy_policy.React(next_state_index);

            result += env_params_.gamma * weight *
                      this->ValueAtIndex(next_state_index)[next_state_reaction];
        }

        result /= weight_sum;

        return result;
    }

private:
    const EnvParams env_params_;
    DiscreteQFuncEst discrete_est_;
    const ImportanceFuncT importance_func_;
};

std::ostream& operator<<(std::ostream& stream, const DiscreteQFuncEst& est);

template <class IF>
inline std::ostream& operator<<(std::ostream& stream, const QFuncEstForGreedy<IF>& est) {
    return (stream << est.discrete_est_);
}
