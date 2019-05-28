#pragma once

#include <src/bellman.hpp>
#include <src/particle.hpp>
#include <src/bellman_operators/environment.hpp>

#include <optional>

class DiscreteQFuncEst : public AbstractQFuncEstimate<DiscreteQFuncEst> {
    friend class AbstractQFuncEstimate<DiscreteQFuncEst>;

public:
    DiscreteQFuncEst(size_t num_particles, size_t dim)
        : values_(num_particles * dim, 0), dim_(dim) {
    }

    void SetZero() {
        for (auto& val : values_) {
            val = 0;
        }
    }

    template <class RandomDistT, class RandomDeviceT>
    void SetRandom(RandomDeviceT* rd, RandomDistT distr) {
	for (auto& val: values_) {
	    val = distr(*rd);
	}
    }

    template <class ClusterT>
    void SetParticleCluster(ClusterT&& other) {
        particle_cluster_ = std::forward<ClusterT>(other);
    }

    const ParticleCluster& GetParticleCluster() const {
        return particle_cluster_.value();
    }

    ParticleCluster& GetParticleCluster() {
        return particle_cluster_.value();
    }

    size_t NumActions() const {
        return dim_;
    }

    friend std::ostream& operator<<(std::ostream&, const DiscreteQFuncEst&);

private:
    FloatT& ValueAtIndexImpl(size_t state_ix, size_t action_number) {
        return values_[state_ix * dim_ + action_number];
    }

    FloatT ValueAtIndexImpl(size_t state_ix, size_t action_number) const {
        return values_[state_ix * dim_ + action_number];
    }

protected:
    std::vector<FloatT> values_;
    size_t dim_;
    std::optional<ParticleCluster> particle_cluster_;
};

std::ostream& operator<<(std::ostream& stream, const DiscreteQFuncEst& est) {
    for (size_t i = 0; i < est.values_.size() / est.NumActions(); ++i) {
        stream << "{";
        for (size_t action_number = 0; action_number < est.NumActions(); ++action_number) {
            stream << est.ValueAtIndex(i, action_number) << ", ";
        }
        stream << "} ";
        if (est.particle_cluster_) {
            assert(i < est.particle_cluster_.value().size());
            stream << est.particle_cluster_.value()[i];
        } else {
            stream << "{}";
        }
        stream << "\n";
    }
    return stream;
}

template <class RewardFuncT, class ImportanceFuncT, class... T>
class QFuncEstForGreedy
    : public AbstractQFuncEstimate<QFuncEstForGreedy<RewardFuncT, ImportanceFuncT, T...>> {
    friend class AbstractQFuncEstimate<QFuncEstForGreedy<RewardFuncT, ImportanceFuncT, T...>>;

public:
    QFuncEstForGreedy(EnvParams<RewardFuncT, T...> env_params, DiscreteQFuncEst dqf,
                      ImportanceFuncT importance_func)
        : env_params_{std::move(env_params)},
          discrete_est_{std::move(dqf)},
          importance_func_{std::move(importance_func)} {
    }

    size_t NumActions() const {
        return discrete_est_.NumActions();
    }

    template <class S, class RF, class IF>
    friend std::ostream& operator<<(std::ostream&, const QFuncEstForGreedy<S, RF, IF>&);

private:
    template <class S>
    FloatT ValueAtPointImpl(const Particle<S>& state, size_t action) const {
        const ParticleCluster& cluster = discrete_est_.GetParticleCluster();
        GreedyPolicy greedy_policy(*this);

        FloatT result = env_params_.reward_function(state, action);
        for (size_t next_state_index = 0; next_state_index < cluster.size(); ++next_state_index) {
            const Particle<MemoryView>& next_state = cluster[next_state_index];
            size_t next_state_reaction = greedy_policy.React(next_state_index);
            result +=
                env_params_.kGamma *
                env_params_.ac_kernel.GetTransDensityConditionally(state, next_state, action) *
                this->ValueAtIndex(next_state_index, next_state_reaction) *
                importance_func_(next_state_index);
        }

        return result;
    }

    FloatT ValueAtIndexImpl(size_t index, size_t action) const {
        return discrete_est_.ValueAtIndex(index, action);
    }

    FloatT& ValueAtIndexImpl(size_t index, size_t action) {
        return discrete_est_.ValueAtIndex(index, action);
    }

    EnvParams<RewardFuncT, T...> env_params_;
    DiscreteQFuncEst discrete_est_;
    ImportanceFuncT importance_func_;
};

template <class T, class RF, class IF>
std::ostream& operator<<(std::ostream& stream, const QFuncEstForGreedy<T, RF, IF>& est) {
    return (stream << est.discrete_est_);
}
