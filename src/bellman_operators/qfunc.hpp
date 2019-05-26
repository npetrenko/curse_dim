#pragma once

#include <src/bellman.hpp>
#include <src/particle.hpp>

#include <optional>

class DiscreteQFuncEst : public AbstractQFuncEstimate<DiscreteQFuncEst> {
public:
    DiscreteQFuncEst(size_t num_particles, size_t dim)
        : values_(num_particles * dim, 0), dim_(dim) {
    }

    FloatT& ValueAtIndex(size_t state_ix, size_t action_number) {
        return values_[state_ix * dim_ + action_number];
    }

    FloatT ValueAtIndex(size_t state_ix, size_t action_number) const {
        return values_[state_ix * dim_ + action_number];
    }

    void SetZero() {
        for (auto& val : values_) {
            val = 0;
        }
    }

    template<class ClusterT>
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

protected:
    std::vector<FloatT> values_;
    size_t dim_;
    std::optional<ParticleCluster> particle_cluster_;
};
std::ostream& operator<<(std::ostream& stream, const DiscreteQFuncEst& est) {
    for (size_t i = 0; i < est.values_.size(); ++i) {
        stream << "{";
        for (size_t action_number = 0; action_number < est.NumActions(); ++action_number) {
            stream << est.ValueAtIndex(i, action_number) << ", ";
        }
        stream << "} ";
        if (est.particle_cluster_) {
            stream << est.particle_cluster_.value()[i];
        } else {
            stream << "{}";
        }
        stream << "\n";
    }
    return stream;
}

template <class T, class ImportanceFuncT>
class QFuncEstForGreedy : public AbstractQFuncEstimate<QFuncEstForGreedy<T, ImportanceFuncT>> {
public:
    QFuncEstForGreedy(const AbstractConditionedKernel<T>& conditioned_kernel,
                      DiscreteQFuncEst dqf,
                      ImportanceFuncT importance_func)
        : discrete_est_{std::move(dqf)},
          conditioned_kernel_{conditioned_kernel},
          importance_func_{std::move(importance_func)} {
    }

    template <class S>
    FloatT ValueAtPoint(const Particle<S>& state, size_t action) const {
        const ParticleCluster& cluster = discrete_est_.GetParticleCluster();
        GreedyPolicy greedy_policy(*this);
        FloatT result = 0;
        for (size_t i = 0; i < cluster.size(); ++i) {
            const Particle<MemoryView>& next_state = cluster[i];
            size_t next_state_reaction = greedy_policy.React(i);
            result += conditioned_kernel_.GetTransDensityConditionally(state, next_state, action) *
                      this->ValueAtIndex(i, next_state_reaction) * importance_func_(next_state);
        }

        return result;
    }

    FloatT ValueAtIndex(size_t index, size_t action) const {
	return discrete_est_.ValueAtIndex(index, action);
    }

    FloatT& ValueAtIndex(size_t index, size_t action) {
	return discrete_est_.ValueAtIndex(index, action);
    }

    size_t NumActions() const {
	return discrete_est_.NumActions();
    }

    template <class S, class IF>
    friend std::ostream& operator<<(std::ostream&, const QFuncEstForGreedy<S,IF>&);

private:
    DiscreteQFuncEst discrete_est_;
    const AbstractConditionedKernel<T>& conditioned_kernel_;
    ImportanceFuncT importance_func_;
};

template <class T, class IF>
std::ostream& operator<<(std::ostream& stream, const QFuncEstForGreedy<T, IF>& est) {
    return (stream << est.discrete_est_);
}
