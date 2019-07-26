#include <include/bellman_operators/qfunc.hpp>
DiscreteQFuncEst::DiscreteQFuncEst(size_t num_particles, size_t num_actions)
    : values_(num_particles * num_actions, 0), num_actions_(num_actions) {
}

void DiscreteQFuncEst::SetZero() {
    for (auto& val : values_) {
        val = 0;
    }
}

FloatT DiscreteQFuncEst::ValueAtPoint(TypeErasedParticleRef, size_t) const {
    throw NotImplementedError();
}

std::ostream& operator<<(std::ostream& stream, const DiscreteQFuncEst& est) {
    for (size_t i = 0; i < est.values_.size() / est.NumActions(); ++i) {
        stream << est.ValueAtIndex(i);
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
