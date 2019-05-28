#pragma once

#include <src/particle.hpp>

template<class DerivedT>
class AbstractAgentPolicy : public CRTPDerivedCaster<DerivedT> {
public:
    template <class S>
    inline size_t React(const Particle<S>& state) {
	return this->GetDerived()->React(state);
    }

    inline size_t React(size_t state_index) {
	return this->GetDerived()->React(state_index);
    }
};

#include <src/util.hpp>
