#pragma once

#include <src/particle.hpp>
#include <src/util.hpp>

template<class DerivedT>
class AbstractAgentPolicy : public CRTPDerivedCaster<DerivedT> {
public:
    template <class S>
    size_t React(const Particle<S>& state) {
	return this->GetDerived()->React(state);
    }
};
