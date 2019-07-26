#pragma once

#include "particle.hpp"

class IAgentPolicy {
public:
    virtual size_t React(TypeErasedParticleRef state) const = 0;
    virtual size_t React(size_t state_index) const = 0;
    virtual ~IAgentPolicy() = default;
};
