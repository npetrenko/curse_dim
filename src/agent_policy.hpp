#include <src/particle.hpp>

class AbstractAgentPolicy {
public:
    virtual size_t React(const Particle& state) = 0;
    virtual ~AbstractAgentPolicy() = default;
};
