#include <src/initializer.hpp>
#include <src/particle.hpp>
#include <src/kernel.hpp>

template <int step>
class DetermKernel : public AbstractKernel<DetermKernel<step>> {
public:
    inline void Evolve(const Particle& from, Particle* to) const {
        for (size_t i = 0; i < from.GetDim(); ++i) {
            (*to)[i] = from[i] + step;
        }
    }
};

class DummyPolicy : public AbstractAgentPolicy {
public:
    size_t React(const Particle&) override {
        return (step++) != 0;
    }

private:
    size_t step = 0;
};

int main() {
    DummyPolicy policy;
    MDPKernel kernel{ActionConditionedKernel{DetermKernel<1>(), DetermKernel<2>()}, &policy};

    Particle test_particle{ZeroInitializer(8)};

    kernel.Evolve(test_particle, &test_particle);
    kernel.Evolve(test_particle, &test_particle);

    std::cout << test_particle << "\n";
    return 0;
}
