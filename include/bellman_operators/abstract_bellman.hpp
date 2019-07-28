#pragma once

#include "qfunc.hpp"
#include "../density_estimators/stationary_estimator.hpp"
#include "environment.hpp"

class AbstractBellmanOperator {
public:
    template <class Derived>
    class Builder;

    virtual void MakeIteration() = 0;
    virtual const DiscreteQFuncEst& GetQFunc() const = 0;
    virtual const AbstractWeightedParticleCluster& GetSamplingDistribution() const = 0;
    virtual ~AbstractBellmanOperator() = default;
};

template <class Derived>
class AbstractBellmanOperator::Builder : public CRTPDerivedCaster<Derived> {
public:
    Builder() = default;

    Derived& SetEnvParams(EnvParams params) {
	env_params_ = std::move(params);
	return *this->GetDerived();
    }

    Derived& SetRandomDevice(std::mt19937* rd) {
	random_device_ = rd;
	return *this->GetDerived();
    }

    Derived& SetNumParticles(size_t num_particles) {
	num_particles_ = num_particles;
	return *this->GetDerived();
    }

    auto Build() && {
        try {
            return static_cast<Derived&&>(*this).BuildImpl();
        } catch (std::bad_optional_access&) {
            throw BuilderNotInitialized();
        }
    };

protected:
    std::optional<EnvParams> env_params_;
    std::optional<std::mt19937*> random_device_;
    std::optional<size_t> num_particles_;
};
