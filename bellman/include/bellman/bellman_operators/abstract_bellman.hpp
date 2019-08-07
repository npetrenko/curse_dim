#pragma once

#include "qfunc.hpp"
#include "environment.hpp"
#include "../density_estimators/weighted_particle_clusters.hpp"

class IBellmanOperator {
public:
    virtual void MakeIteration() = 0;
    virtual const DiscreteQFuncEst& GetQFunc() const& = 0;
    virtual DiscreteQFuncEst GetQFunc() && = 0;
    virtual const AbstractWeightedParticleCluster& GetSamplingDistribution() const = 0;
    virtual ~IBellmanOperator() = default;
};

class AbstractBellmanOperator : public IBellmanOperator {
public:
    template <class Derived>
    class Builder;

protected:
    struct Params {
        size_t num_particles;
        std::mt19937* random_device;
        EnvParams env_params;
    };

    inline size_t GetNumParticles() const {
        return kParams.num_particles;
    }

    inline std::mt19937* GetRD() const {
        return kParams.random_device;
    }

    inline const EnvParams& GetEnvParams() const {
        return kParams.env_params;
    }

    inline FloatT GetGamma() const {
        return kParams.env_params.gamma;
    }

protected:
    template <class Derived>
    AbstractBellmanOperator(Builder<Derived>&& builder);

private:
    const Params kParams;
};

template <class Derived>
class AbstractBellmanOperator::Builder : public CRTPDerivedCaster<Derived> {
    friend class AbstractBellmanOperator;

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
        } catch (std::bad_optional_access& e) {
            throw BuilderNotInitialized();
        }
    };

private:
    Params CreateParams() && {
        try {
            Params params{num_particles_.value(), random_device_.value(),
                          std::move(env_params_.value())};
            return params;
        } catch (std::bad_optional_access& e) {
            throw BuilderNotInitialized();
        }
    }

protected:
    std::optional<EnvParams> env_params_;
    std::optional<std::mt19937*> random_device_;
    std::optional<size_t> num_particles_;
};

template <class Derived>
AbstractBellmanOperator::AbstractBellmanOperator(Builder<Derived>&& builder)
    : kParams(std::move(builder).CreateParams()) {
}

template <class WPCType>
struct PrevSampleReweighingHelper {
    PrevSampleReweighingHelper(const WPCType* ps, std::optional<FloatT> default_den)
        : prev_sample{ps}, default_density{default_den} {
    }

    inline FloatT operator()(size_t sample_index) const {
        if (!prev_sample) {
            return default_density.value();
        }
        FloatT dens = prev_sample->GetWeights()[sample_index];
        if (dens < 1e-2) {
            return 0;
        }
        return 1. / dens;
    }

    const WPCType* const prev_sample;
    const std::optional<FloatT> default_density;
};
