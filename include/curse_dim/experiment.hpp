#pragma once

#include <bellman/bellman_operators/environment.hpp>
#include <bellman/bellman_operators/abstract_bellman.hpp>
#include "pendulum.hpp"
#include <chrono>
#include "exceptions.hpp"
#include "throw_helpers.hpp"

EnvParams BuildEnvironment(NumPendulums num_pendulums, std::mt19937* rd);

using NumIterations = NamedValue<size_t, class _NumIterationsTag>;

class AbstractExperiment {
    using EWrapper = ThrowWrap<ExperimentNotRunException>;
public:
    using DurT = std::chrono::milliseconds;
    class Builder;

    FloatT Score();
    std::unique_ptr<IQFuncEstimate> EstimateQFunc();
    DurT GetQFuncEstDuration() const;
    DurT GetSimDuration() const;
    virtual std::string GetName() const = 0;

    virtual ~AbstractExperiment() = default;

    inline size_t GetNumParticles() const {
        return kParams.num_particles;
    }

    inline size_t GetNumIterations() const {
        return kParams.num_iterations;
    }

    inline size_t GetNumPendulums() const {
        return kParams.num_pendulums;
    }

    inline FloatT GetGamma() const {
	return kParams.environment.gamma;
    }

    inline FloatT GetScore() const {
	return EWrapper::Wrap(score_);
    }

    inline std::mt19937* GetRandomDevice() {
	return kParams.random_device.get();
    }

private:
    struct Params {
        NumParticles num_particles;
        NumIterations num_iterations;
        NumPendulums num_pendulums;
        EnvParams environment;
	std::unique_ptr<std::mt19937> random_device;
	const std::mt19937 random_device_for_scoring;
    };

protected:
    virtual std::unique_ptr<IQFuncEstimate> EstimateQFuncImpl() = 0;

    AbstractExperiment(Params params);

    inline const EnvParams& GetEnvParams() const {
        return kParams.environment;
    }

private:
    const Params kParams;
    std::optional<DurT> qfunc_est_duration_;
    std::optional<DurT> sim_duration_;
    std::optional<FloatT> score_;
};

class AbstractExperiment::Builder  {
    using EWrapper = ThrowWrap<BuilderNotInitialized>;
public:
    Builder& SetNumParticles(size_t val);
    Builder& SetNumIterations(size_t val);
    Builder& SetNumPendulums(size_t val);
    Builder& SetRandomDevice(const std::mt19937& rd);

    inline std::mt19937* GetRandomDevice() {
	return EWrapper::Wrap(random_device_).rd_ptr.get();
    }

    Params Build() &&;

private:
    struct RDHolder {
	std::unique_ptr<std::mt19937> rd_ptr;
        RDHolder(std::unique_ptr<std::mt19937> ptr) noexcept : rd_ptr(std::move(ptr)) {
        }

        inline RDHolder(const RDHolder& other) {
	    rd_ptr = std::make_unique<std::mt19937>(*other.rd_ptr);
	}

	inline RDHolder& operator=(const RDHolder& other) {
	    rd_ptr = std::make_unique<std::mt19937>(*other.rd_ptr);
	    return *this;
	}

	RDHolder(RDHolder&&) = default;
	RDHolder& operator=(RDHolder&&) = default;
    };
    std::optional<NumParticles> num_particles_;
    std::optional<NumIterations> num_iterations_;
    std::optional<NumPendulums> num_pendulums_;
    std::optional<RDHolder> random_device_;
};
