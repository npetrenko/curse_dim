#pragma once

#include <include/bellman_operators/environment.hpp>
#include <include/bellman_operators/abstract_bellman.hpp>
#include <main/pendulum.hpp>

EnvParams BuildEnvironment(size_t num_pendulums, std::mt19937* rd);

class AbstractExperiment {
public:
    class Builder;

    FloatT Score();

    virtual std::unique_ptr<IQFuncEstimate> EstimateQFunc() = 0;
    virtual ~AbstractExperiment() = default;

private:
    struct Params {
        size_t num_particles;
        size_t num_iterations;
        size_t num_pendulums;
        EnvParams environment;
        std::mt19937* random_device;
    };

protected:
    AbstractExperiment(Params params);

    inline size_t GetNumParticles() const {
        return params_.num_particles;
    }

    inline size_t GetNumIterations() const {
        return params_.num_iterations;
    }

    inline size_t GetNumPendulums() const {
        return params_.num_pendulums;
    }

    inline const EnvParams& GetEnvParams() const {
        return params_.environment;
    }

    inline std::mt19937* GetRandomDevice() const {
	return params_.random_device;
    }

private:
    const Params params_;
};

class AbstractExperiment::Builder {
public:
    Builder& SetNumParticles(size_t val);
    Builder& SetNumIterations(size_t val);
    Builder& SetNumPendulums(size_t val);
    Builder& SetEnvironment(EnvParams env_params);
    Builder& SetRandomDevice(std::mt19937* rd);

    Params Build() &&;

private:
    std::optional<size_t> num_particles_;
    std::optional<size_t> num_iterations_;
    std::optional<size_t> num_pendulums_;
    std::optional<EnvParams> environment_;
    std::optional<std::mt19937*> random_device_;
};
