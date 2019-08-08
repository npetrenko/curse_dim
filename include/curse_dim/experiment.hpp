#pragma once

#include <bellman/bellman_operators/environment.hpp>
#include <bellman/bellman_operators/abstract_bellman.hpp>
#include <bellman/builder.hpp>
#include "throw_helpers.hpp"
#include "pendulum.hpp"
#include <chrono>
#include "exceptions.hpp"

EnvParams BuildEnvironment(NumPendulums num_pendulums, std::mt19937* rd);

using TargetNumIterations = NamedValue<size_t, class _TargetNumIterationsTag>;

class AbstractExperiment {
    using EWrapper = ThrowWrap<ExperimentNotRunException>;

public:
    enum class IterType { kExhaustion, kSingle };

    using DurT = std::chrono::milliseconds;
    class Builder;

    FloatT Score();

    DurT GetIterDuration() const;
    DurT GetSimDuration() const;
    void MakeIteration(IterType type);

    virtual std::string GetName() const = 0;

    virtual ~AbstractExperiment() = default;

    inline size_t GetNumParticles() const {
        return kParams.num_particles;
    }

    inline size_t GetNumIterations() const {
        return last_iteration_data_.iter_num;
    }

    inline size_t GetNumPendulums() const {
        return kParams.num_pendulums;
    }

    inline FloatT GetGamma() const {
        return kParams.environment.gamma;
    }

    inline FloatT GetScore() const {
        return EWrapper::Wrap(last_score_data_.score);
    }

    inline std::mt19937* GetRandomDevice() {
        return kParams.random_device.get();
    }

private:
    struct Params {
        NumParticles num_particles;
        BuilderOption<TargetNumIterations> target_num_iterations{"target_num_iterations"};
        NumPendulums num_pendulums;
        EnvParams environment;
        std::unique_ptr<std::mt19937> random_device;
        std::mt19937 random_device_for_scoring;
    };

protected:
    virtual std::pair<DurT, FloatT> RunScoring();

    IQFuncEstimate* EstimateQFunc();
    virtual IQFuncEstimate* EstimateQFuncImpl() = 0;
    virtual void MakeIterationImpl() = 0;

    AbstractExperiment(Params params);

    inline const EnvParams& GetEnvParams() const {
        return kParams.environment;
    }

private:
    const Params kParams;

    struct LastIterData {
        inline void Reset() {
            iter_duration = std::nullopt;
        }
        size_t iter_num{0};
        std::optional<DurT> iter_duration;
    };

    struct LastScoreData {
        inline void Reset() {
            score = std::nullopt;
            sim_duration = std::nullopt;
        }
        std::optional<FloatT> score;
        std::optional<DurT> sim_duration;
    };

    LastIterData last_iteration_data_;
    LastScoreData last_score_data_;
};

class AbstractExperiment::Builder {
public:
    Builder() = default;
    Builder& SetNumParticles(size_t val);
    Builder& SetTargetNumIterations(size_t val);
    Builder& SetNumPendulums(size_t val);
    Builder& SetRandomDevice(const std::mt19937& rd);

    inline std::mt19937* GetRandomDevice() {
        return random_device_.Value().rd_ptr.get();
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
    BuilderOption<NumParticles> num_particles_{"num_particles_"};
    BuilderOption<TargetNumIterations> num_iterations_{"num_iterations_"};
    BuilderOption<NumPendulums> num_pendulums_{"num_pendulums_"};
    BuilderOption<RDHolder> random_device_{"random_device_"};
};
