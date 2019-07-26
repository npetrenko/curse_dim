#pragma once

#include "../kernel.hpp"
#include "../agent_policy.hpp"

struct EnvParams {
    using RewardFuncT = std::function<FloatT(TypeErasedParticleRef, size_t)>;

    EnvParams() = default;

    EnvParams(const IActionConditionedKernel& kernel, RewardFuncT reward, FloatT gamma);

    EnvParams(const EnvParams& other);

    EnvParams(EnvParams&&) = default;

    EnvParams& operator=(EnvParams&&) = default;

    EnvParams& operator=(const EnvParams& other);

    std::unique_ptr<IActionConditionedKernel> ac_kernel;
    RewardFuncT reward_function;
    FloatT kGamma;
};
