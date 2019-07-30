#pragma once

#include "../kernel.hpp"
#include "../agent_policy.hpp"

struct EnvParams {
    using RewardFuncT = std::function<FloatT(TypeErasedParticleRef, size_t)>;

    EnvParams() = default;

    template <class ACKernel>
    EnvParams(const ACKernel& kernel, RewardFuncT reward, FloatT gamma) : ac_kernel(kernel.Clone()), reward_function(std::move(reward)), gamma(gamma) {
	mdp_kernel = std::make_unique<MDPKernel<ACKernel>>(kernel, nullptr);
	static_assert(MDPKernel<ACKernel>::holds_kernel_by_value, "It should hold by value here");
    }

    EnvParams(const EnvParams& other);

    EnvParams(EnvParams&&) = default;

    EnvParams& operator=(EnvParams&&) = default;

    EnvParams& operator=(const EnvParams& other);

    std::unique_ptr<IActionConditionedKernel> ac_kernel;
    std::unique_ptr<IMDPKernel> mdp_kernel;
    RewardFuncT reward_function;
    FloatT gamma;
};
