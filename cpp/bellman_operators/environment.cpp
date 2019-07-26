#include <include/bellman_operators/environment.hpp>

EnvParams::EnvParams(const IActionConditionedKernel& kernel, RewardFuncT reward, FloatT gamma)
    : ac_kernel{kernel.Clone()}, reward_function{std::move(reward)}, kGamma{gamma} {
}

EnvParams::EnvParams(const EnvParams& other)
    : reward_function(other.reward_function), kGamma(other.kGamma) {
    ac_kernel = other.ac_kernel->Clone();
}

EnvParams& EnvParams::operator=(const EnvParams& other) {
    ac_kernel = other.ac_kernel->Clone();
    reward_function = other.reward_function;
    kGamma = other.kGamma;
    return *this;
}
