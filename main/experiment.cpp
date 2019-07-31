#include <main/experiment.hpp>

EnvParams BuildEnvironment(size_t num_pendulums, std::mt19937* rd) {
    ActionConditionedKernel action_conditioned_kernel{Pendulum::Kernel<-1>{num_pendulums, rd},
                                                      Pendulum::Kernel<0>{num_pendulums, rd},
                                                      Pendulum::Kernel<1>{num_pendulums, rd}};

    EnvParams env_params{action_conditioned_kernel, Pendulum::RewardFunc{}, 0.95};
    return env_params;
}

FloatT AbstractExperiment::Score() {
    throw NotImplementedError();
}

AbstractExperiment::AbstractExperiment(Params params) : params_(std::move(params)) {
}

using Builder = AbstractExperiment::Builder;

Builder& Builder::SetNumParticles(size_t val) {
    num_particles_ = val;
    return *this;
}

Builder& Builder::SetNumIterations(size_t val) {
    num_iterations_ = val;
    return *this;
}

Builder& Builder::SetNumPendulums(size_t val) {
    num_pendulums_ = val;
    return *this;
}

Builder& Builder::SetEnvironment(EnvParams env_params) {
    environment_ = std::move(env_params);
    return *this;
}

Builder& Builder::SetRandomDevice(std::mt19937* rd) {
    random_device_ = rd;
    return *this;
}

AbstractExperiment::Params Builder::Build() && {
    try {
        return {num_particles_.value(), num_iterations_.value(), num_pendulums_.value(),
                environment_.value(), random_device_.value()};
    } catch (std::bad_optional_access&) {
        throw BuilderNotInitialized();
    }
}
