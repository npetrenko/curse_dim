#include <curse_dim/experiment.hpp>

EnvParams BuildEnvironment(NumPendulums num_pendulums, std::mt19937* rd) {
    ActionConditionedKernel action_conditioned_kernel{Pendulum::Kernel<-1>{num_pendulums, rd},
                                                      Pendulum::Kernel<0>{num_pendulums, rd},
                                                      Pendulum::Kernel<1>{num_pendulums, rd}};

    EnvParams env_params{action_conditioned_kernel, Pendulum::RewardFunc{}, 0.95};
    return env_params;
}

FloatT AbstractExperiment::Score() {
    const size_t kNumRuns = 1000;
    if (score_) {
        return *score_;
    }
    std::mt19937 scoring_rd = kParams.random_device_for_scoring;

    std::unique_ptr<IQFuncEstimate> qfunc_est = EstimateQFunc();
    GreedyPolicy policy{*qfunc_est};
    MDPKernel mdp_kernel{*GetEnvParams().ac_kernel, &policy};

    auto scorer = [&] {
        FloatT reward = 0;
        Particle state{ConstantInitializer(0., ParticleDim{mdp_kernel.GetSpaceDim()})};
        Particle evolve_to(state);

        for (int i = 0; i < 10; ++i) {
            size_t action = mdp_kernel.CalculateHint(state);
            reward += GetEnvParams().reward_function(state, action);
            mdp_kernel.EvolveWithHint(state, &evolve_to, &scoring_rd, &action);
	    std::copy(evolve_to.begin(), evolve_to.end(), state.begin());
        }
        return reward;
    };

    score_ = 0;
    auto begin_time = std::chrono::high_resolution_clock::now();
    score_ = scorer();
    auto end_time = std::chrono::high_resolution_clock::now();
    sim_duration_ = std::chrono::duration_cast<DurT>(end_time - begin_time);
    for (size_t i = 0; i < kNumRuns - 1; ++i) {
        score_.value() += scorer();
    }
    score_.value() /= kNumRuns;
    return *score_;
}

AbstractExperiment::DurT AbstractExperiment::GetQFuncEstDuration() const {
    return EWrapper::Wrap(qfunc_est_duration_);
}

AbstractExperiment::DurT AbstractExperiment::GetSimDuration() const {
    return EWrapper::Wrap(sim_duration_);
}

AbstractExperiment::AbstractExperiment(Params params) : kParams(std::move(params)) {
}

std::unique_ptr<IQFuncEstimate> AbstractExperiment::EstimateQFunc() {
    auto begin_time = std::chrono::high_resolution_clock::now();
    auto ret = EstimateQFuncImpl();
    auto end_time = std::chrono::high_resolution_clock::now();
    qfunc_est_duration_ = std::chrono::duration_cast<DurT>(end_time - begin_time);
    return ret;
}

using Builder = AbstractExperiment::Builder;

Builder& Builder::SetNumParticles(size_t val) {
    num_particles_ = NumParticles(val);
    return *this;
}

Builder& Builder::SetNumIterations(size_t val) {
    num_iterations_ = NumIterations(val);
    return *this;
}

Builder& Builder::SetNumPendulums(size_t val) {
    num_pendulums_ = NumPendulums(val);
    return *this;
}

Builder& Builder::SetRandomDevice(const std::mt19937& rd) {
    random_device_ = RDHolder{std::make_unique<std::mt19937>(rd)};
    return *this;
}

AbstractExperiment::Params Builder::Build() && {
    try {
        EnvParams environment =
            BuildEnvironment(num_pendulums_.value(), random_device_.value().rd_ptr.get());
        std::mt19937 scoring_rd = *random_device_.value().rd_ptr;
        return {num_particles_.value(),
                num_iterations_.value(),
                num_pendulums_.value(),
                std::move(environment),
                std::move(random_device_.value().rd_ptr),
                scoring_rd};
    } catch (std::bad_optional_access&) {
        throw BuilderNotInitialized();
    }
}
