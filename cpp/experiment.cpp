#include <curse_dim/experiment.hpp>
#include <mutex>
#include <thread_pool/for_loop.hpp>
#include <glog/logging.h>

EnvParams BuildEnvironment(NumPendulums num_pendulums, std::mt19937* rd) {
    ActionConditionedKernel action_conditioned_kernel{Pendulum::Kernel<-1>{num_pendulums, rd},
                                                      Pendulum::Kernel<0>{num_pendulums, rd},
                                                      Pendulum::Kernel<1>{num_pendulums, rd}};

    EnvParams env_params{action_conditioned_kernel, Pendulum::RewardFunc{}, 0.95};
    return env_params;
}

FloatT AbstractExperiment::Score() {
    const size_t kNumRuns = 1000;
    if (last_score_data_.score) {
        return *last_score_data_.score;
    }

    IQFuncEstimate* qfunc_est = EstimateQFunc();
    if (!qfunc_est) {
        throw ExperimentNotRunException();
    }

    VLOG(3) << "Making scoring";
    std::vector<std::mt19937> rds;
    rds.reserve(kNumRuns);
    {
	std::mt19937 init = kParams.random_device_for_scoring;
	for (size_t i = 0; i < kNumRuns; ++i) {
	    rds.emplace_back(init());
	}
    }

    GreedyPolicy policy{*qfunc_est};
    MDPKernel mdp_kernel{*GetEnvParams().ac_kernel, &policy};

    std::mutex mut;
    FloatT sum_score = 0;
    auto scorer = [&](size_t i) {
        FloatT reward = 0;
        Particle state{ConstantInitializer(0., ParticleDim{mdp_kernel.GetSpaceDim()})};
        Particle evolve_to(state);

	const int kNumSteps = 200;
        for (int step = 0; step < kNumSteps; ++step) {
            size_t action = mdp_kernel.CalculateHint(state);
            reward += GetEnvParams().reward_function(state, action);
            mdp_kernel.EvolveWithHint(state, &evolve_to, &rds[i], &action);
            std::copy(evolve_to.begin(), evolve_to.end(), state.begin());
        }
        std::lock_guard lock(mut);
        sum_score += reward / kNumSteps;
    };
    ParallelFor{0, kNumRuns - 1, 255}(scorer);

    auto begin_time = std::chrono::high_resolution_clock::now();
    scorer(kNumRuns - 1);
    auto end_time = std::chrono::high_resolution_clock::now();
    last_score_data_.sim_duration = std::chrono::duration_cast<DurT>(end_time - begin_time);
    last_score_data_.score = sum_score / kNumRuns;
    VLOG(3) << "Finished scoring";
    return *last_score_data_.score;
}

void AbstractExperiment::MakeIteration(IterType type) {
    last_iteration_data_.Reset();
    last_score_data_.Reset();
    size_t num_iter;
    switch (type) {
        case IterType::kSingle:
            num_iter = 1;
            break;
        case IterType::kExhaustion:
            num_iter = kParams.target_num_iterations.value();
            break;
        default:
            throw std::runtime_error("Unknown IterType");
    }
    try {
        if (num_iter) {
            auto begin_time = std::chrono::high_resolution_clock::now();
            MakeIterationImpl();
            auto end_time = std::chrono::high_resolution_clock::now();
            last_iteration_data_.iter_duration =
                std::chrono::duration_cast<DurT>(end_time - begin_time);
            --num_iter;
	    ++last_iteration_data_.iter_num;
        }
    } catch(...) {
	last_iteration_data_.Reset();
	throw;
    }

    for (size_t iter = 0; iter < num_iter; ++iter) {
	++last_iteration_data_.iter_num;
	MakeIterationImpl();
    }
}

AbstractExperiment::DurT AbstractExperiment::GetIterDuration() const {
    return EWrapper::Wrap(last_iteration_data_.iter_duration);
}

AbstractExperiment::DurT AbstractExperiment::GetSimDuration() const {
    return EWrapper::Wrap(last_score_data_.sim_duration);
}

AbstractExperiment::AbstractExperiment(Params params) : kParams(std::move(params)) {
}

IQFuncEstimate* AbstractExperiment::EstimateQFunc() {
    auto ret = EstimateQFuncImpl();
    return ret;
}

using Builder = AbstractExperiment::Builder;

Builder& Builder::SetNumParticles(size_t val) {
    num_particles_ = NumParticles(val);
    return *this;
}

Builder& Builder::SetTargetNumIterations(size_t val) {
    num_iterations_ = TargetNumIterations(val);
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
                num_iterations_,
                num_pendulums_.value(),
                std::move(environment),
                std::move(random_device_.value().rd_ptr),
                scoring_rd};
    } catch (std::bad_optional_access&) {
        throw BuilderNotInitialized();
    }
}
