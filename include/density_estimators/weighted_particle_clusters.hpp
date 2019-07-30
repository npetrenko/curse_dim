#pragma once

#include "../particle.hpp"
#include "../memory_view.hpp"

class AbstractWeightedParticleCluster : public ParticleCluster {
public:
    using ParticleCluster::ParticleCluster;
    explicit AbstractWeightedParticleCluster(ParticleCluster cluster)
        : ParticleCluster{std::move(cluster)} {
    }

    virtual ConstStridedMemoryView IGetWeights() const = 0;
    virtual StridedMemoryView IGetMutableWeights() = 0;
    virtual ~AbstractWeightedParticleCluster() = default;
};

class VectorWeightedParticleCluster final : public AbstractWeightedParticleCluster {
public:
    template <class S>
    VectorWeightedParticleCluster(size_t size,
                                  const AbstractInitializer<S, MemoryView>& initializer)
        : AbstractWeightedParticleCluster{size, initializer}, weights_(size) {
    }

    inline ConstStridedMemoryView IGetWeights() const override {
        return GetWeights();
    }

    inline StridedMemoryView IGetMutableWeights() override {
        return GetMutableWeights();
    }

    inline ConstMemoryView GetWeights() const {
        return ConstMemoryView{weights_.data(), weights_.size()};
    }

    inline MemoryView GetMutableWeights() {
        return MemoryView{weights_.data(), weights_.size()};
    }

private:
    std::vector<FloatT> weights_;
};

class ConstantWeightedParticleCluster final : public AbstractWeightedParticleCluster {
public:
    template <class S>
    ConstantWeightedParticleCluster(size_t size,
                                    const AbstractInitializer<S, MemoryView>& initializer,
                                    FloatT weighing_constant)
        : AbstractWeightedParticleCluster{size, initializer},
          weighing_constant_{weighing_constant} {
    }

    ConstantWeightedParticleCluster(ParticleCluster cluster, FloatT weighing_constant)
        : AbstractWeightedParticleCluster{std::move(cluster)},
          weighing_constant_{weighing_constant} {
    }

    inline StridedMemoryView IGetMutableWeights() override {
	return GetMutableWeights();
    }

    inline ConstStridedMemoryView IGetWeights() const override {
	return GetWeights();
    }

    inline StridedMemoryView GetMutableWeights() {
        throw std::runtime_error(
            "GetMutableWeights for ConstantWeightedParticleCluster makes no sense");
    }

    inline ConstStridedMemoryView GetWeights() const {
        return {&weighing_constant_, this->size(), 0};
    }

private:
    const FloatT weighing_constant_;
};
