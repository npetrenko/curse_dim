#pragma once

#include <vector>
#include <random>

#include "types.hpp"
#include "util.hpp"
#include "type_traits.hpp"
#include "particle_storage.hpp"

struct ClusterInitializationTag {};

template <class DerivedT, class StorageT>
class AbstractInitializer : public CRTPDerivedCaster<DerivedT> {
public:
    explicit AbstractInitializer(ParticleDim dim) noexcept : dim_{dim} {
    }

    AbstractInitializer(ParticleDim dim, ClusterInitializationTag) noexcept : dim_{dim} {
    }

    AbstractInitializer(ParticleDim dim, ParticleStorage* storage) noexcept
        : storage_{storage}, dim_{dim} {
    }

    inline void SetStorage(ParticleStorage* storage) const {
        storage_ = storage;
    }

    template <class Container>
    inline void Initialize(Container* data) const {
        this->GetDerived()->Initialize(data);
    }

    inline StorageT CreateStorage() const {
        if constexpr (std::is_same_v<StorageT, MemoryView>) {
            assert(storage_);
            return storage_->AllocateForParticle(dim_);
        } else {
            static_assert(std::is_same_v<StorageT, ParticleStorage>,
                          "Initializers can only use ParticleStorage or MemoryView");
            return ParticleStorage{dim_};
        }
    }

    inline size_t GetDim() const {
        return dim_;
    }

private:
    mutable ParticleStorage* storage_
#ifndef NDEBUG
    {
        nullptr
    }
#endif
    ;
    size_t dim_;
};

template <class StorageT>
class EmptyInitializer : public AbstractInitializer<EmptyInitializer<StorageT>, StorageT> {
    using BaseT = AbstractInitializer<EmptyInitializer<StorageT>, StorageT>;
public:
    using BaseT::BaseT;

    template <class Container>
    inline void Initialize(Container*) const {
    }
};

EmptyInitializer(ParticleDim)->EmptyInitializer<ParticleStorage>;
EmptyInitializer(ParticleDim, ClusterInitializationTag)->EmptyInitializer<MemoryView>;
EmptyInitializer(ParticleDim, ParticleStorage*)->EmptyInitializer<MemoryView>;

template <class OtherContainer>
class ValueInitializer : public AbstractInitializer<ValueInitializer<MemoryView>, MemoryView> {
public:
    ValueInitializer(const OtherContainer& other, ParticleStorage* storage)
        : BaseT{ParticleDim{other.GetDim()}, storage}, other_{other} {
    }

    template <class Container>
    inline void Initialize(Container* data) const {
        size_t i = 0;
        for (auto& elem : *data) {
            elem = other_[i];
            ++i;
        }
    }

private:
    using BaseT = AbstractInitializer<ValueInitializer<MemoryView>, MemoryView>;
    const OtherContainer& other_;
};

template <class DerivedT, class StorageT>
class VectorizingInitializer
    : public CRTPDerivedCaster<DerivedT>,
      public AbstractInitializer<VectorizingInitializer<DerivedT, StorageT>, StorageT> {

    using BaseT = AbstractInitializer<VectorizingInitializer<DerivedT, StorageT>, StorageT>;
public:
    using BaseT::BaseT;

public:
    template <class Container>
    inline void Initialize(Container* data) const {
        size_t i = 0;
        for (auto& elem : *data) {
            elem = GetIthElem(i);
            ++i;
        }
    }

protected:
    inline FloatT GetIthElem(size_t i) const {
        return CRTPDerivedCaster<DerivedT>::GetDerived()->GetIthElemImpl(i);
    }
};

template <class StorageT, class FuncT>
class LambdaInitializer : public VectorizingInitializer<LambdaInitializer<StorageT, FuncT>, StorageT> {
    friend class VectorizingInitializer<LambdaInitializer<StorageT, FuncT>, StorageT>;
    using BaseT = VectorizingInitializer<LambdaInitializer<StorageT, FuncT>, StorageT>;
public:
    LambdaInitializer(FuncT func, ParticleDim dim) : BaseT(dim), func_(std::move(func)) {
    }

    LambdaInitializer(FuncT func, ParticleDim dim, ClusterInitializationTag) : BaseT(dim), func_(std::move(func)) {
    }

    LambdaInitializer(FuncT func, ParticleDim dim, ParticleStorage* storage) : BaseT(dim, storage), func_(std::move(func)) {
    }
protected:
    inline FloatT GetIthElemImpl(size_t i) const {
	return func_(i);
    }
private:
    FuncT func_;
};

template <class FuncT>
LambdaInitializer(FuncT, ParticleDim)->LambdaInitializer<ParticleStorage, FuncT>;

template <class FuncT>
LambdaInitializer(FuncT, ParticleDim, ClusterInitializationTag)->LambdaInitializer<MemoryView, FuncT>;

template <class FuncT>
LambdaInitializer(FuncT, ParticleDim, ParticleStorage*)->LambdaInitializer<MemoryView, FuncT>;

template <class StorageT>
class ConstantInitializer : public VectorizingInitializer<ConstantInitializer<StorageT>, StorageT> {
public:
    ConstantInitializer(FloatT ct, ParticleDim dim) : BaseT{dim}, ct_{ct} {
    }

    ConstantInitializer(FloatT ct, ParticleDim dim, ClusterInitializationTag) : BaseT{dim}, ct_{ct} {
    }

    ConstantInitializer(FloatT ct, ParticleDim dim, ParticleStorage* storage)
        : BaseT{dim, storage}, ct_{ct} {
    }

    friend class VectorizingInitializer<ConstantInitializer<StorageT>, StorageT>;

protected:
    using BaseT = VectorizingInitializer<ConstantInitializer<StorageT>, StorageT>;
    inline FloatT GetIthElemImpl(size_t) const {
        return ct_;
    }
    FloatT ct_;
};

ConstantInitializer(FloatT, ParticleDim)->ConstantInitializer<ParticleStorage>;
ConstantInitializer(FloatT, ParticleDim, ClusterInitializationTag)->ConstantInitializer<MemoryView>;
ConstantInitializer(FloatT, ParticleDim, ParticleStorage*)->ConstantInitializer<MemoryView>;

template <class StorageT>
class ZeroInitializer : public ConstantInitializer<StorageT> {
public:
    ZeroInitializer(ParticleDim dim) : BaseT{0., dim} {
    }

    ZeroInitializer(ParticleDim dim, ClusterInitializationTag) : BaseT{0., dim} {
    }

    ZeroInitializer(ParticleDim dim, ParticleStorage* storage) : BaseT{0., dim, storage} {
    }

private:
    using BaseT = ConstantInitializer<StorageT>;
};
ZeroInitializer(ParticleDim)->ZeroInitializer<ParticleStorage>;
ZeroInitializer(ParticleDim, ClusterInitializationTag)->ZeroInitializer<MemoryView>;
ZeroInitializer(ParticleDim, ParticleStorage*)->ZeroInitializer<MemoryView>;

template <class StorageT, class RandomDistT, class RandomDevT>
class RandomVectorizingInitializer final
    : public VectorizingInitializer<RandomVectorizingInitializer<StorageT, RandomDistT, RandomDevT>,
                                    StorageT> {
public:
    RandomVectorizingInitializer(ParticleDim dim, RandomDevT* rd, RandomDistT dist)
        : BaseT{dim}, rd_{rd}, dist_{std::move(dist)} {
    }

    RandomVectorizingInitializer(ParticleDim dim, RandomDevT* rd, RandomDistT dist, ClusterInitializationTag)
        : BaseT{dim}, rd_{rd}, dist_{std::move(dist)} {
    }

    RandomVectorizingInitializer(ParticleDim dim, ParticleStorage* storage, RandomDevT* rd,
                                 RandomDistT dist)
        : BaseT{dim, storage}, rd_{rd}, dist_{std::move(dist)} {
    }

    friend class VectorizingInitializer<
        RandomVectorizingInitializer<StorageT, RandomDistT, RandomDevT>, StorageT>;

private:
    inline FloatT GetIthElemImpl(size_t) const {
        return dist_(*rd_);
    }

    using BaseT =
        VectorizingInitializer<RandomVectorizingInitializer<StorageT, RandomDistT, RandomDevT>,
                               StorageT>;
    RandomDevT* rd_;
    mutable RandomDistT dist_;
};

template <class RandomDistT, class RandomDevT>
RandomVectorizingInitializer(ParticleDim, RandomDevT*, RandomDistT)
    ->RandomVectorizingInitializer<ParticleStorage, RandomDistT, RandomDevT>;

template <class RandomDistT, class RandomDevT>
RandomVectorizingInitializer(ParticleDim, RandomDevT*, RandomDistT, ClusterInitializationTag)
    ->RandomVectorizingInitializer<MemoryView, RandomDistT, RandomDevT>;

template <class RandomDistT, class RandomDevT>
RandomVectorizingInitializer(ParticleDim, ParticleStorage*, RandomDevT*, RandomDistT)
    ->RandomVectorizingInitializer<MemoryView, RandomDistT, RandomDevT>;
