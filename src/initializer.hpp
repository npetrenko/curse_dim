#pragma once

#include <vector>
#include <random>

#include <src/types.hpp>
#include <src/util.hpp>
#include <src/particle_storage.hpp>

template <class DerivedT, class StorageT>
class AbstractInitializer : public CRTPDerivedCaster<DerivedT> {
public:
    AbstractInitializer(size_t dim) noexcept : dim_{dim} {
    }

    AbstractInitializer(size_t dim, ParticleStorage* storage) noexcept
        : storage_{storage}, dim_{dim} {
    }

    void SetStorage(ParticleStorage* storage) const {
        storage_ = storage;
    }

    template <class Container>
    void Initialize(Container* data) const {
        this->GetDerived()->Initialize(data);
    }

    StorageT CreateStorage() const {
        if constexpr (std::is_same_v<StorageT, MemoryView>) {
            assert(storage_);
            return storage_->AllocateForParticle(dim_);
        } else {
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
public:
    EmptyInitializer(size_t dim) : BaseT{dim} {
    }

    EmptyInitializer(size_t dim, ParticleStorage* storage) : BaseT{dim, storage} {
    }

    template <class Container>
    inline void Initialize(Container*) const {
    }

private:
    using BaseT = AbstractInitializer<EmptyInitializer<StorageT>, StorageT>;
};

template <class OtherContainer>
class ValueInitializer : public AbstractInitializer<ValueInitializer<MemoryView>, MemoryView> {
public:
    ValueInitializer(const OtherContainer& other, ParticleStorage* storage)
        : BaseT{other.GetDim(), storage}, other_{other} {
    }

    template <class Container>
    inline void Initialize(Container* data) const {
	for (size_t i = 0; i < data->size(); ++i) {
	    (*data)[i] = other_[i];
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
public:
    VectorizingInitializer(size_t dim) : BaseT{dim} {
    }

    VectorizingInitializer(size_t dim, ParticleStorage* storage) : BaseT{dim, storage} {
    }

    template <class Container>
    inline void Initialize(Container* data) const {
        for (size_t i = 0; i < data->size(); ++i) {
            (*data)[i] = GetIthElem(i);
        }
    }

protected:
    using BaseT = AbstractInitializer<VectorizingInitializer<DerivedT, StorageT>, StorageT>;
    FloatT GetIthElem(size_t i) const {
        return CRTPDerivedCaster<DerivedT>::GetDerived()->GetIthElemImpl(i);
    }
};

template <class StorageT>
class ConstantInitializer : public VectorizingInitializer<ConstantInitializer<StorageT>, StorageT> {
public:
    ConstantInitializer(FloatT ct, size_t dim) : BaseT{dim}, ct_{ct} {
    }

    ConstantInitializer(FloatT ct, size_t dim, ParticleStorage* storage)
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

ConstantInitializer(FloatT, size_t)->ConstantInitializer<ParticleStorage>;
ConstantInitializer(FloatT, size_t, ParticleStorage*)->ConstantInitializer<MemoryView>;

template <class StorageT>
class ZeroInitializer : public ConstantInitializer<StorageT> {
public:
    ZeroInitializer(size_t dim) : ConstantInitializer<StorageT>{0., dim} {
    }
    ZeroInitializer(size_t dim, ParticleStorage* storage) : ConstantInitializer<StorageT>{0., dim, storage} {
    }
};
ZeroInitializer(size_t)->ZeroInitializer<ParticleStorage>;
ZeroInitializer(size_t, ParticleStorage*)->ZeroInitializer<MemoryView>;

template <class StorageT, class RandomDistT, class RandomDevT>
class RandomVectorizingInitializer final
    : public VectorizingInitializer<RandomVectorizingInitializer<StorageT, RandomDistT, RandomDevT>,
                                    StorageT> {
public:
    RandomVectorizingInitializer(size_t dim, RandomDevT* rd, RandomDistT dist)
        : BaseT{dim}, rd_{rd}, dist_{std::move(dist)} {
    }

    RandomVectorizingInitializer(size_t dim, ParticleStorage* storage, RandomDevT* rd,
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
