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

    void SetStorage(ParticleStorage* storage) {
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
    ParticleStorage* storage_
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
class ZeroInitializer final : public VectorizingInitializer<ZeroInitializer<StorageT>, StorageT> {
public:
    ZeroInitializer(size_t dim) : BaseT{dim} {
    }

    ZeroInitializer(size_t dim, ParticleStorage* storage) : BaseT{dim, storage} {
    }

    friend class VectorizingInitializer<ZeroInitializer<StorageT>, StorageT>;

protected:
    using BaseT = VectorizingInitializer<ZeroInitializer<StorageT>, StorageT>;
    inline FloatT GetIthElemImpl(size_t) const {
        return 0;
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

    friend class BaseT;

private:
    inline FloatT GetIthElemImpl(size_t) const {
        return dist_(*rd_);
    }

    using BaseT =
        VectorizingInitializer<RandomVectorizingInitializer<RandomDistT, RandomDevT, StorageT>,
                               StorageT>;
    RandomDevT* rd_;
    RandomDistT dist_;
};
