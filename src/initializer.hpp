#pragma once

#include <vector>
#include <random>

#include <src/types.hpp>
#include <src/util.hpp>
#include <src/particle_storage.hpp>

template <class DerivedT, class StorageT>
class AbstractInitializer : public CRTPDerivedCaster<DerivedT> {
public:
    AbstractInitializer(size_t dim) noexcept : dim_(dim) {
        static_assert(std::is_same_v<StorageT, ParticleStorage>,
                      "Cannot create View Initializer without pointer to ParticleStorage");
    }

    AbstractInitializer(size_t dim, ParticleStorage* storage) noexcept : storage_(storage) , dim_(dim) {
    }

    template <class Container>
    void Initialize(Container* data) const {
        this->GetDerived()->Initialize(data);
    }

    StorageT CreateStorage() const {
        if constexpr (std::is_same_v<StorageT, MemoryView>) {
            return storage_->AllocateForParticle(dim_);
        } else {
            return ParticleStorage{dim_};
        }
    }

    inline size_t GetDim() const {
        return dim_;
    }

private:
    ParticleStorage* storage_;
    size_t dim_;
};

template <class DerivedT, class StorageT>
class VectorizingInitializer : public CRTPDerivedCaster<DerivedT>, public AbstractInitializer<VectorizingInitializer<DerivedT, StorageT>, StorageT> {
public:
    VectorizingInitializer(size_t dim)
        : AbstractInitializer<VectorizingInitializer<DerivedT, StorageT>, StorageT>{dim} {
    }

    VectorizingInitializer(size_t dim, ParticleStorage* storage) : AbstractInitializer<VectorizingInitializer<DerivedT, StorageT>, StorageT>(dim, storage) {
    }

    template <class Container>
    inline void Initialize(Container* data) const {
        for (size_t i = 0; i < data->size(); ++i) {
            (*data)[i] = GetIthElem(i);
        }
    }

protected:
    FloatT GetIthElem(size_t i) const {
        return CRTPDerivedCaster<DerivedT>::GetDerived()->GetIthElemImpl(i);
    }
};

template <class StorageT>
class ZeroInitializer final : public VectorizingInitializer<ZeroInitializer<StorageT>, StorageT> {
public:
    ZeroInitializer(size_t dim) : VectorizingInitializer<ZeroInitializer<StorageT>, StorageT>(dim) {
    }

    ZeroInitializer(size_t dim, ParticleStorage* storage) : VectorizingInitializer<ZeroInitializer<StorageT>, StorageT>(dim, storage) {
    }

    friend class VectorizingInitializer<ZeroInitializer<StorageT>, StorageT>;
protected:
    inline FloatT GetIthElemImpl(size_t) const {
        return 0;
    }
};

ZeroInitializer(size_t) -> ZeroInitializer<ParticleStorage>;
ZeroInitializer(size_t, ParticleStorage*) -> ZeroInitializer<MemoryView>;

/*
template <class RandomDistT, class RandomDevT>
class RandomVectorizingInilializer final
    : public VectorizingInitializer<RandomVectorizingInilializer<RandomDistT, RandomDevT>> {
public:
    RandomVectorizingInilializer(size_t dim, RandomDevT* rd, RandomDistT dist)
        : VectorizingInitializer<RandomVectorizingInilializer<RandomDistT, RandomDevT>>(dim),
          rd_(rd),
          dist_(dist) {
    }

protected:
    inline FloatT GetIthElem(size_t) const {
        return dist_(*rd_);
    }

private:
    RandomDevT* rd_;
    RandomDistT dist_;
};
*/
