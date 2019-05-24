#pragma once

#include <vector>
#include <random>

#include <src/types.hpp>

class AbstractInitializer {
public:
    AbstractInitializer(size_t dim) : dim_(dim) {
    }
    virtual void Initialize(std::vector<FloatT>* data) const = 0;
    virtual ~AbstractInitializer() = default;

    inline size_t GetDim() const {
        return dim_;
    }

private:
    size_t dim_;
};

class VectorizingInitializer : public AbstractInitializer {
public:
    VectorizingInitializer(size_t dim) : AbstractInitializer(dim) {
    }

    inline void Initialize(std::vector<FloatT>* data) const override {
        data->resize(GetDim());
        for (size_t i = 0; i < data->size(); ++i) {
            (*data)[i] = GetIthElem(i);
        }
    }

protected:
    virtual FloatT GetIthElem(size_t i) const = 0;
};

class ZeroInitializer final : public VectorizingInitializer {
public:
    ZeroInitializer(size_t dim) : VectorizingInitializer(dim) {
    }

protected:
    inline FloatT GetIthElem(size_t) const override {
        return 0;
    }
};

template <class RandomDistT, class RandomDevT>
class RandomVectorizingInilializer final : public VectorizingInitializer {
public:
    RandomVectorizingInilializer(size_t dim, RandomDevT* rd, RandomDistT dist)
        : VectorizingInitializer(dim), rd_(rd), dist_(dist) {
    }

protected:
    inline FloatT GetIthElem(size_t) const override {
        return dist_(*rd_);
    }

private:
    RandomDevT* rd_;
    RandomDistT dist_;
};
