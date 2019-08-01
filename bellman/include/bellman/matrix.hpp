#pragma once

#include <vector>
#include <array>
#include <cassert>
#include <utility>

#include "types.hpp"

using MatrixDims = std::array<uint32_t, 2>;

template <class StorageT = std::vector<FloatT>>
class Matrix {
public:
    using index_type = uint32_t;
    using value_type = typename StorageT::value_type;

    Matrix() = default;

    Matrix(MatrixDims dims) : dims_(dims), data_(dims[0]*dims[1]) {
    }

    template <class ST>
    Matrix(ST&& storage, MatrixDims dims) : dims_(dims), data_(std::forward<ST>(storage)) {
	assert(data_.size() == dims[0] * dims[1]);
    }

    value_type& operator()(index_type row, index_type column) {
        return data_[row * dims_[1] + column];
    }

    const value_type& operator()(index_type row, index_type column) const {
        return data_[row * dims_[1] + column];
    }

    MatrixDims Dims() const {
	return dims_;
    }
private:
    MatrixDims dims_;
    StorageT data_;
};

template <class ST>
Matrix(ST&&, MatrixDims)->Matrix<std::remove_cv_t<std::remove_reference_t<ST>>>;
