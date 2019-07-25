#pragma once

#include "types.hpp"
#include "contiguous_iterator.hpp"

#include <type_traits>

using std::size_t;

template <bool is_const>
class StridedMemoryViewTemplate {
public:
    using iterator = ContiguousStridedIterator<FloatT, false>;
    using const_iterator = ContiguousStridedIterator<FloatT, true>;
    using DataPtrT = typename iterator::finT;
    using reference = typename iterator::retT;

    StridedMemoryViewTemplate(DataPtrT data, size_t size, size_t stride) noexcept : data_(data), size_(size), stride_(stride) {
    }

    reference operator[](size_t i) const {
	return *(data_ + i * stride_);
    }

    iterator begin() {
        return {data_, stride_};
    }

    iterator end() {
	return begin() + size_;
    }

    const_iterator begin() const {
        return {data_, stride_};
    }

    const_iterator end() const {
	return begin() + size_;
    }

private:
    DataPtrT data_;
    size_t size_;
    size_t stride_;
};

template <bool is_const>
class MemoryViewTemplate : public StridedMemoryViewTemplate<is_const> {
    using BaseT = StridedMemoryViewTemplate<is_const>;

public:
    MemoryViewTemplate(typename BaseT::DataPtrT data, size_t size) noexcept : BaseT(data, size, 1) {
    }
};

#define CREATE_ALIAS(AName, Base)           \
    class AName : public Base {             \
    public:                                 \
        using Base::Base;                   \
                                            \
        AName(Base origin) : Base{origin} { \
        }                                   \
    };

CREATE_ALIAS(MemoryView, MemoryViewTemplate<false>)
CREATE_ALIAS(ConstMemoryView, MemoryViewTemplate<true>)
CREATE_ALIAS(StridedMemoryView, StridedMemoryViewTemplate<false>)
CREATE_ALIAS(ConstStridedMemoryView, StridedMemoryViewTemplate<true>)

#undef CREATE_ALIAS

