#pragma once

#include "types.hpp"
#include "contiguous_iterator.hpp"

#include <type_traits>

using std::size_t;

template <bool is_const>
class StridedMemoryViewTemplate {
public:
    using iterator = ContiguousStridedIterator<FloatT, is_const>;
    using const_iterator = ContiguousStridedIterator<FloatT, true>;
    using DataPtrT = typename iterator::finT;
    using reference = typename iterator::retT;

    StridedMemoryViewTemplate(DataPtrT data, size_t size, size_t stride) noexcept : data_(data), size_(size), stride_(stride) {
    }

    reference operator[](size_t i) const {
	return *(data_ + i * stride_);
    }

    iterator begin() {
        return {data_, size_, stride_};
    }

    typename iterator::EndProxy end() const {
	return {};
    }

    size_t size() const {
	return size_;
    }

private:
    DataPtrT data_;
    size_t size_;
    size_t stride_;
};

#define CREATE_ALIAS(AName, Base, Modifier) \
    class AName : public Base<Modifier> {   \
    public:                                 \
        using Base<Modifier>::Base;         \
                                            \
        AName(Base origin) : Base{origin} { \
        }                                   \
    };

CREATE_ALIAS(StridedMemoryView, StridedMemoryViewTemplate, false)
CREATE_ALIAS(ConstStridedMemoryView, StridedMemoryViewTemplate, true)

template <bool is_const>
class MemoryViewTemplate : public std::conditional_t<is_const, ConstStridedMemoryView, StridedMemoryView> {
    using BaseT = std::conditional_t<is_const, ConstStridedMemoryView, StridedMemoryView>;

public:
    using iterator = ContiguousIterator<FloatT, is_const>;
    using const_iterator = ContiguousIterator<FloatT, true>;

    MemoryViewTemplate(typename BaseT::DataPtrT data, size_t size) noexcept : BaseT(data, size, 1) {
    }

    iterator begin() const {
        return {&(*this)[0]};
    }

    iterator end() const {
	return begin() + this->size();
    }
};

CREATE_ALIAS(MemoryView, MemoryViewTemplate, false)
CREATE_ALIAS(ConstMemoryView, MemoryViewTemplate, true)

#undef CREATE_ALIAS
