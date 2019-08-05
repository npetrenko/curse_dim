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

    inline reference operator[](size_t i) const {
	return *(data_ + i * stride_);
    }

    inline iterator begin() {
        return {data_, size_, stride_};
    }

    inline typename iterator::EndProxy end() const {
	return {};
    }

    inline size_t size() const {
	return size_;
    }

private:
    DataPtrT data_;
    size_t size_;
    size_t stride_;
};

using StridedMemoryView = StridedMemoryViewTemplate<false>;
using ConstStridedMemoryView = StridedMemoryViewTemplate<true>;

template <bool is_const>
class MemoryViewTemplate : public StridedMemoryViewTemplate<is_const> {
    using BaseT = StridedMemoryViewTemplate<is_const>;

    template <bool ic>
    friend class MemoryViewTemplate;

public:
    using iterator = ContiguousIterator<FloatT, is_const>;
    using const_iterator = ContiguousIterator<FloatT, true>;

    inline MemoryViewTemplate(typename BaseT::DataPtrT data, size_t size) noexcept : BaseT(data, size, 1) {
    }

    inline MemoryViewTemplate(MemoryViewTemplate<false> other) noexcept : BaseT(&*other.begin(), other.size(), 1) {
    }

    inline iterator begin() const {
        return iterator{&(*this)[0]};
    }

    inline iterator end() const {
	return begin() + this->size();
    }
};

using MemoryView = MemoryViewTemplate<false>;
using ConstMemoryView = MemoryViewTemplate<true>;

