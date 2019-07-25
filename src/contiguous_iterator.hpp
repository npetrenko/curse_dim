#pragma once

#include <type_traits>
#include <cstdint>

using std::size_t;

template <class T, bool is_const>
class ContiguousStridedIterator {
public:
    using finT = std::conditional_t<is_const, const T*, T*>;
    using retT = std::conditional_t<is_const, const T&, T&>;
    using value_type = T;

    ContiguousStridedIterator(finT data, size_t stride) noexcept : data_(data), stride_(stride) {
    }

    inline retT operator*() const {
        return *data_;
    }

    inline ContiguousStridedIterator& operator++() {
        data_ += stride_;
        return *this;
    }

    inline ContiguousStridedIterator operator++(int) {
        ContiguousStridedIterator prev = *this;
        data_ += stride_;
        return prev;
    }

    inline ContiguousStridedIterator operator+(int64_t val) const {
        return {data_ + val*stride_};
    }

    template <class K, bool ic>
    inline bool operator!=(ContiguousStridedIterator<K, ic> other) {
        return other.data_ != data_;
    }

private:
    finT data_;
    size_t stride_;
};

template <class T, bool is_const>
class ContiguousIterator : public ContiguousStridedIterator<T, is_const> {
    using BaseT = ContiguousStridedIterator<T, is_const>;
public:
    ContiguousIterator(typename BaseT::finT data) noexcept : BaseT(data, 1) {
    }
};
