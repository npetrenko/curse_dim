#pragma once

#include "type_traits.hpp"
#include <cstdint>

using std::size_t;

template <class T, bool is_const>
class ContiguousStridedIterator;

template <class T, bool is_const>
class ContiguousStridedIterator {
public:
    class EndProxy {
        bool operator!=(ContiguousStridedIterator<T, is_const>) const;
    };

    using finT = std::conditional_t<is_const, const T*, T*>;
    using retT = std::conditional_t<is_const, const T&, T&>;
    using value_type = T;

    ContiguousStridedIterator(finT data, size_t range_size, size_t stride) noexcept
        : data_(data), stride_(stride), range_size_(range_size) {
    }

    inline retT operator*() const {
        return *data_;
    }

    inline ContiguousStridedIterator& operator++() {
        data_ += stride_;
        ++current_range_position_;
        return *this;
    }

    inline ContiguousStridedIterator operator++(int) {
        ContiguousStridedIterator prev = *this;
        data_ += stride_;
        ++current_range_position_;
        return prev;
    }

    inline ContiguousStridedIterator operator+(int64_t val) const {
        return {data_ + val * stride_};
    }

    template <class K, bool ic>
    inline bool operator!=(EndProxy) const {
        return range_size_ != current_range_position_;
    }

private:
    finT data_;
    size_t stride_;
    size_t current_range_position_{0};
    size_t range_size_;
};

template <class T, bool ic>
bool ContiguousStridedIterator<T, ic>::EndProxy::operator!=(
    ContiguousStridedIterator<T, ic> iter) const {
    return iter != EndProxy{};
}

template <class T, bool is_const>
class ContiguousIterator {
public:
    template <class S, bool c>
    friend class ContiguousIterator;

    using pointer = std::conditional_t<is_const, const T*, T*>;
    using reference = std::conditional_t<is_const, const T&, T&>;
    using value_type = T;
    using difference_type = void;
    using iterator_category = std::forward_iterator_tag;    

    explicit ContiguousIterator(pointer data) noexcept : data_(data) {
    }

    ContiguousIterator(const ContiguousIterator<T, false>& other) noexcept {
	data_ = other.data_;
    }

    inline reference operator*() const {
        return *data_;
    }

    inline ContiguousIterator& operator++() {
        ++data_;
        return *this;
    }

    inline ContiguousIterator operator++(int) {
        ContiguousIterator prev = *this;
        ++(*this);
        return prev;
    }

    inline ContiguousIterator operator+(int64_t val) const {
        return ContiguousIterator{data_ + val};
    }

    template <class K, bool ic>
    inline bool operator!=(ContiguousIterator<K, ic> other) const {
        return &*other != data_;
    }

private:
    pointer data_;
};
