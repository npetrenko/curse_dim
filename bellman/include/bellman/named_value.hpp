#pragma once

#include <utility>

template <class ValueType, class Tag>
class NamedValue {
public:
    explicit NamedValue(const ValueType& val) noexcept(noexcept(ValueType(val))) : value_(val) {
    }

    explicit NamedValue(ValueType&& val) noexcept(noexcept(ValueType(std::move(val))))
        : value_(std::move(val)) {
    }

    operator const ValueType&() const& noexcept {
        return value_;
    }

    operator ValueType&() & noexcept {
        return value_;
    }

    operator ValueType &&() && noexcept {
        return std::move(value_);
    }

private:
    ValueType value_;
};
