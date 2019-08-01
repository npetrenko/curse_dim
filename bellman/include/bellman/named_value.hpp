#pragma once

#include <utility>

template <class ValueType, class Tag>
class NamedValue {
public:
    explicit NamedValue(const ValueType& val) : value_(val) {
    }

    explicit NamedValue(ValueType&& val) : value_(std::move(val)) {
    }

    operator const ValueType&() const& {
        return value_;
    }

    operator ValueType&() & {
        return value_;
    }

    operator ValueType &&() && {
        return std::move(value_);
    }

private:
    ValueType value_;
};
