#pragma once

#include <string>
#include <exception>

class NotImplementedError : public std::exception {
    inline const char* what() const noexcept override {
	return "Not implemented";
    }
};

class OutOfStorage : public std::exception {
    inline const char* what() const noexcept override {
	return "Out of preallocated storage";
    }
};
