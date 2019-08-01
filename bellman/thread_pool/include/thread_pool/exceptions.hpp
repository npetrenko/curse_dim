#pragma once

#include <exception>

class ThreadPoolExeption : public std::exception {
public:
    const char* what() const noexcept override {
	return "Unable to submit task to threadpool";
    }
};
