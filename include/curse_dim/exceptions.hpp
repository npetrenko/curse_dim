#pragma once

#include <exception>

class ExperimentNotRunException final : public std::exception {
public:
    const char* what() const noexcept override {
	return "Experiment has not been run yet, illegal action";
    }
};
