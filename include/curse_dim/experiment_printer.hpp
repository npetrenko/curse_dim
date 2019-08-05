#pragma once

#include "experiment.hpp"

class Printer {
public:
    static std::ostream& PrintHeader(std::ostream& stream);
    std::ostream& PrintBody(std::ostream& stream, const AbstractExperiment& experiment);
private:
};

