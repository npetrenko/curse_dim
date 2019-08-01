#pragma once

#include "experiment.hpp"

namespace UniformExperiment {
std::unique_ptr<AbstractExperiment> Make(AbstractExperiment::Builder builder);
}
