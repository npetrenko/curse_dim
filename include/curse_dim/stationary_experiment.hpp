#pragma once

#include "experiment.hpp"

namespace StationaryExperiment {
std::unique_ptr<AbstractExperiment> Make(AbstractExperiment::Builder builder);
}
