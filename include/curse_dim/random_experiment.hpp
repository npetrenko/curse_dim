
#pragma once

#include "experiment.hpp"

namespace RandomExperiment {
std::unique_ptr<AbstractExperiment> Make(AbstractExperiment::Builder builder);
}
