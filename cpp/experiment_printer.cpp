#include <curse_dim/experiment_printer.hpp>

std::ostream& Printer::PrintHeader(std::ostream& stream) {
    stream << "ExperimentName,Score,NumParticles,NumPendulums,NumIterations,Gamma,IterDur(ms),"
              "SimDur(ms)\n";
    return stream;
}

std::ostream& Printer::PrintBody(std::ostream& stream, const AbstractExperiment& experiment) {
    auto print_ms = [](std::ostream& stream, std::chrono::milliseconds ms) {
        stream << ms.count();
    };

    stream << experiment.GetName() << "," << experiment.GetScore() << ","
           << experiment.GetNumParticles() << "," << experiment.GetNumPendulums() << ","
           << experiment.GetNumIterations() << "," << experiment.GetGamma();

    stream << ",";
    print_ms(stream, experiment.GetIterDuration());
    stream << ",";
    print_ms(stream, experiment.GetSimDuration());
    stream << "\n";
    return stream;
}
