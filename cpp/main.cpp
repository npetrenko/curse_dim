#include <curse_dim/experiment.hpp>
#include <curse_dim/uniform_experiment.hpp>
#include <curse_dim/stationary_experiment.hpp>
#include <curse_dim/experiment_printer.hpp>

#include <glog/logging.h>
#include <gflags/gflags.h>

#include <fstream>

DEFINE_string(output_file, "/tmp/output.csv", "File to output experiment results to");

int main(int argc, char** argv) {
    google::LogToStderr();
    google::InitGoogleLogging(argv[0]);
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    std::mt19937 rd{1234};
    Printer printer;

    std::ofstream output_file;
    output_file.open(FLAGS_output_file.c_str(), std::ios::trunc);

    printer.PrintHeader(output_file);
    for (const size_t kNumPendulums : {2}) {
        constexpr size_t kNumIterations = 4;
        for (const size_t kNumParticles : {128, 256, 2048*16}) {
            AbstractExperiment::Builder builder;
            builder.SetNumIterations(kNumIterations)
                .SetNumParticles(kNumParticles)
                .SetNumPendulums(kNumPendulums)
                .SetRandomDevice(std::mt19937(rd));

            std::unique_ptr uniform_experiment = UniformExperiment::Make(builder);
            uniform_experiment->Score();
            printer.PrintBody(output_file, *uniform_experiment);

            std::unique_ptr stationary_experiment = StationaryExperiment::Make(builder);
            stationary_experiment->Score();
            printer.PrintBody(output_file, *stationary_experiment);
        }
    }
    return 0;
}
