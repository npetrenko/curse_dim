#include <gtest/gtest.h>
#include <src/particle.hpp>
#include <src/particle_storage.hpp>

TEST(Cluster, CopyConstruct) {
    ParticleCluster first_cluster{2, ZeroInitializer<MemoryView>(1)};
    ParticleCluster second_cluster{first_cluster};

    for (auto& part : second_cluster) {
        part[0] = 1.;
    }

    for (const auto& part: first_cluster) {
	ASSERT_EQ(part[0], 0.);
    }
}

TEST(Cluster, CopyAssing) {
    ParticleCluster first_cluster{2, ZeroInitializer<MemoryView>(1)};
    ParticleCluster second_cluster{1, ConstantInitializer<MemoryView>(1., 2)};

    second_cluster = first_cluster;

    for (auto& part : second_cluster) {
	ASSERT_EQ(part.GetDim(), 1);
        part[0] = 1.;
    }

    for (const auto& part: first_cluster) {
	ASSERT_EQ(part.GetDim(), 1);
	ASSERT_EQ(part[0], 0.);
    }

    ASSERT_EQ(second_cluster.size(), 2);
}

TEST(Cluster, MoveConstruct) {
    ParticleCluster first_cluster{2, ZeroInitializer<MemoryView>(1)};
    ParticleCluster second_cluster{std::move(first_cluster)};

    for (const auto& part: second_cluster) {
	ASSERT_EQ(part.GetDim(), 1);
	ASSERT_EQ(part[0], 0.);
    }

    ASSERT_EQ(second_cluster.size(), 2);
}

TEST(Cluster, MoveAssign) {
    ParticleCluster first_cluster{2, ZeroInitializer<MemoryView>(1)};
    ParticleCluster second_cluster{1, ConstantInitializer<MemoryView>(1., 2)};

    second_cluster = std::move(first_cluster);

    for (const auto& part: second_cluster) {
	ASSERT_EQ(part.GetDim(), 1);
	ASSERT_EQ(part[0], 0.);
    }

    ASSERT_EQ(second_cluster.size(), 2);
}