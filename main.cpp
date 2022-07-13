#include "cs/SIG.hpp"
#include "headers/Particle.hpp"
#include "geometry/Universe.hpp"
#include "geometry/Plane.hpp"

#include <alpaka/alpaka.hpp>
#include <alpaka/example/ExampleDefaultAcc.hpp>

#include <cstdint>
#include <cstdlib>
#include <iostream>

using Dim = alpaka::DimInt<1>;

struct Kernel
{
    template<typename TAcc>
    ALPAKA_FN_ACC auto operator()(
        TAcc const& acc,
		mxmc::SIG* const ptrHostParticles) const -> void
    {
        // Get the global linearized thread idx.
        auto const globalThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
        auto const globalThreadExtent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);

        auto const linearizedGlobalThreadIdx = alpaka::mapIdx<1u>(globalThreadIdx, globalThreadExtent)[0];

        if (linearizedGlobalThreadIdx == 1)
        {
				ptrHostParticles[1].get_fission();
        }

        auto engine = alpaka::rand::engine::createDefault(
            acc,
            linearizedGlobalThreadIdx,
            0);

        auto dist(alpaka::rand::distribution::createUniformReal<float>(acc));

        float random_value = dist(engine);


    }
};

void simpleParticlesInitialisation(mxmc::Particle* const ptrHostParticles, size_t num_particles)
{
	for (size_t i = 0; i < num_particles; i++)
		ptrHostParticles[i] = mxmc::Particle(0., 1., 2.);
}

double calculate_weights(mxmc::Particle* const ptrHostParticles, size_t num_particles)
{
	double sum_weights = 0;
	for (size_t i = 0; i < num_particles; i++)
		sum_weights += ptrHostParticles[i].weight;

	return sum_weights;
}

auto main() -> int
{


    using Dim = alpaka::DimInt<1>;
    using Idx = std::size_t;
    using Vec = alpaka::Vec<Dim, Idx>;
   // using Acc = alpaka::ExampleDefaultAcc<Dim, Idx>;
    using Host = alpaka::DevCpu;
    using Acc = alpaka::AccCpuSerial<Dim, Idx>;
    std::cout << "Using alpaka accelerator: " << alpaka::getAccName<Acc>() << std::endl;
    auto const dev_host = alpaka::getDevByIdx<Host>(0u);
    auto const dev_acc = alpaka::getDevByIdx<Acc>(0u);


    using QueueProperty = alpaka::Blocking;
    using QueueAcc = alpaka::Queue<Acc, QueueProperty>;
    QueueAcc queue{dev_acc};

    Idx number_groups = 10;


    mxmc::SIG_storage test_storage(number_groups, dev_acc, dev_host);

    using WorkDiv = alpaka::WorkDivMembers<Dim, Idx>;

    constexpr size_t numThreads = 100u;
    constexpr size_t numAlpakaElementsPerThread = 1;

    WorkDiv workdiv{alpaka::getValidWorkDiv<Acc>(
    		dev_acc,
        Vec(numThreads),
        Vec(numAlpakaElementsPerThread),
        false,
        alpaka::GridBlockExtentSubDivRestrictions::Unrestricted)};

    auto device_sig = test_storage.get_pointer_host();
    for (int i = 0; i < number_groups; i++)
    	device_sig[i].set_fission(12.2);

    test_storage.copy_from_host_to_device(queue);
    auto pointer_to_device = test_storage.get_pointer_device();

    Kernel kernel;
    alpaka::exec<Acc>(queue, workdiv, kernel, pointer_to_device);
    test_storage.copy_from_device_to_host(queue);


    return EXIT_SUCCESS;
}
