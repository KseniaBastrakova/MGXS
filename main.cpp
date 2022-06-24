#include "cs/SIG.hpp"
#include "headers/Particle.hpp"
#include "geometry/Universe.hpp"
#include "geometry/Plane.hpp"

#include <alpaka/alpaka.hpp>
#include <alpaka/example/ExampleDefaultAcc.hpp>

#include <cstdint>
#include <cstdlib>
#include <iostream>

struct Function
{
    template<typename TAcc>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc, float const x) -> float
    {
        return alpaka::math::sqrt(acc, (1.0f - x * x));
    }
};

struct Kernel
{
    template<typename TAcc>
    ALPAKA_FN_ACC auto operator()(
        TAcc const& acc,
		mxmc::Particle* const particles) const -> void
    {
        // Get the global linearized thread idx.
        auto const globalThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
        auto const globalThreadExtent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);

        auto const linearizedGlobalThreadIdx = alpaka::mapIdx<1u>(globalThreadIdx, globalThreadExtent)[0];

        auto engine = alpaka::rand::engine::createDefault(
            acc,
            linearizedGlobalThreadIdx,
            0);

        auto dist(alpaka::rand::distribution::createUniformReal<float>(acc));

        float random_value = dist(engine);
        if (random_value > 0.5)
			particles[linearizedGlobalThreadIdx].weight = 0;

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
    using Acc = alpaka::ExampleDefaultAcc<Dim, Idx>;
    using Host = alpaka::DevCpu;
    auto const devAcc = alpaka::getDevByIdx<Acc>(0u);
    auto const devHost = alpaka::getDevByIdx<Host>(0u);
    using QueueProperty = alpaka::Blocking;
    using QueueAcc = alpaka::Queue<Acc, QueueProperty>;
    QueueAcc queue{devAcc};

    using BufHostParticles = alpaka::Buf<Host, mxmc::Particle, Dim, Idx>;
    using BufAccParticles = alpaka::Buf<Acc, mxmc::Particle, Dim, Idx>;
    using WorkDiv = alpaka::WorkDivMembers<Dim, Idx>;

    constexpr size_t numThreads = 100u;
    constexpr size_t numAlpakaElementsPerThread = 1;


    WorkDiv workdiv{alpaka::getValidWorkDiv<Acc>(
        devAcc,
        Vec(numThreads),
        Vec(numAlpakaElementsPerThread),
        false,
        alpaka::GridBlockExtentSubDivRestrictions::Unrestricted)};


    constexpr size_t number_particles = 100u;
    BufHostParticles bufHostParticles{alpaka::allocBuf<mxmc::Particle, Idx>(devHost, number_particles)};
    mxmc::Particle* const ptrBufHostParticles{alpaka::getPtrNative(bufHostParticles)};

    BufAccParticles bufAccParticles{alpaka::allocBuf<mxmc::Particle, Idx>(devAcc, number_particles)};
    mxmc::Particle* const ptrBufAccParticles{alpaka::getPtrNative(bufAccParticles)};


    simpleParticlesInitialisation(ptrBufHostParticles, number_particles);
    alpaka::memcpy(queue, bufAccParticles, bufHostParticles);




    double sum_weights_before = calculate_weights(ptrBufHostParticles, number_particles);
    Kernel kernel;
    alpaka::exec<Acc>(queue, workdiv, kernel, ptrBufAccParticles);
    alpaka::memcpy(queue, bufHostParticles, bufAccParticles);
    alpaka::wait(queue);

    double sum_weights_after = calculate_weights(ptrBufHostParticles, number_particles);

    std::cout << "sum weights before "<< sum_weights_before<< "\n";
    std::cout << "sum weights after "<< sum_weights_after << "\n";
    std::cout << "relative difference "<< sum_weights_before /sum_weights_after << "\n";
    return EXIT_SUCCESS;
}
