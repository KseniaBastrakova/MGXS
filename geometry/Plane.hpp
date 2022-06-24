#pragma once

#include <alpaka/alpaka.hpp>
#include <alpaka/example/ExampleDefaultAcc.hpp>

namespace mxmc{

struct Plane
{
	Plane():
		A(0.), B(0.), C(0.), D(0.){}
	Plane(double A_, double B_, double C_, double D_):
		A(A_), B(B_), C(C_), D(D_){}

	template<typename TAcc, typename TParticle>
	ALPAKA_FN_ACC auto distance(TAcc const& acc, TParticle particle)
	{
		double vp = A * particle.px + B * particle.py + C * particle.pz;

		if (vp == 0)
			return -1;

		distance = (D  - A * particle.x - B * particle.y - C * particle.z) / vp;
		return distance;
	}
	template<typename TAcc, typename TParticle>
	ALPAKA_FN_ACC auto get_sign(TAcc const& acc, TParticle particle)
	{
		double sign = (A * particle.x + B * particle.y + C * particle.z - D);

		return sign > 0;
	}

private:

	double A;
	double B;
	double C;
	double D;
};

}

