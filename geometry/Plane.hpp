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

		if vp == 0:
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


	double A;
	double B;
	double C;
	double D;
};
/*
    def __init__(self, A, B, C, D):
        self.A = A
        self.B = B
        self.C = C
        self.D = D

    def distance(self, particle):
        vp = (self.A * particle.direction.tetta_x + self.B * particle.direction.tetta_y +
              self.C * particle.direction.tetta_z)


        if (abs(vp) < 1e-9):
            return -1

        distance = (self.D - self.A * particle.coordinates.x - self.B * particle.coordinates.y
                    - self.C * particle.coordinates.z) / vp

        return distance

    def get_normal(self):
        sq = math.sqrt(self.A * self.A + self.B * self.B + self.C * self.C)
        a_n = self.A / sq
        b_n = self.B / sq
        c_n = self.C / sq
        return [a_n, b_n, c_n]


    def get_sign(self, particle):

        sign = (self.A * particle.coordinates.x + self.B * particle.coordinates.y +
        self.C * particle.coordinates.z - self.D)

        if sign == 0:
            return sign

        if sign < 0:
            return -1
        else:
            return 1

    def get_xml(self):

        obj_xml = " Plane with A "+ str(self.A) + " B "+ str(self.B) + " C "+ str(self.C) + " D "+ str(self.D)

        return obj_xml

}
*/
