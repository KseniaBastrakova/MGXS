#pragma once
#include <alpaka/alpaka.hpp>
#include <alpaka/example/ExampleDefaultAcc.hpp>
namespace mxmc{

struct Particle
{
	Particle():
		x(0.), y(0.), z(0),
		px(0.), py(0.), pz(0.),
		weight(1.)
	{}
	Particle(double x_, double y_, double z_):
		x(x_), y(y_), z(z_),
		px(0.), py(0.), pz(0.),
		weight(1.)
	{}
	template<typename TAcc>
	ALPAKA_FN_ACC auto set_coordinates(TAcc const& acc, double x_, double y_, double z_) -> void
	{
		x = x_;
		y = y_;
		z = z_;
	}


	double x, y, z;
	double px, py, pz;
	double weight;

};

}




/*
def set_direction(self, ets, phi):
    self.direction = Direction(ets, phi)

def set_direction_angle(self, angle):
    self.direction = angle

def set_energy_groups(self, energy_groups):
    self.energy_groups = energy_groups

def get_energy_group(self):

    return self.energy_group
 #   res = next(x for x, val in enumerate(self.energy_groups)
  #                            if val > self.energy)
   # return res - 1

def set_energy_group(self, energy_group):

    self.energy_group = energy_group

def set_terminated(self):
    self.terminated = True

def is_terminated(self):
    return self.terminated

def get_weight(self):
    return self.weight

def set_weight(self, weight):
    self.weight = weight

def set_particle_deleted(self):
    self.weight = self.weight * 0

def set_particle_fission(self, additional_weight):
    self.weight = self.weight * additional_weight

def set_multiplicity(self, additional_weight):
    self.weight = self.weight * additional_weight

def is_particle_deleted(self):
    return self.weight < 0.0001

def add_path(self, path):
    self.path += path

def get_path(self):
    return self.path

def set_path(self, path):
    self.path = path
    */
