#pragma once

#include <alpaka/alpaka.hpp>
#include <alpaka/example/ExampleDefaultAcc.hpp>

using Idx = std::size_t;

namespace mxmc{

template<typename TCells, typename TCS>
struct Universe{
	Universe(TCells cells_, TCS cross_sections_, Idx number_of_groups_, Idx number_of_materials_):
		cells(cells_), cross_sections(cross_sections_),
		number_of_groups(number_of_groups_), number_of_materials(number_of_materials_)
		{
			set_delta_cs();
		}

	template<typename TAcc>
	ALPAKA_FN_ACC auto get_delta_cs(Idx energy_group_idx) const
	{
		return delta_cross_sections[energy_group_idx];
	}

	template<typename TAcc, typename TParticle, typename CS>
	ALPAKA_FN_ACC auto get_cs(TAcc const& acc, const TParticle& particle) const
	{
		CS current_material = CS();
		for (int i = 0; i < number_of_materials; i++)
		{
			if (cells[i].is_inside(particle))
			{
				particle.set_last_cell(i);
				current_material = cross_sections[i];
			}
		}
		return current_material;
	}
	template<typename TAcc, typename TParticle>
	ALPAKA_FN_ACC auto is_inside(TAcc const& acc, const TParticle& particle) const
	{
		bool is_inside = 0;
		for (int i = 0; i < number_of_materials; i++)
		{
			if (cells[i].is_inside(particle))
					is_inside = 1;
		}
		return is_inside;
	}

private:
	auto set_delta_cs()
	{
		delta_cross_sections = new double[number_of_groups];
		double delta_cs = 0.;
		for (int i = 0; i < number_of_groups; i++)
		{
			double delta_cs = 0.;
			for (int j = 0; j < number_of_materials; j++)
				delta_cs = std::max(delta_cs, cross_sections[j]);
			delta_cross_sections[i] = delta_cs;
		}
	}

	TCells cells;
	TCS    cross_sections;
	double * delta_cross_sections;
	Idx number_of_groups;
	Idx number_of_materials;
};



}

/*
 *



    def is_boudary_reflective(self, particle):

        minumum = np.inf
        idx_of_min = 0.
        for i in range(0, len(self.cells)):
            distance, surface_idx = self.cells[i].get_minimum_distance(particle)
            if abs(distance) < minumum:
                minumum = distance
                idx_of_min = i
        if self.cells[idx_of_min].is_boudary_reflective(particle):
            return True
        else:
            return False

    def reflect_particle(self,  particle):
        self.cells[particle.last_cell].reflect_particle(particle)

       */
