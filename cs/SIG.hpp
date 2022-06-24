#pragma once

#include <alpaka/alpaka.hpp>
#include <alpaka/example/ExampleDefaultAcc.hpp>
namespace mxmc{

using Idx = std::size_t;

template<typename TBuf>
struct SIG{
	SIG(){}

	auto set_pointers(TBuf sig_f_,
			TBuf capture_,
			TBuf scattering_,
			TBuf total_,
			TBuf number_production_neutrons_,
			TBuf cs_virtual_)
	{
		sig_f = sig_f_;
		capture = capture_;
		scattering = scattering_;
		total = total_;
		number_production_neutrons = number_production_neutrons_;
		cs_virtual = cs_virtual_;
	}

	TBuf sig_f;
	TBuf capture;
	TBuf scattering;
	TBuf total;
	TBuf number_production_neutrons;

	TBuf cs_virtual;
};

template<typename TBuf>
using SIG_type = SIG<TBuf>;

template<typename TAcc>
struct SIG_storage{
	SIG_storage(Idx number_groups):
		number_groups(number_groups){}

	using Dim = alpaka::DimInt<1>;
	using Host = alpaka::DevCpu;
	using Data = double;
	using Buf_host_double = alpaka::Buf<Host, Data, Dim, Idx>;
	using Buf_acc_double = alpaka::Buf<TAcc, Data, Dim, Idx>;

	auto allocate_memory_host()
	{
		auto const dev_host = alpaka::getDevByIdx<Host>(0u);

		Buf_host_double sig_f_host {alpaka::allocBuf<Data, Idx>(dev_host, number_groups)};
		SIG_host.sig_f = sig_f_host;

		Buf_host_double capture_host {alpaka::allocBuf<Data, Idx>(dev_host, number_groups)};
		SIG_host.capture = capture_host;

		Buf_host_double scattering_host {alpaka::allocBuf<Data, Idx>(dev_host, number_groups)};
		SIG_host.scattering = scattering_host;

		Buf_host_double total_host {alpaka::allocBuf<Data, Idx>(dev_host, number_groups)};
		SIG_host.total = total_host;

		Buf_host_double number_production_neutrons_host {alpaka::allocBuf<Data, Idx>(dev_host, number_groups)};
		SIG_host.number_production_neutrons = number_production_neutrons_host;

		Buf_host_double cs_virtual_host {alpaka::allocBuf<Data, Idx>(dev_host, number_groups)};
		SIG_host.cs_virtual = cs_virtual_host;
	}

	auto allocate_memory_device()
	{
		auto const dev_acc = alpaka::getDevByIdx<TAcc>(0u);

		Buf_acc_double sig_f_acc {alpaka::allocBuf<Data, Idx>(dev_acc, number_groups)};
		SIG_device.sig_f = sig_f_acc;

		Buf_acc_double capture_acc {alpaka::allocBuf<Data, Idx>(dev_acc, number_groups)};
		SIG_device.capture = capture_acc;

		Buf_acc_double scattering_acc {alpaka::allocBuf<Data, Idx>(dev_acc, number_groups)};
		SIG_device.scattering = scattering_acc;

		Buf_acc_double total_acc {alpaka::allocBuf<Data, Idx>(dev_acc, number_groups)};
		SIG_device.total = total_acc;

		Buf_acc_double number_production_neutrons_acc {alpaka::allocBuf<Data, Idx>(dev_acc, number_groups)};
		SIG_device.number_production_neutrons = number_production_neutrons_acc;

		Buf_acc_double cs_virtual_acc {alpaka::allocBuf<Data, Idx>(dev_acc, number_groups)};
		SIG_device.cs_virtual = cs_virtual_acc;

	}

	template<typename TQueueAcc>
	auto copy_from_host_to_device(const TQueueAcc& queue)
	{
		alpaka::memcpy(queue, SIG_device.sig_f, SIG_host.sig_f);
		alpaka::memcpy(queue, SIG_device.capture, SIG_host.capture);
		alpaka::memcpy(queue, SIG_device.scattering, SIG_host.scattering);
		alpaka::memcpy(queue, SIG_device.total, SIG_host.total);
		alpaka::memcpy(queue, SIG_device.number_production_neutrons, SIG_host.number_production_neutrons);
		alpaka::memcpy(queue, SIG_device.cs_virtual, SIG_host.cs_virtual);

	}

	template<typename TQueueAcc>
	auto copy_from_device_to_host(const TQueueAcc& queue)
	{
		alpaka::memcpy(queue, SIG_host.sig_f, SIG_device.sig_f);
		alpaka::memcpy(queue, SIG_host.capture, SIG_device.capture);
		alpaka::memcpy(queue, SIG_host.scattering, SIG_device.scattering);
		alpaka::memcpy(queue, SIG_host.total, SIG_device.total);
		alpaka::memcpy(queue, SIG_host.number_production_neutrons, SIG_device.number_production_neutrons);
		alpaka::memcpy(queue, SIG_host.cs_virtual, SIG_device.cs_virtual);
	}

	auto get_pointer_host()
	{
		return SIG_host;
	}

	auto get_pointer_device()
	{
		return SIG_device;
	}


private:

	Idx number_groups;
	SIG_type<Buf_host_double> SIG_host;
	SIG_type<Buf_acc_double> SIG_device;
};

}

/*
class Sig:
    def __init__(self):


    def __init__(self, number_groups, sig_f, sig_c, sig_scattering_matrix, number_of_production_neutrons,
                 cs_xi_table, sig_t):
        self.number_groups = number_groups
        self.sig_f = sig_f
        self.sig_c = sig_c
        self.sig_s = []
        self.sig_scattering_matrix = sig_scattering_matrix

        for i in range(0, len(self.sig_scattering_matrix)):
            self.sig_s.append(sum(self.sig_scattering_matrix[i]))

        self.number_of_production_neutrons = number_of_production_neutrons
        self.cs_xi_table = cs_xi_table
        self.sig_t = sig_t
        self.delta_xs = max(sig_t)/2.
        max_total_cs = max(self.sig_t)
        self.virtual = [self.delta_xs]


    def get_virtual_cs(self):
        max_total_cs = max(self.sig_t)
        self.virtual = [cross_section - max_total_cs + self.delta_xs for cross_section in self.sig_t]

    def return_virtual(self):
        return self.virtual

    def get_fission_probability(self, energy_group_idx):
        return self.sig_f[energy_group_idx] / self.sig_t[energy_group_idx]

    def get_capture_probability(self, energy_group_idx):
        return self.sig_c[energy_group_idx] / self.sig_t[energy_group_idx]

    def get_scatter_probability(self, energy_group_idx):
        return self.sig_s[energy_group_idx] / self.sig_t[energy_group_idx]

*/
