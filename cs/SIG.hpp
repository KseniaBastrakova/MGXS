#pragma once

#include <alpaka/alpaka.hpp>
#include <alpaka/example/ExampleDefaultAcc.hpp>
namespace mxmc{

using Idx = std::size_t;

struct SIG{

	using Data = double;

	SIG(){}

	auto set_fission(double fission_)
	{
		fission = fission_;
	}

	auto set_capture(double capture_)
	{
		 capture = capture_;
	}

	auto set_scattering(double scattering_)
	{
		scattering = scattering_;
	}

	auto set_total(double total_)
	{
		total = total_;
	}

	auto set_number_production_neutrons(double number_production_neutrons_)
	{
		number_production_neutrons = number_production_neutrons_;
	}

	ALPAKA_FN_ACC
	auto get_fission()
	{
		return fission;
	}

	ALPAKA_FN_ACC
	auto get_capture()
	{
		return capture;
	}

	ALPAKA_FN_ACC
	auto get_scattering()
	{
		return scattering;
	}

	ALPAKA_FN_ACC
	auto get_total()
	{
		return total;
	}

	ALPAKA_FN_ACC
	auto get_number_production_neutrons()
	{
		return number_production_neutrons;
	}

	ALPAKA_FN_ACC
	auto get_fission_probability()
	{
		return fission / total;
	}

	ALPAKA_FN_ACC
	auto get_capture_probability()
	{
		return capture / total;
	}

	ALPAKA_FN_ACC
	auto get_scatter_probability()
	{
		return scattering / total;
	}

	double fission;
    double capture;
    double scattering;
    double total;
    double number_production_neutrons;
    double cs_virtual;
};



template<typename TBufAcc, typename TBufHost>
struct SIG_storage{
	SIG_storage(Idx number_groups, TBufAcc dev_acc, TBufHost dev_host):
		number_groups(number_groups),
		SIG_host(alpaka::allocBuf<SIG, Idx>(dev_host, number_groups)),
		SIG_device(alpaka::allocBuf<SIG, Idx>(dev_acc, number_groups))
	{

	}

    using Dim = alpaka::DimInt<1>;
    using Vec = alpaka::Vec<Dim, Idx>;
    using Host = alpaka::DevCpu;
    using Acc = alpaka::AccCpuSerial<Dim, Idx>;
	using Data = double;
	using Buf_host_double = alpaka::Buf<Host, SIG, Dim, Idx>;
	using Buf_acc_double = alpaka::Buf<Acc, SIG, Dim, Idx>;

	template<typename TQueueAcc>
	auto copy_from_host_to_device(TQueueAcc queue)
	{
		alpaka::memcpy(queue, SIG_device, SIG_host);
	}

	template<typename TQueueAcc>
	auto copy_from_device_to_host(TQueueAcc queue)
	{
		alpaka::memcpy(queue, SIG_host, SIG_device);
	}

	auto get_pointer_host()
	{
		SIG* const ptr_buf_host{alpaka::getPtrNative(SIG_host)};
		return ptr_buf_host;
	}

	auto get_pointer_device()
	{
		SIG* const ptr_buf_acc{alpaka::getPtrNative(SIG_device)};
		return ptr_buf_acc;
	}


private:

	Idx number_groups;
	Buf_host_double SIG_host;
	Buf_acc_double SIG_device;
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
