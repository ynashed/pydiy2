#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <diy/reduce.hpp>
#include <diy/partners/merge.hpp>
#include <assert.h>
#include "pydiy2.h"

using namespace pydiy2;

std::vector<int> PyDIY2::decompose(int dims,
									const std::vector<int>& dmin,
									const std::vector<int>& dmax,
									const std::vector<bool>& share_face,
									const std::vector<bool>& wrap,
									const std::vector<int>& ghosts)
{
	assert (dims<=DIY_MAX_DIM);

	int procs = m_mpiCommunicator.size();
	int rank = m_mpiCommunicator.rank();

	diy::RoundRobinAssigner assigner(procs, procs);
	Bounds domain;

	for(int d=0; d<dims; ++d)
	{
		domain.min[d] = dmin[d];
		domain.max[d] = dmax[d];
	}
	
	m_decomposer = new Decomposer(dims, domain, procs, share_face, wrap, ghosts);
	m_decomposer->decompose(rank, assigner, *this);

	Bounds bounds;
	m_decomposer->fill_bounds(bounds, rank, true);

	std::vector<int> scanBounds;
	scanBounds.push_back(bounds.min[0]); scanBounds.push_back(bounds.max[0]);
	scanBounds.push_back(bounds.min[1]); scanBounds.push_back(bounds.max[1]);
	return scanBounds;
}


// PYTHON BINDINGS

namespace py = pybind11;
using namespace py::literals;

PYBIND11_PLUGIN(pydiy2) 
{
	py::module m("pydiy2", "pybind11 example plugin");

	py::class_<DIY_MSG>(m, "DIY_MSG")
			.def("__init__", [](DIY_MSG &dm, py::buffer b)
					{
			            py::buffer_info info = b.request();
			            if (info.format != py::format_descriptor<bytes>::format() || info.ndim != 1)
			                throw std::runtime_error("Incompatible buffer format!");
			            new (&dm) DIY_MSG((bytes*)info.ptr, info.shape[0]);
			        })
			 /// Provide buffer access
			.def_buffer([](DIY_MSG &dm) -> py::buffer_info
			{
				return py::buffer_info
				(
					(void*)dm.buff,							/* Pointer to buffer */
					sizeof(bytes),                          /* Size of one scalar */
					py::format_descriptor<bytes>::format(), /* Python struct-style format descriptor */
					1,                                      /* Number of dimensions */
					{ dm.size }, 		                	/* Buffer dimensions */
					{ sizeof(bytes)} 			            /* Strides (in bytes) for each index */
				);
			});

	py::class_<Block>(m, "Block")
			.def("getNeighborNum", &Block::getNeighborNum);

	py::class_<PyDIY2>(m, "PyDIY2")
		.def(py::init<BlockFunc,MSGRcvdFunc,MPI_Comm>(), "c"_a, "mr"_a, "comm"_a=MPI_COMM_WORLD)
		.def("decompose", &PyDIY2::decompose, 	"dims"_a,
												"boundsMin"_a,
												"boundsMax"_a,
												"share_face"_a=std::vector<bool>(),
												"wrap"_a=std::vector<bool>(),
												"ghosts"_a=std::vector<int>())
		.def("sendToNeighbors", &PyDIY2::sendToNeighbors)
		.def("recvFromNeighbors", &PyDIY2::recvFromNeighbors);
		//.def("popMessage", &PyDIY2::popMessage);

    return m.ptr();
}
