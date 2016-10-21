#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <diy/reduce.hpp>
#include <diy/partners/merge.hpp>
#include <assert.h>
#include "pydiy2.h"

using namespace pydiy2;

void PyDIY2::decompose(	int dims,
						const std::vector<int>& dmin,
						const std::vector<int>& dmax,
						const std::vector<bool>& share_face,
						const std::vector<bool>& wrap,
						const std::vector<int>& ghosts)
{
	assert (dims<=DIY_MAX_DIM);

	int procs = m_mpiCommunicator.size();
	int rank = m_mpiCommunicator.rank();

	m_assigner = new diy::RoundRobinAssigner(procs, procs); //TODO: add actual block configuration parameter and assigner type
	Bounds domain;

	for(int d=0; d<dims; ++d)
	{
		domain.min[d] = dmin[d];
		domain.max[d] = dmax[d];
	}
	
	m_decomposer = new Decomposer(dims, domain, procs, share_face, wrap, ghosts);
	m_decomposer->decompose(rank, *m_assigner, *this);
}


// PYTHON BINDINGS

namespace py = pybind11;
using namespace py::literals;
PYBIND11_DECLARE_HOLDER_TYPE(DIY_MSG, std::shared_ptr<DIY_MSG>);

PYBIND11_PLUGIN(pydiy2) 
{
	py::module m("pydiy2", "pybind11 example plugin");

	py::class_<DIY_MSG, std::shared_ptr<DIY_MSG> >(m, "DIY_MSG")
			.def("__init__", [](DIY_MSG &dm, py::buffer b)
					{
						py::buffer_info info = b.request();
						new (&dm) DIY_MSG(info);
					})
			 /// Provide buffer access
			.def_buffer([](DIY_MSG &dm) -> py::buffer_info
					{return py::buffer_info(
					        dm.ptr,
					        dm.itemsize,
					        dm.format,
					        dm.ndim,
					        dm.shape,
					        dm.strides);})
			.def("getFromGID", &DIY_MSG::getFromGID);


	py::class_<IBlock>(m, "IBlock")
			.def("getBoundsMin", &IBlock::getBoundsMin)
			.def("getBoundsMax", &IBlock::getBoundsMax)
			.def("getCoreBoundsMin", &IBlock::getCoreBoundsMin)
			.def("getCoreBoundsMax", &IBlock::getCoreBoundsMax)
			.def("getNeighborNum", &IBlock::getNeighborNum);

	py::class_<PyDIY2>(m, "PyDIY2")
		.def(py::init<BlockFunc,MPI_Comm>(), "c"_a, "comm"_a=MPI_COMM_WORLD)
		.def("decompose", &PyDIY2::decompose, 	"dims"_a,
												"boundsMin"_a,
												"boundsMax"_a,
												"share_face"_a=std::vector<bool>(),
												"wrap"_a=std::vector<bool>(),
												"ghosts"_a=std::vector<int>())
		.def("gidToCoords", &PyDIY2::gidToCoords)
		.def("sendToNeighbors", &PyDIY2::sendToNeighbors)
		.def("recvFromNeighbors", &PyDIY2::recvFromNeighbors)
		.def("mergeReduce", &PyDIY2::mergeReduce)
		.def("swapReduce", &PyDIY2::swapReduce)
		.def("a2aReduce", &PyDIY2::a2aReduce);

    return m.ptr();
}
