/*
 * pydiy2.h
 *
 *  Created on: Jun 30, 2016
 *      Author: ynashed
 */

#ifndef PYDIY2_H_
#define PYDIY2_H_

#include <pybind11/pybind11.h>
#include <diy/mpi.hpp>
#include <diy/master.hpp>
#include <diy/decomposition.hpp>
#include <diy/serialization.hpp>
#include <functional>
#include <vector>
#include <string>

typedef diy::DiscreteBounds Bounds;
typedef diy::RegularLink<Bounds> RLink;
typedef diy::RegularDecomposer<Bounds> Decomposer;
typedef unsigned char bytes;
namespace py = pybind11;

struct DIY_MSG : py::buffer_info
{
	int fromGID;
	bool owned;
	DIY_MSG(const py::buffer_info& info) : py::buffer_info(	info.ptr,
															info.itemsize,
															info.format,
															info.ndim,
															info.shape,
															info.strides),
									fromGID(-1), owned(false)
	{}
	DIY_MSG(): py::buffer_info(), fromGID(-1)
	{}
	DIY_MSG(const DIY_MSG &) = delete;
	DIY_MSG& operator=(const DIY_MSG &) = delete;
	~DIY_MSG()
	{if(owned) delete[] (bytes*)ptr;}
	int getFromGID() const {return fromGID;}
};
namespace diy
{
template<>
struct Serialization<DIY_MSG>
{
	static
	void save(diy::BinaryBuffer& bb, const DIY_MSG& dm)
		{
			diy::save(bb, dm.size);
			diy::save(bb, dm.itemsize);
			diy::save(bb, dm.format);
			diy::save(bb, dm.ndim);
			diy::save(bb, dm.shape);
			diy::save(bb, dm.strides);
			diy::save(bb, (bytes*)dm.ptr, dm.size*dm.itemsize);
			//printf("Saved (%ld,%ld,%s,%ld,%ld,%ld,%ld,%ld)\n", dm.size, dm.itemsize, dm.format.c_str(), dm.ndim, dm.shape[0], dm.shape[1], dm.strides[0], dm.strides[1]);
		}
	static
	void load(diy::BinaryBuffer& bb, DIY_MSG& dm)
		{
			diy::load(bb, dm.size);
			diy::load(bb, dm.itemsize);
			diy::load(bb, dm.format);
			diy::load(bb, dm.ndim);
			diy::load(bb, dm.shape);
			diy::load(bb, dm.strides);
			dm.ptr = new bytes[dm.size*dm.itemsize];
			dm.owned = true;
			diy::load(bb, (bytes*)dm.ptr, dm.size*dm.itemsize);
			//printf("Loaded (%ld,%ld,%s,%ld,%ld,%ld,%ld,%ld)\n", dm.size, dm.itemsize, dm.format.c_str(), dm.ndim, dm.shape[0], dm.shape[1], dm.strides[0], dm.strides[1]);
		}
};
}
typedef std::shared_ptr<DIY_MSG> MSGPtr;
typedef std::function<void(const DIY_MSG*,int)> MSGRcvdFunc;
typedef std::function<const MSGPtr(int)> MSGSentFunc;
/////////////////////////////////////////////////////////////////////////////////////////////////

namespace pydiy2
{

class IBlock
{
private:
	size_t m_neighborNum;
	Bounds m_bounds;
	Bounds m_coreBounds;
public:
	IBlock(size_t n=0) : m_neighborNum(n)
	{}
	virtual ~IBlock()
	{}

	void setNeighborNum(size_t n) {m_neighborNum=n;}
	void setBounds(const Bounds& b, const Bounds& c)
	{
		m_bounds = b;
		m_coreBounds = c;
	}

	std::vector<int> getBoundsMin()		const {return std::vector<int>(m_bounds.min, m_bounds.min+DIY_MAX_DIM);}
	std::vector<int> getBoundsMax() 	const {return std::vector<int>(m_bounds.max, m_bounds.max+DIY_MAX_DIM);}
	std::vector<int> getCoreBoundsMin() const {return std::vector<int>(m_coreBounds.min, m_coreBounds.min+DIY_MAX_DIM);}
	std::vector<int> getCoreBoundsMax() const {return std::vector<int>(m_coreBounds.max, m_coreBounds.max+DIY_MAX_DIM);}
	size_t getNeighborNum() 			const {return m_neighborNum;}

	void diy2enqueue(const diy::Master::ProxyWithLink& cp, void* msg)
	{
		diy::Link*    l = cp.link();
		for (size_t i=0; i<l->size(); ++i)
			if(cp.gid() != l->target(i).gid)
				cp.enqueue(l->target(i),(*(DIY_MSG*)msg));
	}

	void diy2dequeue(const diy::Master::ProxyWithLink& cp, void* fptr)
	{
		MSGRcvdFunc msgFunc = *static_cast<MSGRcvdFunc*>(fptr);
		std::vector<int> in;

		cp.incoming(in);
		for (size_t i=0; i<in.size(); ++i)
			if(cp.gid() != in[i])
			{
				DIY_MSG rcvd;
				cp.dequeue(in[i], rcvd);
				rcvd.fromGID = in[i];
				msgFunc(&rcvd, cp.gid());
			}
	}
};
typedef std::function<IBlock*(int)> BlockFunc;

struct MergeFunctor
{
	MSGRcvdFunc rcvFPtr;
	MSGSentFunc sendFPtr;

	MergeFunctor(MSGRcvdFunc mr, MSGSentFunc ms): rcvFPtr(mr), sendFPtr(ms)
	{}

	void operator()(void* b_, const diy::ReduceProxy& rp, const diy::RegularMergePartners& partners) const
	{
		IBlock*     b        = static_cast<IBlock*>(b_);
		unsigned   round    = rp.round();               // current round number

		// step 1: dequeue and merge
		for (int i = 0; i < rp.in_link().size(); ++i)
		{
			int nbr_gid = rp.in_link().target(i).gid;
			if (nbr_gid != rp.gid())
			{
				DIY_MSG rcvd;
				rp.dequeue(nbr_gid, rcvd);
				rcvd.fromGID = nbr_gid;
				rcvFPtr(&rcvd, rp.gid());
			}
		}

		// step 2: enqueue
		for (int i = 0; i < rp.out_link().size(); ++i)    // redundant since size should equal to 1
		{
			// only send to root of group, but not self
			if (rp.out_link().target(i).gid != rp.gid())
			{
				const MSGPtr rcvd = sendFPtr(rp.gid());
				rp.enqueue(rp.out_link().target(i), *rcvd);
			}
		}
	}
};

class PyDIY2
{
private:
	BlockFunc		 		m_blockCreator;
	diy::mpi::communicator 	m_mpiCommunicator;
	diy::Master 			m_diyMaster;
	Decomposer* 			m_decomposer;

public:
	PyDIY2(BlockFunc c, MPI_Comm comm = MPI_COMM_WORLD) :	m_blockCreator(c),
															m_mpiCommunicator(comm),
															m_diyMaster(m_mpiCommunicator),
															m_decomposer(0)
	{}
	virtual ~PyDIY2()
	{if(m_decomposer) delete m_decomposer;}


	void  operator()(int gid,                 // block global id
				   const Bounds& core,        // block bounds without any ghost added
				   const Bounds& bounds,      // block bounds including any ghost region added
				   const Bounds& domain,      // global data bounds
				   const RLink& link)         // neighborhood
	const
	{
		IBlock*         b	= m_blockCreator(gid);
		RLink*          l   = new RLink(link);
		diy::Master*    m   = const_cast<diy::Master*>(&m_diyMaster);
		int             lid = m->add(gid, b, l); // add block to the master (mandatory)

		b->setBounds(bounds, core);
		b->setNeighborNum(link.size());
		// process any additional args here, using them to initialize the block
	}

	void decompose(	int dims,
					const std::vector<int>& min,
					const std::vector<int>& max,
					const std::vector<bool>& share_face=std::vector<bool>(),
					const std::vector<bool>& wrap=std::vector<bool>(),
					const std::vector<int>& ghosts=std::vector<int>());
	void sendToNeighbors(const MSGPtr msg)
	{
		m_diyMaster.foreach(&IBlock::diy2enqueue, (void*)(msg.get()));
	}

	std::vector<int> gidToCoords(int gid) const
	{
		std::vector<int> pos;
		m_decomposer->gid_to_coords(gid, pos);
		return pos;
	}

	void recvFromNeighbors(MSGRcvdFunc msgReceiver)
	{
		m_diyMaster.exchange();
		m_diyMaster.foreach(&IBlock::diy2dequeue, (void*)(&msgReceiver));
	}
	void mergeReduce(int k, MSGRcvdFunc mr, MSGSentFunc ms)
	{
		int procs = m_mpiCommunicator.size();
		diy::RegularMergePartners  partners(*m_decomposer, k, true);
		diy::RoundRobinAssigner assigner(procs, procs);
		diy::reduce(m_diyMaster, assigner, partners, MergeFunctor(mr,ms));
	}
};

} /* namespace pydiy2 */

#endif /* PYDIY2_H_ */
