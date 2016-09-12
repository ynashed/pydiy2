/*
 * pydiy2.h
 *
 *  Created on: Jun 30, 2016
 *      Author: ynashed
 */

#ifndef PYDIY2_H_
#define PYDIY2_H_

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

struct DIY_MSG
{
	bytes* buff;
	size_t size;
	int fromGID;
	bool toFree;

	DIY_MSG(bytes* b=0, size_t n=0, int gid=-1): buff(b), size(n), fromGID(gid), toFree(false)
	{}
	~DIY_MSG()
	{if(toFree) delete[] buff;}
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
			diy::save(bb, dm.buff, dm.size);
		}
	static
	void load(diy::BinaryBuffer& bb, DIY_MSG& dm)
		{
			diy::load(bb, dm.size);
			dm.buff = new bytes[dm.size];
			/*dm.toFree = true; //Generates a double free error in python*/
			diy::load(bb, dm.buff, dm.size);
		}
};
}
typedef std::function<void(const DIY_MSG&)> MSGRcvdFunc;
/////////////////////////////////////////////////////////////////////////////////////////////////

namespace pydiy2
{
class Block
{
private:
	size_t m_neighborNum;

public:
	Block(size_t n=0) : m_neighborNum(n)
	{}
	virtual ~Block()
	{}

	size_t getNeighborNum() const {return m_neighborNum;}
	void diy2enqueue(const diy::Master::ProxyWithLink& cp, void* msg)
	{
		DIY_MSG msgToSend = *static_cast<DIY_MSG*>(msg);

		diy::Link*    l = cp.link();
		for (size_t i=0; i<l->size(); ++i)
			if(cp.gid() != l->target(i).gid)
				cp.enqueue(l->target(i),msgToSend);
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
				msgFunc(rcvd);
			}
	}
};
typedef std::function<void(const Block&)> BlockFunc;

class PyDIY2
{
private:
	BlockFunc		 		m_blockReceiver;
	MSGRcvdFunc 			m_msgReceiver;
	diy::mpi::communicator 	m_mpiCommunicator;
	diy::Master 			m_diyMaster;
	Decomposer* 			m_decomposer;

public:
	PyDIY2(BlockFunc c, MSGRcvdFunc mr, MPI_Comm comm = MPI_COMM_WORLD) :	m_blockReceiver(c),
																			m_msgReceiver(mr),
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
		Block*         	b	= new Block(link.size());
		RLink*          l   = new RLink(link);
		diy::Master*    m   = const_cast<diy::Master*>(&m_diyMaster);
		int             lid = m->add(gid, b, l); // add block to the master (mandatory)
		m_blockReceiver(*b);
		// process any additional args here, using them to initialize the block
	}

	std::vector<int> decompose(int dims,
								const std::vector<int>& min,
								const std::vector<int>& max,
								const std::vector<bool>& share_face=std::vector<bool>(),
								const std::vector<bool>& wrap=std::vector<bool>(),
								const std::vector<int>& ghosts=std::vector<int>());
	void sendToNeighbors(const DIY_MSG* msg)
	{
		m_diyMaster.foreach(&Block::diy2enqueue, (void*)(msg));
	}

	void recvFromNeighbors()
	{
		m_diyMaster.exchange();
		m_diyMaster.foreach(&Block::diy2dequeue, (void*)(&m_msgReceiver));
	}

	/*std::string popMessage()
	{
		std::string msg;
		if(m_msgQueue.size()>0)
		{
			msg = m_msgQueue.back();
			m_msgQueue.pop_back();
		}
		return msg;
	}*/
};

} /* namespace pydiy2 */

#endif /* PYDIY2_H_ */
