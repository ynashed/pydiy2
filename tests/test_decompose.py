from pydiy2 import *
from mpi4py import MPI
try:
    import cPickle as pickle  # Use cPickle on Python 2.7
except ImportError:
    import pickle
import numpy as np
import binascii

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
procs = comm.Get_size()

class ToSerialize():
    def __init__(self, ID):
    	self.id = ID+1000

class myBlock(IBlock):
    def __init__(self):
        self.id = "Derived"


blocks = {}
def createBlock(gid):
    print "Block Created!"
    b = myBlock()
    blocks[gid] = b
    return b

def msgReceived(m,gid):
	dd = np.array(m, copy = False)
	print pickle.loads(binascii.rledecode_hqx(dd)).id


myDIY = PyDIY2(createBlock)
myDIY.decompose(2, [0,0], [20,20], ghosts=[2,2] )
print "Neighbor# : " + str(blocks.values()[0].getNeighborNum())
print "BoundsMin : " + str(blocks.values()[0].getBoundsMin())
print "BoundsMax : " + str(blocks.values()[0].getBoundsMax())

msgStr = binascii.rlecode_hqx(pickle.dumps(ToSerialize(rank), 2))
myDIY.sendToNeighbors(DIY_MSG(msgStr))
myDIY.recvFromNeighbors(msgReceived)


# import ctypes
# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()
# procs = comm.Get_size()
# comm_ptr = MPI._addressof(comm)
# if MPI._sizeof(MPI.Comm) == ctypes.sizeof(ctypes.c_int):
#     MPI_Comm = ctypes.c_int
# else:
#     MPI_Comm = ctypes.c_void_p
# print decompose(MPI_Comm.from_address(comm_ptr).value, rank, procs, 20, 20)

