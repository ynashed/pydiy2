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

blocks = []
def blockCreated(b):
	print "Block Created!"
	blocks.append(b)

def msgReceived(m):
	dd = np.array(m, copy = False)
	print pickle.loads(binascii.rledecode_hqx(dd)).id


myDIY = PyDIY2(blockCreated,msgReceived)
myDIY.decompose(2, [0,0], [20,20], ghosts=[4,4] )
print "Neighbor# : " + str(blocks[0].getNeighborNum())

msgStr = binascii.rlecode_hqx(pickle.dumps(ToSerialize(rank), 2))
myDIY.sendToNeighbors(DIY_MSG(msgStr))
myDIY.recvFromNeighbors()


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

