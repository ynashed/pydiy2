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

from mpi4py import MPI
import tensorflow as tf
import os, time, sys
import numpy as np
from pydiy2 import *

os.chdir('/home/ynashed/workspace/sandbox/')

blocks = []
def blockCreated(b):
	print "Block Created!"
	blocks.append(b)

def msgReceived(m):
	dd = np.array(m, copy = False)
	feed_dict[neighborW] += pickle.loads(binascii.rledecode_hqx(dd))


def tfposlist(shifts, diffSize, sigSize, in2D=False):
    if in2D:
        return (np.mgrid[shifts[0]:shifts[0]+diffSize[0], shifts[1]:shifts[1]+diffSize[1]].swapaxes(0,2)).swapaxes(0,1)
    else:
        result = []
        for i in xrange(diffSize[0]):
            for j in xrange(diffSize[1]):
                result.append( ((i+shifts[0])*sigSize[1])+shifts[1]+j )
        return np.asarray(result)

iterations = 200
runs = 1
simulated = False
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
procs = comm.Get_size()
sigSize=np.asarray([340,340])       #[256,256])
diffSize=np.asarray([256,256])      #[64,64])
cartesian_scan_dims=[26,26]         #[11, 11] # Used to generate Cartesian scan mesh
diffNum = cartesian_scan_dims[0]*cartesian_scan_dims[1]
ePIEshifts = (np.genfromtxt('ptycho/tp_pos.csv', delimiter=',')).astype(int)
probe = np.genfromtxt('ptycho/tp_probe.csv', delimiter=',', dtype=complex)
diffs = np.fromfile('ptycho/tp_diffs.bin', dtype=np.float32).reshape( (diffNum, diffSize[0], diffSize[1]) )

myDIY = PyDIY2(blockCreated,msgReceived)
bounds = myDIY.decompose(2, [0,0], cartesian_scan_dims )
neighborNum = blocks[0].getNeighborNum()
xMin = bounds[0]
xMax = bounds[1]
yMin = bounds[2]
yMax = bounds[3]
diffIndeces = []
for i in xrange(xMin,xMax):
	for j in xrange(yMin,yMax):
		diffIndeces.append( (i*cartesian_scan_dims[1])+j )
diffNum = len(diffIndeces)
diffs = diffs[diffIndeces]
ePIEshifts = ePIEshifts[diffIndeces]

shifts = []
for i in xrange(diffNum):
    print 'Expanding indeces for pattern# {:d}'.format(i)
    shifts.append(tfposlist(ePIEshifts[i], diffSize, sigSize, False))
shifts = np.asarray(shifts)

##TF
neighborW = tf.placeholder(tf.float32 if simulated else tf.complex64, shape=sigSize)
W = tf.get_variable("weights", sigSize, initializer=tf.random_uniform_initializer())
if not simulated:
    W = tf.complex(W, tf.ones_like(W))
WShared = (W + neighborW)/(neighborNum+1)
WSerialized = tf.reshape(WShared, [-1])
D = tf.constant(diffs, tf.float32)
pos = tf.constant(shifts, tf.int32)
P = tf.constant(probe, tf.float32 if simulated else tf.complex64)

views = tf.gather(WSerialized, pos)
views2D = P*tf.reshape(views, [diffNum, diffSize[0], diffSize[1]])
if simulated:
    views2D = tf.complex(views2D, tf.zeros_like(views2D))
ppgt = tf.complex_abs( tf.batch_fft2d(views2D) )
loss = tf.reduce_mean(tf.square(ppgt - D))

# Minimize the mean squared errors.
optimizer = tf.train.AdamOptimizer(learning_rate=0.8, beta1=0.9, beta2=0.99)
train = optimizer.minimize(loss)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

feed_dict={neighborW: np.zeros(sigSize, dtype=np.float32 if simulated else np.complex64)}
tf.get_variable_scope().reuse_variables()

for r in xrange(runs):
    init = tf.initialize_all_variables()
    sess = tf.InteractiveSession(config=config)
    init.run()

    for step in xrange(iterations):
    	print "Run : " + str(r) + " Iteration : " + str(step)

    	train.run(feed_dict=feed_dict)
    	reconTensor = tf.get_variable("weights")
    	tfrecon = reconTensor.eval()

    	if procs > 1:
    		feed_dict[neighborW] = np.zeros_like(feed_dict[neighborW])
    		neighborNum = 0
    		msgStr = binascii.rlecode_hqx(pickle.dumps(tfrecon))
    		myDIY.sendToNeighbors(DIY_MSG(msgStr))
    		myDIY.recvFromNeighbors()

    reconTensor =  tf.get_variable("weights")
    tfrecon = reconTensor.eval()
    np.savetxt('recon%d_%d.csv'%(rank,r), tfrecon, delimiter=',')
    sess.close()

finalRecon = np.zeros_like(tfrecon)
comm.Reduce(tfrecon, finalRecon, root=0)
if rank == 0:
    np.savetxt('finalRecon.csv', finalRecon, delimiter=',')