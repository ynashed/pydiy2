from pydiy2 import *
from mpi4py import MPI
try:
    import cPickle as pickle  # Use cPickle on Python 2.7
except ImportError:
    import pickle
import numpy as np
import scipy.fftpack as spf
import binascii
import tensorflow as tf
import os

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
procs = comm.Get_size()


os.chdir('/home/ynashed/workspace/sandbox/')

class myBlock(IBlock):
    def __init__(self):
        self.myRecon = None

def createBlock(gid):
    print "Block Created!"
    b = myBlock()
    blocks[gid] = b
    return b

def msgReceived(m,gid):
	dd = np.array(m, copy = False)
	gradsList.append( pickle.loads(binascii.rledecode_hqx(dd)) )

def mergeMSGReceived(m,gid):
    otherRecon = np.array(m, copy = False)
    myRecon = blocks[gid].myRecon

    G0 = np.gradient(abs(myRecon))[0] + np.gradient(abs(myRecon))[1]*1j
    G1 = np.gradient(abs(otherRecon))[0] + np.gradient(abs(otherRecon))[1]*1j
    F1 = spf.fft2(abs(G0))
    F2 = spf.fft2(abs(G1))

    pdm = F1*(F2.conjugate())/(abs(F1)*abs(F2.conjugate()))
    pcf = abs(spf.ifft2(pdm))
    px = pcf.argmax()/pcf.shape[1]
    py = pcf.argmax()%pcf.shape[1]

    I2 = myRecon[:-px]
    I1 = otherRecon[px:]
    # g1 = I1.mean()/I2.mean()
    # otherRecon*=g1

    blocks[gid].myRecon = np.concatenate((myRecon[:px+I1.shape[0]/2], otherRecon[(I1.shape[0]/2):]))

def sendMergeMSG(gid):
    block = blocks[gid]
    return DIY_MSG(block.myRecon)

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
simulated = True
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
procs = comm.Get_size()
sigSize=np.asarray([256,256])
diffSize=np.asarray([64,64])
cartesian_scan_dims=[11, 11] # Used to generate Cartesian scan mesh
diffNum = cartesian_scan_dims[0]*cartesian_scan_dims[1]
ePIEshifts = (np.genfromtxt('ptycho/sd_pos.csv', delimiter=',')).astype(int)
probe = np.genfromtxt('ptycho/sd_probe.csv', delimiter=',', dtype=complex)
diffs = np.fromfile('ptycho/sd_diffs.bin', dtype=np.float32).reshape( (diffNum, diffSize[0], diffSize[1]) )

blocks = {}
myDIY = PyDIY2(createBlock)
myDIY.decompose(2, [0,0], cartesian_scan_dims )
neighborNum = blocks.values()[0].getNeighborNum()
boundsMin = blocks.values()[0].getBoundsMin()
boundsMax = blocks.values()[0].getBoundsMax()
xMin, yMin = boundsMin[0], boundsMin[1]
xMax, yMax = boundsMax[0], boundsMax[1]

diffIndeces = []
for i in xrange(xMin,xMax):
	for j in xrange(yMin,yMax):
		diffIndeces.append( (i*cartesian_scan_dims[1])+j )
diffNum = len(diffIndeces)
diffs = diffs[diffIndeces]
ePIEshifts = ePIEshifts[diffIndeces]
mask = np.zeros(sigSize)
mask[   ePIEshifts[0][0]:ePIEshifts[-1][0]+diffSize[0],
        ePIEshifts[0][1]:ePIEshifts[-1][1]+diffSize[1]] = 1
shifts = []
for i in xrange(diffNum):
    print 'Expanding indeces for pattern# {:d}'.format(i)
    shifts.append(tfposlist(ePIEshifts[i], diffSize, sigSize, False))
shifts = np.asarray(shifts)

##TF
neighborGrads = [tf.placeholder_with_default(np.zeros(sigSize,dtype=np.float32),sigSize) for _ in xrange(neighborNum) ]
mag = tf.Variable(tf.random_uniform(sigSize), tf.float32)
if not simulated:
    phase = tf.Variable(tf.random_uniform(sigSize), tf.float32)
    mag = tf.complex(mag, tf.zeros_like(mag)) * tf.exp(tf.complex(tf.zeros_like(phase), phase))
#WShared = (mag + neighborW)/(neighborNum+1)
WSerialized = tf.reshape(mag, [-1])
D = tf.constant(diffs, tf.float32)
pos = tf.constant(shifts, tf.int32)
P = tf.constant(probe, tf.float32 if simulated else tf.complex64)

views = tf.gather(WSerialized, pos)
views2D = P*tf.reshape(views, [diffNum, diffSize[0], diffSize[1]])
if simulated:
    views2D = tf.complex(views2D, tf.zeros_like(views2D))
ppgt = tf.complex_abs( tf.batch_fft2d(views2D) )
loss = tf.reduce_mean(tf.square(ppgt - D))

optimizer = tf.train.AdamOptimizer(learning_rate=0.8, beta1=0.9, beta2=0.99)

# myGrad = optimizer.compute_gradients(loss)
# grads = [ tf.expand_dims(myGrad[0][0], 0) ]
# # Average neighbors gradients
# for grad in neighborGrads:
#     grads.append( tf.expand_dims(grad, 0) )

# grads = tf.reduce_sum(grads, 0)
# grads = [ (grads[0],myGrad[0][1]) ]
# # Apply the gradients to adjust the shared variables.
# train = optimizer.apply_gradients(grads)

# Minimize the mean squared errors.
train = optimizer.minimize(loss)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

for r in xrange(runs):
    init = tf.initialize_all_variables()
    sess = tf.InteractiveSession(config=config)
    init.run()

    for step in xrange(iterations):
        print "Run : " + str(r) + " Iteration : " + str(step)
        # if procs > 1:
        #     gradArray = myGrad[0][0].eval() * mask
        #     msgStr = binascii.rlecode_hqx(pickle.dumps(gradArray))
        #     myDIY.sendToNeighbors(DIY_MSG(msgStr))
        #     gradsList = []
        #     myDIY.recvFromNeighbors(msgReceived)

        train.run()#feed_dict={i: d for i, d in zip(neighborGrads, gradsList)})

    tfrecon = mag.eval()[   ePIEshifts[0][0]:ePIEshifts[-1][0]+diffSize[0],
                            ePIEshifts[0][1]:ePIEshifts[-1][1]+diffSize[1]]
    np.savetxt('recon%d_%d.csv'%(rank,r), tfrecon, delimiter=',')
    sess.close()

blocks.values()[0].myRecon = tfrecon
myDIY.mergeReduce(2, mergeMSGReceived, sendMergeMSG)
if rank == 0:
    np.savetxt('finalRecon%d.csv'%procs, blocks.values()[0].myRecon, delimiter=',')
