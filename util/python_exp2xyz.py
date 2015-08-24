import numpy as np
import sys
import math

connection = [
            (0, 1), 
            (0, 5), 
            (0, 9), 
            (1, 2), 
            (2, 3), 
            (3, 4), 
            (5, 6), 
            (6, 7), 
            (7, 8), 
            (9, 10), 
            (9, 14), 
            (10, 11), 
            (11, 12), 
            (12, 13),
            (14, 15),
            (15, 16),
            (16, 17)
            ]

joint_idx = {
			'pelvis': 0,
				'lfemur': 1,
					'ltibia': 2,
						'lfoot': 3,
							'ltoes': 4,
			'rfemur': 5,
				'rtibia': 6,
					'rfoot': 7,
						'rtoes': 8,
			'thorax': 9,
				'lclavicle': 10,
					'lhumerus': 11,
						'lradius': 12,
							'lhand': 13,
			'rclavicle': 14,
				'rhumerus': 15,
					'rradius': 16,
						'rhand': 17
			}

class MitSkeleton():
	"""Skeleton join structure of MIT dataset"""

	def __init__(self):
		self.numNodes = 18
		self.tree = [{} for i in range(self.numNodes)]
		self.type = 'mit'

		self.buildChildren()
		self.buildParents()
		self.buildDataIndices()

	def buildChildren(self):
		self.tree[0]['children'] = [1, 5, 9]
		self.tree[1]['children'] = [2]
		self.tree[2]['children'] = [3]
		self.tree[3]['children'] = [4]
		self.tree[4]['children'] = []
		self.tree[5]['children'] = [6]
		self.tree[6]['children'] = [7]
		self.tree[7]['children'] = [8]
		self.tree[8]['children'] = []
		self.tree[9]['children'] = [10, 14]
		self.tree[10]['children'] = [11]
		self.tree[11]['children'] = [12]
		self.tree[12]['children'] = [13]
		self.tree[13]['children'] = []
		self.tree[14]['children'] = [15]
		self.tree[15]['children'] = [16]
		self.tree[16]['children'] = [17]
		self.tree[17]['children'] = []

	def buildParents(self):
		self.tree[0]['parent'] = None
		self.tree[1]['parent'] = 0
		self.tree[2]['parent'] = 1
		self.tree[3]['parent'] = 2
		self.tree[4]['parent'] = 3
		self.tree[5]['parent'] = 0
		self.tree[6]['parent'] = 5
		self.tree[7]['parent'] = 6
		self.tree[8]['parent'] = 7
		self.tree[9]['parent'] = 0
		self.tree[10]['parent'] = 9
		self.tree[11]['parent'] = 10
		self.tree[12]['parent'] = 11
		self.tree[13]['parent'] = 12
		self.tree[14]['parent'] = 9
		self.tree[15]['parent'] = 14
		self.tree[16]['parent'] = 15
		self.tree[17]['parent'] = 16

	def buildDataIndices(self):
		for i in range(self.numNodes):
			self.tree[i]['or'] = np.arange(i*6, i*6+3)
			self.tree[i]['offset'] = np.arange(i*6+3, (i+1)*6)

def exp2xyz(channels):
	skel = MitSkeleton()
	if len(channels.shape) == 1:
		channels = channels.reshape((1, -1))

	channels = channels.T
	xyzStruct = [{} for i in range(skel.numNodes)]
	rotVal = channels[skel.tree[0]['or']]

	xyzStruct[0]['rot'] = expmap2rotmat(rotVal)
	xyzStruct[0]['xyz'] = channels[skel.tree[0]['offset']]

	for i in range(len(skel.tree[0]['children'])):
		node = skel.tree[0]['children'][i]
		updateChildXyz(skel, xyzStruct, node, channels)

	xyz = np.vstack(tuple([xyzStruct[i]['xyz'] for i in range(skel.numNodes)]))

	rot = [xyzStruct[i]['rot'] for i in range(skel.numNodes)]

	xyz = xyz.T
	channels = channels.T
	return xyz, rot

def updateChildXyz(skel, xyzStruct, node, channels):
	parent = skel.tree[node]['parent']
	children = skel.tree[node]['children']

	tdof = expmap2rotmat(channels[skel.tree[node]['or']]).transpose((1, 0, 2))

	xyzStruct[node]['rot'] = manyFrameDot(tdof, xyzStruct[parent]['rot'])

	conversion = manyFrameDot(channels[skel.tree[node]['offset']], xyzStruct[parent]['rot'])

	xyzStruct[node]['xyz'] = xyzStruct[parent]['xyz'] + conversion

	for i in range(len(children)):
		child = children[i]
		updateChildXyz(skel, xyzStruct, child, channels)

def expmap2rotmat(r):
	theta = np.linalg.norm(r, axis=0)
	r0 = r / (theta + np.spacing(1))

	r0x = np.zeros((3, 3, r.shape[1]))
	r0x[0, 1] = -r0[2]
	r0x[0, 2] = r0[1]
	r0x[1, 2] = -r0[0]

	identity = manyFrameEye(3, r.shape[1])

	r0x = r0x - np.transpose(r0x, axes=[1, 0, 2])
	r0x_squared = manyFrameDot(r0x, r0x)
	R = identity + np.sin(theta)*r0x + (1 - np.cos(theta))*r0x_squared

	return R

def manyFrameDot(a, b):
	lenA = len(a.shape)
	lenB = len(b.shape)

	a = a.transpose(tuple([lenA-1]) + tuple(range(lenA-1)))
	b = b.transpose(tuple([lenB-1]) + tuple(range(lenB-1)))

	result = np.array(map(lambda i: a[i].dot(b[i]), range(a.shape[0]))).transpose(tuple(range(1, lenA-1+lenB-2)) + tuple([0]))

	a = a.transpose(tuple(range(1, lenA)) + tuple([0]))
	b = b.transpose(tuple(range(1, lenB)) + tuple([0]))

	return result

def manyFrameEye(a, numFrames):
	identity = np.zeros((a, a, numFrames))
	for i in range(a):
		identity[i, i] += 1

	return identity