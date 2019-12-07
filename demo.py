'''
This code is an implementation for the system state model specified in the paper
Dataset Model to Predict Bound Error for Particle Filter based Kullback-Leibler Distance
by L.T.Nga, T.L.Thuong, P.V.Qui.
'''

import numpy as np 

class Simulation:
	def __init__(self,init_state=np.array([[0,0,.4,-.05]]).T,v_min=1,v_max=5,Pref=-23,K=45,R=.5,Q=.5,time_seg=40,use_seed=True):
		'''
		--- Constructor ------
		Attributes:
			init_state - array-like of shape (4,1) 
			initial state
			Default: [[0,0,0.4,-0.05]].T

			v_min: float
			the minimum velocity of considered object
			Default: 1

			v_max: float
			the maximum velocity of considered object
			Default: 5

			Pref: float 
			reference value of RSS
			Default: -23

			K: float 
			the factor in path loss
			Default: 45

			R: float 
			standard noise deviation for measurement
			Default: 0.5

			Q: float 
			standard noise deviation for state
			Default: 0.5

			time_seg: int
			time segment
			Default: 40

			use_seed: boolean
			allow to use deterministic or not (useful for debugging).
			Default: True

			state_mat: array-like of shape (4,4)
			indicated in the paper

			noise_mat: array-like of shape(4,2)
			indicated in the paper
		'''
		self.state_mat = np.array([[1,1,0,0],
								   [0,1,0,0],
								   [0,0,1,1],
								   [0,0,0,1]])

		self.noise_mat = np.array([[.5,0],
			                       [1,0],
			                       [0,.5],
			                       [0,1]])

		self.init_state = init_state
		self.v_min = v_min
		self.v_max = v_max
		self.Pref = Pref
		self.K = K
		self.R = R
		self.Q = Q
		self.time_seg = time_seg

		if use_seed:
			np.random.seed(0)

	def uniform_sampling(self,low,high,size=None):
		'''
		Uniform sampling: U[low, high]

		Params:
			low: float
			lower bound

			high: float
			upper bound

			size: None or tuple
			sample size

		Returns: array-like/float
			if None, return an array with given size. Otherwise, return a float
		'''
		if size is None:
			return (low - high)*np.random.random() + low
		return (low - high)*np.random.random(size) + low

	def run(self,t=10):
		'''
		Execute simulation

		Params:
			t: int
			timestep

		Returns: tuple of lists
			xs: list
			contains state info

			zs: list
			contains measurement info
		'''
		xs = []
		zs = []
		for i in range(t):
			current_velocity = self.uniform_sampling(self.v_min,self.v_max,size=(4,1))
			if i==0:
				current_state = self.state_mat.dot(self.init_state + current_velocity*self.time_seg) + self.Q*self.noise_mat.dot(np.random.normal(size=(2,1)))
			else:
				current_state = self.state_mat.dot(prev_state + current_velocity*self.time_seg) + self.Q*self.noise_mat.dot(np.random.normal(size=(2,1)))
			measure = self.Pref + self.K*np.log(np.arctan(current_state[0,0]/current_state[2,0])) + self.R*np.random.normal()

			xs.append(current_state)
			zs.append(measure)

			prev_state = current_state
		return xs, zs

simulation = Simulation()
print(simulation.run())
