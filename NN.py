import random
import numpy as np
from copy import deepcopy
class NN(object):
	"""docstring for NN"""
	def __init__(self, structure, bias):#Structure is an array, e.g. '[5,15,1]',Bias is an array, e.g. '[1,1]'
		self.structure = structure
		self.bias = bias
	def randSWM(self,): #initializs_a_sequence_of_weight_matrixes
		self.swm=[]
		numberoflayers=len(self.structure)
		for i in range(numberoflayers-1): 	# loop through each layer and create weighted edge matrix between every two neighbored layers 
			self.swm.append(np.matrix([[random.random()*2-1 for y in range(self.structure[i+1])] for x in range(self.structure[i]+(self.bias[i]!=0))]))
	def zeroSDWM(self,): #initializs_a_sequence_of_delta_weight_matrixes (All zeros)
		self.sdwm=[]
		numberoflayers=len(self.structure)
		for i in range(numberoflayers-1): 	# loop through each layer and create weighted edge matrix between every two neighbored layers
			self.sdwm.append(np.matrix([[0 for y in range(self.structure[i+1])] for x in range(self.structure[i]+(self.bias[i]!=0))]))
	def sig(self,x):#logsig_function
		return 1/(1+np.exp(-x))
	def ff(self,inputx): #feed_forward_function
		x=deepcopy(inputx)
		l=len(self.swm)
		self.soa=[]
		for i in range(l-1):
			if self.bias[i]!=0:
				x.append(self.bias[i])
				x=self.sig(np.matrix(x)*self.swm[i])
				x=x.tolist()[0]
			else:
				x=self.sig(np.matrix(x)*self.swm[i])
				x=x.tolist()[0]
			self.soa.append(deepcopy(x))
		if self.bias[-1]!=0:
			x.append(self.bias[-1])
			x=np.matrix(x)*self.swm[-1]
			x=x.tolist()[0]
		else:
			x=np.matrix(x)*self.swm[-1]
			x=x.tolist()[0]
		self.soa.append(deepcopy(x)) # ouput is a sequence_of_output_arrays
	def backp(self,inputx,target,learninggain,momentumgain): #Backpropagation
		x=deepcopy(inputx)
		self.ff(x)
		l=len(self.soa)
		e=target-np.matrix(self.soa[-1])	# Error between ouput and target
		for i in range(l-2,-1,-1):
			tia=deepcopy(self.soa[i])
			if self.bias[i+1]!=0:
				tia.append(self.bias[i+1])
			tia=np.matrix(tia)
			self.sdwm[i+1]=learninggain*(tia.transpose())*e+momentumgain*self.sdwm[i+1]
			self.tv= e.transpose()
			e=(self.swm[i+1]*np.matrix(e).transpose()).transpose().tolist()[0]
			e=np.multiply(np.multiply(np.matrix(self.soa[i]),(1-np.matrix(self.soa[i]))).tolist()[0],e[0:len(e)-(self.bias[i+1]!=0)])
		tia=deepcopy(inputx)
		if self.bias[0]!=0:
			tia=np.append(tia,self.bias[0])
		tia=np.matrix(tia)
		self.sdwm[0]=learninggain*(tia.transpose())*e+momentumgain*self.sdwm[0]
	def batchtraining(self,):
		self.zeroSDWM()
		tszwm=deepcopy(self.sdwm)
		for ep in range(100):
			tsdwm=deepcopy(tszwm)
			for x in range(-100,100,1):
				self.backp([float(x)/100],[float(x*x)/10000],0.1,0.5)
				tsdwm=[tsdwm[i]+self.sdwm[i] for i in range(len(tsdwm))]
			tsdwm=[tsdwm[i]/199 for i in range(len(tsdwm))]
			self.swm=[self.swm[i]+tsdwm[i] for i in range(len(tsdwm))]

a=NN([1,5,1],[1,0])
a.randSWM()
#a.zeroSDWM()
a.batchtraining()
