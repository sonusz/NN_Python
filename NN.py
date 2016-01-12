import random
import numpy as np

class NN(object):
	"""docstring for NN"""
	def __init__(self, structure, bias):#Structure is an array, e.g. '[5,15,1]',Bias is an array, e.g. '[1,1]'
		self.structure = structure
		self.bias = bias
	def randSWM(): #initializs_a_sequence_of_weight_matrixes
		self.swm=[]
		numberoflayers=len(self.structure)
		for i in range(numberoflayers-2): 	# loop through each layer and create weighted edge matrix between every two neighbored layers 
			self.swm.append(np.matrix([[random.random()*2-1 for y in range(structure(i+1))] for x in range(structure(i))]))
	def zeroSDWM(): #initializs_a_sequence_of_delta_weight_matrixes (All zeros)
		self.sdwm=[]
		numberoflayers=len(structure)
		for i in range(numberoflayers-2): 	# loop through each layer and create weighted edge matrix between every two neighbored layers
			self.sdwm.append(np.matrix([[0 for y in range(structure(i+1))] for x in range(structure(i))]))
	def sig(x):#logsig_function
		return 1/(1+np.exp(-x))
	def ff(x): #feed_forward_function
		l=len(self.swm)
		self.soa=[]
		for i in range(l-2):
			if self.bias[i]!=0:
				x=sig(np.append(x,self.bias[i])*self.swm[i])
			else:
				x=sig(x*self.swm[i])
			self.soa.append(x)
		if self.bias[-1]!=0:
			x=np.append(x,self.bias[-1])*self.swm[-1]
		else:
			x=x*self.swm[-1]
		self.soa.append(x) # ouput is a sequence_of_output_arrays
	def backp(x,target,learninggain,momentumgain): #Backpropagation
		l=len(self.soa)
		e=target-self.soa[-1]	# Error between ouput and target
		for i in range(l-2,-1,-1):
			tia=self.soa[i]										# temp_input_array
			if self.bias[i+1]!=0:
				tia=np.append(tia,self.bias[i+1])
			self.sdwm(i+1)=learninggain*(tia.transpose())*e+momentumgain*self.sdwm(i+1)
			e=self.swm(i+1).transpose()*e
			e=np.multiply(np.multiply(self.soa[i],(1-self.soa[i])),e[0:-1-(self.bias[i+1]!=0)])
		tia=x
		if self.bias[0]!=0:
			tia=np.append(tia,self.bias[0])
		self.sdwm[0]=learninggain*(tia.transpose())*e+momentumgain*self.sdwm[0]
		