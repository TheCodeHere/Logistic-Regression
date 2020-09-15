import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.axes as ax

def GetData():
	file = open("logReg_data.txt","r")

	data = []
	target = []
	for line in file:
		*X, y = (map(float,line.split()))

		#add bias
		data.append([1]+list(X))
		target.append([y])

	data = np.array(data)
	target = np.array(target)

	#Centering Data
	mean = np.mean(data, axis=0)
	mean[0] = 0
	data -= mean

	return data,target

def Ploting(tit = "default"):
	# add grid
	plt.grid(True,linestyle='--')

	# add title
	plt.title(tit)
	plt.tight_layout()

	# add x,y axes labels
	plt.xlabel('X1')
	plt.ylabel('X2')
	#plt.legend(loc='upper left')


sigmoid = lambda x: 1 / (1 + np.exp(-x)) #FUNCION SIGMOIDE

def Grad_LogReg(X,Y,epochs = 2000,alpha=0.01):
	print("##########################  Gradient Descent ##########################")
	# Constants
	m,n = X.shape
	epsilon = 0.00001 # Suppress runtimewarning divide by zero encountered in log

	Theta = np.random.rand(n, 1)

	cost = []
	for _ in range(epochs):

		H = sigmoid(np.dot(X, Theta))

		# Calculating Cost
		J = -( np.dot(Y.T,np.log(H)) / m + np.dot((1-Y).T,np.log(1 - H + epsilon)) / m )

		cost.append(*J[0])
		#print("COST FUNCTION: ",cost[-1])

		Error = H - Y

		# Calculate New Theta
		DJ = np.dot(X.T,Error)
		Theta = Theta - (alpha/m) * DJ

	print("Final Cost Function value: ", cost[-1])

	print("\nFinal Theta value:\n", Theta, "\n")

	# plot Decision Boundary
	eje_x = np.linspace(X[:, 1].min(), X[:, 1].max(), 100)
	eje_y = - (Theta[0] + Theta[1] * eje_x) / Theta[2] ### x2
	plt.plot(eje_x, eje_y , '--r', label='Decision Boundary')
	Ploting("Logistic Regression Gradient")

	# plot the cost function
	plt.figure()
	plt.plot(list(range(epochs)), cost, '-r')
	Ploting("Cost Function - LogR Gradient")

def Grad_Reg_LogReg(X,Y,epochs=2000,alpha=0.01,lamb=100):
	print("##########################  Gradient + Regularization ##########################")

	#constants
	m, n = X.shape
	epsilon = 0.00001  # Suppress runtimewarning divide by zero encountered in log
	diag_Mat = np.diag([1] + [1-lamb*(alpha/m)]*(n-1))

	Theta = np.random.rand(n, 1) + 0.1

	cost = []
	for _ in range(epochs):
		H = sigmoid(np.dot(X, Theta))

		# Calculating Cost
		J = -( np.dot(Y.T, np.log(H)) / m + np.dot((1 - Y).T, np.log(1 - H + epsilon)) / m )

		J += (lamb/(2*m))*np.dot(Theta[1:].T,Theta[1:])

		cost.append(*J[0])
		# print("COST FUNCTION: ",cost[-1])

		# Calculate New Theta
		Error = H - Y

		DJ = np.dot(X.T, Error)
		Theta = np.dot(diag_Mat,Theta) - (alpha / m) * DJ

	print("Final Cost Function value: ", cost[-1])

	print("\nFinal Theta value:\n", Theta, "\n")

	# plot line aproximation
	eje_x = np.linspace(X[:, 1].min(), X[:, 1].max(), 100)
	eje_y = - (Theta[0] + Theta[1] * eje_x) / Theta[2]  ### x2
	plt.plot(eje_x, eje_y, '--r', label='Decision Boundary')
	Ploting("Logistic Regression Gradient + Regularization")

	# plot the cost function
	plt.figure()
	plt.plot(list(range(epochs)), cost, '-r')
	Ploting("Cost Function - LogR Gradient + Regularization")

if __name__ == '__main__':
	X,y = GetData()

	classes = [int(i[0]) for i in y]
	#print("X:\n", X, "\ny:\n", y, "\nyp:\n", classes)

	# New figure
	plt.figure()
	plt.scatter(X.T[1], X.T[2], s=35, marker='.', c=classes)
	Grad_LogReg(X,y)

	plt.figure()
	plt.scatter(X.T[1], X.T[2], s=35, marker='.', c=classes)
	Grad_Reg_LogReg(X, y)

	plt.show()
	

	

	