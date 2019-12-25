#!/usr/bin/env python
# coding: utf-8

# In[128]:


import numpy as np
################################ BEGIN STARTER CODE ################################################
def sigmoid(x):
	#Numerically stable sigmoid function.
	#Taken from: https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
	if x >= 0:
		z = np.exp(-x)
		return 1 / (1 + z)
	else:
		# if x is less than zero then z will be small, denom can't be
		# zero because it's 1+z.
		z = np.exp(x)
		return z / (1 + z)


def sample_logistic_distribution(x,a):
	#np.random.seed(1)
	num_samples = len(x)
	y = np.empty(num_samples)
	for i in range(num_samples):
		y[i] = np.random.binomial(1,logistic_positive_prob(x[i],a))
	return y

def create_input_values(dim,num_samples):
	#np.random.seed(100)
	x_inputs = []
	for i in range(num_samples):
		x = 10*np.random.rand(dim)-5
		x_inputs.append(x)
	return x_inputs


def create_dataset():
	x= create_input_values(2,100)
	a= np.array([12,12])
	y= sample_logistic_distribution(x,a)

	return x,y

################################ END STARTER CODE ################################################


# In[129]:


# PROBLEM 1
def logistic_positive_prob(x,a):
    return sigmoid(np.dot(x,a))


# In[130]:


# PROBLEM 2
def logistic_derivative_per_datapoint(y_i,x_i,a,j):
    return -(y_i-logistic_positive_prob(x_i,a))*x_i[j]


# In[131]:


# PROBLEM 3
def logistic_partial_derivative(y,x,a,j):
    n=len(y)
    loss_per_datapoint=0
    for i in range(n):
        loss_per_datapoint=loss_per_datapoint+logistic_derivative_per_datapoint(y[i],x[i,:],a,j)
    return loss_per_datapoint


# In[132]:


# PROBLEM 4
def compute_logistic_gradient(a,y,x):
    k=len(a)
    logistic_gradient=np.zeros(k)
    for j in range(k):
        logistic_gradient[j]=logistic_partial_derivative(y,x,a,j)
    return logistic_gradient


# In[133]:


# PROBLEM 5
def gradient_update(a,lr,gradient):
    return a-lr*gradient


# In[134]:


# PROBLEM 6
def gradient_descent_logistic(initial_a,lr,num_iterations,y,x):
    a=initial_a
    for i in range(num_iterations):
        gradient=compute_logistic_gradient(a,y,x)
        a=gradient_update(a,lr,gradient)
    return a


# In[144]:


# PROBLEM 7
# Please include your generated graphs in this zipped folder when you submit.
# Comment out your calls to matplotlib (e.g. plt.show()) before submitting

#import matplotlib.pyplot as plt
#f=create_dataset()
#a=f[0]
#a = np.array(a)
#a[:,0]
#plt.scatter(a[:,0], a[:,1],c=f[1])


# In[159]:


#Problem 8
#Free Response: values of a are shown below. But this suggests that gradient descent
#does not necessarily accurately solves the classification problem.

#for i in range(10):
#   f=create_dataset()
#   a=gradient_descent_logistic(np.array([-2,-2]),0.01,1000,np.array(f[1]),np.array(f[0]))
#   print(a)
#   List of answers 
#[5.70208223 5.52446843]
#[5.65381557 5.70444864]
#[6.22523563 6.40725262]
#[6.07101146 5.98213173]
#[5.88896766 5.58980851]
#[5.82799887 6.22079515]
#[5.40395399 5.37112151]
#[6.10035299 5.75511076]
#[5.20209542 5.23994593]
#[5.92037476 5.20224912]


# In[167]:


#Problem 9
# Free Response: For different values of initial_a, we got similar final results of a
# This suggests that gradient descent on logistic regression gradually approaches and converges to the optimal value.

#for i in range(10):
#    f=create_dataset()
#   a=gradient_descent_logistic(np.array([-1,0]),0.01,1000,np.array(f[1]),np.array(f[0]))
#   print(a)
#[5.61597428 5.73067626]
#[5.7402167  5.83870345]
#[6.74457053 6.65791552]
#[5.6644828  5.69451326]
#[5.27610915 5.56228019]
#[5.31553746 5.17765432]
#[5.94497813 6.16127001]
#[5.91896499 6.1198036 ]
#[5.97128423 6.23808712]
#[5.00344348 4.77139051]


# In[168]:


#for i in range(10):
#    f=create_dataset()
#    a=gradient_descent_logistic(np.array([0,-1]),0.01,1000,np.array(f[1]),np.array(f[0]))
#    print(a)
#[6.87902176 6.87498866]
#[5.42546393 5.34634858]
#[5.38245792 5.65915505]
#[5.70911194 5.66495141]
#[5.42650492 5.85262491]
#[5.88403338 5.78474049]
#[6.07259861 6.100518  ]
#[6.11121326 5.66715694]
#[6.08261205 5.99898046]
#[5.91092637 6.2596206 ]


# In[277]:


#Problem 10
#Free Response: different learning rates will leads to different results of convergence; if the learning rate is too
#large, it may cause over-shootting

#for i in range(10):
#   f=create_dataset()
#   a=gradient_descent_logistic(np.array([-2,-2]),0.0005,1000,np.array(f[1]),np.array(f[0]))
#   print(a)
#[2.12259201 2.06244364]
#[2.23384523 1.79235841]
#[2.04917296 2.1725694 ]
#[2.28327914 2.22634385]
#[2.30438868 2.3387156 ]
#[2.18837957 2.0218544 ]
#[2.0847979  2.23281578]
#[2.20188291 2.27327705]
#[1.93325729 2.0668768 ]
#[2.04800164 2.41937009]


# In[298]:


#for i in range(10):
#   f=create_dataset()
#   a=gradient_descent_logistic(np.array([-2,-2]),0.1,1000,np.array(f[1]),np.array(f[0]))
#   print(a)
#[17.42222481 16.24319007]
#[16.3224952  16.62717695]
#[18.01523448 16.70389157]
#[16.44561349 17.16793928]
#[21.03442249 22.40611219]
#[21.72966263 21.13322373]
#[7.39401358 6.83291083]
#[16.67976693 19.4572748 ]
#[16.00350929 17.44229513]
#[17.17448206 16.44159458]


# In[353]:


# Problem 11
def logistic_l2_partial_derivative(y,x,a,j):
    loss = logistic_partial_derivative(y,x,a,j)
    for i in a:
        loss=loss+2*0.15*i
    return loss


# In[354]:


# Problem 12
def compute_logistic_l2_gradient(a,y,x):
    k=len(a)
    logistic_l2_gradient=np.zeros(k)
    for j in range(k):
        logistic_l2_gradient[j]=logistic_l2_partial_derivative(y,x,a,j)
    return logistic_l2_gradient


# In[355]:


# PROBLEM 13
def gradient_descent(initial_a,lr,num_iterations,y,x,gradient_fn):
    a=initial_a
    for i in range(num_iterations):
        gradient=gradient_fn(a,y,x)
        a=gradient_update(a,lr,gradient)
    return a


# In[337]:


# Problem 14
#Free Response: The gradient values are smaller compared to the results in Problem 8
#This is because the added penality term prevents the results from going too high and avoids overfitting 

#f=create_dataset()
#gradient_descent([-2,-2],0.01,1000,np.array(f[1]),np.array(f[0]),compute_logistic_l2_gradient)
#array([2.45629553, 2.39825753])


# In[338]:


#f=create_dataset()
#gradient_descent([-2,-2],0.01,1000,np.array(f[1]),np.array(f[0]),compute_logistic_l2_gradient)
#array([2.20677192, 2.31944198])


# In[339]:


#f=create_dataset()
#gradient_descent([-2,-2],0.01,1000,np.array(f[1]),np.array(f[0]),compute_logistic_l2_gradient)
#array([2.27352728, 2.26159292])


# In[379]:


#Problem 15
#Free Response: With increasing lamda, the final results of a is decreasing; this is because bigger lambda leads 
#to larger level of penality and prevent weights from growing too large.

#f=create_dataset()
#gradient_descent([-2,-2],0.01,1000,np.array(f[1]),np.array(f[0]),compute_logistic_l2_gradient) # lambda=0.01
#array([5.12961112, 5.01813834])


# In[381]:


#f=create_dataset()
#gradient_descent([-2,-2],0.01,1000,np.array(f[1]),np.array(f[0]),compute_logistic_l2_gradient) # lambda=1
#array([1.07467529, 1.24119452])


# In[373]:


#f=create_dataset()
#gradient_descent([-2,-2],0.01,1000,np.array(f[1]),np.array(f[0]),compute_logistic_l2_gradient) # lambda=10
#array([0.52085363, 0.46639503])

