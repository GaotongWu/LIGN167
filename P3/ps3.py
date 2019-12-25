import numpy as np
import torch


######################################## BEGIN STARTER CODE ########################################

def relu(x):
	if x<0:
		return 0
	else:
		return x

def loss(y_predicted, y_observed):
	return (y_predicted - y_observed)**2


def mlp(x,W0,W1,W2):
	

	r0_0 = x*W0[0]
	r0_1 = x*W0[1]
	r0_2 = x*W0[2]
	r0 = np.array([r0_0,r0_1,r0_2])

	h0_0 = relu(r0_0)
	h0_1 = relu(r0_1)
	h0_2 = relu(r0_2)
	h0 = np.array([h0_0,h0_1,h0_2])

	

	r1_0 = h0_0*W1[0,0] + h0_1*W1[0,1]+ h0_2*W1[0,2]
	r1_1 = h0_0*W1[1,0] + h0_1*W1[1,1]+ h0_2*W1[1,2]
	r1_2 = h0_0*W1[2,0] + h0_1*W1[2,1]+ h0_2*W1[2,2]
	r1 = np.array([r1_0,r1_1,r1_2])

	h1_0 = relu(r1_0)
	h1_1 = relu(r1_1)
	h1_2 = relu(r1_2)
	h1 = np.array([h1_0,h1_1,h1_2])

	y_predicted = h1_0*W2[0] + h1_1*W2[1]+ h1_2*W2[2]

	variable_dict = {}
	variable_dict['x'] = x
	variable_dict['r0'] = r0
	variable_dict['h0'] = h0
	variable_dict['r1'] = r1
	variable_dict['h1'] = h1
	variable_dict['y_predicted'] = y_predicted

	return variable_dict


# x = 10
# W0 = np.array([1,2,3])
# W1 = np.array([[3,4,5],[-5,4,3],[3,4,1]])
# W2 = np.array([1,3,-3])

#print(mlp(x,W0,W1,W2))

###########  PyTorch code   ###########
def torch_mlp(x,W0,W1,W2):
	m = torch.nn.ReLU()
	h0 = m(torch.mul(W0,x))

	h1 = m(torch.matmul(W1,h0))

	y_predicted = torch.dot(W2,h1)

	return y_predicted

def torch_loss(y_predicted,y_observed):
	return torch.pow(y_predicted-y_observed,2)


# x_torch = torch.tensor(x,dtype=torch.float)
# W0_torch = torch.tensor(W0,dtype=torch.float,requires_grad=True)
# W1_torch = torch.tensor(W1,dtype=torch.float,requires_grad=True)
# W2_torch = torch.tensor(W2,dtype=torch.float,requires_grad=True)
# output = torch_mlp(x_torch,W0_torch,W1_torch,W2_torch)

########### END PyTorch code  ###########




######################################## END STARTER CODE ########################################


# NOTICE: DO NOT EDIT FUNCTION SIGNATURES 
# PLEASE FILL IN FREE RESPONSE AND CODE IN THE PROVIDED SPACES



#PROBLEM 1
def d_loss_d_ypredicted(variable_dict,y_observed):
    return 2*(variable_dict[‘y_predicted’]-y_observed)


#PROBLEM 2
def d_loss_d_W2(variable_dict,y_observed):
    return d_loss_d_ypredicted(variable_dict,y_observed)*variable_dict['h1']


#PROBLEM 3
def d_loss_d_h1(variable_dict,W2,y_observed):
    return d_loss_d_ypredicted(variable_dict,y_observed)*W2


#PROBLEM 4
def relu_derivative(x):
    x_d=np.zeros(len(x))
    for i in range(len(x)):
        if x[i]>0:
            x_d[i]=1
        else:
            x_d[i]=0
    return x_d


#PROBLEM 5
def d_loss_d_r1(variable_dict,W2,y_observed):
    return d_loss_d_h1(variable_dict,W2,y_observed)*relu_derivative(variable_dict['r1'])


#PROBLEM 6
def d_loss_d_W1(variable_dict,W2,y_observed):
    return np.outer(d_loss_d_r1(variable_dict,W2,y_observed),variable_dict['h0'])


#PROBLEM 7
def d_loss_d_h0(variable_dict,W1,W2,y_observed):
    return np.dot(d_loss_d_r1(variable_dict,W2,y_observed), W1)



#PROBLEM 8
def d_loss_d_r0(variable_dict,W1,W2,y_observed):
    return d_loss_d_h0(variable_dict,W1,W2,y_observed)*relu_derivative(variable_dict['r0'])


#PROBLEM 9
def d_loss_d_W0(variable_dict,W1,W2,y_observed):
    return d_loss_d_r0(variable_dict,W1,W2,y_observed)*variable_dict['x']


