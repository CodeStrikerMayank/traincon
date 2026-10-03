import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))


def Activate_neuron():
    # user data
    x = np.array([[2, 3, 4], [4, 5, 6], [5, 6, 7], [6, 7, 8]])
    print(x.shape)
    y = np.array([[1], [0], [1], [1]])
    # layer 1
    """Layer 1 it will contain 4 neuron """
    l1w = np.random.randn(3, 4) * 0.1
    b1w = np.zeros((1,4))
    n1 = np.dot(x,l1w)+b1w
    result1 = sigmoid(n1)
    # layer 2
    """LAyer 2 it will contain 1 neuron"""
    l2w = np.random.randn(4,1)*.1
    b2w = np.zeros((1,1))
    n2 = np.dot(result1,l2w)+b2w
    result2 = sigmoid(n2)
    print(result2)
    print(result2.shape)
    # layer 3
    """ Layer 3 it will conatin 1 neuron """
    l3w = np.random.randn(1,1)*.1
    b3w = np.zeros((4,1))
    result3=sigmoid(np.dot(result2,l3w)+b3w)
    print(result3)
    print(result3.shape)
    # final output prediction
    """Layer 3 output to final result in 1 output neuron"""
    o1 = np.random.randn(1,1)*.1
    bg = np.zeros((1,1))
    final_result = sigmoid(np.dot(result3,o1.T))
    print(final_result)
Activate_neuron()
