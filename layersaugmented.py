from layer import NNlayer
import numpy as np

""" Class: Fullyconnected_dropout_layer
 This class defines a fully connected neural network layer with dropout.
 Inputs:
 - input_dim: The number of input features to the layer.
 - output_dim: The number of neurons (output units) in the layer.
 - activation_function: The activation function to apply to the layer's output.
 - function_derivative: The derivative of the activation function for backpropagation.
 - learningrate: The learning rate used for weight updates during training (default 0.1).
 - drop_properpility: The probability of dropping a neuron during training (dropout rate, default 0.5).
 It initializes weights, biases, and handles forward and backward propagation with dropout.
 Another difference to the basic NNlayer class is that it applies L2 regularisation
 The generate_random_row_matrix is used to generate a matrix where each row is randomly with the drop_properbility assigned either 1s or Os 
 allowing the deactivation of nods during the training process
 """
class Fullyconnected_dropout_layer(NNlayer):
    def __init__(self, input_dim, output_dim, activation_function, function_derivative, learningrate=0.1, drop_properpility=0.5):
        self.weigth_Array = np.random.randn(output_dim, input_dim) * np.sqrt(2. / input_dim)
        self.bias = np.zeros(output_dim)
        self.f = activation_function
        self.dF = function_derivative
        self.lr = learningrate
        self.drop_prop=drop_properpility
        self.cache = None
        self.mask=self.generate_random_row_matrix(self.weigth_Array.shape,self.drop_prop)
    def generate_random_row_matrix(self,shape, prob):
        while True:
                matrix = np.zeros(shape, dtype=int) 
                for i in range(shape[0]):
                    if np.random.rand() < prob:
                        matrix[i, :] = 1 
                if np.any(matrix == 1):
                    break  
        return matrix

    def forward(self, input,training=False):

        matrix_result = np.dot(self.weigth_Array, input) + self.bias
        output = self.f(matrix_result)
        if training:
            output=output*self.mask[:, 0]*(1.0/1.0-self.drop_prop)

        self.cache = (input, matrix_result, output)
        return output

    def backward(self, gradientchain, regulartion=0):
        X, Z, A = self.cache
        dZ = gradientchain * self.dF(Z)
        dW = np.outer(dZ, X) + regulartion * self.weigth_Array
        db = np.sum(dZ, axis=0)*self.mask[:, 0]
        dX = np.dot(self.weigth_Array.T, dZ)

        dW = np.clip(dW, -1e5, 1e5)
        dW=dW*self.mask
        db = np.clip(db, -1e5, 1e5)

        self.weigth_Array -= self.lr * dW
        self.bias -= self.lr * db

        return dX
"""Class: Layer_V2
    This class extends Fullyconnected_dropout_layer by adding the Adam optimizer for adaptive weight updates.
    Inputs:
        - beta1, beta2: Adam optimizer hyperparameters for momentum and RMSProp (default values 0.9, 0.999).
        - epsi: Small value to avoid division by zero in Adam updates (default 0.000001).
        - use_adam: Flag indicating whether to use Adam optimizer or regular gradient descent.
"""   
class Layer_V2(Fullyconnected_dropout_layer):
    def __init__(self, input_dim, output_dim, activation_function, function_derivative, learningrate, drop_properpility,beta1=0.9,beta2=0.999,epsi=0.000001,use_adam=True):
        super().__init__(input_dim, output_dim, activation_function, function_derivative, learningrate, drop_properpility)
        self.beta1=beta1
        self.beta2=beta2
        self.epsi=epsi
        self.use_adam = use_adam
        self.m = np.zeros_like(self.weigth_Array)
        self.v = np.zeros_like(self.weigth_Array)  
    def backward(self, gradientchain, regulartion=0,timestep=1):
        X, Z, A = self.cache
        dZ = gradientchain * self.dF(Z)
        dW = np.outer(dZ, X) + regulartion * self.weigth_Array
        db = np.sum(dZ, axis=0)*self.mask[:, 0]
        dX = np.dot(self.weigth_Array.T, dZ)

        dW = np.clip(dW, -1e5, 1e5)
        dW=dW*self.mask
        db = np.clip(db, -1e5, 1e5)
        if(self.use_adam):
            self.m=self.beta1*self.m+(1-self.beta1)*dW
            self.v=self.beta2*self.v+(1-self.beta2)*(dW**2)
            m_hat=self.m/(1-(self.beta1)**timestep)
            v_hat=self.v/(1-(self.beta2)**timestep)
            self.weigth_Array -= self.lr * m_hat/(np.sqrt(v_hat)+self.epsi)
            self.bias -= self.lr * db
        else:
            self.weigth_Array -= self.lr * dW
            self.bias -= self.lr * db
        return dX
    
""" Class: FinalLayer_V2
 This class represents the final output layer of a neural network, extended to support Adam optimizer.
 Inputs are similar to Layer_V2 but without dropout.
"""    
class FinalLayer_V2(NNlayer):
    def __init__(self, input_dim, output_dim, activation_function, function_derivative, learningrate, beta1=0.9,beta2=0.999,epsi=0.000001,use_adam=True):
        super().__init__(input_dim, output_dim, activation_function, function_derivative, learningrate)
        self.beta1=beta1
        self.beta2=beta2
        self.epsi=epsi
        self.use_adam = use_adam,
        self.m = np.zeros_like(self.weigth_Array) 
        self.v = np.zeros_like(self.weigth_Array)  
    def backward(self, gradientchain, regulartion=0,timestep=1):
        X, Z, A = self.cache
        dZ = gradientchain * self.dF(Z)
        dW = np.outer(dZ, X) + regulartion * self.weigth_Array
        db = np.sum(dZ, axis=0)
        dX = np.dot(self.weigth_Array.T, dZ)

        dW = np.clip(dW, -1e5, 1e5)

        db = np.clip(db, -1e5, 1e5)
        if(self.use_adam):
            self.m=self.beta1*self.m+(1-self.beta1)*dW
            self.v=self.beta2*self.v+(1-self.beta2)*(dW**2)
            m_hat=self.m/(1-(self.beta1)**timestep)
            v_hat=self.v/(1-(self.beta2)**timestep)
            self.weigth_Array -= self.lr * m_hat/(np.sqrt(v_hat)+self.epsi)
            self.bias -= self.lr * db
        else:
            self.weigth_Array -= self.lr * dW
            self.bias -= self.lr * db
        return dX