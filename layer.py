
import numpy as np
"""class layer:
    Inputs:
        -input_dim: the dimension / number of nodes of the prior layer,
        -output_dim: the number of nodes in the this layer
        -activation_function: a function used on the sum over input*weight_Array
        -function_derivative: the derivderivative of the activation function used in back propergation
        -learningrate: hyperparameter
    for a more efficent calculation the input and the matrixresult are cached and used inthe backward function
    depending on the used activaiton function diffrent initial weights should be used
"""
class NNlayer:
    _id_counter = 0
    def __init__(self,input_dim,output_dim,activation_function,function_derivative,learningrate=0.1):
       #init of sigmoid:
       #self.weigth_Array=np.random.normal(0.0,pow(input_dim,-0.5),size=( output_dim,input_dim))
       #for relu:
       self.weigth_Array = np.random.randn(output_dim, input_dim) * np.sqrt(2. / input_dim)
       self.bias=np.zeros(output_dim)
       self.f=activation_function
       self.dF=function_derivative
       self.lr=learningrate
       self.id=NNlayer._get_next_id()
       self.cache= None
    def forward(self,input):
        matrix_result= np.dot(self.weigth_Array,input)+self.bias
        output=self.f(matrix_result)
        self.cache=(input,matrix_result,output)
        return output
    def backward(self,gradientchain,regulartion=0):
        X, Z, A = self.cache
        dZ = gradientchain * self.dF(Z)
        dW = np.outer( dZ,X)+regulartion * self.weigth_Array
        db = np.sum(dZ, axis=0)
        dX = np.dot( self.weigth_Array.T,dZ)

        self.weigth_Array -= self.lr * dW
        self.bias -= self.lr * db

        return dX
    @classmethod
    def _get_next_id(cls):
        unique_id = cls._id_counter
        cls._id_counter += 1
        return unique_id
