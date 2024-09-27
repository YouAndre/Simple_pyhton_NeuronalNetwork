from layer import NNlayer
import layersaugmented
import numpy as np
from matplotlib import pyplot as plt
class NeuronalNetwork:
    def __init__(self, initial_input, final_output, layers_number, activation_function, function_derivative, learning_rate,lambda_reg=0.001,decay_rate=0.001):
        self.initial_input = initial_input                # The initial input to the neural network
        self.final_output = final_output                  # The desired output from the neural network
        self.activation_function = activation_function    # The activation function to be used in each layer
        self.function_derivative = function_derivative    # The derivative of the activation function, used for backpropagation
        self.learning_rate = learning_rate                # The learning rate for the training process
        self.losschache=0.0
        self.regular_cache=0.0
        self.lambda_reg = lambda_reg
        self.decay_rate=decay_rate
        self.total_loss=[]
        self.validate_loss=[]
        self.layers=[]
        self.layers.append(NNlayer(initial_input,layers_number[0],activation_function,function_derivative,learning_rate))
        for x in range(len(layers_number)-1):
            self.layers.append(NNlayer(layers_number[x],layers_number[x+1],activation_function,function_derivative,learning_rate))
        self.layers.append(NNlayer(layers_number[-1],final_output,activation_function,function_derivative,learning_rate))
        
    def train(self,input_train,input_validate,epochs=10):
        assert epochs >= 2, "Number of epochs must be at least 2 for training."
        interval_size=int(np.log10(epochs))
        self.total_loss=[]
        self.validate_loss=[]
        for epoch in range(epochs):
            
            helpervar=0
            self.losschache=0
            for x in input_train.values():
                input,truth=x
                layer_outputs = input
                for layer in self.layers:
                    layer_outputs = layer.forward(layer_outputs)
                
                helpervar+=self.loss_function(layer_outputs,truth)
                
                self.backprop()
            self.learning_rate= (self.learning_rate/(1.0+np.exp(-self.decay_rate * epoch)))
            self.total_loss.append(helpervar)
            if (epoch%interval_size ==0):
                print(f'Epoch {epoch + 1}, Loss: {helpervar}',flush=True)
            self.validate_loss.append(self.validate(input_validate))
    
    def validate(self,input):
        loss=0
        for x in input.values():
            a,b=x
            layer_outputs = a
            for layer in self.layers:
                layer_outputs = layer.forward(layer_outputs)
            loss+=self.loss_function(layer_outputs,b)
        return loss      
    def backprop(self):
        gradient_chain =self.losschache
        #gradient_chain=self.output_layer.backward(gradient_chain)
        for layer in reversed(self.layers):
            #gradient_chain = layer.backward(gradient_chain,self.regular_cache)
            gradient_chain = layer.backward(gradient_chain)
    def querry(self,input):
        for layer in self.layers:
            input = layer.forward(input)
        return input
    def loss_function(self,prediciton,expected):
        loss=0
        loss=np.mean((prediciton-expected)**2)
        self.losschache+=2 * (prediciton-expected)
        return loss
    def softmax(self,input):
        exp_values = np.exp(input - np.max(input))  
        probabilities = exp_values / np.sum(exp_values)
        return probabilities
    def softmax_deriverative(input):
        helper= len(input)
        result=np.zeros((helper,helper))
        for x in range(len(input)):
            for y in range(len(input)):
                if(x==y):
                    result[x,y]=input[x]*(1-input[x])
                else:
                    result[x,y]=-(input[x]*input[y])
        return result
    def cross_entropy_derivative(predictions, labels):
        return predictions - labels
    def visualize_weights(self):
        num_layers = len(self.layers)
        fig = plt.figure(figsize=(14, 10)) 
        for i, layer in enumerate(self.layers):
            weight_array = layer.weigth_Array  # Get the weight array from the layer

            # 3D subplot for weight array
            ax3d = fig.add_subplot(2, num_layers, i + 1, projection='3d')
            x_axis = np.arange(weight_array.shape[1])
            y_axis = np.arange(weight_array.shape[0])
            x, y = np.meshgrid(x_axis, y_axis)
            ax3d.plot_surface(x, y, weight_array, cmap='viridis')
            ax3d.set_title(f'Layer {i + 1} Weights (3D)')

            # 2D subplot for weight array
            ax2d = fig.add_subplot(2, num_layers, i + 1 + num_layers)
            cax = ax2d.imshow(weight_array, cmap='viridis', aspect='auto')
            ax2d.set_title(f'Layer {i + 1} Weights (2D)')
            fig.colorbar(cax, ax=ax2d)

            # Add grid to the 2D plot with black color and thicker lines
            ax2d.grid(color='black', linestyle='-', linewidth=.05)  # Thicker lines
            
            # Add ticks to make grid lines more apparent
            ax2d.set_xticks(np.arange(weight_array.shape[1]))
            ax2d.set_yticks(np.arange(weight_array.shape[0]))
            
            ax2d.set_xticks(np.arange(weight_array.shape[1]), minor=True)
            ax2d.set_yticks(np.arange(weight_array.shape[0]), minor=True)

            ax2d.grid(which='minor', color='black', linestyle='-', linewidth=1.5)

        plt.tight_layout()
        plt.show()
    def evaluate_network_accuracy(self,valid_data):
        correct_pred=0
        samplenumber=len(valid_data)
        for x in range (samplenumber):
            image,lable=valid_data.get(x)
            prediction=self.querry(image)
            if(np.argmax(lable)==np.argmax(prediction)):
                correct_pred+=1
            
        print(f'accuracy on {samplenumber} samples:{correct_pred*1.0/samplenumber}')


class NeuronalNetwork_and_softmax (NeuronalNetwork):
    def __init__(self, initial_input, final_output, layers_number, activation_function, function_derivative, learning_rate,lambda_reg=0.001,decay_rate=0.001):
        super().__init__(initial_input, final_output, layers_number, activation_function, function_derivative, learning_rate,lambda_reg,decay_rate)
    
    def loss_function(self, prediction, expected):
        predictions = self.softmax(prediction)  
        predictions = np.clip(predictions, 1e-15, 1 - 1e-15)
        loss =np.sum (-1.0*expected * np.log(predictions))
        self.losschache = predictions - expected
        return loss
    def softmax(self,input):
        exp_values = np.exp(input - np.max(input))  
        probabilities = exp_values / np.sum(exp_values)
        return probabilities
    def train(self,input_train,input_validate,epochs=10):
        assert epochs >= 2, "Number of epochs must be at least 2 for training."
        interval_size=int(np.log10(epochs))
        self.total_loss=[]
        self.validate_loss=[]
        for epoch in range(epochs):
            helpervar=0
            self.losschache=0
            for x in input_train.values():
                a,b=x
                layer_outputs = a
                for layer in self.layers:
                    layer_outputs = layer.forward(layer_outputs)
                helpervar+=self.loss_function(layer_outputs,b)
                self.backprop()
            self.learning_rate=self.learning_rate*0.8
            self.total_loss.append(helpervar)
            if (epoch%interval_size ==0):
                print(f'Epoch {epoch + 1}, Loss: {helpervar}',flush=True)
            self.validate_loss.append(self.validate(input_validate))
    def querry(self,input):
        for layer in self.layers:
            input = layer.forward(input)
        input=self.softmax(input)
        return input
    def backprop(self):
        gradient_chain =self.losschache
        for layer in reversed(self.layers):
             gradient_chain = layer.backward(gradient_chain)
class NeuronalNetwork_softmax_dropout(NeuronalNetwork_and_softmax):
        def __init__(self, initial_input, final_output, layers_number, activation_function, function_derivative, learning_rate,lambda_reg=0.01,decay_rate=0.0,properbility=0.5,L2_reg=False,subset=32):
            self.initial_input = initial_input                # The initial input to the neural network
            self.final_output = final_output                  # The desired output from the neural network
            self.activation_function = activation_function    # The activation function to be used in each layer
            self.function_derivative = function_derivative    # The derivative of the activation function, used for backpropagation
            self.learning_rate = learning_rate                # The learning rate for the training process
            self.losschache=0.0
            self.regular_cache=0.0
            self.lambda_reg = lambda_reg
            self.decay_rate=decay_rate
            self.total_loss=[]
            self.validate_loss=[]
            self.layers=[]
            self.regular_L2=L2_reg
            self.subset_range=subset
            self.layers.append(layersaugmented.Fullyconnected_dropout_layer(initial_input,layers_number[0],activation_function,function_derivative,learning_rate,drop_properpility=properbility))
            for x in range(len(layers_number)-1):
                self.layers.append(layersaugmented.Fullyconnected_dropout_layer(layers_number[x],layers_number[x+1],activation_function,function_derivative,learning_rate,drop_properpility=properbility))
            self.layers.append(layersaugmented.NNlayer(layers_number[-1],final_output,activation_function,function_derivative,learning_rate))
        def train(self,input_train,input_validate,epochs=10):
            assert epochs >= 2, "Number of epochs must be at least 2 for training."
            interval_size=int(np.log10(epochs))
            self.total_loss=[]
            self.validate_loss=[]
            current_loss=0.0
            for epoch in range(epochs):
                helpervar=0
                self.losschache=0
                counter=0
                self.generate_new_dropout_masks()
                loss_without_reg=0    
        
                for x in input_train.values():
                    if(counter%self.subset_range==0):
                        self.generate_new_dropout_masks()
                    input,truth=x
                    layer_outputs = input
                    if (self.regular_L2):
                        l2_loss = 0.0
                        for layer in self.layers:
                            l2_loss += np.sum(layer.weigth_Array ** 2)
                        self.regular_cache=self.lambda_reg * l2_loss
                    for i, layer in enumerate(self.layers):
                        if i == len(self.layers) - 1:
                            layer_outputs = layer.forward(layer_outputs)
                        else:
                            layer_outputs = layer.forward(layer_outputs, training=True)
                    current_loss=self.loss_function(layer_outputs,truth)
                    loss_without_reg +=current_loss    
                    helpervar +=current_loss+self.regular_cache
                    
                    self.backprop()
                    counter+=1
                self.learning_rate*= np.exp(-self.decay_rate)
                self.total_loss.append(helpervar)
                if (epoch%interval_size ==0):
                    print(f'Epoch {epoch + 1}, Loss: {helpervar}, Loss without reg: {loss_without_reg}',flush=True)
                self.validate_loss.append(self.validate(input_validate))
        def backprop(self):
                gradient_chain =self.losschache  
                if(self.regular_L2):
                    for layer in reversed(self.layers):
                        gradient_chain = layer.backward(gradient_chain,2*self.lambda_reg)
                else:
                    for layer in reversed(self.layers):
                        gradient_chain = layer.backward(gradient_chain)
        def generate_new_dropout_masks(self):
            for layer in self.layers[:-1]:
                        layer.mask = layer.generate_random_row_matrix(layer.weigth_Array.shape, layer.drop_prop)
    

class NeuronalNetwork_Adam_softmax_dropout(NeuronalNetwork_softmax_dropout):
        def __init__(self, initial_input, final_output, layers_number, activation_function, function_derivative, learning_rate,lambda_reg=0.01,decay_rate=0.0,properbility=0.5,L2_reg=False,subset=32,beta1=0.9,beta2=0.999,epsi=0.000001,use_adam=True):
            self.initial_input = initial_input               
            self.final_output = final_output                  
            self.activation_function = activation_function    
            self.function_derivative = function_derivative    
            self.learning_rate = learning_rate                
            self.losschache=0.0
            self.regular_cache=0.0
            self.lambda_reg = lambda_reg
            self.decay_rate=decay_rate
            self.total_loss=[]
            self.validate_loss=[]
            self.layers=[]
            self.regular_L2=L2_reg
            self.subset_range=subset
            self.layers.append(layersaugmented.Layer_V2(initial_input,layers_number[0],activation_function,function_derivative,learning_rate,properbility,beta1,beta2,epsi,use_adam))
            for x in range(len(layers_number)-1):
                self.layers.append(layersaugmented.Layer_V2(layers_number[x],layers_number[x+1],activation_function,function_derivative,learning_rate,properbility,beta1,beta2,epsi,use_adam))
            self.layers.append(layersaugmented.FinalLayer_V2(layers_number[-1],final_output,activation_function,function_derivative,learning_rate,beta1,beta2,epsi,use_adam))



class NeuronalNetwork_softmax_and_minibatch(NeuronalNetwork_and_softmax):
    def __init__(self, initial_input, final_output, layers_number, activation_function, function_derivative, learning_rate,lambda_reg=0.001,decay_rate=0.001,batch_size=100):
        super().__init__(initial_input, final_output, layers_number, activation_function, function_derivative, learning_rate,lambda_reg,decay_rate)
        self.batch_size=batch_size
    def train(self,input_train,input_validate,epochs=10):
        assert epochs >= 2, "Number of epochs must be at least 2 for training."
        self.total_loss=[]
        self.validate_loss=[]
        for epoch in range(epochs):
            helpervar=0
            self.losschache=0
            counter=1
            for x in input_train.values():
                a,b=x
                layer_outputs = a
                for layer in self.layers:
                    layer_outputs = layer.forward(layer_outputs)
                helpervar+=self.loss_function(layer_outputs,b)
                if(counter%self.batch_size==0):
                    self.losschache/=self.batch_size
                    self.backprop()
                    self.losschache=0
                counter+=1
            if (counter % self.batch_size != 0):
                    self.losschache /= (counter % self.batch_size)  # Average over remaining samples
                    self.backprop()
                    self.losschache = 0
            self.learning_rate=self.learning_rate*0.95
            self.total_loss.append(helpervar)
            if (epoch%10 ==0):
                print(f'Epoch {epoch + 1}, Loss: {helpervar}',flush=True)
            self.validate_loss.append(self.validate(input_validate))
    def loss_function(self, prediction, expected):
        predictions = self.softmax(prediction)  
        predictions = np.clip(predictions, 1e-15, 1 - 1e-15)
        loss =np.sum (-1.0*expected * np.log(predictions))
        self.losschache -= np.abs(predictions - expected)
        return loss