from abc import abstractmethod
import numpy as np

class Layer():
    def __init__(self) -> None:
        self.optimizable = True
    
    @abstractmethod
    def forward():
        pass

    @abstractmethod
    def backward():
        pass


class Linear(Layer):
    """
    The linear layer for a neural network. You need to implement the forward function and the backward function.
    """
    def __init__(self, in_dim, out_dim, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.W = initialize_method(size=(in_dim, out_dim))
        self.b = initialize_method(size=(1, out_dim))
        self.grads = {'W' : None, 'b' : None}
        self.input = None # Record the input for backward process.

        self.params = {'W' : self.W, 'b' : self.b}

        self.weight_decay = weight_decay # whether using weight decay
        self.weight_decay_lambda = weight_decay_lambda # control the intensity of weight decay
            
    
    def __call__(self, X) -> np.ndarray:
        return self.forward(X)

    def forward(self, X):
        """
        input: [batch_size, in_dim]
        out: [batch_size, out_dim]
        """
        self.input = X
        return np.dot(X, self.W) + self.b  # Y = WX + b
        #pass

    def backward(self, grad : np.ndarray):
        """
        input: [batch_size, out_dim] the grad passed by the next layer.
        output: [batch_size, in_dim] the grad to be passed to the previous layer.
        This function also calculates the grads for W and b.
        """
        batch_size=self.input.shape[0]
        self.grads['W'] = np.dot(self.input.T, grad)
        if self.weight_decay:  #  L2 regulation
            self.grads['W'] += 2 * self.weight_decay_lambda * self.W
        self.grads['b'] = np.sum(grad, axis=0, keepdims=True)
        return np.dot(grad, self.W.T)
        #pass
    
    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}

class ReLU(Layer):
    """
    An activation layer.
    """
    def __init__(self) -> None:
        super().__init__()
        self.input = None
        self.optimizable =False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input = X
        output = np.where(X<0, 0, X)
        return output
    
    def backward(self, grads):
        if self.input.shape != grads.shape:
            print(f"ReLU backward shape mismatch! self.input.shape = {self.input.shape}, grads.shape = {grads.shape}")
        assert self.input.shape == grads.shape
        output = np.where(self.input < 0, 0, grads)
        return output

class MultiCrossEntropyLoss(Layer):
    """
    A multi-cross-entropy loss layer, with Softmax layer in it, which could be cancelled by method cancel_softmax
    """
    def __init__(self, model = None, max_classes = 10) -> None:
        super().__init__()
        self.model = model
        self.has_softmax = True
        self.predicts = None
        self.labels = None
        #pass

    def __call__(self, predicts, labels):
        return self.forward(predicts, labels)
    
    def forward(self, predicts, labels):
        """
        predicts: [batch_size, D]
        labels : [batch_size, ]
        This function generates the loss.
        """
        # / ---- your codes here ----/
        self.predicts = predicts
        self.labels = labels
        batch_size = predicts.shape[0]
        if self.has_softmax:
            probs = softmax(predicts)
        else:
            probs = predicts  # I suppose without softmax, the input is already the softmax output

        probs = np.clip(probs, 1e-10, 1 - 1e-10)  # avoid log(0)
        log_probs = -np.log(probs[np.arange(batch_size), labels])
        loss = np.sum(log_probs) / batch_size # calculate the multi-cross-entropy loss and do average
        return loss
        #pass
    
    def backward(self):
        # first compute the grads from the loss to the input
        # / ---- your codes here ----/
        batch_size = self.predicts.shape[0]
        if self.has_softmax:
            probs = softmax(self.predicts)
            self.grads = probs.copy()
            self.grads[np.arange(batch_size), self.labels] -= 1
            self.grads /= batch_size  # calculate the average gradient as the formula 
        else:
            # If softmax is cancelled, I suppose the input predicts are already the softmax output
            num_classes = self.predicts.shape[1]
            one_hot_labels = np.eye(num_classes)[self.labels]  # [batch_size, num_classes]
            self.grads = (self.predicts - one_hot_labels) / batch_size
        # Then send the grads to model for back propagation
        self.model.backward(self.grads)

    def cancel_soft_max(self):
        self.has_softmax = False
        return self
    
class L2Regularization(Layer):
    """
    L2 Reg can act as weight decay that can be implemented in class Linear.
    """
    def __init__(self, layers, lambda_=1e-8):
        super().__init__()
        self.layers = layers
        self.lambda_ = lambda_

    def forward(self): # calculate L2 loss
        loss = 0
        for layer in self.layers:
            if hasattr(layer, 'params') and 'W' in layer.params:
                loss += np.sum(np.square(layer.params['W']))
        return 0.5 * self.lambda_ * loss

    def backward(self): # calculate the gradient of L2 loss
        for layer in self.layers:
            if hasattr(layer, 'grads') and 'W' in layer.grads and hasattr(layer, 'params') and 'W' in layer.params:
                layer.grads['W'] += self.lambda_ * layer.params['W']
    #pass
       
def softmax(X):
    x_max = np.max(X, axis=1, keepdims=True)
    x_exp = np.exp(X - x_max)
    partition = np.sum(x_exp, axis=1, keepdims=True)
    return x_exp / partition


class Logistic(Layer):
    """
    An Logistic/sigmod layer. This is because the Logistic layer is used in the model.py
    """
    def __init__(self) -> None:
        super().__init__()
        self.input = None
        self.optimizable = False
    
    def __call__(self, X):
        return self.forward(X)
    
    def forward(self, X):
        self.input = X  
        output = 1 / (1 + np.exp(-X))  # calculate Loigistic function
        return output

    def backward(self, grads):
        assert self.input.shape == grads.shape, "The input must be consistent with the gradient shape"
        sigmoid_output = self.forward(self.input)  
        grad = grads * sigmoid_output * (1 - sigmoid_output)
        return grad

class Dropout(Layer):
    def __init__(self, dropout_rate=0.5):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.mask = None
        self.training = True
        self.optimizable = False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        if self.training:
            self.mask = np.random.rand(*X.shape) > self.dropout_rate
            return X * self.mask / (1 - self.dropout_rate)
        else:
            return X

    def backward(self, grad):
        if self.training:
            return grad * self.mask / (1 - self.dropout_rate)
        else:
            return grad
        
    def clear_grad(self):
        pass

class Softmax(Layer):
    def __init__(self) -> None:
        super().__init__()
        self.input = None
        self.output = None
        self.optimizable = False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input = X
        exp_X = np.exp(X - np.max(X, axis=1, keepdims=True)) # minus max for numerical stability
        self.output = exp_X / np.sum(exp_X, axis=1, keepdims=True)
        return self.output

    def backward(self, grad):
        # backward pass for softmax
        return self.output * (grad - np.sum(self.output * grad, axis=1, keepdims=True))

class Flatten(Layer):
    def __init__(self):
        super().__init__()
        self.input_shape = None
        self.optimizable = False
    
    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        input: [batch_size, channels, height, width]
        out: [batch_size, channels * height * width]
        """
        self.input_shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, grad):
        """
        input: [batch_size, channels * height * width]
        output: [batch_size, channels, height, width]
        """
        return grad.reshape(self.input_shape)


class Conv2D(Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, initialize_method=np.random.normal,
                 weight_decay=False, weight_decay_lambda=1e-8):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.weight = initialize_method(size=(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = initialize_method(size=(out_channels,))
        self.grads = {'weight': None, 'bias': None}
        self.input = None
        self.params = {'weight': self.weight, 'bias': self.bias}
        self.weight_decay = weight_decay
        self.weight_decay_lambda = weight_decay_lambda

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        input: [batch_size, in_channels, height, width]
        out: [batch_size, out_channels, out_height, out_width]
        """
        self.input = x
        batch_size, in_channels, height, width = x.shape
        # output shape
        out_height = (height + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_width = (width + 2 * self.padding - self.kernel_size) // self.stride + 1
        out = np.zeros((batch_size, self.out_channels, out_height, out_width))

        padded_x = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')

        # convolution operation
        for b in range(batch_size):
            for c_out in range(self.out_channels):
                for h_out in range(out_height):
                    for w_out in range(out_width):
                        h_start = h_out * self.stride
                        h_end = h_start + self.kernel_size
                        w_start = w_out * self.stride
                        w_end = w_start + self.kernel_size
                        patch = padded_x[b, :, h_start:h_end, w_start:w_end]
                        out[b, c_out, h_out, w_out] = np.sum(patch * self.weight[c_out]) + self.bias[c_out]

        return out

    def backward(self, grad):
        """
        input: [batch_size, out_channels, out_height, out_width]
        output: [batch_size, in_channels, height, width]
        This function also calculates the grads for weight and bias.
        """
        batch_size, out_channels, out_height, out_width = grad.shape
        _, in_channels, height, width = self.input.shape
        padded_x = np.pad(self.input, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')
        padded_grad = np.zeros_like(padded_x)
        self.grads['weight'] = np.zeros_like(self.weight)
        self.grads['bias'] = np.zeros_like(self.bias)

        # convolution operation for gradient
        for b in range(batch_size):
            for c_out in range(out_channels):
                for h_out in range(out_height):
                    for w_out in range(out_width):
                        h_start = h_out * self.stride
                        h_end = h_start + self.kernel_size
                        w_start = w_out * self.stride
                        w_end = w_start + self.kernel_size
                        patch = padded_x[b, :, h_start:h_end, w_start:w_end]
                        self.grads['weight'][c_out] += grad[b, c_out, h_out, w_out] * patch
                        self.grads['bias'][c_out] += grad[b, c_out, h_out, w_out]
                        padded_grad[b, :, h_start:h_end, w_start:w_end] += grad[b, c_out, h_out, w_out] * self.weight[c_out]

        # L2 regulation
        if self.weight_decay:
            self.grads['weight'] += 2 * self.weight_decay_lambda * self.weight

        grad_input = padded_grad[:, :, self.padding:self.padding + height, self.padding:self.padding + width]
        return grad_input

    def clear_grad(self):
        self.grads = {'weight': None, 'bias': None}



class MaxPool2D(Layer):
    def __init__(self, kernel_size=2, stride=2, padding=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.input = None  
        self.max_indices = None 
        self.optimizable = False
    
    def __call__(self, x):
        return self.forward(x)
    
    def forward(self, x):
        """
        input: [batch_size, in_channels, height, width]
        out: [batch_size, in_channels, out_height, out_width]
        """
        self.input = x
        batch_size, channels, height, width = x.shape
        out_height = (height + 2*self.padding - self.kernel_size) // self.stride + 1
        out_width = (width + 2*self.padding - self.kernel_size) // self.stride + 1
        out = np.zeros((batch_size, channels, out_height, out_width))
        # keep track of (h, w) positions
        self.max_indices = np.zeros((batch_size, channels, out_height, out_width, 2), dtype=int) 
        
        padded_x = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')
        
        for b in range(batch_size):
            for c in range(channels):
                for h_out in range(out_height):
                    for w_out in range(out_width):
                        h_start = h_out * self.stride
                        h_end = h_start + self.kernel_size
                        w_start = w_out * self.stride
                        w_end = w_start + self.kernel_size
                        patch = padded_x[b, c, h_start:h_end, w_start:w_end]
                        max_val = np.max(patch)
                        # record the max value position 
                        max_h, max_w = np.unravel_index(np.argmax(patch), (self.kernel_size, self.kernel_size))
                        out[b, c, h_out, w_out] = max_val
                        self.max_indices[b, c, h_out, w_out] = (max_h, max_w)
        
        return out
    
    def backward(self, grad):
        """
        input: [batch_size, channels, out_height, out_width]
        output: [batch_size, channels, height, width]
        """
        batch_size, channels, out_height, out_width = grad.shape
        _, _, height, width = self.input.shape
        grad_input = np.zeros_like(self.input)
        padded_grad_input = np.pad(grad_input, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')
        
        for b in range(batch_size):
            for c in range(channels):
                for h_out in range(out_height):
                    for w_out in range(out_width):
                        h_start = h_out * self.stride
                        h_end = h_start + self.kernel_size
                        w_start = w_out * self.stride
                        w_end = w_start + self.kernel_size
                        max_h, max_w = self.max_indices[b, c, h_out, w_out]
                        padded_grad_input[b, c, h_start + max_h, w_start + max_w] += grad[b, c, h_out, w_out]
        
        # remove padding    
        grad_input = padded_grad_input[:, :, self.padding:self.padding+height, self.padding:self.padding+width]
        return grad_input
    
    def clear_grad(self):
        pass

class AvgPool2D(Layer):
    def __init__(self, kernel_size=2, stride=2, padding=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.input = None  # save original input shape to restore gradient shape
        self.optimizable = False  # this layer has no parameters
    
    def __call__(self, x):
        return self.forward(x)
    
    def forward(self, x):
        """
        input: [batch_size, channels, height, width]
        output: [batch_size, channels, out_height, out_width]
        """
        self.input = x  
        batch_size, channels, height, width = x.shape
        # calculate output shape
        out_height = (height + 2*self.padding - self.kernel_size) // self.stride + 1
        out_width = (width + 2*self.padding - self.kernel_size) // self.stride + 1
        out = np.zeros((batch_size, channels, out_height, out_width))
        
        padded_x = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')
        
        for b in range(batch_size):
            for c in range(channels):
                for h_out in range(out_height):
                    for w_out in range(out_width):
                        h_start = h_out * self.stride
                        h_end = h_start + self.kernel_size
                        w_start = w_out * self.stride
                        w_end = w_start + self.kernel_size
                        patch = padded_x[b, c, h_start:h_end, w_start:w_end]
                        # calculate average value in the pooling window
                        out[b, c, h_out, w_out] = np.mean(patch)
        
        return out
    
    def backward(self, grad):
        """
        input: [batch_size, channels, out_height, out_width]
        output: [batch_size, channels, height, width]
        """
        batch_size, channels, out_height, out_width = grad.shape
        _, _, height, width = self.input.shape  
        grad_input = np.zeros_like(self.input)  
        pool_area = self.kernel_size * self.kernel_size  # pooling area
        
        padded_grad_input = np.pad(grad_input, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')
        
        for b in range(batch_size):
            for c in range(channels):
                for h_out in range(out_height):
                    for w_out in range(out_width):
                        h_start = h_out * self.stride
                        h_end = h_start + self.kernel_size
                        w_start = w_out * self.stride
                        w_end = w_start + self.kernel_size
                        # Average gradient value in the pooling window
                        grad_val = grad[b, c, h_out, w_out] / pool_area
                        padded_grad_input[b, c, h_start:h_end, w_start:w_end] += grad_val
        
        # remove padding
        grad_input = padded_grad_input[:, :, self.padding:self.padding+height, self.padding:self.padding+width]
        return grad_input
    
    def clear_grad(self):
        pass

class BatchNorm2D(Layer):
    def __init__(self, num_features, eps=1e-5, momentum=0.9):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.gamma = np.ones((num_features,))
        self.beta = np.zeros((num_features,))
        self.running_mean = np.zeros((num_features,))
        self.running_var = np.ones((num_features,))
        self.training = True
        self.params = {'gamma': self.gamma, 'beta': self.beta}
        self.grads = {'gamma': None, 'beta': None}
        self.x_norm = None
        self.batch_mean = None
        self.batch_var = None
    
    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        if self.training:
            batch_size, channels, height, width = x.shape
            self.batch_mean = np.mean(x, axis=(0, 2, 3))
            self.batch_var = np.var(x, axis=(0, 2, 3))
            x_norm = (x - self.batch_mean[None, :, None, None]) / np.sqrt(self.batch_var[None, :, None, None] + self.eps)
        else:
            x_norm = (x - self.running_mean[None, :, None, None]) / np.sqrt(self.running_var[None, :, None, None] + self.eps)
        self.x_norm = x_norm
        return self.gamma[None, :, None, None] * x_norm + self.beta[None, :, None, None]

    def backward(self, grad):
        batch_size, channels, height, width = grad.shape
        n = batch_size * height * width
        d_gamma = np.sum(grad * self.x_norm, axis=(0, 2, 3))
        d_beta = np.sum(grad, axis=(0, 2, 3))
        d_x_norm = grad * self.gamma[None, :, None, None]
        mean_dx_norm = np.mean(d_x_norm, axis=(0, 2, 3))
        var_dx_norm = np.var(d_x_norm, axis=(0, 2, 3))
        x_centered = self.x_norm * np.sqrt(self.batch_var[None, :, None, None] + self.eps)
        d_input = (d_x_norm - x_centered * mean_dx_norm[None, :, None, None] / (self.batch_var[None, :, None, None] + self.eps) - mean_dx_norm[None, :, None, None]) / n
        self.grads['gamma'] = d_gamma
        self.grads['beta'] = d_beta
        return d_input

    def clear_grad(self):
        self.grads = {'gamma': None, 'beta': None}

