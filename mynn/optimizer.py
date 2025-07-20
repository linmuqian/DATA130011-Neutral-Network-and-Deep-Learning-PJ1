from abc import abstractmethod
import numpy as np

class Optimizer:
    def __init__(self, init_lr, model) -> None:
        self.init_lr = init_lr
        self.lr = init_lr  # changeable learning rate
        self.model = model

    @abstractmethod
    def step(self):
        pass


class SGD(Optimizer):
    def __init__(self, init_lr, model):
        super().__init__(init_lr, model)
    
    def step(self):
        for layer in self.model.layers:
            if layer.optimizable == True:
                for key in layer.params.keys():
                    if layer.weight_decay:
                        layer.params[key] *= (1 - self.lr * layer.weight_decay_lambda)
                    layer.params[key] -= self.lr * layer.grads[key]


class MomentGD(Optimizer):
    def __init__(self, init_lr, model, mu=0.9):
        super().__init__(init_lr, model)
        self.mu = mu  
        self.velocities = {}  # save the momentum for each layer
    
    def step(self):
        if not self.velocities:
            for layer in self.model.layers:
                if layer.optimizable:
                    layer_name = id(layer)  
                    self.velocities[layer_name] = {
                        key: np.zeros_like(layer.params[key]) 
                        for key in layer.params.keys()
                    }
        
        for layer in self.model.layers:
            if layer.optimizable:
                layer_name = id(layer)
                for key in layer.params.keys():
                    # Moment：v = mu * v + lr * grad
                    self.velocities[layer_name][key] = (
                        self.mu * self.velocities[layer_name][key] + 
                        self.lr * layer.grads[key]  
                    )
                    # Parameter：param = param - v
                    if layer.weight_decay:
                        layer.params[key] *= (1 - self.lr * layer.weight_decay_lambda)
                    layer.params[key] -= self.velocities[layer_name][key]


class Adam(Optimizer):
    def __init__(self, init_lr=1e-3, model=None, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__(init_lr, model)
        self.beta1 = beta1    # the first moment decay rate (momentum)
        self.beta2 = beta2    # the second moment decay rate (variance)
        self.eps = eps        # avoid division by zero
        self.t = 0
        self.m = {}           # the first moment (momentum)
        self.v = {}           # the second moment (variance)
        self.lr = init_lr     
    def step(self):
        self.t += 1
        if not self.m:
            for layer in self.model.layers:
                if self._has_params(layer):
                    layer_id = id(layer)
                    self.m[layer_id] = {key: np.zeros_like(layer.params[key]) for key in layer.params.keys()}
                    self.v[layer_id] = {key: np.zeros_like(layer.params[key]) for key in layer.params.keys()}

        for layer in self.model.layers:
            if not layer.optimizable:
                continue
            if self._has_params(layer):
                layer_id = id(layer)
                for key in layer.params.keys():
                    if not hasattr(layer, 'grads') or key not in layer.grads:
                        print(f"Warning: Layer {layer.__class__.__name__} does not have gradient for parameter {key}.")
                        continue
                    grad = layer.grads[key]
                    param = layer.params[key]

                    # Calculate the first moment (momentum): exponential moving average
                    self.m[layer_id][key] = self.beta1 * self.m[layer_id][key] + (1 - self.beta1) * grad
                    # Calculate the second moment (variance): exponential moving average
                    self.v[layer_id][key] = self.beta2 * self.v[layer_id][key] + (1 - self.beta2) * (grad ** 2)

                    # Calculate the bias-corrected first moment and second moment
                    m_hat = self.m[layer_id][key] / (1 - self.beta1 ** self.t)
                    v_hat = self.v[layer_id][key] / (1 - self.beta2 ** self.t)

                    # L2 regularization (if weight_decay exists)
                    if hasattr(layer, 'weight_decay') and layer.weight_decay:
                        param *= (1 - self.lr * layer.weight_decay_lambda)  # Apply weight decay

                    # Update the parameters
                    param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
                    layer.params[key] = param

    def _has_params(self, layer):
        return hasattr(layer, 'params') and layer.params
    
