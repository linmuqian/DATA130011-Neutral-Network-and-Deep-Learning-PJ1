from mynn.op import *
import pickle

def he_init(size):
    fan_in = size[0]
    std = np.sqrt(2.0 / fan_in)
    return np.random.normal(0, std, size)

class Model_MLP(Layer):
    """
    A model with linear layers. We provied you with this example about a structure of a model.
    """
    def __init__(self, size_list=None, act_func=None, lambda_list=None):
        self.size_list = size_list
        self.act_func = act_func

        if size_list is not None and act_func is not None:
            self.layers = []
            for i in range(len(size_list) - 1):
                layer = Linear(in_dim=size_list[i], out_dim=size_list[i + 1], initialize_method=he_init)
                if lambda_list is not None:
                    layer.weight_decay = True
                    layer.weight_decay_lambda = lambda_list[i]
                if act_func == 'Logistic':
                    layer_f = Logistic()
                    #raise NotImplementedError
                elif act_func == 'ReLU':
                    layer_f = ReLU()
                self.layers.append(layer)
                if i < len(size_list) - 2:
                    self.layers.append(layer_f)

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        assert self.size_list is not None and self.act_func is not None, 'Model has not initialized yet. Use model.load_model to load a model or create a new model with size_list and act_func offered.'
        outputs = X
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs

    def backward(self, loss_grad):
        grads = loss_grad
        for layer in reversed(self.layers):
            grads = layer.backward(grads)
        return grads

    def load_model(self, param_list):
        with open(param_list, 'rb') as f:
            param_list = pickle.load(f)
        self.size_list = param_list[0]
        self.act_func = param_list[1]

        self.layers = []
        for i in range(len(self.size_list) - 1):
            layer = Linear(in_dim=self.size_list[i], out_dim=self.size_list[i + 1])
            layer.W = param_list[i + 2]['W']
            layer.b = param_list[i + 2]['b']
            layer.params['W'] = layer.W
            layer.params['b'] = layer.b
            layer.weight_decay = param_list[i + 2]['weight_decay']
            layer.weight_decay_lambda = param_list[i+2]['lambda']
            if self.act_func == 'Logistic':
                raise NotImplemented
            elif self.act_func == 'ReLU':
                layer_f = ReLU()
            self.layers.append(layer)
            if i < len(self.size_list) - 2:
                self.layers.append(layer_f)
        
    def save_model(self, save_path):
        param_list = [self.size_list, self.act_func]
        for layer in self.layers:
            if layer.optimizable:
                param_list.append({'W' : layer.params['W'], 'b' : layer.params['b'], 'weight_decay' : layer.weight_decay, 'lambda' : layer.weight_decay_lambda})
        
        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)

    def clear_grad(self):  # clear the gradient of all optimizable layers
        for layer in self.layers:
            if layer.optimizable:
                layer.clear_grad()
        

class Model_MLP_dropout(Layer):
    """
    A model with linear layers and dropout layers.
    """
    def __init__(self, size_list=None, act_func=None, lambda_list=None, dropout_rate=0.5, dropout=True):
        self.size_list = size_list
        self.act_func = act_func
        self.dropout_rate = dropout_rate
        self.dropout = dropout

        if size_list is not None and act_func is not None:
            self.layers = []
            for i in range(len(size_list) - 1):
                layer = Linear(in_dim=size_list[i], out_dim=size_list[i + 1], initialize_method=he_init)
                if lambda_list is not None:
                    layer.weight_decay = True
                    layer.weight_decay_lambda = lambda_list[i]
                if act_func == 'Logistic':
                    layer_f = Logistic()
                    # raise NotImplementedError
                elif act_func == 'ReLU':
                    layer_f = ReLU()
                self.layers.append(layer)
                if i < len(size_list) - 2:
                    self.layers.append(layer_f)
                    if self.dropout:
                        dropout_layer = Dropout(dropout_rate)
                        self.layers.append(dropout_layer)

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        assert self.size_list is not None and self.act_func is not None, 'Model has not initialized yet. Use model.load_model to load a model or create a new model with size_list and act_func offered.'
        outputs = X
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs

    def backward(self, loss_grad):
        grads = loss_grad
        for layer in reversed(self.layers):
            grads = layer.backward(grads)
        return grads

    def load_model(self, param_list):
        with open(param_list, 'rb') as f:
            param_list = pickle.load(f)
        self.size_list = param_list[0]
        self.act_func = param_list[1]
        self.dropout_rate = param_list[-2]
        self.dropout = param_list[-1]

        self.layers = []
        param_index = 2
        for i in range(len(self.size_list) - 1):
            layer = Linear(in_dim=self.size_list[i], out_dim=self.size_list[i + 1])
            layer.W = param_list[param_index]['W']
            layer.b = param_list[param_index]['b']
            layer.params['W'] = layer.W
            layer.params['b'] = layer.b
            layer.weight_decay = param_list[param_index]['weight_decay']
            layer.weight_decay_lambda = param_list[param_index]['lambda']
            self.layers.append(layer)
            if i < len(self.size_list) - 2:
                if self.act_func == 'Logistic':
                    raise NotImplemented
                elif self.act_func == 'ReLU':
                    layer_f = ReLU()
                self.layers.append(layer_f)
                if self.dropout:
                    dropout_layer = Dropout(self.dropout_rate)
                    self.layers.append(dropout_layer)
            param_index += 1

    def save_model(self, save_path):
        param_list = [self.size_list, self.act_func]
        for layer in self.layers:
            if layer.optimizable and isinstance(layer, Linear):
                param_list.append({'W': layer.params['W'], 'b': layer.params['b'],
                                   'weight_decay': layer.weight_decay, 'lambda': layer.weight_decay_lambda})
        param_list.extend([self.dropout_rate, self.dropout])
        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)

    def clear_grad(self):  # clear the gradient of all optimizable layers
        for layer in self.layers:
            if layer.optimizable:
                layer.clear_grad()

    def set_dropout_enabled(self, enabled):
        self.dropout = enabled
        for layer in self.layers:
            if isinstance(layer, Dropout):
                layer.set_enabled(enabled)

class Model_MLP_softmax(Layer):
    '''
    A model with linear layers and dropout layers. And the last layer is softmax.
    '''
    def __init__(self, size_list=None, act_func=None, lambda_list=None, dropout_rate=0.5, dropout=True):
        self.size_list = size_list
        self.act_func = act_func
        self.dropout_rate = dropout_rate
        self.dropout = dropout

        if size_list is not None and act_func is not None:
            self.layers = []
            for i in range(len(size_list) - 1):
                layer = Linear(in_dim=size_list[i], out_dim=size_list[i + 1], initialize_method=he_init)
                if lambda_list is not None:
                    layer.weight_decay = True
                    layer.weight_decay_lambda = lambda_list[i]
                if act_func == 'Logistic':
                    layer_f = Logistic()
                    # raise NotImplementedError
                elif act_func == 'ReLU':
                    layer_f = ReLU()
                self.layers.append(layer)
                if i < len(size_list) - 2:
                    self.layers.append(layer_f)
                    if self.dropout:
                        dropout_layer = Dropout(dropout_rate)
                        self.layers.append(dropout_layer)
            self.layers.append(Softmax())  # Add softmax layer at the end

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        assert self.size_list is not None and self.act_func is not None, 'Model has not initialized yet. Use model.load_model to load a model or create a new model with size_list and act_func offered.'
        outputs = X
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs

    def backward(self, loss_grad):
        grads = loss_grad
        for layer in reversed(self.layers):
            grads = layer.backward(grads)
        return grads

    def load_model(self, param_list):
        with open(param_list, 'rb') as f:
            param_list = pickle.load(f)
        self.size_list = param_list[0]
        self.act_func = param_list[1]
        self.dropout_rate = param_list[-2]
        self.dropout = param_list[-1]

        self.layers = []
        param_index = 2
        for i in range(len(self.size_list) - 1):
            layer = Linear(in_dim=self.size_list[i], out_dim=self.size_list[i + 1])
            layer.W = param_list[param_index]['W']
            layer.b = param_list[param_index]['b']
            layer.params['W'] = layer.W
            layer.params['b'] = layer.b
            layer.weight_decay = param_list[param_index]['weight_decay']
            layer.weight_decay_lambda = param_list[param_index]['lambda']
            self.layers.append(layer)
            if i < len(self.size_list) - 2:
                if self.act_func == 'Logistic':
                    raise NotImplemented
                elif self.act_func == 'ReLU':
                    layer_f = ReLU()
                self.layers.append(layer_f)
                if self.dropout:
                    dropout_layer = Dropout(self.dropout_rate)
                    self.layers.append(dropout_layer)
            param_index += 1
        self.layers.append(Softmax())  # Add softmax layer at the end

    def save_model(self, save_path):
        param_list = [self.size_list, self.act_func]
        for layer in self.layers:
            if layer.optimizable and isinstance(layer, Linear):
                param_list.append({'W': layer.params['W'], 'b': layer.params['b'],
                                   'weight_decay': layer.weight_decay, 'lambda': layer.weight_decay_lambda})
        param_list.extend([self.dropout_rate, self.dropout])
        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)

    def clear_grad(self):  # clear the gradient of all optimizable layers
        for layer in self.layers:
            if layer.optimizable:
                layer.clear_grad()

    def set_dropout_enabled(self, enabled):
        self.dropout = enabled
        for layer in self.layers:
            if isinstance(layer, Dropout):
                layer.set_enabled(enabled)


class Model_CNN(Layer):
    def __init__(self, conv_configs, fc_configs):
        super().__init__()
        self.conv_configs = conv_configs
        self.fc_configs = fc_configs

        # Initialize layers
        self.conv1 = Conv2D(*conv_configs[0])
        self.flatten = Flatten()
        self.fc1 = Linear(*fc_configs[0])

        self.layers = [self.conv1, self.flatten, self.fc1]

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        out = self.conv1(x)
        out = self.flatten(out)
        out = self.fc1(out)
        return out

    def backward(self, grad_output):
        grad = self.fc1.backward(grad_output)
        grad = self.flatten.backward(grad)
        grad = self.conv1.backward(grad)
        return grad

    def save_model(self, save_path):
        param_list = [self.conv_configs, self.fc_configs]
        for layer in self.layers:
            if isinstance(layer, Conv2D):
                # 保存卷积层参数及配置
                param_dict = {
                    'weight': layer.weight,
                    'bias': layer.bias,
                    'weight_decay': layer.weight_decay,
                    'lambda': layer.weight_decay_lambda,
                    'in_channels': layer.in_channels,
                    'out_channels': layer.out_channels,
                    'kernel_size': layer.kernel_size,
                    'stride': layer.stride,
                    'padding': layer.padding
                }
                param_list.append(param_dict)
            elif isinstance(layer, Linear):
                param_dict = {
                    'weight': layer.W,
                    'bias': layer.b,
                    'weight_decay': layer.weight_decay,
                    'lambda': layer.weight_decay_lambda,
                    'in_dim': layer.in_dim,
                    'out_dim': layer.out_dim
                }
                param_list.append(param_dict)
        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)

    def load_model(self, param_list_path):
        with open(param_list_path, 'rb') as f:
            param_list = pickle.load(f)
        self.conv_configs = param_list[0]
        self.fc_configs = param_list[1]
        self.layers = []
        param_index = 2

        conv_params = param_list[param_index]
        conv1 = Conv2D(
            in_channels=conv_params['in_channels'],
            out_channels=conv_params['out_channels'],
            kernel_size=conv_params['kernel_size'],
            stride=conv_params['stride'],
            padding=conv_params['padding'],
            weight_decay=conv_params['weight_decay'],
            weight_decay_lambda=conv_params['lambda']
        )
        conv1.weight = conv_params['weight']
        conv1.bias = conv_params['bias']
        self.layers.append(conv1)
        param_index += 1

        self.layers.append(Flatten())  

        fc_params = param_list[param_index]
        fc1 = Linear(
            in_dim=fc_params['in_dim'],
            out_dim=fc_params['out_dim'],
            weight_decay=fc_params['weight_decay'],
            weight_decay_lambda=fc_params['lambda']
        )
        fc1.W = fc_params['weight']
        fc1.b = fc_params['bias']
        self.layers.append(fc1)

    def clear_grad(self):
        for layer in self.layers:
            if hasattr(layer, 'clear_grad'):
                layer.clear_grad()


class Model_LeNet(Layer):
    def __init__(self, in_channels=1, num_classes=10):
        """
        Conv1 -> ReLU -> Pool2 -> Conv3 -> ReLU -> Pool4 -> Conv5 -> ReLU -> Flatten -> FC6 -> ReLU -> FC7 -> output
        """
        self.conv1 = Conv2D(in_channels=in_channels, out_channels=6, kernel_size=5, stride=1, padding=0)
        self.relu1 = ReLU()
        self.pool2 = MaxPool2D(kernel_size=2, stride=2)
        self.conv3 = Conv2D(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.relu3 = ReLU()
        self.pool4 = AvgPool2D(kernel_size=2, stride=2)
        self.conv5 = Conv2D(in_channels=16, out_channels=120, kernel_size=5, stride=1, padding=0)
        self.relu5 = ReLU()
        self.fc6 = Linear(in_dim=120, out_dim=84)
        self.relu6 = ReLU()
        self.fc7 = Linear(in_dim=84, out_dim=num_classes)

        self.layers = [self.conv1, self.relu1, self.pool2, 
                       self.conv3, self.relu3, self.pool4, 
                       self.conv5, self.relu5, 
                       self.fc6, self.relu6, self.fc7]

    def __call__(self, x):
        return self.forward(x)
    
    def forward(self, x):
        """
        input: (N, in_channels, 32, 32)
        output: (N, num_classes)
        """
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.pool2.forward(out)  # Use forward explicitly
        out = self.conv3(out)
        out = self.relu3(out)
        out = self.pool4.forward(out)  # Use forward explicitly
        out = self.conv5(out)
        out = self.relu5(out)
        N = out.shape[0]
        out = out.reshape(N, -1)  # shape (N, 120)
        out = self.fc6(out)
        out = self.relu6(out)
        out = self.fc7(out)
        return out
    
    def backward(self, grad_output):
        """
        grad_output: (N, num_classes)
        grad_input: (N, in_channels, 32, 32)
        """
        grad = self.fc7.backward(grad_output)
        grad = self.relu6.backward(grad)
        grad = self.fc6.backward(grad)
        grad = grad.reshape(-1, 120, 1, 1)
        grad = self.relu5.backward(grad)
        grad = self.conv5.backward(grad)
        grad = self.pool4.backward(grad)
        grad = self.relu3.backward(grad)
        grad = self.conv3.backward(grad)
        grad = self.pool2.backward(grad)
        grad = self.relu1.backward(grad)
        grad_input = self.conv1.backward(grad)
        return grad_input
    
    def save_model(self, file_path):
        params = {
            'conv1_weight': self.conv1.weight,
            'conv1_bias': self.conv1.bias,
            'conv3_weight': self.conv3.weight,
            'conv3_bias': self.conv3.bias,
            'conv5_weight': self.conv5.weight,
            'conv5_bias': self.conv5.bias,
            'fc6_W': self.fc6.W,
            'fc6_b': self.fc6.b,
            'fc7_W': self.fc7.W,
            'fc7_b': self.fc7.b
        }
        # 使用 pickle 保存参数
        with open(file_path, 'wb') as f:
            pickle.dump(params, f)

    def load_model(self, file_path):
        with open(file_path, 'rb') as f:
            params = pickle.load(f)
        self.conv1.weight = params['conv1_weight']
        self.conv1.bias = params['conv1_bias']
        self.conv3.weight = params['conv3_weight']
        self.conv3.bias = params['conv3_bias']
        self.conv5.weight = params['conv5_weight']
        self.conv5.bias = params['conv5_bias']
        self.fc6.W = params['fc6_W']
        self.fc6.b = params['fc6_b']
        self.fc7.W = params['fc7_W']
        self.fc7.b = params['fc7_b']

    def clear_grad(self):
        for layer in self.layers:
            if hasattr(layer, 'clear_grad'):
                layer.clear_grad() 

class ResidualBlock(Layer):
    '''residual block for ResNet: conv1 -> bn1 -> relu -> conv2 -> bn2 + identity -> relu'''
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.downsample = downsample
        self.optimizable = False
        self.conv1 = Conv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=1
        )
        self.bn1 = BatchNorm2D(num_features=out_channels)
        self.relu1 = ReLU()
        self.conv2 = Conv2D(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.bn2 = BatchNorm2D(num_features=out_channels)
        self.relu2 = ReLU()

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            for layer in self.downsample:
                identity = layer(identity)
        out += identity
        out = self.relu2(out)
        return out

    def backward(self, grad):
        grad = self.relu2.backward(grad)
        
        original_grad = grad.copy() # identity connection gradient
        
        grad = self.bn2.backward(grad)  
        grad = self.conv2.backward(grad) 
        grad = self.relu1.backward(grad)
        grad = self.bn1.backward(grad)
        grad = self.conv1.backward(grad)
        
        if self.downsample is not None:
            for layer in reversed(self.downsample):
                original_grad = layer.backward(original_grad)
            grad += original_grad  # add the identity connection gradient
        return grad



class Model_ResNet(Layer):
    '''ResNet18: conv1 -> bn1 -> relu -> maxpool -> layer1 -> layer2 -> layer3 -> avgpool -> flatten -> fc'''
    def __init__(self, num_classes=10):
        super().__init__()
        self.in_channels = 16
        self.layers = []
        self.conv1 = Conv2D(
            in_channels=1,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.bn1 = BatchNorm2D(num_features=16)
        self.relu = ReLU()
        self.maxpool = MaxPool2D(kernel_size=3, stride=2, padding=1)
        self.layers.extend([self.conv1, self.bn1, self.relu, self.maxpool])
        # 3 layers: each have 2 Residual blocks and perhaps downsample layer
        self.layer1 = self._make_layer(out_channels=16, stride=1, blocks=2)
        self.layer2 = self._make_layer(out_channels=32, stride=2, blocks=2)
        self.layer3 = self._make_layer(out_channels=64, stride=2, blocks=2)
        self.avgpool = AvgPool2D(kernel_size=4, stride=1)
        self.flatten = Flatten()
        self.fc = Linear(in_dim=64, out_dim=num_classes)
        self.layers.extend([self.avgpool, self.flatten, self.fc])

    def __call__(self, x):
        return self.forward(x)

    def _make_layer(self, out_channels, stride, blocks):
        downsample = None # downsample is to equalize the input and output dimensions of the residual block
        if self.in_channels != out_channels or stride != 1:
            downsample = [
                Conv2D(
                    in_channels=self.in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=stride,
                    padding=0
                ),
                BatchNorm2D(num_features=out_channels)
            ]
        layers = []
        layers.append(ResidualBlock(
            in_channels=self.in_channels,
            out_channels=out_channels,
            stride=stride,
            downsample=downsample
        ))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(ResidualBlock(
                in_channels=out_channels,
                out_channels=out_channels,
                stride=1,
                downsample=None
            ))
        return layers

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        for block in self.layer1: x = block(x)
        for block in self.layer2: x = block(x)
        for block in self.layer3: x = block(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

    def backward(self, grad):
        grad = self.fc.backward(grad)
        grad = self.flatten.backward(grad)
        grad = self.avgpool.backward(grad)
        for block in reversed(self.layer3): grad = block.backward(grad)
        for block in reversed(self.layer2): grad = block.backward(grad)
        for block in reversed(self.layer1): grad = block.backward(grad)
        grad = self.maxpool.backward(grad)
        grad = self.relu.backward(grad)
        grad = self.bn1.backward(grad)
        grad = self.conv1.backward(grad)
        return grad

    def save_model(self, save_path):
        param_list = []
        param_list.append({
            'conv1_weight': self.conv1.params['weight'],
            'conv1_bias': self.conv1.params['bias'],
            'bn1_gamma': self.bn1.params['gamma'],
            'bn1_beta': self.bn1.params['beta']
        })
        for layer in [self.layer1, self.layer2, self.layer3]:
            for block in layer:
                param_list.append({
                    'conv1_weight': block.conv1.params['weight'],
                    'conv1_bias': block.conv1.params['bias'],
                    'bn1_gamma': block.bn1.params['gamma'],
                    'bn1_beta': block.bn1.params['beta'],
                    'conv2_weight': block.conv2.params['weight'],
                    'conv2_bias': block.conv2.params['bias'],
                    'bn2_gamma': block.bn2.params['gamma'],
                    'bn2_beta': block.bn2.params['beta']
                })
                if block.downsample is not None:
                    param_list.append({
                        'downsample_conv_weight': block.downsample[0].params['weight'],
                        'downsample_conv_bias': block.downsample[0].params['bias'],
                        'downsample_bn_gamma': block.downsample[1].params['gamma'],
                        'downsample_bn_beta': block.downsample[1].params['beta']
                    })
        param_list.append({
            'fc_weight': self.fc.params['W'],
            'fc_bias': self.fc.params['b']
        })
        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)

    def load_model(self, load_path):
        with open(load_path, 'rb') as f:
            param_list = pickle.load(f)
        init_params = param_list[0]
        self.conv1.params['weight'] = init_params['conv1_weight']
        self.conv1.params['bias'] = init_params['conv1_bias']
        self.bn1.params['gamma'] = init_params['bn1_gamma']
        self.bn1.params['beta'] = init_params['bn1_beta']
        idx = 1
        for layer in [self.layer1, self.layer2, self.layer3]:
            for block in layer:
                block_params = param_list[idx]
                idx += 1
                block.conv1.params['weight'] = block_params['conv1_weight']
                block.conv1.params['bias'] = block_params['conv1_bias']
                block.bn1.params['gamma'] = block_params['bn1_gamma']
                block.bn1.params['beta'] = block_params['bn1_beta']
                block.conv2.params['weight'] = block_params['conv2_weight']
                block.conv2.params['bias'] = block_params['conv2_bias']
                block.bn2.params['gamma'] = block_params['bn2_gamma']
                block.bn2.params['beta'] = block_params['bn2_beta']
                if block.downsample is not None:
                    downsample_params = param_list[idx]
                    idx += 1
                    block.downsample[0].params['weight'] = downsample_params['downsample_conv_weight']
                    block.downsample[0].params['bias'] = downsample_params['downsample_conv_bias']
                    block.downsample[1].params['gamma'] = downsample_params['downsample_bn_gamma']
                    block.downsample[1].params['beta'] = downsample_params['downsample_bn_beta']
        fc_params = param_list[idx]
        self.fc.params['W'] = fc_params['fc_weight']
        self.fc.params['b'] = fc_params['fc_bias']

    def clear_grad(self):
        for layer in self.layers:
            if hasattr(layer, 'clear_grad'):
                layer.clear_grad()