import numpy as np

def relu(x):
    return np.maximum(0, x)

def softmax(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)

    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def flatten(x):
    B, _, _, _ = x.shape

    x = x.reshape(B, -1)

    return x

class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.in_channels = in_channels   # C(channels) for first conv otherwise out_channels of previuos conv layer
        self.out_channels = out_channels # number if filters
        self.kernel_size = kernel_size      
        self.stride = stride
        self.padding = padding

        self.init_params()

    def init_params(self):
        self.weights = np.random.randn(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)
        self.bias = np.random.randn(self.out_channels)

    def get_kernel_matrix(self, c, j, k):
            kernel_matrix = np.zeros(shape=(self.kernel_size, self.kernel_size))
            r = 0

            for i in range(j, j + self.kernel_size):
                kernel_matrix[r] = c[i][k:k + self.kernel_size]

                r += 1

            return kernel_matrix

    def __call__(self, x):
        B, C, H, W = x.shape

        H_out = ((H - self.kernel_size + self.padding * 2) // self.stride) + 1
        W_out = ((W - self.kernel_size + self.padding * 2) // self.stride) + 1

        if self.padding > 0:
            feature_maps = np.zeros(shape=(B, self.out_channels, H_out, W_out)) # (B, out, H', W')
            padded_x = np.zeros(shape=(B, C, H + self.padding * 2, W + self.padding * 2))

            padded_x[:, :, self.padding:self.padding + H, self.padding:self.padding + W] = x
            x = padded_x
        else:
            feature_maps = np.zeros(shape=(B, self.out_channels, H_out, W_out)) # (B, out, H', W')

        for bi, inp in enumerate(x):
            # inp: (C, H, W)
            
            for fi, (w, b) in enumerate(zip(self.weights, self.bias)):
                # w: (out, K, K) 
                # b: (out)
                # for the first conv later C=out 

                for i, c in enumerate(inp):
                    # c: (H, W)

                    for j in range(H_out):
                        for k in range(W_out):
                            kernel_part = self.get_kernel_matrix(c, j * self.stride, k * self.stride)
                            val = (kernel_part * w[i]).sum()
                            feature_maps[bi][fi][j][k] += val

                feature_maps[bi][fi] += b # (B, out, H', W')

        return feature_maps

class MaxPool2D:
    def __init__(self, kernel_size, stride=1, padding=0):
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def get_kernel_matrix(self, c, j, k):
            kernel_matrix = np.zeros(shape=(self.kernel_size, self.kernel_size))
            r = 0
      
            for i in range(j, j + self.kernel_size):
                kernel_matrix[r] = c[i][k:k + self.kernel_size]

                r += 1

            return kernel_matrix
    
    def __call__(self, x):
        B, F, H, W = x.shape # F is number of filters so out_channels in conv is C (because the first input C stands for channel)

        H_out = ((H - self.kernel_size) // self.stride) + 1
        W_out = ((W - self.kernel_size) // self.stride) + 1
        feature_maps_pooled = np.zeros(shape=(B, F, H_out, W_out))

        for i in range(B):
            for j in range(F):
                feature_map = x[i][j] # (H, W)

                for k in range(H_out):
                    for r in range(W_out):
                        kernel_part = self.get_kernel_matrix(feature_map, k * self.stride, r * self.stride)
                        val = np.max(kernel_part)
                        feature_maps_pooled[i][j][k][r] = val
   
        return feature_maps_pooled

class FNN:
    def __init__(self, in_features, out_features):
        self.w = np.random.randn(in_features, out_features)
        self.b = np.random.randn(out_features)

    def __call__(self, x):
        logits = x @ self.w + self.b

        # probs = softmax(logits)

        return logits
