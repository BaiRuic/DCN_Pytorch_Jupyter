import math

from typing import Union, Tuple

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch.nn.modules.utils import _pair
from torch.nn.functional import unfold, relu, linear

# Python类型提示（Type Hints）   基本类型有str list dict等等
# 参考 https://blog.csdn.net/weixin_41931548/article/details/89640223?spm=1001.2101.3001.4242   
#      https://blog.csdn.net/ypgsh/article/details/84992461
Pairable = Union[int, Tuple[int, int]]


def conv2d_local(input: torch.Tensor, 
                 weight: torch.Tensor,
                 bias=None,
                 padding: Pairable=0,
                 stride: Pairable=1,
                 dilation: Pairable=1,
                 data_format: str="channels_first"):
    """Calculate the local convolution.
    Args:
        input:
        weight:
        bias:
        padding:
        stride:
        dilation:
        data_format: For Keras compatibility
    Returns:
    """
    if input.dim() != 4:
        raise NotImplementedError("Input Error: Only 4D input Tensors supported (got {}D)".format(input.dim()))
    if weight.dim() != 6:
        # outH x outW x outC x inC x kH x kW
        raise NotImplementedError("Input Error: Only 6D weight Tensors supported (got {}D)".format(weight.dim()))

    out_height, out_width, out_channels, in_channels, kernel_height, kernel_width = weight.size()
    kernel_size = (kernel_height, kernel_width)

    # N x [in_channels * kernel_height * kernel_width] x [out_height * out_width]
    if data_format == "channels_first":
        cols = unfold(input, kernel_size, dilation=dilation, padding=padding, stride=stride)
        reshaped_input = cols.view(cols.size(0), cols.size(1), cols.size(2), 1).permute(0, 2, 3, 1)
    elif data_format == "channels_last":
        # Taken from `keras.backend.tensorflow_backend.local_conv2d`
        stride_y, stride_x = _pair(stride)
        feature_dim = in_channels * kernel_height * kernel_width
        xs = []
        for i in range(out_height):
            for j in range(out_width):
                y = i * stride_y
                slice_row = slice(y, y + kernel_size[0])
                x = j * stride_x
                slice_col = slice(x, x + kernel_size[1])
                val = input[:, slice_row, slice_col, :].contiguous()
                xs.append(val.view(input.shape[0], 1, -1, feature_dim))
        concated = torch.cat(xs, dim=1)
        reshaped_input = concated
    else:
        raise NotImplementedError("must be channels_last or channels_last")

    output_size = out_height * out_width
    input_size = in_channels * kernel_height * kernel_width
    weights_view = weight.view(output_size, out_channels, input_size)
    permuted_weights = weights_view.permute(0, 2, 1)

    out = torch.matmul(reshaped_input, permuted_weights)
    out = out.view(reshaped_input.shape[0], out_height, out_width, out_channels).permute(0, 3, 1, 2)
    if data_format == "channels_last":
        out = out.permute(0, 2, 3, 1)

    if bias is not None:
        # 这里可以用广播机制实现
        final_bias = bias.expand_as(out)
        out = out + final_bias

    return out


class Conv2dLocal(Module):
    """A 2D locally connected layer.
    Attributes:
        weight (torch.Tensor): The weights. out_height x out_width x out_channels x in_channels x kernel_height x kernel_width
        kernel_size (Tuple[int, int]): The height and width of the convolutional kernels.
        stride (Tuple[int, int]): The stride height and width.
    """

    def __init__(self, in_height: int, in_width: int, 
                 in_channels: int, out_channels: int,
                 kernel_size: Pairable,
                 stride: Pairable = 1,
                 padding: Pairable = 0,
                 bias: bool = True,
                 dilation: Pairable = 1,
                 data_format="channels_first"):
        super(Conv2dLocal, self).__init__()

        self.data_format = data_format
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)

        self.in_height = in_height
        self.in_width = in_width
        self.out_height = int(math.floor(
            (in_height + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) / self.stride[0] + 1))
        self.out_width = int(math.floor(
            (in_width + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) / self.stride[1] + 1))
        self.out_channels = out_channels

        self.weight = Parameter(torch.Tensor(
            self.out_height, self.out_width,
            out_channels, in_channels, *self.kernel_size))
        if bias:
            if self.data_format == "channels_first":
                self.bias = Parameter(torch.Tensor(out_channels, self.out_height, self.out_width))
            else:
                self.bias = Parameter(torch.Tensor(self.out_height, self.out_width, out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()
    
    # 使用装饰器将 将方法转换成属性， 使得属性不用直接暴漏在外面 
    # 参考 https://blog.csdn.net/dxk_093812/article/details/83212231
    @property 
    def input_shape(self):
        """The expected input shape for this module."""
        if self.data_format == "channels_first":
            shape = (self.in_channels, self.in_height, self.in_width)
        else:
            shape = (self.in_height, self.in_width, self.in_channels)
        return torch.Tensor(shape)

    def reset_parameters(self):
        """Reset the parameters of the layer."""
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def __repr__(self):
        s = ('{name}({in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.bias is None:
            s += ', bias=False'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)

    def forward(self, input: torch.Tensor):
        return conv2d_local(input=input, 
                            weight=self.weight, 
                            bias=self.bias,
                            stride=self.stride,
                            padding=self.padding,
                            dilation=self.dilation,
                            data_format=self.data_format
                            )


class Conv1dLocal(Conv2dLocal):
    """A 1D locally connected layer.
    input.shape must be [batch_size, time_steps, 1 ,in_channels],
    if input.shape is [batch_size, in_channels, time_steps, 1], the parameter that data_format need to be modefied to channels_first
    """

    def __init__(self, in_height, in_channels, out_channels,
                 kernel_size, stride=1, padding=0, bias=True, dilation=1):
        two_dimensional_kernel = (kernel_size, 1)
        two_dimensional_stride = (stride, 1)
        two_dimensional_padding = (padding, 0)
        two_dimensional_dilation = (dilation, 1)
        super().__init__(in_height, 1, in_channels, out_channels, two_dimensional_kernel,
                         stride=two_dimensional_stride,
                         padding=two_dimensional_padding,
                         dilation=two_dimensional_dilation,
                         bias=bias,
                         data_format="channels_first")


class Flatten(Module):

    def forward(self, input: torch.Tensor):
        return input.view(-1)




class Dense_conv(nn.Module):
    '''
    定义一个 block
    args:
        input_channels:  该block的输入通道
        output_channels：该block的输出通道
    '''    
    def __init__(self,input_channels, output_channels):
        super(Dense_conv,self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.unshared_conv_first = Conv1dLocal(in_height=8, in_channels=self.input_channels, out_channels=self.output_channels, kernel_size=3, padding=1)
        self.unshared_conv_after = Conv1dLocal(in_height=8, in_channels=self.output_channels, out_channels=self.output_channels, kernel_size=3, padding=1)
        self.batchnorm = nn.BatchNorm2d(self.output_channels)  # 传入通道数
        
    def forward(self, inputs):
        # inputs.shape[batch_size, channels(features), time_step, 1]  [batch_size, 1,10,1]
        if inputs.shape[1] != self.output_channels:
            # print(f'befor_1:{inputs.shape}')
            inputs = self.unshared_conv_first(inputs)  # inputs.shape=[batch_size,channels(8),time_step(8),1]
        dense1 = self.unshared_conv_after(inputs) + inputs
        dense1 = self.batchnorm(dense1).relu()
        
        dense2 = self.unshared_conv_after(dense1) + dense1 + inputs
        dense2 = self.batchnorm(dense2).relu()
        # print(f'after_1:{dense2.shape}')
        return dense2

class My_UCNN(nn.Module):
    def __init__(self):
        super(My_UCNN, self).__init__()
        self.batch_size=128
        self.con1 = nn.Sequential(
                        Dense_conv(input_channels=1, output_channels=8),
                        Dense_conv(input_channels=8, output_channels=16),
                        Dense_conv(input_channels=16, output_channels=32),
                        Dense_conv(input_channels=32, output_channels=64),
                        nn.Dropout(p=0.5),
                        Dense_conv(input_channels=64, output_channels=1),
                        )
        self.fc = nn.Linear(in_features=8*1 ,out_features=1)
    
    def forward(self,inputs):
        # inputs.shape=[batch_size, time_step(height), 1(width)]
        # after unsqueeze:
        #     inputs.shape = [batch_size, channels(features), time_step(height), 1(width)]
        inputs = inputs.unsqueeze(axis=1)
        # print(f"before_con1{x.shape}")
        x = self.con1(inputs)
        # print(f"after_con1{x.shape}")
        x = x.view(x.size(0),-1)
        # print(f"after_con1_view{x.shape}")
        out = self.fc(x)
        # print(f"out.shape:{out.shape}")
        return out
    
    def predict(self,inputs):
        with torch.no_grad():
            predictions = self.forward(inputs)
        return predictions

if __name__ == '__main__':
    
    model = My_UCNN()
    
    # 输入[batch_size, channels(features), time_step, 1] 1是固定的，因为序列数据的维度本来就是[num_steps, 1]
    inputs = torch.randn(128,8,1)
    # 前向传播
    output_1 = model(inputs)
    



    print(f'input.shape:{inputs.shape}')
    print(f'output_1.shape:{output_1.shape}')
  
   
