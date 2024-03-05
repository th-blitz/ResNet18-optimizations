import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn

import torchvision
import torchvision.transforms as transforms

import torch.nn.functional as Functions


class Conv_Layer(nn.Module):

    def __init__(self, input_channels, output_channels, kernel_size, stride, padding, bias, use_batch_norm = True):
        super().__init__()
        
        if use_batch_norm == True:
            self.conv = nn.Sequential(
                nn.Conv2d(
                    in_channels = input_channels, out_channels = output_channels,
                    kernel_size = kernel_size, stride = stride, padding = padding, bias = bias
                ),
                nn.BatchNorm2d(output_channels) 
            ) 
        else:
            self.conv = nn.Conv2d(
                in_channels = input_channels, out_channels = output_channels,
                kernel_size = kernel_size, stride = stride, padding = padding, bias = bias
            )


    def forward(self, x):
        x = self.conv(x)
        return x


class Basic_Block(nn.Module):

    def __init__(self, input_channels, output_channels, kernel_size, stride, padding, bias, use_batch_norm = True):
        super().__init__()
       
        self.conv_layer_0 = Conv_Layer(input_channels, output_channels, kernel_size, stride, padding, bias, use_batch_norm)
        self.relu_0 = nn.ReLU()
        self.conv_layer_1 = Conv_Layer(output_channels, output_channels, kernel_size, 1, padding, bias, use_batch_norm)
        self.shortcut = nn.Sequential()

        if stride > 1:
            self.shortcut = Conv_Layer(input_channels, output_channels, (1, 1), stride, 0, bias, use_batch_norm) 

        self.relu_1 = nn.ReLU()
    
    def forward(self, x):
        identity = x
        x = self.conv_layer_0(x) 
        x = self.relu_0(x)
        x = self.conv_layer_1(x)
        x += self.shortcut(identity)
        x = self.relu_1(x)
        return x


class ResNet18(nn.Module):
    
    def __init__(self, use_batch_norm = True):

        super().__init__()

        self.initial_layer = Conv_Layer( input_channels = 3, output_channels = 64, 
                kernel_size = (3, 3), stride = 1, padding = 1, bias = False, use_batch_norm = use_batch_norm
            )

        self.sub_group_0 = nn.Sequential(
            Basic_Block( 64, 64, (3, 3), 1, 1, False, use_batch_norm),
            Basic_Block( 64, 64, (3, 3), 1, 1, False, use_batch_norm)
        )
        self.sub_group_1 = nn.Sequential(
            Basic_Block( 64, 128, (3, 3), 2, 1, False, use_batch_norm),
            Basic_Block( 128, 128, (3, 3), 1, 1, False, use_batch_norm)
        )
        self.sub_group_2 = nn.Sequential(
            Basic_Block( 128, 256, (3, 3), 2, 1, False, use_batch_norm),
            Basic_Block( 256, 256, (3, 3), 1, 1, False, use_batch_norm)
        )
        self.sub_group_3 = nn.Sequential(
            Basic_Block( 256, 512, (3, 3), 2, 1, False, use_batch_norm),
            Basic_Block( 512, 512, (3, 3), 1, 1, False, use_batch_norm)
        )
        
        self.avg_pooling = nn.AvgPool2d(4)
        self.linear = nn.Linear(512, 10) 
        # self.softmax = nn.Softmax(dim = 0)

    def forward(self, x):
        x = self.initial_layer(x)
        x = self.sub_group_0(x)
        x = self.sub_group_1(x)
        x = self.sub_group_2(x)
        x = self.sub_group_3(x)
        x = self.avg_pooling(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


# class ResNet18_old(nn.Module):
#     
#     def __init__(self):
#         
#         super().__init__()
# 
#         self.conv_0 = nn.Sequential(
#             nn.Conv2d(
#                 in_channels = input_channels, out_channels = output_channels, 
#                 kernel_size = kernel_size, stride = stride, padding = 1, bias = False
#             ),
#             nn.BatchNorm2d(output_channels),
#             nn.ReLU()
#         )
# 
#         self.sub_group_0 = SubGroup(
#             input_channels = 64, output_channels = 64, kernel_size = (3, 3), stride = 1, padding = 1
#         )
# 
#         self.sub_group_1 = SubGroup(
#             input_channels = 64, output_channels = 128, kernel_size = (3, 3), stride = 2, padding = 1
#         )
# 
#         self.sub_group_2 = SubGroup(
#             input_channels = 128, output_channels = 256, kernel_size = (3, 3), stride = 2, padding = 1
#         )
# 
#         self.sub_group_3 = SubGroup(
#             input_channels = 256, output_channels = 512, kernel_size = (3, 3), stride = 2, padding = 1
#         )
#         
#         self.flatten = nn.Flatten()
#         self.linear = nn.Linear(512 * 4 * 4,  10)
# 
#     def forward(self, x):
# 
#         x = self.conv_0(x)
#         x = self.sub_group_0.forward(x)
#         x = self.sub_group_1.forward(x)
#         x = self.sub_group_2.forward(x)
#         x = self.sub_group_3.forward(x)
#         x = self.flatten(x)
# 
#         x = self.linear(x)
# 
#         return x 
# 
# class SubGroup(nn.Module):
# 
#     def __init__(self, input_channels, output_channels, kernel_size, stride, padding):
# 
#         super().__init__()
#         
#         self.basic_block_0 = nn.Sequential(
#             nn.Conv2d(
#                 in_channels = input_channels, out_channels = output_channels, kernel_size = kernel_size, stride = stride, padding = padding, bias = False
#             ),
#             nn.BatchNorm2d(output_channels),
#             nn.ReLU(), 
#             nn.Conv2d(
#                 in_channels = output_channels, out_channels = output_channels, kernel_size = kernel_size, stride = 1, padding = padding, bias = False
#             ),
#             nn.BatchNorm2d(output_channels)
#         )
# 
#         self.shortcut_0 = nn.Sequential(
#             nn.Conv2d(
#                 in_channels = input_channels, out_channels = output_channels, kernel_size = (1, 1), stride = stride, bias = False 
#             ),
#             nn.BatchNorm2d(output_channels)
#         )
# 
#         self.basic_block_1 = nn.Sequential(
#             nn.Conv2d(
#                 in_channels = output_channels, out_channels = output_channels, kernel_size = kernel_size, stride = 1, padding = padding, bias = False
#             ),
#             nn.BatchNorm2d(output_channels),
#             nn.ReLU(),
#             nn.Conv2d(
#                 in_channels = output_channels, out_channels = output_channels, kernel_size = kernel_size, stride = 1, padding = padding, bias = False
#             ),
#             nn.BatchNorm2d(output_channels)
#         )
# 
#         self.shortcut_1 = nn.Sequential()
# 
# 
#     def forward(self, x):
# 
#         projection = x 
#         x = self.basic_block_0(x)
#         x += self.shortcut_0(projection)
#         x = Functions.relu(x)
#        
#         identity = x
#         x = self.basic_block_1(x)
#         x += self.shortcut_1(identity)
#         x = Functions.relu(x)
# 
#         return x


