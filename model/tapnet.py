import torch
import torch.nn as nn
import torch.nn.functional as F


class InitialModule(nn.Module):
    def __init__(self, nIn, nOut):  # nIn = 3, nOut = 19
        super().__init__()
        n_int = int(nOut - nIn)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.conv1 = CBR(nIn, n_int, 3, stride=2)
        self.conv2 = CBR(n_int, n_int, 3, stride=1)
        self.conv3 = CBR(n_int, n_int, 3, stride=1)
        self.act = nn.PReLU(nOut)

    def forward(self, input):
        output = self.conv3(self.conv2(self.conv1(input)))
        output = torch.cat([output, self.pool(input)], dim = 1)
        output = self.act(output)
        return output

class CBR(nn.Module):
    '''
    This class defines the convolution layer with batch normalization and PReLU activation
    '''
    def __init__(self, nIn, nOut, kSize, stride=1):
        '''

        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: stride rate for down-sampling. Default is 1
        '''
        super().__init__()
        padding = int((kSize - 1)/2)
        #self.conv = nn.Conv2d(nIn, nOut, kSize, stride=stride, padding=padding, bias=False)
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), bias=False)
        #self.conv1 = nn.Conv2d(nOut, nOut, (1, kSize), stride=1, padding=(0, padding), bias=False)
        self.bn = nn.BatchNorm2d(nOut, eps=1e-03)
        self.act = nn.PReLU(nOut)

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.conv(input)
        #output = self.conv1(output)
        output = self.bn(output)
        output = self.act(output)
        return output

class BR(nn.Module):
    '''
        This class groups the batch normalization and PReLU activation
    '''
    def __init__(self, nOut):
        '''
        :param nOut: output feature maps
        '''
        super().__init__()
        self.bn = nn.BatchNorm2d(nOut, eps=1e-03)
        self.act = nn.PReLU(nOut)

    def forward(self, input):
        '''
        :param input: input feature map
        :return: normalized and thresholded feature map
        '''
        output = self.bn(input)
        output = self.act(output)
        return output

class CB(nn.Module):
    '''
       This class groups the convolution and batch normalization
    '''
    def __init__(self, nIn, nOut, kSize, stride=1):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optinal stide for down-sampling
        '''
        super().__init__()
        padding = int((kSize - 1)/2)
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), bias=False)
        self.bn = nn.BatchNorm2d(nOut, eps=1e-03)

    def forward(self, input):
        '''

        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.conv(input)
        output = self.bn(output)
        return output

class C(nn.Module):
    '''
    This class is for a convolutional layer.
    '''
    def __init__(self, nIn, nOut, kSize, stride=1, groups = 1):
        '''

        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optional stride rate for down-sampling
        '''
        super().__init__()
        padding = int((kSize - 1)/2)
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), bias=False, groups = groups)

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.conv(input)
        return output

class FacDilatedConv(nn.Module):
    '''
    This class defines the factorized dilated convolution.
    '''
    def __init__(self, nIn, nOut, kSize, stride=1, d=1):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optional stride rate for down-sampling
        :param d: optional dilation rate
        '''
        super().__init__()
        padding = int((kSize - 1)/2) * d         
        self.conv31 = nn.Conv2d(nIn, nOut, (kSize, 1), stride=stride, padding=(padding, 0), bias=False, dilation=d)
        self.conv13 = nn.Conv2d(nOut, nOut, (1, kSize), stride=stride, padding=(0, padding), bias=False, dilation=d)
        self.act = nn.PReLU(nOut)

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.conv31(input)
        output = self.act(output)
        output = self.conv13(output)
        return output

class DownSamplerBlock(nn.Module):
    def __init__(self, nIn, nOut):  # lets say. nIn = 35, nOut = 64 
        super().__init__() 
        n_squeezed = int(nOut/5)                                           # n_squeezed = 12
        n1 = nOut - 4*n_squeezed                                           # n1 = 16
        self.conv_1x1 = nn.Conv2d(nIn, n1, 1)                      # 19 --> 12
        self.conv = C(n1, n1, 3, 2, groups = n1)                        # 12 --> 12
        self.conv_d2 = FacDilatedConv(n1, n_squeezed, 3, 1, 2)             # 12 --> 16
        self.conv_d4 = FacDilatedConv(n1, n_squeezed, 3, 1, 4)     # 12 --> 12
        self.conv_d8 = FacDilatedConv(n1, n_squeezed, 3, 1, 8)     # 12 --> 12
        self.conv_d16 = FacDilatedConv(n1, n_squeezed, 3, 1, 16)   # 12 --> 12
        self.bn = nn.BatchNorm2d(nOut, eps=1e-3)
        self.act = nn.PReLU(nOut)

    def forward(self, input):
        output1 = self.conv(self.conv_1x1(input))
        d2 = self.conv_d2(output1)
        d4 = self.conv_d4(output1)
        d8 = self.conv_d8(output1)
        d16 = self.conv_d16(output1)
        
        d4 = d4 + d2
        d8 = d8 + d16
        combine = torch.cat([output1, d2, d4, d8, d16],1)
        output = self.bn(combine)
        output = self.act(output)
        return output

class BasicBlock(nn.Module):
    '''
    This class defines the Factorized block, which is based on the following principle
        Reduce ---> Split ---> Factorized Transform --> Merge
    '''
    def __init__(self, nIn, nOut, add=True):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param add: if true, add a residual connection through identity operation. You can use projection too as
                in ResNet paper, but we avoid to use it if the dimensions are not the same because we do not want to
                increase the module complexity
        '''
        super().__init__()
        n_squeezed = int(nOut/5)   # n_squeezed = 16
        n1 = nOut - 4*n_squeezed
        self.conv_1x1 = C(nIn, n1, 1, 1)                           # 19 --> 12
        self.conv = C(n1, n1, 3, 1, groups = n1)                        # 12 --> 12
        self.conv_d2 = FacDilatedConv(n1, n_squeezed, 3, 1, 2)             # 12 --> 16
        self.conv_d4 = FacDilatedConv(n1, n_squeezed, 3, 1, 4)     # 12 --> 12
        self.conv_d8 = FacDilatedConv(n1, n_squeezed, 3, 1, 8)     # 12 --> 12
        self.conv_d16 = FacDilatedConv(n1, n_squeezed, 3, 1, 16)   # 12 --> 12
        self.bn = BR(nOut)
        self.add = add  # bool

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        # reduce
        output1 = self.conv(self.conv_1x1(input))
        d2 = self.conv_d2(output1)
        d4 = self.conv_d4(output1)
        d8 = self.conv_d8(output1)
        d16 = self.conv_d16(output1)

        d4 = d4 + d2
        d8 = d8 + d16
        combine = torch.cat([output1, d2, d4, d8, d16],1)

        # if residual version
        if self.add:
            combine = input + combine
        output = self.bn(combine)
        return output


class PARSE_SPP(nn.Module):
    """
    PARSE_SPP: Pairwise-Aggregated Reinforced Scale-Enlarged Spatial Pyramid Pooling
    """
    def __init__(self, in_channels):
        super(PARSE_SPP, self).__init__()

        out_channels = int(in_channels/4)

        self.conv0 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                nn.BatchNorm2d(out_channels, eps=1e-03),
                                nn.ReLU(True))
        self.conv2_r12 = nn.Sequential(nn.Conv2d(out_channels, out_channels, 3, padding = 12, bias=False, dilation = 12),
                                nn.BatchNorm2d(out_channels, eps=1e-03),
                                nn.ReLU(True))
        self.conv3_r24 = nn.Sequential(nn.Conv2d(out_channels, out_channels, 3, padding = 24, bias=False, dilation = 24),
                                nn.BatchNorm2d(out_channels, eps=1e-03),
                                nn.ReLU(True))
        self.conv4_r36 = nn.Sequential(nn.Conv2d(out_channels, out_channels, 3, padding = 36,  bias=False, dilation = 36),
                                nn.BatchNorm2d(out_channels, eps=1e-03),
                                nn.ReLU(True))

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.compress = nn.Sequential(nn.Conv2d(2*in_channels, in_channels, 1, bias=False),
                                nn.BatchNorm2d(in_channels, eps=1e-03),
                                nn.ReLU(True))

    def forward(self, x):
        _, _, h, w = x.size()
        feat0 = self.conv0(x)

        feat1 = self.conv2_r12(feat0)
        feat2 = self.conv3_r24(feat0)
        feat3 = self.conv4_r36(feat0)
        feat4 = F.interpolate(self.pool(feat0), (h, w), mode='bilinear', align_corners = True)
        
        agg1 = feat1 # agg means aggregated feature
        agg2 = agg1 + feat2
        agg3 = feat3
        agg4 = agg3 + feat4

        output = self.compress(torch.cat([agg1, agg2, agg3, agg4, x], 1))
        return output
        
        
class Network(nn.Module):

    def __init__(self, num_classes=19):

        super().__init__()
        self.stageE1 = InitialModule(3, 35)  
        self.bn_relu1 = BR(35)    

        self.stageE2 = nn.ModuleList()
        self.stageE2.append(DownSamplerBlock(35, 64))
        self.stageE2.append(BasicBlock(64 , 64))
        self.stageE2.append(BasicBlock(64 , 64))
        self.stageE2.append(BasicBlock(64 , 64))
        self.stageE2.append(BasicBlock(64 , 64))
        self.bn_relu2 = BR(64)  
        
        self.stageE3 = nn.ModuleList()
        self.stageE3.append(DownSamplerBlock(64, 128))
        self.stageE3.append(BasicBlock(128 , 128))
        self.stageE3.append(BasicBlock(128 , 128))
        self.stageE3.append(BasicBlock(128 , 128))
        self.stageE3.append(BasicBlock(128 , 128))
        self.stageE3.append(BasicBlock(128 , 128))
        self.stageE3.append(BasicBlock(128 , 128))
        self.stageE3.append(BasicBlock(128 , 128))
        self.stageE3.append(BasicBlock(128 , 128))
        self.bn_relu3 = BR(128)

        self.context = PARSE_SPP(128)

        self.classifier = C(128, num_classes, 1, 1)

    def forward(self, input):
 
        output = self.bn_relu1(self.stageE1(input))  
 
        for layer in self.stageE2:
            output = layer(output) 
        output = self.bn_relu2(output)

        for layer in self.stageE3:
            output = layer(output)  
        output = self.bn_relu3(output)
        
        output = self.context(output)
        output = self.classifier(output)
        output = F.interpolate(output, scale_factor=8, mode='bilinear', align_corners = True)
        return output
        

