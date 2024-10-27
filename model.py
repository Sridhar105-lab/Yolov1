import torch
import torch.nn as nn

"""
--------------------------- WORKFLOW --------------------------
1) create of define a model here 
    1-->intial configuration as per the paper  
    2-->Bounding boc selection  
    3-->YoLo algorithm here
    
2) creat a seperate "loss function" in another python Module
    --> Most probably Box loss
    --> IoU methodology (bring it as close to 1)

3) import the loss.py here
4)  

"""


# the arch_config is a list that sets the layers of convolution Network to defined sizes as per paper

architecture_config = [
    # Tuple : (kernal_size, num_filters , stride, padding)
    (7, 64, 2, 3),
    "M",
    (3, 192, 1, 1),
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",
    
    # List : tuples and then last integer represents number of repeats 
    [(1, 256, 1, 0),(3, 512, 1 ,1),4], # repeat 4 times
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    
    "M",
    [(1, 512, 1, 0),(3, 1024, 1 ,1),2], # repeat twice 
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1)   # leaky relu helps with vanishing gradients 
        
    def forward(self, x):
        return self.leakyrelu(self.batchnorm(self.conv(x)))
    
class Yolov1(nn.Module):
    def __init__(self, in_channels=3, **kwargs):
        super(Yolov1,self).__init__()
        self.architecture = architecture_config
        self.in_channels = in_channels
        self.darknet = self._create_conv_layers(self.architecture)
        self.fcs = self._create_fcs(**kwargs)
        
    def forward(self,x):
        x = self.darknet(x)
        return self.fcs(torch.flatten(x,start_dim=1))

    # here's the hard part 
    # Gonna create a darknet 

    def _create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels
        
        for x in architecture:
            if type(x) == tuple:     
                # here outchannels=x[1]
                layers += [CNNBlock(in_channels, x[1], kernel_size = x[0], stride=x[2], padding = x[3],)]
                in_channels = x[1] 
            elif type(x) == str:
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            
            
            elif type(x) == list:
                conv1 = x[0] #tuple
                conv2 = x[1] #tuple 
                num_repreats = x[2] #integer 
                
                for _ in range (num_repreats):
                    layers += [
                        CNNBlock(
                            in_channels, 
                            conv1[1], #outchannels
                            kernel_size = conv1[0],
                            stride = conv1[2],
                            padding = conv1[3],
                            )
                    ]
                    layers += [
                        CNNBlock(
                            conv1[1],
                            conv2[1],
                            kernel_size = conv2[0],
                            stride = conv2[2],
                            padding = conv2[3],
                        )
                    ]
                    # the output of the conv channels is the input to the next channel
                    in_channels = conv2[1]
        return nn.Sequential(*layers)

    # here we go for fully connected layers 
    def _create_fcs(self, split_size, num_boxes, num_classes):
        S,B,C = split_size, num_boxes, num_classes  #here we have assigned properly
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024*S*S, 496), # given as 4096 in original paper
            nn.Dropout(0.0),
            nn.LeakyReLU(0.1),
            nn.Linear(496, S*S*(C+B*5)), #(S,S,30) where C+B*5 = 30
        )                

def tester(S=7, B=2, C=20):
    model = Yolov1(split_size = S, num_boxes = B, num_classes = C)
    x = torch.randn((2,3,448,448)) # need to check with theses dimensions
    print(model(x).shape) 

# print("check\n")
#tester(7,2,20)              

