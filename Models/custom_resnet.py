import torch.nn as nn
import torch.nn.functional as F

def _get_conv_block(in_channels, out_channels, is_pool, num_layers):
    """Get convolution blocks

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        is_pool (bool): True if maxpooling is required else False
        num_layers (int): Number of convolution layers requied

    Returns:
        Object: Object of convolution block
    """
    block_list = []
    for _ in range(num_layers):
        block_list.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False))
        if is_pool:
            block_list.append(nn.MaxPool2d(2, 2))
        block_list.append(nn.BatchNorm2d(out_channels))
        block_list.append(nn.ReLU())
        conv_block = nn.Sequential(*block_list)
    return conv_block

class ResBlock(nn.Module):
    """Class for resnet block 
    """
    def __init__(self, in_channels, out_channels) -> None:
        """Constructor for ResNet class.

        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
        """
        super(ResBlock, self).__init__()
        self.conv_block = _get_conv_block(in_channels, out_channels, is_pool=True, num_layers=1)
        self.res_block = _get_conv_block(out_channels, out_channels, is_pool=False, num_layers=2)
    
    def forward(self, x):
        """Method to define forward computation pass.

        Args:
            x (Tensor): Batch of images or single image

        Returns:
            Tensor: Tensor representing probability of each class
        """
        x = self.conv_block(x)
        r1 = self.res_block(x)
        x = x + r1
        return x
    
class Model(nn.Module):
    """Class to initialize model
    """
    def __init__(self) -> None:
        """Creates instance of ResNet model used by david page for Dawn Bench
        """
        super(Model, self).__init__()
        self.prep_layer = _get_conv_block(3, 64, is_pool=False, num_layers=1)
        self.layer1 = ResBlock(64, 128)
        self.layer2 = _get_conv_block(128, 256, is_pool=True, num_layers=1)
        self.layer3 = ResBlock(256, 512)
        self.maxpool = nn.MaxPool2d(4)
        self.flat = nn.Flatten()
        self.fc = nn.Linear(512, 10)

    def forward(self, x):
        """Method to define forward computation pass.

        Args:
            x (Tensor): Batch of images or single image

        Returns:
            Tensor: Tensor representing probability of each class
        """
        x = self.prep_layer(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.maxpool(x)
        x = self.flat(x)
        x = self.fc(x)
        return F.log_softmax(x, dim=-1)
