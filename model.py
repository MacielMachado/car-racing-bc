from torch import nn
import torch

class Model_original(nn.Module):
    def __init__(self, input_ch=4, ch=8):
        super().__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=input_ch, out_channels=8*ch, kernel_size=(7,7)),
            nn.ReLU(),
            nn.Conv2d(in_channels=8*ch, out_channels=ch*16, kernel_size=(5,5), stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=ch*16, out_channels=ch*32, kernel_size=(3,3), stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=ch*32, out_channels=ch*32, kernel_size=(3,3), stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=ch*32, out_channels=ch*64, kernel_size=(3,3), stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=ch*64, out_channels=ch*64, kernel_size=(3,3), stride=2),
            nn.ReLU(),
        )
        self.flat_layer = nn.Sequential(
            nn.Linear(64*ch*1*1, 1024),
            nn.ReLU()
        )
        self.output = nn.Linear(in_features=1024, out_features=3)

    def forward(self, x):
        x = x.permute(0, 3, 2, 1)
        x = self.conv_layer(x.to(torch.float32))
        x = x.view(x.size(0), -1)
        x = self.flat_layer(x)
        x = self.output(x)

        x[:,0] = torch.tanh(x[:,0])
        x[:,1] = torch.sigmoid(x[:,1])
        x[:,2] = torch.sigmoid(x[:,2])

        return x


class Model_residual(nn.Module):
    def __init__(self, x_shape, n_hidden, y_dim, embed_dim, net_type, output_dim=None):
        super(Model_residual, self).__init__()

        self.x_shape = x_shape
        self.n_hidden = n_hidden
        self.y_dim = y_dim
        self.embed_dim = embed_dim
        self.n_feat = 64
        self.net_type = net_type

        if output_dim is None:
            self.output_dim = y_dim  # by default, just output size of action space
        else:
            self.output_dim = output_dim  # sometimes overwrite, eg for discretised, mean/variance, mixture density models

        # set up CNN for image
        self.conv_down1 = nn.Sequential(
            ResidualConvBlock(self.x_shape[-1], self.n_feat, is_res=True),
            nn.MaxPool2d(2),
        )
        self.conv_down3 = nn.Sequential(
            ResidualConvBlock(self.n_feat, self.n_feat * 2, is_res=True),
            nn.MaxPool2d(2),
        )
        self.imageembed = nn.Sequential(nn.AvgPool2d(8))
        

        self.output = nn.Linear(in_features=output_dim,
                                out_features=3)
        # it is the flattened size after CNN layers, and average pooling

    def forward(self, x):
        x = self.embed_context(x)

        return x

    def embed_context(self, x):
        x = x.permute(0, 3, 2, 1)
        x1 = self.conv_down1(x)
        x3 = self.conv_down3(x1)  # [batch_size, 128, 35, 18]
        x_embed = self.imageembed(x3)
        x_embed = x_embed.view(x.shape[0], -1)
        x = self.output(x_embed)
        return x
    

class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, is_res=False):
        super().__init__()
        self.same_channels = in_channels == out_channels
        self.is_res = is_res
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        if self.is_res:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            # this adds on correct residual in case channels have increased
            if self.same_channels:
                x = x + x2
            else:
                x = x1 + x2
            return x / 1.414
        else:
            x = self.conv1(x)
            x = self.conv2(x)
            return x