import torch
import torch.nn as nn
import torch.nn.functional as F

class CConv2d(nn.Module):
    """
    Depthwise separable convolution block for Zero-DCE++
    """
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super(CConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size=kernel_size, stride=stride, 
                                   padding=padding, groups=in_ch, bias=True)
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

class DCE_Net(nn.Module):
    """
    Zero-DCE++ Network Architecture
    Lightweight zero-reference enhancement network with depthwise separable convolutions.
    """
    def __init__(self, n_iterations=8, scale_factor=1, base_channels=32):
        super(DCE_Net, self).__init__()
        self.n_iterations = n_iterations
        self.scale_factor = scale_factor

        self.e_conv1 = CConv2d(3, base_channels, 3, 1, 1)
        self.e_conv2 = CConv2d(base_channels, base_channels, 3, 1, 1)
        self.e_conv3 = CConv2d(base_channels, base_channels, 3, 1, 1)
        self.e_conv4 = CConv2d(base_channels, base_channels, 3, 1, 1)
        self.e_conv5 = CConv2d(base_channels * 2, base_channels, 3, 1, 1)
        self.e_conv6 = CConv2d(base_channels * 2, base_channels, 3, 1, 1)
        self.e_conv7 = CConv2d(base_channels * 2, 3 * n_iterations, 3, 1, 1)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # Downsample for faster execution if scale_factor > 1
        if self.scale_factor > 1:
            x_down = F.interpolate(x, scale_factor=1.0 / self.scale_factor, mode='bilinear', align_corners=False)
        else:
            x_down = x

        x1 = self.relu(self.e_conv1(x_down))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))

        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], dim=1)))
        x6 = self.relu(self.e_conv6(torch.cat([x2, x5], dim=1)))
        A = torch.tanh(self.e_conv7(torch.cat([x1, x6], dim=1)))

        if self.scale_factor > 1:
            A = F.interpolate(A, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)

        # Iterative curve enhancement: LE_n(x) = LE_{n-1}(x) + a_n * LE_{n-1}(x) * (1 - LE_{n-1}(x))
        r1, r2, r3, r4, r5, r6, r7, r8 = torch.split(A, 3, dim=1)

        x_enhanced = x
        x_enhanced = x_enhanced + r1 * (torch.pow(x_enhanced, 2) - x_enhanced)
        x_enhanced = x_enhanced + r2 * (torch.pow(x_enhanced, 2) - x_enhanced)
        x_enhanced = x_enhanced + r3 * (torch.pow(x_enhanced, 2) - x_enhanced)
        x_enhanced = x_enhanced + r4 * (torch.pow(x_enhanced, 2) - x_enhanced)
        x_enhanced = x_enhanced + r5 * (torch.pow(x_enhanced, 2) - x_enhanced)
        x_enhanced = x_enhanced + r6 * (torch.pow(x_enhanced, 2) - x_enhanced)
        x_enhanced = x_enhanced + r7 * (torch.pow(x_enhanced, 2) - x_enhanced)
        x_enhanced = x_enhanced + r8 * (torch.pow(x_enhanced, 2) - x_enhanced)

        # Ensure enhanced image stays strictly within [0, 1]
        x_enhanced = torch.clamp(x_enhanced, 0.0, 1.0)

        return x_enhanced, A
