import torch
import torch.nn as nn
import torch.nn.functional as F

class L_spatial(nn.Module):
    """
    Spatial Consistency Loss for Zero-DCE / Zero-DCE++
    Preserves structural/spatial consistency between input and enhanced images.
    """
    def __init__(self):
        super(L_spatial, self).__init__()
        kernel_left = torch.FloatTensor([[0, 0, 0], [-1, 1, 0], [0, 0, 0]]).unsqueeze(0).unsqueeze(0)
        kernel_right = torch.FloatTensor([[0, 0, 0], [0, 1, -1], [0, 0, 0]]).unsqueeze(0).unsqueeze(0)
        kernel_up = torch.FloatTensor([[0, -1, 0], [0, 1, 0], [0, 0, 0]]).unsqueeze(0).unsqueeze(0)
        kernel_down = torch.FloatTensor([[0, 0, 0], [0, 1, 0], [0, -1, 0]]).unsqueeze(0).unsqueeze(0)

        self.weight_left = nn.Parameter(data=kernel_left, requires_grad=False)
        self.weight_right = nn.Parameter(data=kernel_right, requires_grad=False)
        self.weight_up = nn.Parameter(data=kernel_up, requires_grad=False)
        self.weight_down = nn.Parameter(data=kernel_down, requires_grad=False)

        self.pool = nn.AvgPool2d(4, stride=4)

    def forward(self, org, enhance):
        b, c, h, w = org.shape

        org_mean = torch.mean(org, dim=1, keepdim=True)
        enhance_mean = torch.mean(enhance, dim=1, keepdim=True)

        org_pool = self.pool(org_mean)
        enhance_pool = self.pool(enhance_mean)

        w_left = self.weight_left.to(org.device)
        w_right = self.weight_right.to(org.device)
        w_up = self.weight_up.to(org.device)
        w_down = self.weight_down.to(org.device)

        d_org_left = F.conv2d(org_pool, w_left, padding=1)
        d_org_right = F.conv2d(org_pool, w_right, padding=1)
        d_org_up = F.conv2d(org_pool, w_up, padding=1)
        d_org_down = F.conv2d(org_pool, w_down, padding=1)

        d_enhance_left = F.conv2d(enhance_pool, w_left, padding=1)
        d_enhance_right = F.conv2d(enhance_pool, w_right, padding=1)
        d_enhance_up = F.conv2d(enhance_pool, w_up, padding=1)
        d_enhance_down = F.conv2d(enhance_pool, w_down, padding=1)

        d_left = torch.pow(d_org_left - d_enhance_left, 2)
        d_right = torch.pow(d_org_right - d_enhance_right, 2)
        d_up = torch.pow(d_org_up - d_enhance_up, 2)
        d_down = torch.pow(d_org_down - d_enhance_down, 2)

        return torch.mean(d_left + d_right + d_up + d_down)

class L_exp(nn.Module):
    """
    Exposure Control Loss
    Measures difference between average intensity of local patches and target exposure level E (default 0.6).
    """
    def __init__(self, patch_size=16, target_exp=0.6):
        super(L_exp, self).__init__()
        self.pool = nn.AvgPool2d(patch_size)
        self.target_exp = target_exp

    def forward(self, enhance):
        mean_intensity = torch.mean(enhance, dim=1, keepdim=True)
        patch_means = self.pool(mean_intensity)
        return torch.mean(torch.pow(patch_means - self.target_exp, 2))

class L_color(nn.Module):
    """
    Color Constancy Loss based on Gray-World hypothesis.
    Eliminates color deviations between R, G, B channels.
    """
    def __init__(self):
        super(L_color, self).__init__()

    def forward(self, enhance):
        mean_rgb = torch.mean(enhance, dim=[2, 3])
        r, g, b = mean_rgb[:, 0], mean_rgb[:, 1], mean_rgb[:, 2]
        d_rg = torch.pow(r - g, 2)
        d_rb = torch.pow(r - b, 2)
        d_gb = torch.pow(g - b, 2)
        return torch.mean(torch.sqrt(d_rg + d_rb + d_gb + 1e-6))

class L_TV(nn.Module):
    """
    Illumination Smoothness Loss (Total Variation Loss)
    Preserves monotonicity between neighboring pixels in estimated curve maps A.
    """
    def __init__(self):
        super(L_TV, self).__init__()

    def forward(self, A):
        b, c, h, w = A.shape
        count_h = (h - 1) * w
        count_w = h * (w - 1)
        h_tv = torch.pow((A[:, :, 1:, :] - A[:, :, :-1, :]), 2).sum()
        w_tv = torch.pow((A[:, :, :, 1:] - A[:, :, :, :-1]), 2).sum()
        return 2 * (h_tv / count_h + w_tv / count_w) / b

class ZeroDCELoss(nn.Module):
    """
    Combined Zero-DCE / Zero-DCE++ Loss Function
    L_total = w_spa * L_spa + w_exp * L_exp + w_col * L_col + w_tv * L_tv
    """
    def __init__(self, w_spa=1.0, w_exp=10.0, w_col=5.0, w_tv=200.0, target_exp=0.6):
        super(ZeroDCELoss, self).__init__()
        self.w_spa = w_spa
        self.w_exp = w_exp
        self.w_col = w_col
        self.w_tv = w_tv

        self.loss_spa = L_spatial()
        self.loss_exp = L_exp(patch_size=16, target_exp=target_exp)
        self.loss_col = L_color()
        self.loss_tv = L_TV()

    def forward(self, org, enhance, A):
        l_spa = self.loss_spa(org, enhance)
        l_exp = self.loss_exp(enhance)
        l_col = self.loss_col(enhance)
        l_tv = self.loss_tv(A)

        total_loss = (self.w_spa * l_spa +
                      self.w_exp * l_exp +
                      self.w_col * l_col +
                      self.w_tv * l_tv)

        return total_loss, {
            "spatial_loss": l_spa.item(),
            "exposure_loss": l_exp.item(),
            "color_loss": l_col.item(),
            "tv_loss": l_tv.item(),
            "total_loss": total_loss.item()
        }
