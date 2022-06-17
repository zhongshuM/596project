import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

class PixelWiseDotProduct(nn.Module):
    def __init__(self):
        super(PixelWiseDotProduct, self).__init__()

    def forward(self, x, K):
        n, c, h, w = x.size()
        _, cout, ck = K.size()
        assert c == ck, "Number of channels in x and Embedding dimension (at dim 2) of K matrix must match"
        y = torch.matmul(x.view(n, c, h * w).permute(0, 2, 1), K.permute(0, 2, 1))  # .shape = n, hw, cout
        return y.permute(0, 2, 1).view(n, cout, h, w)

class PositionalEncoding2D(nn.Module):
    def __init__(self, channels, dropout=0.1):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding2D, self).__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 4) * 2)
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)
        self.cached_penc = None

    def forward(self, tensor):
        """
        :param tensor: A 4d tensor of size (batch_size, x, y, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, ch)
        """
        if len(tensor.shape) != 4:
            raise RuntimeError("The input tensor has to be 4d!")

        if self.cached_penc is not None and self.cached_penc.shape == tensor.shape:
            return self.cached_penc

        self.cached_penc = None
        batch_size, x, y, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
        pos_y = torch.arange(y, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1).unsqueeze(1)
        emb_y = torch.cat((sin_inp_y.sin(), sin_inp_y.cos()), dim=-1)
        emb = torch.zeros((x, y, self.channels * 2), device=tensor.device).type(
            tensor.type()
        )
        emb[:, :, : self.channels] = emb_x
        emb[:, :, self.channels : 2 * self.channels] = emb_y

        self.cached_penc = emb[None, :, :, :orig_ch].repeat(tensor.shape[0], 1, 1, 1)
        return self.cached_penc

class HRMerge(nn.Module):
    def __init__(self,
                 in_channels=(32, 64, 128, 256),
                 out_channels=256,
                 patch_size=16,
                 normalize=None):
        super(HRMerge, self).__init__()
        assert isinstance(in_channels, list) or isinstance(in_channels, tuple)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.with_bias = normalize is None
        self.reduction_conv = nn.Sequential(
            nn.Conv2d(in_channels=sum(in_channels),
                      out_channels=out_channels,
                      kernel_size=1),
        )

        self.fpn_conv = nn.Conv2d(in_channels=out_channels,
                                  out_channels=out_channels,
                                  kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        self.reduction_conv = nn.Sequential(
            nn.Conv2d(in_channels=sum(in_channels),
                      out_channels=int(out_channels),
                      kernel_size=1),
        )
        self.embedding_patch = nn.Conv2d(out_channels, out_channels,
                                        kernel_size=patch_size, stride=patch_size, padding=0)
        self.positional_encoding = PositionalEncoding2D(channels=out_channels)
        encoder_layers = nn.TransformerEncoderLayer(out_channels, 4, dim_feedforward=1024)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=2)

        # default init_weights for conv(msra) and norm in ConvModule

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)
        outs = []
        outs.append(inputs[0])
        for i in range(1, len(inputs)):
            outs.append(F.interpolate(
                inputs[i], scale_factor=2 ** i, mode='bilinear'))
        out = torch.cat(outs, dim=1)

        out = self.reduction_conv(out)
        out = self.relu(out)
        out = self.fpn_conv(out)

        return out

class PatchMerge(nn.Module):
    def __init__(self,
                 in_channels=(32, 64, 128, 256),
                 out_channels=128,
                 patch_size=16,
                 normalize=None):
        super(PatchMerge, self).__init__()
        assert isinstance(in_channels, list) or isinstance(in_channels, tuple)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.with_bias = normalize is None
        self.reduction_conv = nn.Sequential(
            nn.Conv2d(in_channels=sum(in_channels),
                      out_channels=out_channels,
                      kernel_size=1),
        )
        self.relu = nn.ReLU(inplace=True)

        self.reduction_conv = nn.Sequential(
            nn.Conv2d(in_channels=sum(in_channels),
                      out_channels=int(out_channels),
                      kernel_size=1),
        )
        self.conv3x3 = nn.Conv2d(in_channels=out_channels,
                            out_channels=out_channels,
                            kernel_size=3, padding=1)

        self.embedding_patch = nn.Conv2d(out_channels, out_channels,
                                        kernel_size=patch_size, stride=patch_size, padding=0)
        self.positional_encoding = PositionalEncoding2D(channels=out_channels)
        encoder_layers = nn.TransformerEncoderLayer(out_channels, 4, dim_feedforward=1024)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=2)

        self.dot_product_layer = PixelWiseDotProduct()

        # heads 
        self.heads_channel = (3, 5, 7, 11, 15, 16, 18)
        self.plane_center = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=1)
        )

        self.plane_xy = nn.Sequential(
            nn.Conv2d(2, 2, kernel_size=1)
        )

        self.plane_wh = nn.Sequential(
            nn.Conv2d(2, 2, kernel_size=1)
        )

        self.plane_params_pixelwise = nn.Sequential(
            nn.Conv2d(4, 4, kernel_size=1)
        )

        self.plane_params_instance = nn.Sequential(
            nn.Conv2d(4, 4, kernel_size=1)
        )

        self.line_region = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=1)
        )

        self.line_params = nn.Sequential(
            nn.Conv2d(2, 2, kernel_size=1)
        )

        self.heads_conv = [self.plane_center, self.plane_xy, self.plane_wh,
            self.plane_params_pixelwise, self.plane_params_instance,
            self.line_region, self.line_params]
        self.heads = 18

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)
        outs = []
        outs.append(inputs[0])
        for i in range(1, len(inputs)):
            outs.append(F.interpolate(
                inputs[i], scale_factor=2 ** i, mode='bilinear'))
        out = torch.cat(outs, dim=1)

        out = self.reduction_conv(out)
        out = self.relu(out)
        out = self.conv3x3(out)
        embeddings = self.embedding_patch(out.clone())
        # patchs_height = embeddings.shape[2]
        # patchs_width = embeddings.shape[3]
        # B, E, H, W -> B, H, W, E
        embeddings = embeddings.permute(0, 2, 3, 1)
        embeddings += self.positional_encoding(embeddings)
        # B, H, W, E -> B, E, H, W -> B, E, N(H*W) -> N, B, E
        embeddings = embeddings.permute(0, 3, 1, 2).flatten(2).permute(2, 0, 1)

        queries = embeddings[:self.heads, ...]
        queries = queries.permute(1, 0, 2)
        x = self.dot_product_layer(out, queries)

        plane_center = self.plane_center(x[:, :self.heads_channel[0], ...])
        plane_xy = self.plane_xy(x[:, self.heads_channel[0]:self.heads_channel[1], ...])
        plane_wh = self.plane_wh(x[:, self.heads_channel[1]:self.heads_channel[2], ...])
        plane_params_pixelwise = self.plane_params_pixelwise(x[:, self.heads_channel[2]:self.heads_channel[3], ...])
        plane_params_instance = self.plane_params_instance(x[:, self.heads_channel[3]:self.heads_channel[4], ...])

        line_region = self.line_region(x[:, self.heads_channel[4]:self.heads_channel[5], ...])
        line_params = self.line_params(x[:, self.heads_channel[5]:self.heads_channel[6], ...])

        out = {
            'plane_center': plane_center,
            'plane_offset': plane_xy,
            'plane_wh': plane_wh,
            'plane_params_pixelwise': plane_params_pixelwise,
            'plane_params_instance': plane_params_instance,
            'line_region': line_region,
            'line_params': line_params,
            'feature': x
        }
        return out
        # return embeddings


class Heads(nn.Module):
    def __init__(self, in_planes=256, out_planes=64):
        super(Heads, self).__init__()
        self.plane_center = nn.Sequential(
            nn.Conv2d(in_planes, in_planes, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_planes, 3, kernel_size=1)
        )

        self.plane_xy = nn.Sequential(
            nn.Conv2d(in_planes, in_planes, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_planes, 2, kernel_size=1)
        )

        self.plane_wh = nn.Sequential(
            nn.Conv2d(in_planes, in_planes, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_planes, 2, kernel_size=1)
        )

        self.plane_params_pixelwise = nn.Sequential(
            nn.Conv2d(in_planes, in_planes, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_planes, 4, kernel_size=1)
        )

        self.plane_params_instance = nn.Sequential(
            nn.Conv2d(in_planes, in_planes, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_planes, 4, kernel_size=1)
        )

        self.line_region = nn.Sequential(
            nn.Conv2d(in_planes, in_planes, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_planes, 1, kernel_size=1)
        )

        self.line_params = nn.Sequential(
            nn.Conv2d(in_planes, in_planes, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_planes, 2, kernel_size=1)
        )

    def forward(self, x):
        print("\nheads input shape:", x.shape)
        plane_params_pixelwise = self.plane_params_pixelwise(x)
        plane_center = self.plane_center(x)
        plane_wh = self.plane_wh(x)
        plane_xy = self.plane_xy(x)
        plane_params_instance = self.plane_params_instance(x)

        line_region = self.line_region(x)
        line_params = self.line_params(x)

        print("heads out shape:")
        print("plane_center:", plane_center.shape)
        print("plane_offset:", plane_xy.shape)
        print("plane_wh:", plane_wh.shape)
        print("plane_params_pixelwise:", plane_params_pixelwise.shape)
        print("plane_params_instance:", plane_params_instance.shape)
        print("line_region:", line_region.shape)
        print("line_params:", line_params.shape)
        print("feature:", x.shape)

        out = {
            'plane_center': plane_center,
            'plane_offset': plane_xy,
            'plane_wh': plane_wh,
            'plane_params_pixelwise': plane_params_pixelwise,
            'plane_params_instance': plane_params_instance,
            'line_region': line_region,
            'line_params': line_params,
            'feature': x
        }
        return out
