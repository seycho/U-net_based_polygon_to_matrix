import torch.nn.functional as F
from torch import nn
import torch, math

__all__ = ["poly2mat"]


class CoordFourierPE(nn.Module):

    def __init__(self, hidden_dim, num_freq=8):
        super().__init__()

        freq = 2.0**torch.arange(num_freq)
        self.register_buffer("freq", freq)

        self.proj = nn.Sequential(
            nn.Linear(2+num_freq*4, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
    def forward(self, data, typeData):
        device = data.device
        if typeData == "mat":
            _b, _, _h, _w = data.shape
            yy, xx = torch.meshgrid(torch.linspace(0, 1, _h+1)[:-1], torch.linspace(0, 1, _w+1)[:-1], indexing="ij")
            data = torch.stack([xx, yy], dim=-1)
            data = data.reshape(1, _h*_w, 2).repeat(_b,1,1).to(device)

        x = data[..., 0:1]
        y = data[..., 1:2]
        freq = self.freq.to(device)
        x_freq = 2 * math.pi * x * freq
        y_freq = 2 * math.pi * y * freq

        fourier = torch.cat([torch.sin(x_freq), torch.cos(x_freq), torch.sin(y_freq), torch.cos(y_freq)], dim=-1)
        feature = torch.cat([data, fourier], dim=-1)
        return self.proj(feature)

class vertexRelativisticEncoding(nn.Module):
    def __init__(self, chan):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(6, chan, bias=False),
            nn.LayerNorm(chan),
            nn.GELU(),
        )
    def forward(self, x):
        x = self.proj(x)
        return x

class PolygonEmbedding(nn.Module):
    def __init__(self, chan):
        super().__init__()
        self.vertexRelativisticEncoder = vertexRelativisticEncoding(chan)
        encoderLayer = nn.TransformerEncoderLayer(
            d_model=chan,
            nhead=4,
            dim_feedforward=chan * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoderLayer,
            num_layers=4
        )
        self.norm = nn.LayerNorm(chan)
    def forward(self, xy, polyPad):
        x = self.vertexRelativisticEncoder(xy)
        x = self.transformer(x, src_key_padding_mask=~polyPad)
        x = self.norm(x)
        return x

class PolyTrans(nn.Module):
    def __init__(self, chanIn, chanOut, numHead, chanPoly):
        super().__init__()
        self.chanMid = chanOut // numHead * numHead
        self.chanEach = self.chanMid // numHead
        self.numHead = numHead
        self.embedPolyQ = nn.Sequential(
            nn.Linear(chanIn+chanPoly, self.chanMid, bias=False),
            nn.LayerNorm(self.chanMid)
        )
        self.embedPolyK = nn.Sequential(
            nn.Linear(chanPoly*2, self.chanMid, bias=False),
            nn.LayerNorm(self.chanMid)
        )
        self.embedPolyV = nn.Sequential(
            nn.Linear(chanPoly*2, self.chanMid, bias=False),
            nn.LayerNorm(self.chanMid)
        )
        self.mlpAttPoly = nn.Sequential(
            nn.Linear(self.chanMid, chanOut, bias=False),
            nn.Dropout(0.1)
        )
        self.FFNExpandPoly = nn.Sequential(
            nn.LayerNorm(chanOut),
            nn.Linear(chanOut, chanOut*4, bias=False),
            nn.Dropout(0.1)
        )
        self.FFNConv2DPoly = nn.Sequential(
            nn.Conv2d(chanOut*4, chanOut*4, kernel_size=3, stride=1, padding=1, groups=chanOut*4, bias=False),
            nn.GELU(),
            nn.Conv2d(chanOut*4, chanOut, kernel_size=1, stride=1, padding=0, bias=False)
        )
        self.normFinPoly = nn.LayerNorm(chanOut)
    def forward(self, pack):
        x, polyEmbed, polyPad, gridPE, polyPE = pack
        b, c, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)
        
        q = self.embedPolyQ(torch.concat([x, gridPE], -1)).reshape(b, -1, self.numHead, self.chanEach).transpose(1, 2)
        k = self.embedPolyK(torch.concat([polyEmbed, polyPE], -1)).reshape(b, -1, self.numHead, self.chanEach).transpose(1, 2)
        v = self.embedPolyV(torch.concat([polyEmbed, polyPE], -1)).reshape(b, -1, self.numHead, self.chanEach).transpose(1, 2)
        attPoly = F.scaled_dot_product_attention(query=q, key=k, value=v, dropout_p=0.1 if self.training else 0.0, is_causal=False, attn_mask=polyPad[:,None,None,:])
        x = x + self.mlpAttPoly(attPoly.transpose(1, 2).reshape(b, -1, self.chanMid))
        xFFN = self.FFNExpandPoly(x)
        xFFN = self.FFNConv2DPoly(xFFN.transpose(1, 2).reshape(b, -1, h, w))
        x = x + xFFN.flatten(2).transpose(1, 2)

        x = self.normFinPoly(x)
        x = x.transpose(1, 2).reshape(b, -1, h, w).contiguous()
        return x, polyEmbed, polyPad, gridPE, polyPE

class poly2mat(nn.Module):
    def __init__(self, chanIn=1, chanPoly=64, chanOut=1, scaleList=[1.0, 0.5, 0.25, 0.125], chanMidList=[32, 64, 128, 256], repeatAttList=[3, 3, 3, 3]):
        super().__init__()
        self.scaleList = scaleList
        self.embedPoly = PolygonEmbedding(chanPoly)
        self.coorinateEmbed = CoordFourierPE(chanPoly)
        self.conv2dEncode1 = nn.Sequential(
            nn.Conv2d(chanIn, chanMidList[0], 3, 1, 1, bias=False),
            # nn.InstanceNorm2d(chanMidList[0]),
            nn.BatchNorm2d(chanMidList[0]),
            nn.LeakyReLU(0.01),
        )
        self.conv2dEncode2 = nn.Sequential(
            nn.Conv2d(chanMidList[0], chanMidList[1], 3, 1, 1, bias=False),
            # nn.InstanceNorm2d(chanMidList[1]),
            nn.BatchNorm2d(chanMidList[1]),
            nn.LeakyReLU(0.01),
        )
        self.conv2dEncode3 = nn.Sequential(
            nn.Conv2d(chanMidList[1], chanMidList[2], 3, 1, 1, bias=False),
            # nn.InstanceNorm2d(chanMidList[2]),
            nn.BatchNorm2d(chanMidList[2]),
            nn.LeakyReLU(0.01),
        )
        self.conv2dEncode4 = nn.Sequential(
            nn.Conv2d(chanMidList[2], chanMidList[3], 3, 1, 1, bias=False),
            # nn.InstanceNorm2d(chanMidList[3]),
            nn.BatchNorm2d(chanMidList[3]),
            nn.LeakyReLU(0.01),
        )
        self.trans1 = nn.Sequential(*[PolyTrans(chanMidList[3], chanMidList[3], 1, chanPoly) for i in range(repeatAttList[0])])
        self.trans2 = nn.Sequential(*[PolyTrans(chanMidList[2], chanMidList[2], 2, chanPoly) for i in range(repeatAttList[1])])
        self.trans3 = nn.Sequential(*[PolyTrans(chanMidList[1], chanMidList[1], 3, chanPoly) for i in range(repeatAttList[2])])
        self.trans4 = nn.Sequential(*[PolyTrans(chanMidList[0], chanMidList[0], 4, chanPoly) for i in range(repeatAttList[3])])
        self.conv2dDecode1 = nn.Sequential(
            nn.Conv2d(chanMidList[3]*2, chanMidList[2], 3, 1, 1, bias=False),
            # nn.InstanceNorm2d(chanMidList[2]),
            nn.BatchNorm2d(chanMidList[2]),
            nn.LeakyReLU(0.01),
        )
        self.conv2dDecode2 = nn.Sequential(
            nn.Conv2d(chanMidList[2]*2, chanMidList[1], 3, 1, 1, bias=False),
            # nn.InstanceNorm2d(chanMidList[1]),
            nn.BatchNorm2d(chanMidList[1]),
            nn.LeakyReLU(0.01),
        )
        self.conv2dDecode3 = nn.Sequential(
            nn.Conv2d(chanMidList[1]*2, chanMidList[0], 3, 1, 1, bias=False),
            # nn.InstanceNorm2d(chanMidList[0]),
            nn.BatchNorm2d(chanMidList[0]),
            nn.LeakyReLU(0.01),
        )
        self.conv2dDecode4 = nn.Sequential(
            nn.Conv2d(chanMidList[0]*2, chanOut, 3, 1, 1),
            nn.Sigmoid()
        )
    def forward(self, x, poly=None, polyPad=None):
        if poly is not None:
            polyPE = self.coorinateEmbed(poly[:,:,:2], "poly")
            polyEmbed = self.embedPoly(poly, polyPad)
        b, c, h, w = x.shape
        xList = []
        idx = 0
        x = F.interpolate(x, size=(int(h*self.scaleList[idx]), int(w*self.scaleList[idx])), mode="bilinear", align_corners=False)
        x = self.conv2dEncode1(x)
        xList.append(x)
        idx = 1
        x = F.interpolate(x, size=(int(h*self.scaleList[idx]), int(w*self.scaleList[idx])), mode="bilinear", align_corners=False)
        x = self.conv2dEncode2(x)
        xList.append(x)
        idx = 2
        x = F.interpolate(x, size=(int(h*self.scaleList[idx]), int(w*self.scaleList[idx])), mode="bilinear", align_corners=False)
        x = self.conv2dEncode3(x)
        xList.append(x)
        idx = 3
        x = F.interpolate(x, size=(int(h*self.scaleList[idx]), int(w*self.scaleList[idx])), mode="bilinear", align_corners=False)
        x = self.conv2dEncode4(x)
        xList.append(x)

        feature = x
        idx = 3
        if poly is not None:
            gridPE = self.coorinateEmbed(feature, "mat")
            feature, polyEmbed, polyPad, gridPE, polyPE = self.trans1([feature, polyEmbed, polyPad, gridPE, polyPE])
        feature = torch.concat([F.interpolate(feature, size=(int(h*self.scaleList[idx]), int(w*self.scaleList[idx])), mode="bilinear", align_corners=False), xList[idx]], 1)
        feature = self.conv2dDecode1(feature)

        idx = 2
        if poly is not None:
            gridPE = self.coorinateEmbed(feature, "mat")
            feature, polyEmbed, polyPad, gridPE, polyPE = self.trans2([feature, polyEmbed, polyPad, gridPE, polyPE])
        feature = torch.concat([F.interpolate(feature, size=(int(h*self.scaleList[idx]), int(w*self.scaleList[idx])), mode="bilinear", align_corners=False), xList[idx]], 1)
        feature = self.conv2dDecode2(feature)

        idx = 1
        if poly is not None:
            gridPE = self.coorinateEmbed(feature, "mat")
            feature, polyEmbed, polyPad, gridPE, polyPE = self.trans3([feature, polyEmbed, polyPad, gridPE, polyPE])
        feature = torch.concat([F.interpolate(feature, size=(int(h*self.scaleList[idx]), int(w*self.scaleList[idx])), mode="bilinear", align_corners=False), xList[idx]], 1)
        feature = self.conv2dDecode3(feature)

        idx = 0
        if poly is not None:
            gridPE = self.coorinateEmbed(feature, "mat")
            feature, polyEmbed, polyPad, gridPE, polyPE = self.trans4([feature, polyEmbed, polyPad, gridPE, polyPE])
        feature = torch.concat([F.interpolate(feature, size=(int(h*self.scaleList[idx]), int(w*self.scaleList[idx])), mode="bilinear", align_corners=False), xList[idx]], 1)
        feature = self.conv2dDecode4(feature)
        x = feature
        return x
