import torch.nn as nn
import torch
import torch.nn.functional as F


class Conv2D(nn.Module):
    def __init__(self, in_ch: int = 3, out_ch: int = 64) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.conv(x))


class Encoder(nn.Module):
    def __init__(
        self,
        in_ch: int = 3,
        out_ch: int = 64,
        dropout_prob: float = 0,
        max_pooling: bool = True,
    ):
        super().__init__()
        self.dropout_prob = dropout_prob
        self.max_pooling = max_pooling
        self.conv1 = Conv2D(in_ch, out_ch)
        self.conv2 = Conv2D(out_ch, out_ch)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        conv = self.conv1(x)
        conv = self.conv2(conv)

        if self.dropout_prob > 0:
            conv = F.dropout(conv, self.dropout_prob)

        if self.max_pooling:
            next_layer = F.max_pool2d(conv, 2)
        else:
            next_layer = conv

        skip = conv

        return next_layer, skip


class Decoder(nn.Module):
    def __init__(self, in_ch: int = 64, out_ch: int = 64) -> None:
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_ch, out_ch, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        self.conv1 = Conv2D(out_ch * 2, out_ch)
        self.conv2 = Conv2D(out_ch, out_ch)

    def forward(self, next: torch.Tensor, skip: torch.Tensor):
        up = self.conv_transpose(next)
        merge = torch.concat([skip, up], dim=1)
        conv = self.conv1(merge)
        conv = self.conv2(conv)
        return conv


class UNet(nn.Module):
    def __init__(self, in_ch: int = 3, out_ch: int = 64) -> None:
        super().__init__()
        self.encode1 = Encoder(in_ch, out_ch)
        self.encode2 = Encoder(out_ch, out_ch * 2)
        self.encode3 = Encoder(out_ch * 2, out_ch * 4)
        self.encode4 = Encoder(out_ch * 4, out_ch * 8, dropout_prob=0.3)
        self.encode5 = Encoder(
            out_ch * 8, out_ch * 16, dropout_prob=0.3, max_pooling=False
        )
        self.decode6 = Decoder(out_ch * 16, out_ch * 8)
        self.decode7 = Decoder(out_ch * 8, out_ch * 4)
        self.decode8 = Decoder(out_ch * 4, out_ch * 2)
        self.decode9 = Decoder(out_ch * 2, out_ch)
        self.conv10 = Conv2D(out_ch, out_ch)
        self.conv11 = nn.Conv2d(out_ch, 1, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor):
        cblock1 = self.encode1(x)
        cblock2 = self.encode2(cblock1[0])
        cblock3 = self.encode3(cblock2[0])
        cblock4 = self.encode4(cblock3[0])
        cblock5 = self.encode5(cblock4[0])
        ublock6 = self.decode6(cblock5[0], cblock4[1])
        ublock7 = self.decode7(ublock6, cblock3[1])
        ublock8 = self.decode8(ublock7, cblock2[1])
        ublock9 = self.decode9(ublock8, cblock1[1])
        conv10 = self.conv10(ublock9)
        return F.sigmoid(self.conv11(conv10))
