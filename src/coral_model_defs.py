
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision.models as tvm

CLASS_NAMES = ['Healthy Coral', 'Dead / Degraded', 'Substrate / CCA']
IMAGENET_MEAN = (0.485, 0.456, 0.406); IMAGENET_STD = (0.229, 0.224, 0.225)
TL_NAMES = ['MobileNetV2', 'ResNet18', 'EfficientNet-B0', 'ShuffleNetV2']
MODEL_ORDER = TL_NAMES + ['CoralCNN-SE', 'DINOv2-ViT-S/14']

class NormWrap(nn.Module):
    """Input: RGB float tensor in [0,1].  Normalisation (mean/std buffers) is part of the model → weights are self-contained."""
    def __init__(self, net, mean, std, channels_last=False):
        super().__init__()
        self.net = net
        self.register_buffer('mean', torch.tensor(list(mean), dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer('std',  torch.tensor(list(std),  dtype=torch.float32).view(1, 3, 1, 1))
        self.channels_last = channels_last
    def forward(self, x):
        x = (x - self.mean) / self.std
        if self.channels_last: x = x.contiguous(memory_format=torch.channels_last)
        return self.net(x)

# ───────────────────────── CNN from scratch: residual + Squeeze-Excitation ─────────────────────────
class ConvBNAct(nn.Sequential):
    def __init__(self, cin, cout, k=3, s=1):
        super().__init__(nn.Conv2d(cin, cout, k, s, k // 2, bias=False), nn.BatchNorm2d(cout), nn.SiLU())
class SEBlock(nn.Module):
    def __init__(self, c, r=8):
        super().__init__()
        self.fc = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c, c // r, 1), nn.SiLU(), nn.Conv2d(c // r, c, 1), nn.Sigmoid())
    def forward(self, x): return x * self.fc(x)
class ResBlock(nn.Module):
    def __init__(self, cin, cout, stride=1, drop_path=0.0):
        super().__init__()
        self.conv1 = ConvBNAct(cin, cout, 3, stride)
        self.conv2 = nn.Sequential(nn.Conv2d(cout, cout, 3, 1, 1, bias=False), nn.BatchNorm2d(cout))
        self.se = SEBlock(cout)
        self.short = nn.Identity() if (cin == cout and stride == 1) else nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))
        self.act = nn.SiLU(); self.drop_path = drop_path
    def forward(self, x):
        out = self.se(self.conv2(self.conv1(x)))
        if self.training and self.drop_path > 0:                      # stochastic depth
            keep = (torch.rand(x.shape[0], 1, 1, 1, device=x.device) >= self.drop_path).to(out.dtype)
            out = out * keep / (1 - self.drop_path)
        return self.act(out + self.short(x))
class CoralCNN(nn.Module):
    def __init__(self, num_classes=3, widths=(32, 64, 128, 192, 256), depths=(1, 2, 2, 2), dropout=0.3, drop_path=0.1):
        super().__init__()
        self.stem = nn.Sequential(ConvBNAct(3, widths[0] // 2, 3, 2), ConvBNAct(widths[0] // 2, widths[0], 3, 1))
        stages, cin = [], widths[0]
        for i, (cout, d) in enumerate(zip(widths[1:], depths)):
            blocks = [ResBlock(cin, cout, stride=2, drop_path=drop_path * i / 3)] + [ResBlock(cout, cout, 1, drop_path=drop_path * i / 3) for _ in range(d - 1)]
            stages.append(nn.Sequential(*blocks)); cin = cout
        self.stage1, self.stage2, self.stage3, self.stage4 = stages
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout), nn.Linear(cin, num_classes))
        for m in self.modules():
            if isinstance(m, nn.Conv2d): nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d): nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear): nn.init.normal_(m.weight, 0, 0.01); nn.init.zeros_(m.bias)
        for m in self.modules():
            if isinstance(m, ResBlock): nn.init.zeros_(m.conv2[1].weight)
    def forward(self, x): return self.head(self.stage4(self.stage3(self.stage2(self.stage1(self.stem(x))))))

# ───────────────────────── Transfer learning backbones ─────────────────────────
def build_tl(name, n=3, pretrained=True):
    W = lambda enum: (enum.DEFAULT if pretrained else None)
    if name == 'MobileNetV2':
        m = tvm.mobilenet_v2(weights=W(tvm.MobileNet_V2_Weights)); m.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(m.last_channel, n))
    elif name == 'ResNet18':
        m = tvm.resnet18(weights=W(tvm.ResNet18_Weights)); m.fc = nn.Sequential(nn.Dropout(0.2), nn.Linear(m.fc.in_features, n))
    elif name == 'EfficientNet-B0':
        m = tvm.efficientnet_b0(weights=W(tvm.EfficientNet_B0_Weights)); m.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(m.classifier[1].in_features, n))
    elif name == 'ShuffleNetV2':
        m = tvm.shufflenet_v2_x1_0(weights=W(tvm.ShuffleNet_V2_X1_0_Weights)); m.fc = nn.Sequential(nn.Dropout(0.2), nn.Linear(m.fc.in_features, n))
    else:
        raise ValueError(name)
    return m

# ───────────────────────── DINOv2 ViT-S/14 (multi-layer pooled, partially unfrozen) ─────────────────────────
def load_dinov2_backbone(pretrained=True):
    try:
        return torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', pretrained=pretrained, trust_repo=True), 'hub'
    except Exception as e:                                   # offline GitHub / hub cache problems → timm mirror of the same weights
        print('torch.hub DINOv2 failed (%s) → falling back to timm' % repr(e)[:100])
        import timm
        return timm.create_model('vit_small_patch14_dinov2.lvd142m', pretrained=pretrained, num_classes=0, img_size=224), 'timm'

class DINOv2Classifier(nn.Module):
    """Mean-pooled patch tokens of the last `n_pool` transformer blocks (4×384 = 1536-d) → MLP head. Top `unfreeze` blocks are fine-tuned."""
    def __init__(self, num_classes=3, n_pool=4, dropout=0.2, pretrained=True, unfreeze=4):
        super().__init__()
        self.backbone, self.kind = load_dinov2_backbone(pretrained)
        self.n_pool = n_pool; dim = self.backbone.embed_dim
        for p in self.backbone.parameters(): p.requires_grad = False
        if unfreeze > 0:
            for blk in list(self.backbone.blocks)[-unfreeze:]:
                for p in blk.parameters(): p.requires_grad = True
            for p in self.backbone.norm.parameters(): p.requires_grad = True
        self.head = nn.Sequential(nn.Linear(dim * n_pool, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(dropout), nn.Linear(256, num_classes))
    def tokens(self, x):
        kw = {'norm': True} if self.kind == 'timm' else {}
        return self.backbone.get_intermediate_layers(x, n=self.n_pool, **kw)       # list of (B, 256, 384) patch-token maps
    def forward(self, x): return self.head(torch.cat([t.mean(1) for t in self.tokens(x)], dim=-1))

def build_model(name, pretrained=True, mean=(0.46, 0.54, 0.51), std=(0.22, 0.21, 0.21), unfreeze=4):
    """Returns a NormWrap model: input RGB in [0,1], output logits (B,3)."""
    if name in TL_NAMES:
        return NormWrap(build_tl(name, 3, pretrained), IMAGENET_MEAN, IMAGENET_STD, channels_last=True).to(memory_format=torch.channels_last)
    if name == 'CoralCNN-SE':
        return NormWrap(CoralCNN(3), mean, std, channels_last=True).to(memory_format=torch.channels_last)
    if name == 'DINOv2-ViT-S/14':
        return NormWrap(DINOv2Classifier(3, pretrained=pretrained, unfreeze=unfreeze), IMAGENET_MEAN, IMAGENET_STD, channels_last=False)
    raise ValueError(name)
