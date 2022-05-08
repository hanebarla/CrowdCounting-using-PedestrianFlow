import torch.nn as nn
import torch
from torch.nn import functional as F
from torchvision import models
from gradcam import GradCAM


class ContextualModule(nn.Module):
    def __init__(self, features, out_features=512, sizes=(1, 2, 3, 6), activate="leaky", do_rate=0.0):
        super(ContextualModule, self).__init__()
        self.scales = []
        self.scales = nn.ModuleList([self._make_scale(features, size) for size in sizes])
        self.bottleneck = nn.Conv2d(features * 2, out_features, kernel_size=1)
        if activate == "leaky":
            self.activate = nn.LeakyReLU()
        elif activate == "sigmoid":
            self.activate = nn.Sigmoid()
        else:
            self.activate = nn.ReLU()
        self.weight_net = nn.Conv2d(features, features, kernel_size=1)

    def __make_weight(self, feature, scale_feature):
        weight_feature = feature - scale_feature
        return torch.sigmoid(self.weight_net(weight_feature))

    def _make_scale(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        # multi_scales = [F.upsample(input=stage(feats), size=(h, w), mode='bilinear') for stage in self.scales]
        multi_scales = [F.interpolate(input=stage(feats), size=(h, w), mode='bilinear', align_corners=False) for stage in self.scales]
        weights = [self.__make_weight(feats, scale_feature) for scale_feature in multi_scales]
        overall_features = [(multi_scales[0]*weights[0]+multi_scales[1]*weights[1]+multi_scales[2]*weights[2]+multi_scales[3]*weights[3])/(weights[0]+weights[1]+weights[2]+weights[3])] + [feats]
        bottle = self.bottleneck(torch.cat(overall_features, 1))
        return self.activate(bottle)


class CANNet2s(nn.Module):
    def __init__(self, load_weights=False, activate="leaky", bn=0, do_rate=0.0):
        super(CANNet2s, self).__init__()
        self.context = ContextualModule(512, 512, activate=activate)
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.backend_feat = [512, 512, 512, 256, 128, 64]
        self.frontend = make_layers(self.frontend_feat, batch_norm=bn, activate=activate, do_rate=do_rate)
        self.backend = make_layers(self.backend_feat, in_channels=1024, batch_norm=True, dilation=True, activate=activate, do_rate=do_rate)
        self.output_layer = nn.Conv2d(64, 10, kernel_size=1)
        if activate == "leaky":
            self.activate = nn.LeakyReLU()
        elif activate == "sigmoid":
            self.activate = nn.Sigmoid()
        else:
            self.activate = nn.ReLU()
        self.relu = nn.ReLU()
        if not load_weights:
            mod = models.vgg16(pretrained=True)
            self._initialize_weights()
            # address the mismatch in key names for python 3
            pretrained_dict = {k[9:]: v for k, v in mod.state_dict().items() if k[9:] in self.frontend.state_dict()}
            self.frontend.load_state_dict(pretrained_dict)
        else:
            self._initialize_weights()

    def forward(self, x_prev, x):
        x_prev = self.frontend(x_prev)
        x = self.frontend(x)

        x_prev = self.context(x_prev)
        x = self.context(x)

        x = torch.cat((x_prev, x), 1)

        x = self.backend(x)
        x = self.output_layer(x)
        x = self.relu(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, in_channels=3, batch_norm=False, dilation=False, activate="leaky", do_rate=0.0):
    activates = {
        "leaky": nn.LeakyReLU(inplace=True),
        "sigmoid": nn.Sigmoid(),
        "relu": nn.ReLU()
    }
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate, dilation=d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), activates[activate]]
            else:
                layers += [conv2d, activates[activate]]

            if do_rate > 0.0:
                layers += [nn.Dropout2d(do_rate)]
            in_channels = v
    return nn.Sequential(*layers)


class CANnetGradCAM(GradCAM):
    def __init__(self, model_dict, verbose=False):
        super(CANnetGradCAM, self).__init__(model_dict, verbose)

    def forward(self, input, class_idx=None, retain_graph=False):
        b, c, h, w = input.size()

        logit = self.model_arch(input)
        if class_idx is None:
            score = logit[:, logit.max(1)[-1]].squeeze()
        else:
            score = logit[:, class_idx].squeeze()

        self.model_arch.zero_grad()
        score.backward(retain_graph=retain_graph)
        gradients = self.gradients['value']  # dS/dA
        activations = self.activations['value']  # A
        b, k, u, v = gradients.size()

        alpha_num = gradients.pow(2)
        alpha_denom = gradients.pow(2).mul(2) + \
            activations.mul(gradients.pow(3)).view(b, k, u*v).sum(-1, keepdim=True).view(b, k, 1, 1)
        alpha_denom = torch.where(alpha_denom != 0.0, alpha_denom, torch.ones_like(alpha_denom))

        alpha = alpha_num.div(alpha_denom+1e-7)
        positive_gradients = F.relu(score.exp()*gradients)  # ReLU(dY/dA) == ReLU(exp(S)*dS/dA))
        weights = (alpha*positive_gradients).view(b, k, u*v).sum(-1).view(b, k, 1, 1)

        saliency_map = (weights*activations).sum(1, keepdim=True)
        saliency_map = F.relu(saliency_map)
        saliency_map = F.upsample(saliency_map, size=(224, 224), mode='bilinear', align_corners=False)
        saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
        saliency_map = (saliency_map-saliency_map_min).div(saliency_map_max-saliency_map_min).data

        return saliency_map, logit


class SimpleCNN(nn.Module):
    def __init__(self, load_model=False):
        super().__init__()
        self.froted = nn.Sequential(
            nn.Conv2d(3, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.outlayer = nn.Sequential(
            nn.Conv2d(128, 10, 3, 2, 1),
            nn.ReLU(inplace=True)
        )

        if not load_model:
            self._init_weight()

    def forward(self, prev_x, x):
        prev_x = self.froted(prev_x)
        x = self.froted(x)

        mid_x = torch.cat((prev_x, x), 1)
        y = self.outlayer(mid_x)

        return y

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
