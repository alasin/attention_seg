import drn

import torch
from torch import nn
from utils import *
from collections import OrderedDict



class DRNDepthSeg(nn.Module):
    def __init__(self, model_name, seg_classes, depth_classes, pretrained_model=None,
                 pretrained=True, use_torch_up=False, train_base=True, train_seg=True, train_depth=False):
        super(DRNDepthSeg, self).__init__()
        model = drn.__dict__.get(model_name)(
            pretrained=pretrained, num_classes=1000)
        
        num_channels = model.out_dim
        pmodel = nn.DataParallel(model)
        if pretrained_model is not None:
            pmodel.load_state_dict(pretrained_model)

        self.base_modules = list(model.children())
        self.num_base_modules = len(self.base_modules)

        self.base = nn.Sequential(*self.base_modules[:-2])
        self.seg = nn.Conv2d(model.out_dim, seg_classes, kernel_size=1, bias=True)

        depth_base = []
        depth_base.extend([nn.Conv2d(in_channels=num_channels, out_channels=256, kernel_size=3,
                            bias=False, padding=1),
                            nn.BatchNorm2d(256),
                            nn.ReLU(inplace=True)])
        depth_base.extend([nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3,
                            bias=False, padding=1),
                            nn.BatchNorm2d(128),
                            nn.ReLU(inplace=True)])

        self.depth_base = nn.Sequential(*depth_base)
        self.depth_cls_layer = nn.Conv2d(in_channels=128, out_channels=depth_classes, kernel_size=1, bias=True)
        self.depth_reg_layer = nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1, bias=True)

        # self.pred_params_conv = nn.Conv2d(num_channels, out_channels=num_channels/2, kernel_size=1, bias=True)
        # self.pred_params = nn.Conv2d(num_channels, out_channels=num_channels/2, kernel_size=1, bias=True)

        self.softmax = nn.LogSoftmax()


        ########## Initialize weights ##########
        fill_conv_weights(self.seg)

        for m in self.depth_base:
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        fill_conv_weights(self.depth_cls_layer)
        fill_conv_weights(self.depth_reg_layer)
        ########## Initialize weights ##########

        if use_torch_up:
            self.up = nn.UpsamplingBilinear2d(scale_factor=8)
        else:
            up = nn.ConvTranspose2d(seg_classes, seg_classes, 16, stride=8, padding=4,
                                    output_padding=0, groups=seg_classes,
                                    bias=False)
            fill_up_weights(up)
            up.weight.requires_grad = False
            self.up = up
        
        self.up_depth = nn.UpsamplingBilinear2d(scale_factor=8)
        self.train_base = train_base
        self.train_seg = train_seg
        self.train_depth = train_depth

    def forward(self, x):
        x = self.forward_base(x, 0, self.num_base_modules - 4)
        x_s = self.forward_base(x, self.num_base_modules - 4, self.num_base_modules - 2)
        
        # x_s = self.seg_base(x)
        y_s = self.seg(x_s)
        y_s = self.up(y_s)

        x_d = self.depth_base(x)
        # print(x_d.size())
        x_d = self.up_depth(x_d)
        # print(x_d.size())
        y_d_cls = self.depth_cls_layer(x_d)
        y_d_reg = self.depth_reg_layer(x_d)

        return self.softmax(y_s), self.softmax(y_d_cls), y_d_reg

    def forward_base(self, x, start, end):
        base_subset = nn.Sequential(*self.base_modules[start:end])
        x = base_subset(x)

        return x

    def optim_parameters(self, memo=None):
        if self.train_base:
            for param in self.base.parameters():
                yield param
        
        if self.train_seg:
            for param in self.seg_base.parameters():
                yield param
            for param in self.seg.parameters():
                yield param
        
        if self.train_depth:
            for m in self.depth_base:
                for param in m.parameters():
                    yield param
            for param in self.depth_cls_layer.parameters():
                yield param
            for param in self.depth_reg_layer.parameters():
                yield param

    def summary(self, input_size):
        def register_hook(module):
            def hook(module, input, output):
                if module._modules:  # only want base layers
                    return
                class_name = str(module.__class__).split('.')[-1].split("'")[0]
                module_idx = len(summary)
                m_key = '%s-%i' % (class_name, module_idx + 1)
                summary[m_key] = OrderedDict()
                summary[m_key]['input_shape'] = list(input[0].size())
                summary[m_key]['input_shape'][0] = None
                if output.__class__.__name__ == 'tuple':
                    summary[m_key]['output_shape'] = list(output[0].size())
                else:
                    summary[m_key]['output_shape'] = list(output.size())
                summary[m_key]['output_shape'][0] = None

                params = 0
                # iterate through parameters and count num params
                for name, p in module._parameters.items():
                    if not p is None:
                        params += torch.numel(p.data)
                        summary[m_key]['trainable'] = p.requires_grad

                summary[m_key]['nb_params'] = params

            if not isinstance(module, torch.nn.Sequential) and \
               not isinstance(module, torch.nn.ModuleList) and \
               not (module == self):
                hooks.append(module.register_forward_hook(hook))

        # check if there are multiple inputs to the network
        if isinstance(input_size[0], (list, tuple)):
            x = [torch.autograd.Variable(torch.rand(1, *in_size).cuda()) for in_size in input_size]
        else:
            x = torch.autograd.Variable(torch.randn(1, *input_size).cuda())

        # x = torch.autograd.Variable(x)

        # create properties
        summary = OrderedDict()
        hooks = []
        # register hook
        self.apply(register_hook)
        # make a forward pass
        self(x)
        # remove these hooks
        for h in hooks:
            h.remove()

        # print out neatly
        def get_names(module, name, acc):
            if not module._modules:
                acc.append(name)
            else:
                for key in module._modules.keys():
                    p_name = key if name == "" else name + "." + key
                    get_names(module._modules[key], p_name, acc)
        names = []
        get_names(self, "", names)

        col_width = 25  # should be >= 12
        summary_width = 61

        def crop(s):
            return s[:col_width] if len(s) > col_width else s

        print('_' * summary_width)
        print('{0: <{3}} {1: <{3}} {2: <{3}}'.format(
            'Layer (type)', 'Output Shape', 'Param #', col_width))
        print('=' * summary_width)
        total_params = 0
        trainable_params = 0
        for (i, l_type), l_name in zip(enumerate(summary), names):
            d = summary[l_type]
            total_params += d['nb_params']
            if 'trainable' in d and d['trainable']:
                trainable_params += d['nb_params']
            print('{0: <{3}} {1: <{3}} {2: <{3}}'.format(
                crop(l_name + ' (' + l_type[:-2] + ')'), crop(str(d['output_shape'])),
                crop(str(d['nb_params'])), col_width))
            if i < len(summary) - 1:
                print('_' * summary_width)
        print('=' * summary_width)
        print('Total params: ' + str(total_params))
        print('Trainable params: ' + str(trainable_params))
        print('Non-trainable params: ' + str((total_params - trainable_params)))
        print('_' * summary_width)