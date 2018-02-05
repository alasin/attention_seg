import drn

import torch
from torch import nn
from utils import *
from collections import OrderedDict


class DRNSeg(nn.Module):
    def __init__(self, model_name, classes, pretrained_model=None,
                 pretrained=True, use_torch_up=False):
        super(DRNSeg, self).__init__()
        model = drn.__dict__.get(model_name)(
            pretrained=pretrained, num_classes=1000)
        pmodel = nn.DataParallel(model)
        if pretrained_model is not None:
            pmodel.load_state_dict(pretrained_model)
        self.base = nn.Sequential(*list(model.children())[:-2])

        self.seg = nn.Conv2d(model.out_dim, classes,
                             kernel_size=1, bias=True)
        self.softmax = nn.LogSoftmax()
        m = self.seg
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        m.bias.data.zero_()
        if use_torch_up:
            self.up = nn.UpsamplingBilinear2d(scale_factor=8)
        else:
            up = nn.ConvTranspose2d(classes, classes, 16, stride=8, padding=4,
                                    output_padding=0, groups=classes,
                                    bias=False)
            fill_up_weights(up)
            up.weight.requires_grad = False
            self.up = up

    def forward(self, x):
        x = self.base(x)
        x = self.seg(x)
        y = self.up(x)
        return self.softmax(y), x

    def optim_parameters(self, memo=None):
        for param in self.base.parameters():
            yield param
        for param in self.seg.parameters():
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