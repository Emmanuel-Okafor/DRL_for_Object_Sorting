#!/usr/bin/env python

from collections import OrderedDict
import numpy as np
from scipy import ndimage
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
import matplotlib.pyplot as plt
import time



def init_trunk_weights(model, branch=None):
    """ Initializes the trunk network weight layer weights.

    # Arguments

        branch: string indicating the specific branch to initialize. Default of None will initialize 'push-', 'grasp-' and 'place-'.
    """
    # Initialize network weights
    for m in model.named_modules():
        if((branch is None and 'push-' in m[0] or 'grasp-' in m[0] or 'place-' in m[0]) or
           (branch is not None and branch in m[0])):
            if isinstance(m[1], nn.Conv2d):
                nn.init.kaiming_normal_(m[1].weight.data)
            elif isinstance(m[1], nn.BatchNorm2d):
                m[1].weight.data.fill_(1)
                m[1].bias.data.zero_()


class reinforcement_net(nn.Module):

    def __init__(self, use_cuda): # , snapshot=None
        super(reinforcement_net, self).__init__()
        self.use_cuda = use_cuda

        # Initialize network trunks with DenseNet pre-trained on ImageNet
        self.push_color_trunk = torchvision.models.mobilenet_v3_large(pretrained=True)
        self.push_depth_trunk =  torchvision.models.mobilenet_v3_large(pretrained=True)
        self.grasp_color_trunk = torchvision.models.mobilenet_v3_large(pretrained=True)
        self.grasp_depth_trunk = torchvision.models.mobilenet_v3_large(pretrained=True)
        self.place_color_trunk = torchvision.models.mobilenet_v3_large(pretrained=True)
        self.place_depth_trunk =torchvision.models.mobilenet_v3_large(pretrained=True)
        self.place_prev_scene_color_trunk = torchvision.models.mobilenet_v3_large(pretrained=True)
        self.place_prev_scene_depth_trunk = torchvision.models.mobilenet_v3_large(pretrained=True)


        self.num_rotations = 16

        # Construct network branches for pushing and grasping
        self.pushnet = nn.Sequential(OrderedDict([
            ('push-norm0', nn.BatchNorm2d(1920)),   #2048
            ('push-relu0', nn.ReLU(inplace=True)),
            ('push-conv0', nn.Conv2d(1920, 64, kernel_size=1, stride=1, bias=False)),
            ('push-norm1', nn.BatchNorm2d(64)),   ##2048
            ('push-relu1', nn.ReLU(inplace=True)),
            ('push-conv1', nn.Conv2d(64, 1, kernel_size=1, stride=1, bias=False)) 
        ]))
        self.graspnet = nn.Sequential(OrderedDict([
            ('grasp-norm0', nn.BatchNorm2d(1920)),    #2048             
            ('grasp-relu0', nn.ReLU(inplace=True)),
            ('grasp-conv0', nn.Conv2d(1920, 64, kernel_size=1, stride=1, bias=False)), #2048
            ('grasp-norm1', nn.BatchNorm2d(64)),
            ('grasp-relu1', nn.ReLU(inplace=True)),
            ('grasp-conv1', nn.Conv2d(64, 1, kernel_size=1, stride=1, bias=False))
        ]))
        self.placenet = nn.Sequential(OrderedDict([
            ('place-norm0', nn.BatchNorm2d(3840)), 
            ('place-relu0', nn.ReLU(inplace=True)),
            ('place-conv0', nn.Conv2d(3840, 64, kernel_size=1, stride=1, bias=False)), 
            ('place-norm1', nn.BatchNorm2d(64)),
            ('place-relu1', nn.ReLU(inplace=True)),
            ('place-conv1', nn.Conv2d(64, 1, kernel_size=1, stride=1, bias=False))
        ]))

        # Initialize network weights
        for m in self.named_modules():
            if 'push-' in m[0] or 'grasp-' in m[0] or 'place-' in m[0]:
                if isinstance(m[1], nn.Conv2d):
                    nn.init.kaiming_normal_(m[1].weight.data)
                elif isinstance(m[1], nn.BatchNorm2d):
                    m[1].weight.data.fill_(1)
                    m[1].bias.data.zero_()

        # Initialize output variable (for backprop)
        self.interm_feat = []
        self.output_prob = []


    def forward(self, input_color_data, input_depth_data, prev_scene_input_color_data, prev_scene_input_depth_data, is_volatile=False, specific_rotation=-1):

        if is_volatile:
            output_prob = []
            interm_feat = []
            with torch.no_grad():

                is_place = False
                if prev_scene_input_color_data is not None and prev_scene_input_depth_data is not None:
                    is_place = True

                number_rotations = self.num_rotations
                if is_place:
                    number_rotations = 1

                # Apply rotations to images
                for rotate_idx in range(number_rotations):
                    rotate_theta = np.radians(rotate_idx * (360 / self.num_rotations))

                    interm_push_feat, interm_grasp_feat, interm_place_feat = self.layers_forward(rotate_theta, input_color_data, input_depth_data, prev_scene_input_color_data, prev_scene_input_depth_data)
                    interm_feat.append([interm_push_feat, interm_grasp_feat, interm_place_feat])

                    # Compute sample grid for rotation AFTER branches
                    affine_mat_after = np.asarray([[np.cos(rotate_theta), np.sin(rotate_theta), 0], [-np.sin(rotate_theta), np.cos(rotate_theta), 0]])
                    affine_mat_after.shape = (2, 3, 1)
                    affine_mat_after = torch.from_numpy(affine_mat_after).permute(2, 0, 1).float()

                    if is_place:
                        if self.use_cuda:
                            flow_grid_after = F.affine_grid(Variable(affine_mat_after, requires_grad=False).cuda(), interm_place_feat.data.size())
                        else:
                            flow_grid_after = F.affine_grid(Variable(affine_mat_after, requires_grad=False), interm_place_feat.data.size())

                        # Forward pass through branches, undo rotation on output predictions, upsample results
                        output_prob.append([None, None, nn.Upsample(scale_factor=16, mode='bilinear').forward(F.grid_sample(self.placenet(interm_place_feat), flow_grid_after, mode='nearest'))])
                    else:
                        if self.use_cuda:
                            flow_grid_after = F.affine_grid(Variable(affine_mat_after, requires_grad=False).cuda(), interm_push_feat.data.size())
                        else:
                            flow_grid_after = F.affine_grid(Variable(affine_mat_after, requires_grad=False), interm_push_feat.data.size())
                        # Forward pass through branches, undo rotation on output predictions, upsample results
                        output_prob.append([nn.Upsample(scale_factor=16, mode='bilinear').forward(F.grid_sample(self.pushnet(interm_push_feat), flow_grid_after, mode='nearest')),
                                            nn.Upsample(scale_factor=16, mode='bilinear').forward(F.grid_sample(self.graspnet(interm_grasp_feat), flow_grid_after, mode='nearest')),
                                            None])

            return output_prob, interm_feat

        else:
            self.output_prob = []
            self.interm_feat = []

            is_place = False
            if prev_scene_input_color_data is not None and prev_scene_input_depth_data is not None:
                is_place = True

            # Apply rotations to intermediate features
            rotate_idx = specific_rotation

            rotate_theta = np.radians(rotate_idx*(360/self.num_rotations))

            interm_push_feat, interm_grasp_feat, interm_place_feat = self.layers_forward(rotate_theta, input_color_data, input_depth_data, prev_scene_input_color_data, prev_scene_input_depth_data)
            self.interm_feat.append([interm_push_feat, interm_grasp_feat, interm_place_feat])

            # Compute sample grid for rotation AFTER branches
            affine_mat_after = np.asarray([[np.cos(rotate_theta), np.sin(rotate_theta), 0], [-np.sin(rotate_theta), np.cos(rotate_theta), 0]])
            affine_mat_after.shape = (2,3,1)
            affine_mat_after = torch.from_numpy(affine_mat_after).permute(2,0,1).float()

            if is_place:
                if self.use_cuda:
                    flow_grid_after = F.affine_grid(Variable(affine_mat_after, requires_grad=False).cuda(), interm_place_feat.data.size())
                else:
                    flow_grid_after = F.affine_grid(Variable(affine_mat_after, requires_grad=False), interm_place_feat.data.size())
                # Forward pass through branches, undo rotation on output predictions, upsample results
                self.output_prob.append([None, None, nn.Upsample(scale_factor=16, mode='bilinear').forward(F.grid_sample(self.placenet(interm_place_feat), flow_grid_after, mode='nearest'))])
            else:
                if self.use_cuda:
                    flow_grid_after = F.affine_grid(Variable(affine_mat_after, requires_grad=False).cuda(), interm_push_feat.data.size())
                else:
                    flow_grid_after = F.affine_grid(Variable(affine_mat_after, requires_grad=False), interm_push_feat.data.size())
                # Forward pass through branches, undo rotation on output predictions, upsample results
                self.output_prob.append([nn.Upsample(scale_factor=16, mode='bilinear').forward(F.grid_sample(self.pushnet(interm_push_feat), flow_grid_after, mode='nearest')),
                                         nn.Upsample(scale_factor=16, mode='bilinear').forward(F.grid_sample(self.graspnet(interm_grasp_feat), flow_grid_after, mode='nearest')),
                                         None])

            return self.output_prob, self.interm_feat


    def layers_forward(self, rotate_theta, input_color_data, input_depth_data, prev_scene_input_color_data, prev_scene_input_depth_data):

        # Compute sample grid for rotation BEFORE neural network
        affine_mat_before = np.asarray([[np.cos(-rotate_theta), np.sin(-rotate_theta), 0], [-np.sin(-rotate_theta), np.cos(-rotate_theta), 0]])
        affine_mat_before.shape = (2, 3, 1)
        affine_mat_before = torch.from_numpy(affine_mat_before).permute(2, 0, 1).float()
        if self.use_cuda:
            flow_grid_before = F.affine_grid(Variable(affine_mat_before, requires_grad=False).cuda(), input_color_data.size())
        else:
            flow_grid_before = F.affine_grid(Variable(affine_mat_before, requires_grad=False), input_color_data.size())

        is_place = False
        if prev_scene_input_color_data is not None and prev_scene_input_depth_data is not None:
            is_place = True

        # Rotate images clockwise
        if self.use_cuda:
            rotate_color = F.grid_sample(Variable(input_color_data).cuda(), flow_grid_before, mode='nearest')
            rotate_depth = F.grid_sample(Variable(input_depth_data).cuda(), flow_grid_before, mode='nearest')
            if prev_scene_input_color_data is not None and prev_scene_input_depth_data is not None:
                prev_scene_rotate_color = F.grid_sample(Variable(prev_scene_input_color_data).cuda(), flow_grid_before, mode='nearest')
                prev_scene_rotate_depth = F.grid_sample(Variable(prev_scene_input_depth_data).cuda(), flow_grid_before, mode='nearest')
        else:
            rotate_color = F.grid_sample(Variable(input_color_data), flow_grid_before, mode='nearest')
            rotate_depth = F.grid_sample(Variable(input_depth_data), flow_grid_before, mode='nearest')
            if prev_scene_input_color_data is not None and prev_scene_input_depth_data is not None:
                prev_scene_rotate_color = F.grid_sample(Variable(prev_scene_input_color_data), flow_grid_before, mode='nearest')
                prev_scene_rotate_depth = F.grid_sample(Variable(prev_scene_input_depth_data), flow_grid_before, mode='nearest')

        # Compute intermediate features
        if is_place:
            interm_place_color_feat = self.place_color_trunk.features(rotate_color)
            interm_place_depth_feat = self.place_depth_trunk.features(rotate_depth)
            interm_prev_scene_place_color_feat = self.place_prev_scene_color_trunk.features(prev_scene_rotate_color)
            interm_prev_scene_place_depth_feat = self.place_prev_scene_depth_trunk.features(prev_scene_rotate_depth)

            interm_place_feat = torch.cat((interm_place_color_feat, interm_place_depth_feat, interm_prev_scene_place_color_feat, interm_prev_scene_place_depth_feat), dim=1)

            return None, None, interm_place_feat

        else:
            interm_push_color_feat = self.push_color_trunk.features(rotate_color)
            interm_push_depth_feat = self.push_depth_trunk.features(rotate_depth)
            interm_push_feat = torch.cat((interm_push_color_feat, interm_push_depth_feat), dim=1)

            interm_grasp_color_feat = self.grasp_color_trunk.features(rotate_color)
            interm_grasp_depth_feat = self.grasp_depth_trunk.features(rotate_depth)
            interm_grasp_feat = torch.cat((interm_grasp_color_feat, interm_grasp_depth_feat), dim=1)

            return interm_push_feat, interm_grasp_feat, None