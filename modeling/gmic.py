# Copyright (C) 2020 Yiqiu Shen, Nan Wu, Jason Phang, Jungkyu Park, Kangning Liu,
# Sudarshini Tyagi, Laura Heacock, S. Gene Kim, Linda Moy, Kyunghyun Cho, Krzysztof J. Geras
#
# This file is part of GMIC.
#
# GMIC is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# GMIC is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with GMIC.  If not, see <http://www.gnu.org/licenses/>.
# ==============================================================================

"""
Module that define the core logic of GMIC
"""

import torch
import torch.nn as nn
import numpy as np
from utilities import tools
import modeling.modules as m


class GMIC(nn.Module):
    def __init__(self, parameters):
        super(GMIC, self).__init__()

        # save parameters
        self.experiment_parameters = parameters
        self.cam_size = parameters["cam_size"]

        # construct networks
        # global network
        self.global_network = m.GlobalNetwork(self.experiment_parameters, self)
        self.global_network.add_layers()

        # aggregation function
        self.aggregation_function = m.TopTPercentAggregationFunction(self.experiment_parameters, self)

        # detection module
        self.retrieve_roi_crops = m.RetrieveROIModule(self.experiment_parameters, self)

        # detection network
        self.local_network = m.LocalNetwork(self.experiment_parameters, self)
        self.local_network.add_layers()

        # MIL module
        self.attention_module = m.AttentionModule(self.experiment_parameters, self)
        self.attention_module.add_layers()

        # clinical data based model
        self.mlp = m.MLP()

        # fusion branch
        # RGB 버전 수정
        self.fusion_dnn = nn.Linear(parameters["post_processing_dim"]+512+32, parameters["num_classes"])  # 32는 clinical_vector

    def _convert_crop_position(self, crops_x_small, cam_size, x_original):
        """
        Function that converts the crop locations from cam_size to x_original
        :param crops_x_small: N, k*c, 2 numpy matrix
        :param cam_size: (h,w)
        :param x_original: N, C, H, W pytorch variable
        :return: N, k*c, 2 numpy matrix
        """
        # retrieve the dimension of both the original image and the small version
        h, w = cam_size
        _, _, H, W = x_original.size()

        # interpolate the 2d index in h_small to index in x_original
        top_k_prop_x = crops_x_small[:, :, 0] / h
        top_k_prop_y = crops_x_small[:, :, 1] / w
        # sanity check
        assert np.max(top_k_prop_x) <= 1.0, "top_k_prop_x >= 1.0"
        assert np.min(top_k_prop_x) >= 0.0, "top_k_prop_x <= 0.0"
        assert np.max(top_k_prop_y) <= 1.0, "top_k_prop_y >= 1.0"
        assert np.min(top_k_prop_y) >= 0.0, "top_k_prop_y <= 0.0"
        # interpolate the crop position from cam_size to x_original
        top_k_interpolate_x = np.expand_dims(np.around(top_k_prop_x * H), -1)
        top_k_interpolate_y = np.expand_dims(np.around(top_k_prop_y * W), -1)
        top_k_interpolate_2d = np.concatenate([top_k_interpolate_x, top_k_interpolate_y], axis=-1)
        return top_k_interpolate_2d

    def _retrieve_crop(self, x_original_pytorch, crop_positions, crop_method):   # x_original_pytorch = img_input_var
        """
        Function that takes in the original image and cropping position and returns the crops
        :param x_original_pytorch: PyTorch Tensor array (N,C,H,W)
        :param crop_positions:
        :return:
        """
        batch_size, num_crops, _ = crop_positions.shape   # batch, 6, 2
        crop_h, crop_w = self.experiment_parameters["crop_shape"]

        output = torch.ones((batch_size, num_crops*x_original_pytorch.size()[1], crop_h, crop_w))
        if self.experiment_parameters["device_type"] == "gpu":
            device = torch.device("cuda:{}".format(self.experiment_parameters["gpu_number"]))
            output = output.cuda().to(device)
        # RGB 버전 수정
        for i in range(batch_size):     # 4
            order=0
            for j in range(num_crops):  # num_crops=6
                for k in range(x_original_pytorch.size()[1]):     # 3 channel   (patch 6개 * 3 channel (rgb)
                    tools.crop_pytorch(x_original_pytorch[i, k, :, :],
                                       self.experiment_parameters["crop_shape"],   # (256, 256)
                                       crop_positions[i,j,:],   # batch, 6, 2
                                       output[i, order,:,:],
                                       method=crop_method)
                    order += 1
        return output


    def forward(self, x_original, clinical_data):   # clinical_data
        """
        :param x_original: N,H,W,C numpy matrix
        """
        # global network: x_small -> class activation map
        h_g, self.saliency_map = self.global_network.forward(x_original)

        # calculate y_global
        # note that y_global is not directly used in inference
        self.y_global = self.aggregation_function.forward(self.saliency_map)

        # region proposal network
        small_x_locations = self.retrieve_roi_crops.forward(x_original, self.cam_size, self.saliency_map)

        # convert crop locations that is on self.cam_size to x_original
        self.patch_locations = self._convert_crop_position(small_x_locations, self.cam_size, x_original)

        # patch retriever
        crops_variable = self._retrieve_crop(x_original, self.patch_locations, self.retrieve_roi_crops.crop_method)   # crop_method = "upper_left"
        self.patches = crops_variable.data.cpu().numpy()
        # print(self.patches.shape)     # [4, 18, 256, 256]

        # detection network
        batch_size, num_crops, I, J = crops_variable.size()
        #print("patches size:", crops_variable.size())    # Size([4, 6, 256, 256]) -> rgb 고려 [4, 18, 256, 256]  (num_crops=18)

        # RGB 버전 수정
        # crops_variable = crops_variable.view(batch_size * num_crops, I, J).unsqueeze(1)     # [72, 1, 256, 256], local network는 3channel이므로 아래 h_crops에서 [72, 3, 256, 256] 변경 (gray scale)
        crops_variable = crops_variable.view(batch_size * self.experiment_parameters["K"], 3, I, J)    # RGB 버전에선 1channel -> 3channel 바꿀 필요 없음

        # h_crops = self.local_network.forward(crops_variable).view(batch_size, num_crops, -1)
        h_crops = self.local_network.forward(crops_variable).view(batch_size, self.experiment_parameters["K"], -1)
        #print("h_crops:", h_crops.size())   # [4,18,512]

        # MIL module
        # y_local is not directly used during inference
        z, self.patch_attns, self.y_local = self.attention_module.forward(h_crops)
        # print(self.patch_attns.size())   # (4, 18)

        # clinical data based model
        clinical_vec = self.mlp(clinical_data)
        # print(clinical_data.size())   # [4, 12]

        # fusion branch
        # use max pooling to collapse the feature map
        g1, _ = torch.max(h_g, dim=2)             # global max pooling
        global_vec, _ = torch.max(g1, dim=2)
        #print('global_hg :', global_vec.size(), 'z :', z.size())    # global_vec : [4, 512] / z : [4, 512]

        concat_vec = torch.cat([global_vec, z, clinical_vec], dim=1)
        # print('concat_vector :', concat_vec.size())   # [4, 1056]

        self.y_fusion = torch.sigmoid(self.fusion_dnn(concat_vec))

        return self.y_fusion, self.y_global, self.y_local, self.saliency_map
