import torch
import torch.nn as nn
from safetensors.torch import load_file
from iopath.common.file_io import g_pathmgr
import logging

from typing import Optional, Dict

import rootutils
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


class Pi3(nn.Module):
    def __init__(
            self,
            pretrained_model_name_or_path: Optional[str] = None,
        ):
        super().__init__()

        if pretrained_model_name_or_path is not None:
            from models.pi3.models.pi3 import Pi3 as Pi3Model
            # self.model = Pi3Model.from_pretrained(pretrained_model_name_or_path)
            self.model = Pi3Model()
            if pretrained_model_name_or_path.endswith('.safetensors'):
                weight = load_file(pretrained_model_name_or_path)
                self.model.load_state_dict(weight)
            elif pretrained_model_name_or_path.endswith('.pt'):
                with g_pathmgr.open(pretrained_model_name_or_path, "rb") as f:
                    checkpoint = torch.load(f, map_location="cpu", weights_only=True)
                # Load model state
                model_state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
                missing, unexpected = self.model.load_state_dict(
                    model_state_dict, strict=False
                )
                logging.info(f"Model state loaded. Missing keys: {missing or 'None'}. Unexpected keys: {unexpected or 'None'}.")

        else:
            raise NotImplementedError

    def forward(self, images: torch.Tensor, get_global_feature_list: bool = False, global_feature_cond: torch.Tensor = None):
        return self.model(images, get_global_feature_list=get_global_feature_list, global_feature_cond=global_feature_cond)


class VGGT(nn.Module):
    def __init__(
            self,
            pretrained_model_name_or_path: Optional[str] = None,
        ):
        super().__init__()

        if pretrained_model_name_or_path is not None:
            from models.vggt.models.vggt import VGGT as VGGTModel
            # self.model = VGGTModel.from_pretrained(pretrained_model_name_or_path)
            self.model = VGGTModel()
            if pretrained_model_name_or_path.endswith('.safetensors'):
                weight = load_file(pretrained_model_name_or_path)
                self.model.load_state_dict(weight)
            elif pretrained_model_name_or_path.endswith('.pt'):
                with g_pathmgr.open(pretrained_model_name_or_path, "rb") as f:
                    checkpoint = torch.load(f, map_location="cpu", weights_only=True)
                # Load model state
                model_state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
                missing, unexpected = self.model.load_state_dict(
                    model_state_dict, strict=False
                )
                logging.info(f"Model state loaded. Missing keys: {missing or 'None'}. Unexpected keys: {unexpected or 'None'}.")
        else:
            raise NotImplementedError

    def forward(self, images: torch.Tensor, query_points: torch.Tensor = None):
        return self.model(images, query_points)
    

class MoGe(nn.Module):
    def __init__(
            self,
            pretrained_model_name_or_path: Optional[str] = None,
            ori_model: bool = True,
        ):
        super().__init__()

        if ori_model and pretrained_model_name_or_path is not None:
            from models.moge.model.v1 import MoGeModel
            self.model = MoGeModel.from_pretrained(pretrained_model_name_or_path)
        else:
            raise NotImplementedError

    def forward(self, image: torch.Tensor, num_tokens: int) -> Dict[str, torch.Tensor]:
        return self.model(image, num_tokens)


# class AetherV1(nn.Module):
#     def __init__(
#             self,
#             pretrained_model_name_or_path: Optional[str] = None,
#             ori_model: bool = True,
#         ):
#         super().__init__()

#         if ori_model and pretrained_model_name_or_path is not None:
#             from models.aether.v1 import AetherV1Model
#             self.model = AetherV1Model.from_pretrained(pretrained_model_name_or_path)
#         else:
#             raise NotImplementedError
#     def forward(self, images: torch.Tensor, query_points: torch.Tensor = None):
#         return self.model(images, query_points)

