from gc import enable
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from functools import partial
from copy import deepcopy
import time

from .dinov2.layers import Mlp
from ..utils.geometry import homogenize_points
from .layers.pos_embed import RoPE2D, PositionGetter
from .layers.block import BlockRope
from .layers.attention import FlashAttentionRope, MemEffAttentionRope
from .layers.transformer_head import TransformerDecoder, LinearPts3d
from .layers.camera_head import CameraHead
from .layers.projectors import MLPProjector, ZeroConvNet, DescProjector
from .dinov2.hub.backbones import dinov2_vitl14, dinov2_vitl14_reg
from huggingface_hub import PyTorchModelHubMixin
import einops
from sklearn.decomposition import PCA

class Pi3(nn.Module, PyTorchModelHubMixin):
    def __init__(
            self,
            pos_type='rope100',
            decoder_size='large',
            rank=4,
            alpha=16,
            detach_conf=True,
            get_scene_descs=True,
        ):
        super().__init__()

        # ----------------------
        #        Encoder
        # ----------------------
        self.encoder = dinov2_vitl14_reg(pretrained=False)
        self.patch_size = 14
        del self.encoder.mask_token

        # ----------------------
        #  Positonal Encoding
        # ----------------------
        self.pos_type = pos_type if pos_type is not None else 'none'
        self.rope=None
        if self.pos_type.startswith('rope'): # eg rope100 
            if RoPE2D is None: raise ImportError("Cannot find cuRoPE2D, please install it following the README instructions")
            freq = float(self.pos_type[len('rope'):])
            self.rope = RoPE2D(freq=freq)
            self.position_getter = PositionGetter()
        else:
            raise NotImplementedError
        

        # ----------------------
        #        Decoder
        # ----------------------
        enc_embed_dim = self.encoder.blocks[0].attn.qkv.in_features        # 1024
        if decoder_size == 'small':
            dec_embed_dim = 384
            dec_num_heads = 6
            mlp_ratio = 4
            dec_depth = 24
        elif decoder_size == 'base':
            dec_embed_dim = 768
            dec_num_heads = 12
            mlp_ratio = 4
            dec_depth = 24
        elif decoder_size == 'large':
            dec_embed_dim = 1024
            dec_num_heads = 16
            mlp_ratio = 4
            dec_depth = 36
        else:
            raise NotImplementedError
        
        decoder_list = []
        for layer_idx in range(dec_depth):
            decoder_list.append(
                BlockRope(
                dim=dec_embed_dim,
                num_heads=dec_num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                proj_bias=True,
                ffn_bias=True,
                drop_path=0.0,
                attn_drop=0.0,
                drop=0.0,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                act_layer=nn.GELU,
                ffn_layer=Mlp,
                init_values=0.01,
                qk_norm=True,
                attn_class=FlashAttentionRope,
                rope=self.rope,
            ))
        self.decoder = nn.ModuleList(decoder_list)
        
        scene_desc_decoder_list = []
        for layer_idx in range(10):
            scene_desc_decoder_list.append(
                BlockRope(
                dim=dec_embed_dim,
                num_heads=dec_num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                proj_bias=True,
                ffn_bias=True,
                drop_path=0.0,
                attn_drop=0.0,
                drop=0.0,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                act_layer=nn.GELU,
                ffn_layer=Mlp,
                init_values=0.01,
                qk_norm=True,
                attn_class=FlashAttentionRope,
                rope=self.rope,
            ))
        self.scene_desc_decoder = nn.ModuleList(scene_desc_decoder_list)

        self.scene_desc_proj = DescProjector(input_dim=dec_embed_dim, output_dim=dec_embed_dim, rope=self.rope)

        self.dec_embed_dim = dec_embed_dim
        self.get_scene_descs = get_scene_descs

        # ----------------------
        #     Register_token
        # ----------------------
        num_register_tokens = 5
        self.patch_start_idx = num_register_tokens
        self.register_token = nn.Parameter(torch.randn(1, 1, num_register_tokens, self.dec_embed_dim))
        nn.init.normal_(self.register_token, std=1e-6)

        # ----------------------
        #  Local Points Decoder
        # ----------------------
        self.point_decoder = TransformerDecoder(
            in_dim=2*self.dec_embed_dim, 
            dec_embed_dim=1024,
            dec_num_heads=16,
            out_dim=1024,
            rope=self.rope,
            use_checkpoint=True,
        )
        self.point_head = LinearPts3d(patch_size=14, dec_embed_dim=1024, output_dim=3)
        # self.point_head = LinearPts3d(patch_size=14, dec_embed_dim=1024, output_dim=3)

        # ----------------------
        #     Conf Decoder
        # ----------------------
        self.conf_decoder = TransformerDecoder(
            in_dim=2*self.dec_embed_dim, 
            dec_embed_dim=1024,
            dec_num_heads=16,
            out_dim=1024,
            rope=self.rope,
            use_checkpoint=True,
            # enable_LoRA=enable_LoRA and enable_head_LoRA,
            # rank=rank,
            # alpha=alpha,
        )
        # self.conf_head = LinearPts3d(patch_size=14, dec_embed_dim=1024, output_dim=1, 
        #                              enable_LoRA=enable_LoRA and enable_head_LoRA, rank=rank, alpha=alpha)
        self.conf_head = LinearPts3d(patch_size=14, dec_embed_dim=1024, output_dim=1)
        self.detach_conf = detach_conf

        # ----------------------
        #  Camera Pose Decoder
        # ----------------------
        self.camera_decoder = TransformerDecoder(
            in_dim=2*self.dec_embed_dim, 
            dec_embed_dim=1024,
            dec_num_heads=16,                # 8
            out_dim=512,
            rope=self.rope,
            use_checkpoint=True,
        )
        self.camera_head = CameraHead(dim=512)
        # self.camera_head = CameraHead(dim=512)

        # For ImageNet Normalize
        image_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        image_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        self.register_buffer("image_mean", image_mean)
        self.register_buffer("image_std", image_std)

    def decode_scene_descs(self, hidden, N, H, W, patch_h, patch_w):
        BN, hw, C = hidden.shape
        B = BN // N
        
        hidden = hidden.reshape(B*N, hw, -1)
        register_token = self.register_token.repeat(B, N, 1, 1).reshape(B*N, *self.register_token.shape[-2:])

        # Concatenate special tokens with patch tokens
        hidden = torch.cat([register_token, hidden], dim=1)
        hw = hidden.shape[1]

        if self.pos_type.startswith('rope'):
            pos = self.position_getter(B * N, H//self.patch_size, W//self.patch_size, hidden.device)

        if self.patch_start_idx > 0:
            # do not use position embedding for special tokens (camera and register tokens)
            # so set pos to 0 for the special tokens
            pos = pos + 1
            pos_special = torch.zeros(B * N, self.patch_start_idx, 2).to(hidden.device).to(pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)
        
        scene_descs_list = []
        for i in range(len(self.scene_desc_decoder)):
            blk = self.scene_desc_decoder[i]

            if i % 2 == 0:
                pos = pos.reshape(B*N, hw, -1)
                hidden = hidden.reshape(B*N, hw, -1)
                # hidden = blk(hidden, xpos=pos)
                if self.training:
                    hidden = checkpoint(blk, hidden, xpos=pos, use_reentrant=False)
                else:
                    hidden = blk(hidden, xpos=pos)
                
            else:
                pos = pos.reshape(B, N*hw, -1)
                hidden = hidden.reshape(B, N*hw, -1)      
                if self.training:
                    hidden = checkpoint(blk, hidden, xpos=pos, use_reentrant=False)
                else:
                    hidden = blk(hidden, xpos=pos)
        # import ipdb; ipdb.set_trace()
        scene_descs = self.scene_desc_proj(hidden.reshape(B*N, hw, -1), xpos=pos.reshape(B*N, hw, -1))
        # import ipdb; ipdb.set_trace()
        
        scene_descs = scene_descs.reshape(B, N, -1)

        return scene_descs

    def decode(self, hidden, N, H, W, patch_h, patch_w, scene_descs):
        BN, hw, C = hidden.shape
        B = BN // N

        final_output = []
        
        hidden = hidden.reshape(B*N, hw, -1)
        register_token = self.register_token.repeat(B, N, 1, 1).reshape(B*N, *self.register_token.shape[-2:])
        # import ipdb; ipdb.set_trace() # check if the shape of scene_descs aligns with register_token
        scene_descs = scene_descs.reshape(B*N, 1, -1)

        # Concatenate special tokens with patch tokens
        hidden = torch.cat([scene_descs, register_token, hidden], dim=1)
        hw = hidden.shape[1]
        # import ipdb; ipdb.set_trace()
        if self.pos_type.startswith('rope'):
            pos = self.position_getter(B * N, H//self.patch_size, W//self.patch_size, hidden.device)

        new_patch_start_idx = self.patch_start_idx + 1

        if new_patch_start_idx > 0:
            # do not use position embedding for special tokens (camera and register tokens)
            # so set pos to 0 for the special tokens
            pos = pos + 1
            pos_special = torch.zeros(B * N, new_patch_start_idx, 2).to(hidden.device).to(pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)
        
        for i in range(len(self.decoder)):
            blk = self.decoder[i]

            if i % 2 == 0:
                pos = pos.reshape(B*N, hw, -1)
                hidden = hidden.reshape(B*N, hw, -1)
                # hidden = blk(hidden, xpos=pos)
                if self.training:
                    hidden = checkpoint(blk, hidden, xpos=pos, use_reentrant=False)
                else:
                    hidden = blk(hidden, xpos=pos)
                
            else:
                pos = pos.reshape(B, N*hw, -1)
                hidden = hidden.reshape(B, N*hw, -1)      
                if self.training:
                    hidden = checkpoint(blk, hidden, xpos=pos, use_reentrant=False)
                else:
                    hidden = blk(hidden, xpos=pos)

            if i+1 in [len(self.decoder)-1, len(self.decoder)]:
                # remove the scene_descs token
                final_output.append(hidden.reshape(B*N, hw, -1)[:, 1:, :])  # remove scene descs token

        ret = {
            "final_output": torch.cat([final_output[0], final_output[1]], dim=-1),
            "pos": pos.reshape(B*N, hw, -1)[:, 1:, :],
        }        

        return ret
    
    def forward(self, images, 
                extrin_cond=None, intrin_cond=None, 
                use_extrin=None, use_intrin=None, 
                edge_mask=None, 
                use_depth_prompt=None, depth_prompt=None,
                ):
        # print(f"start forwarding, images.shape={images.shape}")
        # torch.cuda.synchronize()
        # start = time.time()
        images = (images - self.image_mean) / self.image_std
        B, N, _, H, W = images.shape
        patch_h, patch_w = H // 14, W // 14
        # save images for debugging
        # import torchvision
        # torchvision.utils.save_image(images.reshape(B*N, 3, H, W)[0:1], "debug_input_images.png", nrow=1)
        
        # encode by dinov2
        images = images.reshape(B*N, _, H, W)
        hidden = self.encoder(images, is_training=True)

        if isinstance(hidden, dict):
            hidden = hidden["x_norm_patchtokens"]

        scene_descs = self.decode_scene_descs(hidden, N, H, W, patch_h, patch_w)

        decode_ret = self.decode(hidden, N, H, W, patch_h, patch_w, scene_descs)
        hidden = decode_ret["final_output"]
        pos = decode_ret["pos"]

        point_hidden = self.point_decoder(hidden, xpos=pos)
        if self.detach_conf:
            conf_hidden = self.conf_decoder(hidden.detach(), xpos=pos)
        else:
            conf_hidden = self.conf_decoder(hidden, xpos=pos)
        camera_hidden = self.camera_decoder(hidden, xpos=pos)

        with torch.amp.autocast(device_type='cuda', enabled=False):
            # local points
            point_hidden = point_hidden.float()
            ret = self.point_head([point_hidden[:, self.patch_start_idx:]], (H, W)).reshape(B, N, H, W, -1)
            xy, z = ret.split([2, 1], dim=-1)
            z = torch.exp(z)
            local_points = torch.cat([xy * z, z], dim=-1)

            # confidence
            conf_hidden = conf_hidden.float()
            conf = self.conf_head([conf_hidden[:, self.patch_start_idx:]], (H, W)).reshape(B, N, H, W, -1)

            # camera
            camera_hidden = camera_hidden.float()
            camera_poses = self.camera_head(camera_hidden[:, self.patch_start_idx:], patch_h, patch_w).reshape(B, N, 4, 4)

            # unproject local points using camera poses
            points = None
            points = torch.einsum('bnij, bnhwj -> bnhwi', camera_poses, homogenize_points(local_points))[..., :3]

        # torch.cuda.synchronize()
        # end = time.time()
        # print(f"finish forwarding, {end-start}s used")

        return dict(
            points=points,
            local_points=local_points,
            conf=conf,
            camera_poses=camera_poses,
            scene_descs=scene_descs if self.get_scene_descs else None,
        )
