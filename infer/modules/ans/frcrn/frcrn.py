# Code modified based on open source project 
# "modelscope" (https://github.com/modelscope/modelscope)
# and the implementation of all here is
# modified based on frcrn.py (modelscope/models/audio/ans/frcrn.py)


import os
from typing import Dict
from torch import nn
import torch
from .conv_stft import ConvSTFT, ConviSTFT
from .unet import UNet


class FRCRNDecorator(nn.Module):
    r""" A decorator of FRCRN for integrating into modelscope framework """

    def __init__(self, model_dir: str, *args, **kwargs):
        """initialize the frcrn model from the `model_dir` path.

        Args:
            model_dir (str): the model path.
        """
        super().__init__()
        self.model = FRCRN(*args, **kwargs)
        model_bin_file = os.path.join(model_dir, "frcrn.pt")
        checkpoint = torch.load(model_bin_file, map_location=torch.device('cpu'))
        self.model.load_state_dict(checkpoint, strict=False)

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        result_list = self.model.forward(inputs['noisy'])
        output = {
            'spec_l1': result_list[0],
            'wav_l1': result_list[1],
            'mask_l1': result_list[2],
            'spec_l2': result_list[3],
            'wav_l2': result_list[4],
            'mask_l2': result_list[5]
        }
        return output


class FRCRN(nn.Module):
    r""" Frequency Recurrent CRN """
    def __init__(self,
                complex,
                model_complexity,
                model_depth,
                log_amp,
                padding_mode,
                win_len=400,
                win_inc=100,
                fft_len=512,
                win_type='hann',
                **kwargs):
        r"""
        Args:
            complex: Whether to use complex networks.
            model_complexity: define the model complexity with the number of layers
            model_depth: Only two options are available : 10, 20
            log_amp: Whether to use log amplitude to estimate signals
            padding_mode: Encoder's convolution filter. 'zeros', 'reflect'
            win_len: length of window used for defining one frame of sample points
            win_inc: length of window shifting (equivalent to hop_size)
            fft_len: number of Short Time Fourier Transform (STFT) points
            win_type: windowing type used in STFT, eg. 'hanning', 'hamming'
        """
        super().__init__()
        self.feat_dim = fft_len // 2 + 1

        self.win_len = win_len
        self.win_inc = win_inc
        self.fft_len = fft_len
        self.win_type = win_type

        fix = True
        self.stft = ConvSTFT(
            self.win_len,
            self.win_inc,
            self.fft_len,
            self.win_type,
            feature_type='complex',
            fix=fix)
        self.istft = ConviSTFT(
            self.win_len,
            self.win_inc,
            self.fft_len,
            self.win_type,
            feature_type='complex',
            fix=fix)
        self.unet = UNet(
            1,
            complex=complex,
            model_complexity=model_complexity,
            model_depth=model_depth,
            padding_mode=padding_mode)
        self.unet2 = UNet(
            1,
            complex=complex,
            model_complexity=model_complexity,
            model_depth=model_depth,
            padding_mode=padding_mode)
        
    def forward(self, inputs):
        out_list = []
        # [B, D*2, T]
        cmp_spec = self.stft(inputs)
        # [B, 1, D*2, T]
        cmp_spec = torch.unsqueeze(cmp_spec, 1)

        # to [B, 2, D, T] real_part/imag_part
        cmp_spec = torch.cat([
            cmp_spec[:, :, :self.feat_dim, :],
            cmp_spec[:, :, self.feat_dim:, :],
        ], 1)

        # [B, 2, D, T]
        cmp_spec = torch.unsqueeze(cmp_spec, 4)
        # [B, 1, D, T, 2]
        cmp_spec = torch.transpose(cmp_spec, 1, 4)
        unet1_out = self.unet(cmp_spec)
        cmp_mask1 = torch.tanh(unet1_out)
        unet2_out = self.unet2(unet1_out)
        cmp_mask2 = torch.tanh(unet2_out)
        est_spec, est_wav, est_mask = self.apply_mask(cmp_spec, cmp_mask1)
        out_list.append(est_spec)
        out_list.append(est_wav)
        out_list.append(est_mask)
        cmp_mask2 = cmp_mask2 + cmp_mask1
        est_spec, est_wav, est_mask = self.apply_mask(cmp_spec, cmp_mask2)
        out_list.append(est_spec)
        out_list.append(est_wav)
        out_list.append(est_mask)
        return out_list
    
    def apply_mask(self, cmp_spec, cmp_mask):
        est_spec = torch.cat([
            cmp_spec[:, :, :, :, 0] * cmp_mask[:, :, :, :, 0]
            - cmp_spec[:, :, :, :, 1] * cmp_mask[:, :, :, :, 1],
            cmp_spec[:, :, :, :, 0] * cmp_mask[:, :, :, :, 1]
            + cmp_spec[:, :, :, :, 1] * cmp_mask[:, :, :, :, 0]
        ], 1)
        est_spec = torch.cat([est_spec[:, 0, :, :], est_spec[:, 1, :, :]], 1)
        cmp_mask = torch.squeeze(cmp_mask, 1)
        cmp_mask = torch.cat([cmp_mask[:, :, :, 0], cmp_mask[:, :, :, 1]], 1)

        est_wav = self.istft(est_spec)
        est_wav = torch.squeeze(est_wav, 1)
        return est_spec, est_wav, cmp_mask