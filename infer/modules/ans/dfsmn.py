# Code modified based on open source project 
# "modelscope" (https://github.com/modelscope/modelscope)
# and the implementation of all here is
# modified based on ans_dfsmn_pipeline.py (modelscope/modelscope/pipelines/audio/ans_dfsmn_pipeline.py)
# and uni_deep_fsmn.py (modelscope/models/audio/ans/layers/uni_deep_fsmn.py)


import os
import time
import warnings
import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


HOP_LENGTH = 960
N_FFT = 1920
WINDOW_NAME_HAM = 'hamming'
STFT_WIN_LEN = 1920
WINLEN = 3840
STRIDE = 1920
SAMPLE_RATE = 48000

class UniDeepFsmn(nn.Module):
    def __init__(self, input_dim, output_dim, lorder=1, hidden_size=None):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lorder = lorder
        self.hidden_size = hidden_size

        self.linear = nn.Linear(input_dim, hidden_size)
        self.project = nn.Linear(hidden_size, output_dim, bias=False)
        self.conv1 = nn.Conv2d(
            output_dim,
            output_dim, (lorder, 1), (1, 1),
            groups=output_dim,
            bias=False)

    def forward(self, input):
        """

        Args:
            input: torch with shape: batch (b) x sequence(T) x feature (h)

        Returns:
            batch (b) x channel (c) x sequence(T) x feature (h)
        """
        f1 = F.relu(self.linear(input))
        p1 = self.project(f1)
        x = torch.unsqueeze(p1, 1)
        # x: batch (b) x channel (c) x sequence(T) x feature (h)
        x_per = x.permute(0, 3, 2, 1)
        # x_per: batch (b) x feature (h) x sequence(T) x channel (c)
        y = F.pad(x_per, [0, 0, self.lorder - 1, 0])

        out = x_per + self.conv1(y)
        out1 = out.permute(0, 3, 2, 1)
        # out1: batch (b) x channel (c) x sequence(T) x feature (h)
        return input + out1.squeeze()
    

class DfsmnAns(nn.Module):
    def __init__(self,
                 fsmn_depth=9,
                 lorder=20,
                 *args,
                 **kwargs):
        super().__init__()
        self.lorder = lorder
        self.linear1 = nn.Linear(120, 256)
        self.relu = nn.ReLU()
        repeats = [
            UniDeepFsmn(256, 256, lorder, 256) for i in range(fsmn_depth)
        ]
        self.deepfsmn = nn.Sequential(*repeats)
        self.linear2 = nn.Linear(256, 961)
        self.sig = nn.Sigmoid()

    def forward(self, input):
        """
        Args:
            input: fbank feature [batch_size,number_of_frame,feature_dimension]

        Returns:
            mask value [batch_size, number_of_frame, FFT_size/2+1]
        """
        x1 = self.linear1(input)
        x2 = self.relu(x1)
        x3 = self.deepfsmn(x2)
        x4 = self.linear2(x3)
        x5 = self.sig(x4)
        return x5


class Dfsmn_infer():
    def __init__(self, model_path: str, device="cpu"):
        self.model = DfsmnAns()
        model_bin_file = os.path.join(model_path, "dfsmn.pt")
        checkpoint = torch.load(model_bin_file, map_location='cpu')
        self.model.load_state_dict(checkpoint)
        self.model.to(device).eval()

        window = torch.hamming_window(
            STFT_WIN_LEN, periodic=False, device=device).to(device)

        def stft(x):
            return torch.stft(
                x,
                N_FFT,
                HOP_LENGTH,
                STFT_WIN_LEN,
                center=False,
                window=window,
                return_complex=False
            )

        def istft(x, slen):
            return librosa.istft(
                x,
                hop_length=HOP_LENGTH,
                win_length=STFT_WIN_LEN,
                window=WINDOW_NAME_HAM,
                center=False,
                length=slen
            )

        def istft_torch(stft_matrix, slen):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return torch.istft(
                    stft_matrix,
                    n_fft=N_FFT,
                    hop_length=HOP_LENGTH,
                    win_length=STFT_WIN_LEN,
                    center=False,
                    window=window,
                    length=slen
                )

        self.stft = stft
        self.istft = istft
        self.istft_torch = istft_torch

        
    def _forward(self, origin_audio: torch.Tensor):
        t1 = time.time()
        with torch.no_grad():
            audio_in = origin_audio.unsqueeze(0)
            import torchaudio
            fbanks = torchaudio.compliance.kaldi.fbank(
                audio_in,
                dither=1.0,
                frame_length=40.0,
                frame_shift=20.0,
                num_mel_bins=120,
                sample_frequency=SAMPLE_RATE,
                window_type=WINDOW_NAME_HAM)
            fbanks = fbanks.unsqueeze(0)
            masks = self.model(fbanks)
            # print(f"model time: {time.time() - t1}")
            spectrum = self.stft(origin_audio)
            masks = masks.permute(2, 1, 0)
            masked_spec = (spectrum * masks)
            # print(f"stft time: {time.time() - t1}")
        
        masked_spec_complex = masked_spec[:, :, 0] + 1j * masked_spec[:, :, 1]
        # print(f"before isft time: {time.time() - t1}")
        masked_sig = self.istft_torch(masked_spec_complex, len(origin_audio)).detach()
        # print(f"total inferece time: {time.time() - t1}")
        return masked_sig

    def forward(self, inputs_wave: np.ndarray, **forward_params):
        inputs_wave = torch.tensor(inputs_wave * 32768)
        masked_sig = self._forward(inputs_wave)
        
        return masked_sig
    
    def forward_norm(self, inputs_wave: torch.Tensor, **forward_params):
        inputs_wave = inputs_wave * 32768
        masked_sig = self._forward(inputs_wave) / 32768
        
        return masked_sig