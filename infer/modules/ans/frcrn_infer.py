# Code modified based on open source project 
# "modelscope" (https://github.com/modelscope/modelscope)
# and the implementation of all here is
# modified based on ans_pipeline.py (modelscope/modelscope/pipelines/audio/ans_pipeline.py)


from .frcrn.frcrn import FRCRNDecorator
import torch


model_config = {'complex': True, 'model_complexity': 45, 'model_depth': 14, 'log_amp': False, 'padding_mode': 'zeros', 'win_len': 640, 'win_inc': 320, 'fft_len': 640, 'win_type': 'hann', 'device_map': None, 'device': 'cuda'}
class FRCRN_infer:    
    
    SAMPLE_RATE = 16000

    def __init__(self, model_path="speech_frcrn_ans_cirm_16k", device="cpu"):
        self.device = device
        self.model = FRCRNDecorator(model_path, **model_config).to(self.device)
        self.model.eval()
    
    def forward(self, indata: torch.Tensor) -> torch.Tensor:
        
        indata = indata.unsqueeze(0)

        nsamples = indata.shape[-1]
        decode_do_segement = False
        window = 16000
        stride = int(window * 0.75)
        # print('inputs:{}'.format(ndarray.shape))
        b, t = indata.shape  # size()
        if t > window * 120:
            decode_do_segement = True

        if t < window:
            indata = torch.concatenate(
                [indata, torch.zeros((indata.shape[0], window - t), 
                                     device=self.device)], 1)
        elif t < window + stride:
            padding = window + stride - t
            # print('padding: {}'.format(padding))
            indata = torch.concatenate(
                [indata, torch.zeros((indata.shape[0], padding), 
                                  device=self.device)], 1)
        else:
            if (t - window) % stride != 0:
                padding = t - (t - window) // stride * stride
                # print('padding: {}'.format(padding))
                indata = torch.concatenate(
                    [indata, torch.zeros((indata.shape[0], padding),
                                      device=self.device)], 1)

        # print('inputs after padding:{}'.format(ndarray.shape))

        with torch.no_grad():
            # indata = torch.from_numpy(np.float32(indata)).to(self.device)
            b, t = indata.shape
            if decode_do_segement:
                outputs = torch.zeros(t, device=self.device)
                give_up_length = (window - stride) // 2
                current_idx = 0
                while current_idx + window <= t:
                    # print('current_idx: {}'.format(current_idx))
                    tmp_input = dict(noisy=indata[: , 
                                                current_idx: current_idx + window])
                    tmp_output = self.model(tmp_input)['wav_l2'][0]
                    end_index = current_idx + window - give_up_length

                    if current_idx == 0:
                        outputs[current_idx: end_index] = tmp_output[:-give_up_length]
                    else:
                        outputs[current_idx
                                + give_up_length:end_index] = tmp_output[
                                    give_up_length:-give_up_length]
                
                    current_idx += stride

            else:
                outputs = self.model(dict(noisy=indata))['wav_l2'][0]
                
        outputs = (outputs[:nsamples])

        return outputs