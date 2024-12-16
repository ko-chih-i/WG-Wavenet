import torch
import argparse
import numpy as np
from model.model import Model
from hparams import hparams as hps
from utils.util import mode, to_arr
from utils.audio import load_wav, save_wav, melspectrogram

def load_model(ckpt_pth):
    ckpt_dict = torch.load(ckpt_pth, map_location=None if hps.is_cuda else 'cpu')
    model = Model()
    model.load_state_dict(ckpt_dict['model'])
    model = model.remove_weightnorm(model)
    model = mode(model, True).eval()
    # pre-run
    model.set_inverse()
    with torch.no_grad():
        res = model.infer(mode(torch.zeros((1, 80, 10))))[0]
    torch.cuda.empty_cache()
    return model

def infer_from_mel(model, mel_pth):
    mel = np.load(mel_pth).astype(np.float32)
    mel = mode(torch.Tensor([mel]))
    with torch.no_grad():
        res = model.infer(mel)[0]
    return to_arr(res)

def infer_from_audio(model, src_pth):
    src = load_wav(src_pth, seg=False)
    mel = melspectrogram(src).astype(np.float32)
    mel = mode(torch.Tensor([mel]))
    with torch.no_grad():
        res = model.infer(mel)[0]
    return [src, to_arr(res)]

def save_audio(outputs, res_pth, mel_input=False):
    if not mel_input:
        src = outputs[0]
        save_wav(src, res_pth + '_src.wav')
    res = outputs[1] if not mel_input else outputs
    save_wav(res, res_pth + '_res.wav')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--ckpt_pth', type=str, required=True, help='Path to load checkpoints')
    parser.add_argument('-s', '--src_pth', type=str, default='', help='Path to source audio (optional if using mel)')
    parser.add_argument('-m', '--mel_pth', type=str, default='/content/drive/MyDrive/mel', help='Path to source mel-spectrogram (optional)')
    parser.add_argument('-r', '--res_pth', type=str, required=True, help='Path to save output wavs')

    args = parser.parse_args()

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False

    model = load_model(args.ckpt_pth)

    if args.mel_pth:
        res = infer_from_mel(model, args.mel_pth)
        save_audio(res, args.res_pth, mel_input=True)
    elif args.src_pth:
        outputs = infer_from_audio(model, args.src_pth)
        save_audio(outputs, args.res_pth)
    else:
        raise ValueError("Either --src_pth or --mel_pth must be provided.")
