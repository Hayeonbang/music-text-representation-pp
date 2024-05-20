import os
import argparse
import random
import torch
import pickle
import numpy as np
from tqdm import tqdm
from glob import glob
from torch.utils.data import Dataset
from mtrpp.utils.audio_utils import int16_to_float32, float32_to_int16, load_audio, STR_CH_FIRST
from mtrpp.utils.eval_utils import load_ttmr_pp

class AT_DATASET(Dataset):
    def __init__(self, data_dir, _src_ext_audio=(".mp3", ".wav"), sr=22050, duration=10):
        self.data_dir = data_dir
        self._src_ext_audio = _src_ext_audio
        self.sr = sr        
        self.n_samples = int(sr * duration)
        self.fl = []
        for ext in self._src_ext_audio:
            self.fl.extend(glob(
                os.path.join(self.data_dir, "**", f"*{ext}"),
                recursive=True,
            ))

    def load_wav(self, audio_path):
        audio, _ = load_audio(
            path=audio_path,
            ch_format=STR_CH_FIRST,
            sample_rate=self.sr,
            downmix_to_mono=True)
        if len(audio.shape) == 2:
            audio = audio.squeeze(0)
        audio = int16_to_float32(float32_to_int16(audio))  # for float32 loader
        
        if len(audio) < self.n_samples:
            raise ValueError(f"Audio file {audio_path} is too short.")

        ceil = int(audio.shape[-1] // self.n_samples)
        audio_tensor = torch.from_numpy(np.stack(np.split(audio[:ceil * self.n_samples], ceil)).astype('float32'))
        return audio_tensor

    def __getitem__(self, index):
        audio_path = self.fl[index]
        track_id = os.path.basename(audio_path).split(".")[0]
        try:
            audio_tensor = self.load_wav(audio_path)
            return track_id, audio_tensor
        except ValueError as e:
            print(e)
            return self.__getitem__((index + 1) % len(self.fl))

    def __len__(self):
        return len(self.fl)

def main(args):
    save_dir = f"exp/ttmrpp/meta_tag_caption_sim"
    model, sr, duration = load_ttmr_pp(save_dir)
    model = model.to(args.device)
    model.eval()

    dataset = AT_DATASET(data_dir=args.data_dir)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False,
        num_workers=8, pin_memory=True, drop_last=False
    )
    track_embs = {}
    for i in tqdm(dataloader):
        fname, audio_tensor = i
        B, C, T = audio_tensor.size()
        batch_audio = audio_tensor.view(-1, T)
        with torch.no_grad():
            audio_embs = model.audio_forward(batch_audio.to(args.device))
        unbatch_audio = audio_embs.view(B, C, -1)
        audio_embs = unbatch_audio.mean(1, False).detach().cpu()
        for name, embs in zip(fname, audio_embs):
            track_embs[name] = embs.numpy()
    os.makedirs(os.path.join(save_dir, "embs", "AT"), exist_ok=True)
    torch.save(track_embs, os.path.join(save_dir, "embs", "AT", "track_embs_tr.pt"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data_dir', default='/home/habang/webdemo/AudioFiles', type=str)
    parser.add_argument('--device', default=0, type=str)
    args = parser.parse_args()
    main(args)
