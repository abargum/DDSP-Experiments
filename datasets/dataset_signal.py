import pathlib
import librosa as li
import numpy as np
from tqdm import tqdm
import numpy as np
import torch

def get_file_list(data_location, extension, **kwargs):
    return list(pathlib.Path(data_location).rglob(f"*.{extension}"))

def preprocess(f, sample_rate, block_size, signal_length, oneshot, **kwargs):
    x, sr = li.load(f, sample_rate, mono=True)
    N = (signal_length - len(x) % signal_length) % signal_length
    x = np.pad(x, (0, N))

    if oneshot:
        x = x[..., :signal_length]

    x = x.reshape(-1, signal_length)

    return x

def get_files(config):

    files = get_file_list(**config["data"])
    pb = tqdm(files)

    signals = []

    for f in pb:
        pb.set_description(str(f))
        x = preprocess(f, **config["preprocess"])
        signals.append(x)

    signals = np.concatenate(signals, 0).astype(np.float32)

    return signals

class Dataset(torch.utils.data.Dataset):
    def __init__(self, config_file):
        super().__init__()
        self.signals = get_files(config_file)

    def __len__(self):
        return self.signals.shape[0]

    def __getitem__(self, idx):
        s = torch.from_numpy(self.signals[idx])
        return s
