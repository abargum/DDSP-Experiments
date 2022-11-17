import yaml
import pathlib
import librosa as li
from effortless_config import Config
import numpy as np
from tqdm import tqdm
import numpy as np
from os import makedirs, path
import torch
import crepe
from pathlib import Path
from typing import Optional, Union
from warnings import warn
import numpy as np
from encoder import inference as encoder
from encoder import data_utilities

try:
    import webrtcvad
except:
    warn("Unable to import 'webrtcvad'. This package enables noise removal and is recommended.")
    webrtcvad=None

audio_norm_target_dBFS = -30

def extract_loudness(signal, sample_rate, block_size, n_fft=2048):
    
    S = li.stft(
        signal,
        n_fft=n_fft,
        hop_length=block_size,
        win_length=n_fft,
        center=True,
    )
    S = np.log(abs(S) + 1e-7)
    f = li.fft_frequencies(sample_rate, n_fft)
    a_weight = li.A_weighting(f)

    S = S + a_weight.reshape(-1, 1)

    S = np.mean(S, 0)[..., :-1]

    return S

def extract_pitch(signal, sample_rate, block_size):
    length = signal.shape[-1] // block_size
    f0 = crepe.predict(
        signal,
        sample_rate,
        step_size=int(1000 * block_size / sample_rate),
        verbose=1,
        center=True,
        viterbi=True,
    )
    f0 = f0[1].reshape(-1)[:-1]

    if f0.shape[-1] != length:
        f0 = np.interp(
            np.linspace(0, 1, length, endpoint=False),
            np.linspace(0, 1, f0.shape[-1], endpoint=False),
            f0,
        )

    return f0

def get_file_list(data_location, extension, **kwargs):
    return list(pathlib.Path(data_location).rglob(f"*.{extension}"))

def preprocess(f, sample_rate, block_size, signal_length, oneshot, **kwargs):
    x, _ = li.load(f, sample_rate)
    N = (signal_length - len(x) % signal_length) % signal_length
    x = np.pad(x, (0, N))

    if oneshot:
        x = x[..., :signal_length]

    pitch = extract_pitch(x, sample_rate, block_size)
    loudness = extract_loudness(x, sample_rate, block_size)
    embedding = extract_embedding(x, sample_rate)

    x = x.reshape(-1, signal_length)

    if (x.shape[0] > 1):
        embedding = np.tile(embedding, x.shape[0])

    pitch = pitch.reshape(x.shape[0], -1)
    loudness = loudness.reshape(x.shape[0], -1)
    embedding = embedding.reshape(x.shape[0], -1)

    return x, pitch, loudness, embedding

def get_files(config):
    class args(Config):
        CONFIG = config

    args.parse_args("")
    with open(args.CONFIG, "r") as config:
        config = yaml.safe_load(config)

    files = get_file_list(**config["data"])
    pb = tqdm(files)

    signals = []
    pitchs = []
    loudness = []
    embeddings = []

    for f in pb:
        pb.set_description(str(f))
        x, p, l, e = preprocess(f, **config["preprocess"])
        signals.append(x)
        pitchs.append(p)
        loudness.append(l)
        embeddings.append(e)

    signals = np.concatenate(signals, 0).astype(np.float32)
    pitchs = np.concatenate(pitchs, 0).astype(np.float32)
    loudness = np.concatenate(loudness, 0).astype(np.float32)
    embeddings = np.concatenate(embeddings, 0).astype(np.float32)

    out_dir = config["preprocess"]["out_dir"]
    makedirs(out_dir, exist_ok=True)

    np.save(path.join(out_dir, "signals.npy"), signals)
    np.save(path.join(out_dir, "pitchs.npy"), pitchs)
    np.save(path.join(out_dir, "loudness.npy"), loudness)
    np.save(path.join(out_dir, "embeddings.npy"), embeddings)

# ---------------------------------------------------------------------------------

def extract_embedding(signal, sample_rate):
    
    signal = data_utilities.normalize_volume(signal, audio_norm_target_dBFS, increase_only=True)
    if webrtcvad:
        signal = data_utilities.trim_long_silences(signal, sample_rate)
    
    encoder.load_model(Path("encoder/encoder.pt"))
    embedding = encoder.embed_utterance(signal)
    return embedding

# ---------------------------------------------------------------------------------

class Dataset(torch.utils.data.Dataset):
    def __init__(self, out_dir):
        super().__init__()
        self.signals = np.load(path.join(out_dir, "signals.npy"))
        self.pitchs = np.load(path.join(out_dir, "pitchs.npy"))
        self.loudness = np.load(path.join(out_dir, "loudness.npy"))
        self.embeddings = np.load(path.join(out_dir, "embeddings.npy"))

    def __len__(self):
        return self.signals.shape[0]

    def __getitem__(self, idx):
        return {'signals': torch.from_numpy(self.signals[idx])
               ,'pitches': torch.from_numpy(self.pitchs[idx])
               ,'loudness': torch.from_numpy(self.loudness[idx])
               ,'embeddings': torch.from_numpy(self.embeddings[idx])}