import crepe
import numpy as np
import librosa
import torch
import torchcrepe
import torchaudio

class Pitch_Extractor(torch.nn.Module):
    def __init__(self, sample_rate, block_size, threshold=0.25):
        super().__init__()
        self.crepe = crepe
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.threshold = threshold

    def forward(self, sig):
        with torch.no_grad():
            length = sig.shape[-1] // self.block_size

            _, f0, confidence, _ = crepe.predict(
                                sig.numpy(),
                                self.sample_rate,
                                step_size=int(1000 * self.block_size / self.sample_rate),
                                verbose=1,
                                center=True,
                                viterbi=True,
                                )

            f0[confidence < self.threshold] = 0.0
            if f0.shape != length:
                f0 = np.interp(np.linspace(0, 1, length, endpoint=False), np.linspace(0, 1, f0.shape[-1], endpoint=False), f0)

            f0 = torch.tensor(f0[:-1]).to(sig)

        return f0

class Loudness_Extractor(torch.nn.Module):
    def __init__(self, n_fft, sample_rate, block_size, device='cpu'):
        super().__init__()
        self.n_fft = n_fft
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.device = device

    def forward(self, sig):
        S = librosa.stft(
                    sig.numpy(),
                    n_fft=self.n_fft,
                    hop_length=self.block_size,
                    win_length=self.n_fft,
                    center=True
                    )
                     
        S = np.log(abs(S) + 1e-7)
        f = librosa.fft_frequencies(self.sample_rate, self.n_fft)
        a_weight = librosa.A_weighting(f)
        S = torch.tensor(S + a_weight.reshape(-1, 1), device=self.device)
        S = torch.mean(S, 1)[..., :-1]

        return S

# ----------------------------------------------------------------------------------------------

#Only support sample rate of 16000Hz and fmax at 2006Hz (therefore better with speech)
class Torch_Pitch_Extractor(torch.nn.Module):
    def __init__(self, sample_rate, block_size, device, win_length=3):
        super().__init__()
        self.crepe = crepe
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.hop_length = int(self.sample_rate / 200.)
        self.win_length = win_length
        self.device = device

    def forward(self, sig):
        batch_size = sig.shape[0]
        length = sig.shape[-1] // self.block_size

        with torch.no_grad():
            crepe_out = torchcrepe.predict(audio=sig,
                                        sample_rate=self.sample_rate,
                                        fmin=50,
                                        fmax=550,
                                        model='tiny',
                                        batch_size=batch_size,
                                        return_periodicity=True,
                                        device=self.device)

            pitch = crepe_out[0]
            confidence = crepe_out[0]

            periodicity = torchcrepe.filter.median(confidence, self.win_length)
            pitch = torchcrepe.threshold.At(.21)(pitch, periodicity)
            pitch = torchcrepe.filter.mean(pitch, self.win_length)

        return pitch[:, :length].to(self.device)

class Torch_Loudness_Extractor(torch.nn.Module):
    def __init__(self, n_fft, sample_rate, block_size, device):
        super().__init__()
        self.n_fft = n_fft
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.attenuate_gain = 2
        self.device = device

        self.smoothing_window = torch.hann_window(self.n_fft, dtype = torch.float32, device=self.device)

    def torch_A_weighting(self, frequencies, min_db = -45.0):
        
        # Calculate A-weighting in Decibel scale.
        freq_squared = frequencies ** 2 
        const = torch.tensor([12200, 20.6, 107.7, 737.9]) ** 2.0
        weights_in_db = 2.0 + 20.0 * (torch.log10(const[0]) + 4 * torch.log10(frequencies)
                               - torch.log10(freq_squared + const[0])
                               - torch.log10(freq_squared + const[1])
                               - 0.5 * torch.log10(freq_squared + const[2])
                               - 0.5 * torch.log10(freq_squared + const[3]))
        
        # Set minimum Decibel weight.
        if min_db is not None:
            weights_in_db = torch.max(weights_in_db, torch.tensor([min_db], dtype = torch.float32, device=self.device).to(self.device))
        
        # Transform Decibel scale weight to amplitude scale weight.
        weights = torch.exp(torch.log(torch.tensor([10.], dtype = torch.float32).to(self.device)) * weights_in_db / 10) 
        
        return weights

    def forward(self, sig):

        S = torch.stft(sig,
                    n_fft=self.n_fft,
                    hop_length=self.block_size,
                    win_length=self.n_fft,
                    window=torch.hann_window(2048, device=self.device),
                    pad_mode='constant',
                    center=True,
                    return_complex=True
                    ).to(self.device)
                     
        S = torch.log(abs(S) + 1e-7)
        freqs = torch.from_numpy(librosa.fft_frequencies(self.sample_rate, self.n_fft)).to(self.device)
        a_weight = self.torch_A_weighting(freqs).to(self.device)
        S = S + a_weight.reshape(1, -1, 1)
        S = torch.mean(S, 1, dtype=torch.float32)[..., :-1]

        return S

class Torch_MFCC_Extractor(torch.nn.Module):
    def __init__(self, n_fft, sample_rate, block_size, device):
        super().__init__()
        self.n_fft = n_fft
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.device = device

        self.mfcc = torchaudio.transforms.MFCC(sample_rate=sample_rate,
                                            n_mfcc=30,
                                            log_mels=True,
                                            melkwargs={"n_fft": self.n_fft, "hop_length": int(self.n_fft * (1 - 0.75)), 
                                            "center": True, "n_mels": 128, "f_min": 20.0, "f_max": 8000.0,}
                                            ).to(self.device)

    def forward(self, sig):
        length = sig.shape[-1] // int(self.n_fft * (1 - 0.75))
        mfccs = self.mfcc(sig)
        return mfccs[:, :, :length]