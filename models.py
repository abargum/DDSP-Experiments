
import torch
import torch.nn as nn

from core import harmonic_synth, amp_to_impulse_response, fft_convolve
from core import mlp, gru, scale_function, remove_above_nyquist, upsample
from encoders import Torch_Pitch_Extractor, Torch_Loudness_Extractor, Torch_MFCC_Extractor

class Reverb(nn.Module):
    def __init__(self, length, sample_rate, initial_wet=0, initial_decay=5):
        super().__init__()
        self.length = length
        self.sample_rate = sample_rate

        self.noise = nn.Parameter((torch.rand(length) * 2 - 1).unsqueeze(-1))
        self.decay = nn.Parameter(torch.tensor(float(initial_decay)))
        self.wet = nn.Parameter(torch.tensor(float(initial_wet)))

        t = torch.arange(self.length) / self.sample_rate
        t = t.reshape(1, -1, 1)
        self.register_buffer("t", t)

    def build_impulse(self):
        t = torch.exp(-nn.functional.softplus(-self.decay) * self.t * 500)
        noise = self.noise * t
        impulse = noise * torch.sigmoid(self.wet)
        impulse[:, 0] = 1
        return impulse

    def forward(self, x):
        lenx = x.shape[1]
        impulse = self.build_impulse()
        impulse = nn.functional.pad(impulse, (0, 0, 0, lenx - self.length))

        x = fft_convolve(x.squeeze(-1), impulse.squeeze(-1)).unsqueeze(-1)

        return x

class Encoder(nn.Module):
    def __init__(self, sample_rate, block_size, n_fft, device):
        super().__init__()

        self.extract_pitch = Torch_Pitch_Extractor(sample_rate=sample_rate, block_size=block_size, device=device)
        self.extract_loudness = Torch_Loudness_Extractor(n_fft=n_fft, sample_rate=sample_rate, block_size=block_size, device=device)

    @torch.no_grad()
    def mean_std_loudness(self, loudness):
        mean = 0
        std = 0
        n = 0
        for l in loudness:
            n += 1
            mean += (torch.mean(l) - mean) / n
            std += (torch.std(l) - std) / n
        return mean, std

    def forward(self, input):
        pitch = self.extract_pitch(input)
        loudness = self.extract_loudness(input)

        mean, std = self.mean_std_loudness(loudness)

        pitch = pitch.unsqueeze(-1)
        loudness = loudness.unsqueeze(-1)
        loudness = (loudness - mean) / std

        return pitch, loudness

class Decoder(nn.Module):
    def __init__(self, sample_rate, block_size, hidden_size, n_bands, n_harmonic):
        super().__init__()

        self.in_mlps = nn.ModuleList([mlp(1, hidden_size, 3)] * 2)
        self.gru = gru(2, hidden_size)
        self.out_mlp = mlp(hidden_size + 2, hidden_size, 3)

        self.output_to_harmonic = nn.Linear(hidden_size, n_harmonic + 1)
        self.output_to_noise = nn.Linear(hidden_size, n_bands)

    def forward(self, pitch, loudness):
        hidden = torch.cat([
            self.in_mlps[0](pitch),
            self.in_mlps[1](loudness),
        ], -1)
        hidden = torch.cat([self.gru(hidden)[0], pitch, loudness], -1)
        hidden = self.out_mlp(hidden)

        param_harmonic = scale_function(self.output_to_harmonic(hidden))
        param_noise = scale_function(self.output_to_noise(hidden) - 5)

        return param_harmonic, param_noise

class Latent_Z(nn.Module):
    def __init__(self, sample_rate, block_size, hidden_size, n_fft, num_mfccs, z_dim, device):
        super().__init__()
        self.z_vector = Torch_MFCC_Extractor(n_fft, sample_rate, block_size, device)
        self.norm_layer = nn.InstanceNorm1d(num_mfccs)
        self.gru = nn.GRU(num_mfccs, hidden_size, batch_first=True)
        self.dense_z = nn.Linear(hidden_size, z_dim)

    def forward(self, signal):
        mfccs = self.z_vector(signal)
        mfccs = self.norm_layer(mfccs).permute(0, 2, 1)
        gru_out = self.gru(mfccs)
        latent_z = self.dense_z(gru_out[0])
        return latent_z

class Decoder_with_Z(nn.Module):
    def __init__(self, hidden_size, n_bands, n_harmonic, z_dim):
        super().__init__()

        self.in_mlps = nn.ModuleList([mlp(1, hidden_size, 3)] * 2)
        self.in_mlp_z = nn.Linear(z_dim, hidden_size)
        self.gru = gru(3, hidden_size)
        self.out_mlp = mlp(hidden_size + 2, hidden_size, 3)

        self.output_to_harmonic = nn.Linear(hidden_size, n_harmonic + 1)
        self.output_to_noise = nn.Linear(hidden_size, n_bands)

    def forward(self, pitch, loudness, latent_z):

        hidden = torch.cat([
            self.in_mlps[0](pitch),
            self.in_mlps[1](loudness),
            self.in_mlp_z(latent_z),
        ], -1)

        hidden = torch.cat([self.gru(hidden)[0], pitch, loudness], -1)
        hidden = self.out_mlp(hidden)

        param_harmonic = scale_function(self.output_to_harmonic(hidden))
        param_noise = scale_function(self.output_to_noise(hidden) - 5)

        return param_harmonic, param_noise

class DDSP_signal_only(nn.Module):
    def __init__(self, hidden_size, n_harmonic, n_bands, sample_rate,
                 block_size, n_fft, device):
        super().__init__()
        self.register_buffer("sample_rate", torch.tensor(sample_rate))
        self.register_buffer("block_size", torch.tensor(block_size))

        self.encoder = Encoder(sample_rate, block_size, n_fft, device)
        self.decoder = Decoder(sample_rate, block_size, hidden_size, n_bands, n_harmonic)

        self.reverb = Reverb(sample_rate, sample_rate)

        self.register_buffer("cache_gru", torch.zeros(1, 1, hidden_size))
        self.register_buffer("phase", torch.zeros(1))

    def forward(self, input):

        pitch, loudness = self.encoder(input)
        param_harmonic, param_noise = self.decoder(pitch, loudness)

        total_amp = param_harmonic[..., :1]
        amplitudes = param_harmonic[..., 1:]

        amplitudes = remove_above_nyquist(
            amplitudes,
            pitch,
            self.sample_rate
        )
        amplitudes /= amplitudes.sum(-1, keepdim=True)
        amplitudes *= total_amp

        amplitudes = upsample(amplitudes, self.block_size)
        pitch = upsample(pitch, self.block_size)

        harmonic = harmonic_synth(pitch, amplitudes, self.sample_rate)

        impulse = amp_to_impulse_response(param_noise, self.block_size)
        noise = torch.rand(
            impulse.shape[0],
            impulse.shape[1],
            self.block_size,
        ).to(impulse) * 2 - 1

        noise = fft_convolve(noise, impulse).contiguous()
        noise = noise.reshape(noise.shape[0], -1, 1)

        signal = harmonic + noise

        #reverb part
        signal = self.reverb(signal)

        return signal

class DDSP_with_features(nn.Module):
    def __init__(self, hidden_size, n_harmonic, n_bands, sample_rate,
                 block_size, n_fft, num_mfccs, z_dim, device):
        super().__init__()
        self.register_buffer("sample_rate", torch.tensor(sample_rate))
        self.register_buffer("block_size", torch.tensor(block_size))

        self.latent_z = Latent_Z(sample_rate, block_size, hidden_size, n_fft, num_mfccs, z_dim, device)
        self.decoder = Decoder_with_Z(hidden_size, n_bands, n_harmonic, z_dim)

        self.reverb = Reverb(sample_rate, sample_rate)

        self.register_buffer("cache_gru", torch.zeros(1, 1, hidden_size))
        self.register_buffer("phase", torch.zeros(1))

    def forward(self, signal, pitch, loudness):
        
        latent_z = self.latent_z(signal)
        param_harmonic, param_noise = self.decoder(pitch, loudness, latent_z)

        total_amp = param_harmonic[..., :1]
        amplitudes = param_harmonic[..., 1:]

        amplitudes = remove_above_nyquist(
            amplitudes,
            pitch,
            self.sample_rate
        )
        amplitudes /= amplitudes.sum(-1, keepdim=True)
        amplitudes *= total_amp

        amplitudes = upsample(amplitudes, self.block_size)
        pitch = upsample(pitch, self.block_size)

        harmonic = harmonic_synth(pitch, amplitudes, self.sample_rate)

        impulse = amp_to_impulse_response(param_noise, self.block_size)
        noise = torch.rand(
            impulse.shape[0],
            impulse.shape[1],
            self.block_size,
        ).to(impulse) * 2 - 1

        noise = fft_convolve(noise, impulse).contiguous()
        noise = noise.reshape(noise.shape[0], -1, 1)

        signal = harmonic + noise

        #reverb part
        signal = self.reverb(signal)

        return signal