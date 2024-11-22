import numpy as np
import random
import librosa

class AudioAugmentation:
    def __init__(self, sample_rate=4000):
        self.sample_rate = sample_rate
        
    def time_shift(self, audio, shift_limit=0.1):
        shift = int(random.random() * shift_limit * len(audio))
        return np.roll(audio, shift)
        
    def add_noise(self, audio, noise_factor=0.005):
        noise = np.random.randn(len(audio))
        return audio + noise_factor * noise
        
    def change_pitch(self, audio, pitch_factor=0.7):
        return librosa.effects.pitch_shift(audio, sr=self.sample_rate, n_steps=pitch_factor)
        
    def time_stretch(self, audio, rate=1.2):
        return librosa.effects.time_stretch(audio, rate=rate)