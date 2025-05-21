import numpy as np
import random
from tqdm import tqdm
import colorednoise as cn
import scipy
from scipy import interpolate
class RandomChannelAugmentation:
    def __init__(self, transforms, max_channels=None, probabilities=[0.5, 0.5, 0.5, 0.5]):
        """
        Apply a list of transforms to a random subset of channels with a given probability.

        Args:
            transforms (list): List of transforms to apply to the selected channels.
            max_channels (int, optional): Maximum number of channels to apply the transforms. If None, uses all channels.
            probabilities (list, optional): List of probabilities for each transform to be applied.
        """
        assert transforms is None or probabilities is None or len(transforms) == len(probabilities), \
            "Transforms and probabilities must have the same length."
        
        self.transforms = transforms
        self.max_channels = max_channels
        self.probabilities = probabilities

    def __call__(self, waveform):
        channels = waveform.shape[1]
        if self.max_channels is None:
            self.max_channels = channels

        num_channels_to_augment = random.randint(1, min(self.max_channels, channels))

        selected_channels = random.sample(range(channels), num_channels_to_augment)
        
        for channel in selected_channels:
            random.seed()
            for transform, prob in zip(self.transforms, self.probabilities):
                if random.random() < prob:
                    waveform[:, channel] = transform(waveform[:, channel])  
        
        return waveform

class AddWithNoise:
    def __init__(self, noise_level=0.1):
        self.noise_level = noise_level

    def __call__(self, signal):
        #noise = cn.powerlaw_psd_gaussian(1,signal.shape[0])*self.noise_level
        noise = np.random.randn(*signal.shape).astype(np.float32)*self.noise_level
        noised_signal = noise + signal

        return noised_signal

class RandomAmplitudeScaling:
    def __init__(self, scale_range=[0.5, 1.5]):
        self.scale_range = scale_range

    def __call__(self, signal):
        scale_factor = random.uniform(self.scale_range[0], self.scale_range[1])

        return signal * scale_factor

class RandomTimeShift:
    '''Temporal translation for waveform dataset augmentation.'''

    def __init__(self, window_size=35, shift_range=5):
        self.window_size = window_size
        self.shift_range = shift_range
    
    def __call__(self, signal):
        min_idx = np.argmin(signal)
        window_start = max(0, min_idx - self.window_size // 2)
        window_end = min(len(signal), min_idx + self.window_size // 2 + 1)
        window = signal[window_start:window_end]
        
        shift_amount = np.random.randint(-self.shift_range, self.shift_range + 1)
        
        shifted_signal = np.copy(signal)
        
        transition_length = min(5, (window_end - window_start) // 2)
        if transition_length > 0:
            # smooth the edges of the window
            shifted_signal[window_start:window_start + transition_length] *= np.linspace(1, 0, transition_length)
            shifted_signal[window_end - transition_length:window_end] *= np.linspace(0, 1, transition_length)
        
        # make sure that the window doesn't go out of bounds
        new_window_start = max(0, min(len(signal) - len(window), window_start + shift_amount))
        new_window_end = new_window_start + len(window)
        
        # move the window to the new location
        shifted_signal[new_window_start:new_window_end] = window
        
        return shifted_signal
    
class RandomTimeJitter:
    '''Temporal Jitter for signal sequence dataset augmentation.'''

    def __init__(self, shift_range=5, up_factor=8):
        self.shift_range = shift_range
        self.up_factor = up_factor

    def __call__(self, signal):
        original_length = len(signal)
        upsampled_signal = scipy.signal.resample(signal, original_length * self.up_factor)
        shift_amount = np.random.randint(-self.shift_range * self.up_factor, self.shift_range * self.up_factor + 1)
        shifted_signal = np.roll(upsampled_signal, shift_amount)
        downsampled_signal = scipy.signal.resample(shifted_signal, original_length)
        return downsampled_signal
    
class RandomTimeStretch:
    def __init__(self, stretch_range=0.2):
        """
        Args:
            stretch_range: float
                The range by which to stretch or compress the entire waveform (e.g., 0.2 means up to 20% stretch/compress).
        """
        self.stretch_range = stretch_range

    def __call__(self, signal):
        original_size = len(signal)

        stretch_factor = 1 + np.random.uniform(-self.stretch_range, 0)

        x_original = np.linspace(0, original_size - 1, num=original_size)
        new_size = int(original_size * stretch_factor)
        x_stretched = np.linspace(0, original_size - 1, num=new_size)
        interpolator = interpolate.interp1d(x_original, signal, kind='linear', fill_value="extrapolate")
        stretched_waveform = interpolator(x_stretched)

        if len(stretched_waveform) > original_size:

            final_waveform = stretched_waveform[:original_size]
        else:
            final_waveform = np.zeros(original_size)
            final_waveform[:len(stretched_waveform)] = stretched_waveform
            final_waveform[len(stretched_waveform):] = stretched_waveform[-1]

        return final_waveform

def generate_augmented_data(original_data, original_labels, transform, augmentation_factor=2, label_offset=2000):

    augmented_data = []
    augmented_labels = []
    
    for i in tqdm(range(original_data.shape[0])):
        for _ in range(augmentation_factor):
            # new_sample = transform(original_data[i].copy(), seed=i)
            new_sample = transform(original_data[i].copy())
            augmented_data.append(new_sample)
            # augmented_labels.append(original_labels[i] + label_offset + i)
            augmented_labels.append(original_labels[i])
    
    return np.array(augmented_data), np.array(augmented_labels)