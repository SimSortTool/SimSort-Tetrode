import numpy as np
import neuron
import random
import copy

import numpy as np
import random
import copy

import numpy as np
import matplotlib.pyplot as plt

class Ornstein_Uhlenbeck_Noise:
    def __init__(self, mu=np.zeros(1), sigma=0.25, theta=0.05, dt=1e-1, x0=None, shock_prob=0.01, shock_mag=2.0):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.shock_prob = shock_prob 
        self.shock_mag = shock_mag   
        self.reset()

    def __call__(self):
        x = self.x_prev + \
            self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)

        if np.random.rand() < self.shock_prob:
            x += np.random.normal(scale=self.shock_mag) 
        self.x_prev = x
        return x

    def reset(self):
        if self.x0 is not None:
            self.x_prev = self.x0
        else:
            self.x_prev = np.zeros_like(self.mu)

def generate_OU_noise(N=10000, mu=np.zeros(1), sigma=0.25, theta=0.02, dt=0.1, amplitude=1.0, shock_prob=0.01, shock_mag=2.0):
    ou_noise = Ornstein_Uhlenbeck_Noise(mu=mu, sigma=sigma, theta=theta, dt=dt, shock_prob=shock_prob, shock_mag=shock_mag)
    samples = np.zeros(N)
    for i in range(N):
        samples[i] = ou_noise()
    return samples * amplitude

def generate_pink_noise(N, amplitude=1.0):
    white_noise = np.random.normal(0, 1, N)
    fourier_transform = np.fft.rfft(white_noise)
    frequencies = np.fft.rfftfreq(N)
    scaling_factor = np.where(frequencies == 0, 0, 1 / np.sqrt(frequencies))  
    pink_fourier_transform = fourier_transform * scaling_factor
    pink_noise = np.fft.irfft(pink_fourier_transform, n=N)
    # Normalize the noise to the desired amplitude
    pink_noise = amplitude * pink_noise / np.std(pink_noise)
    
    return pink_noise

def set_input(cell, amplitude=None, **kwargs):
    if amplitude is None:
        amplitude = kwargs['weight']
    else:
        amplitude = amplitude

    dt = kwargs['dt']
    T = kwargs['T']
    delay = kwargs['delay']
    stim_length = T - delay
    tot_ntsteps = int(round(T / dt + 1))
    I = np.ones(tot_ntsteps) * amplitude # current in nA
    noiseVec = neuron.h.Vector(I)
    syn = None
    for sec in cell.allseclist:
        if "soma" in sec.name():
            # mid_point of each section
            syn = neuron.h.IClamp(0.5, sec=sec)
    syn.dur = stim_length
    syn.delay = delay
    noiseVec.play(syn._ref_amp, dt)

    return noiseVec, cell, syn

def set_noise_input(cell, noise_type='pink_noise', 
                    rheobase_current=None, amplitude=1.0, dt=0.1, end_T=1000, delay=10, **kwargs):
    T = end_T
    stim_length = T - delay
    tot_ntsteps = int(round(stim_length / dt + 1))
    if noise_type == 'pink_noise':
        if rheobase_current is not None:
            noise = generate_pink_noise(N=tot_ntsteps, amplitude=rheobase_current)
                                       
        else:
            noise = generate_pink_noise(N=tot_ntsteps, amplitude=amplitude)
                                        
    elif noise_type == 'white_noise':
        noise = np.random.normal(0, 1, tot_ntsteps)

    elif noise_type == 'OU_noise':
        if rheobase_current is not None:
            noise = generate_OU_noise(N=tot_ntsteps, dt=0.1, amplitude=rheobase_current)
        else:
            noise = generate_OU_noise(N=tot_ntsteps, dt=0.1, amplitude=amplitude)

    syn = None
    for sec in cell.allseclist:
        if "soma" in sec.name():
            syn = neuron.h.IClamp(0.5, sec=sec)  # Stimulation electrode  
    syn.dur = stim_length
    syn.delay = delay
    noiseVec = neuron.h.Vector(noise)
    noiseVec.play(syn._ref_amp, dt)

    return noiseVec, cell, syn, noise

def get_rheobase_current(cell, min_current, max_current, current_step):
    rheobase_current = None
    cell.tstop = 1000
    dt = 0.1
    T = 1000
    delay = 0
    for current in np.arange(min_current, max_current, current_step):  
        noiseVec, cell, syn = set_input(cell, amplitude=current, dt=dt, T=T, delay=delay)
        cell.simulate()
        if np.max(cell.somav) > 0:
            rheobase_current = current
            break
    return rheobase_current

def generate_multi_sta_stimulation(sta_waveform, spike_times, **kwargs):
    dt = kwargs['dt']
    delay = kwargs['delay']
    stim_length = kwargs['T'] - delay
    input_current = np.zeros(int(stim_length / dt))  
    for spike_time in spike_times:  
        start_index = int(spike_time / dt)  
        end_index = start_index + len(sta_waveform)  
        input_current[start_index:end_index] += sta_waveform  
    return input_current

def apply_sta_stimulation(cell, current, dt, delay):
    for sec in cell.allseclist:
        if "soma" in sec.name():
            syn = neuron.h.IClamp(0.5, sec=sec)
    syn.dur = len(current) * dt
    syn.delay = delay
    current_vec = neuron.h.Vector(current)
    current_vec.play(syn._ref_amp, dt)
    return current_vec, cell, syn



