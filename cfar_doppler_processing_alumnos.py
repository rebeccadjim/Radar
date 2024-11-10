# -*- coding: utf-8 -*-
"""

@author: fdadam
"""

#%% Libs and functions
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.fft import fft, ifft, fftshift, fftfreq
import sys
from mpl_toolkits.mplot3d import Axes3D
sys.stdout.reconfigure(encoding='utf-8')

def fastconv(A,B):
    out_len = len(A)+len(B)-1
    
    # Next nearest power of 2
    sizefft = int(2**(np.ceil(np.log2(out_len))))
    
    Afilled = np.concatenate((A,np.zeros(sizefft-len(A))))
    Bfilled = np.concatenate((B,np.zeros(sizefft-len(B))))
    
    fftA = fft(Afilled)
    fftB = fft(Bfilled)
    
    fft_out = fftA * fftB
    out = ifft(fft_out)
    
    out = out[0:out_len]
    
    return out

#%% Parameters

c = 3e8 # speed of light [m/s]
k = 1.380649e-23 # Boltzmann

fc = 1.3e9 # Carrier freq
fs = 10e6 # Sampling freq
Np = 100 # Intervalos de sampling
Nint = 10
NPRIs = Nint*Np
ts = 1/fs

Te = 5e-6 # Tx recovery Time[s]
Tp = 10e-6 # Tx Pulse Width [s]
BW = 2e6 # Tx Chirp bandwidth [Hz]
PRF = 1500 # Pulse repetition Frequency [Hz]

wlen = c/fc # Wavelength [m]
kwave = 2*np.pi/wlen # Wavenumber [rad/m]
PRI = PRF**(-1) # Pulse repetition interval [s]
ru = (c*(PRI-Tp-Te))/2 # Unambigous Range [m]
vu_ms = wlen*PRF/2 # Unambigous Velocity [m/s]
vu_kmh = vu_ms*3.6 # Unambigous Velocity [km/h]

rank_min = (Tp/2+Te)*c/2 # Minimum Range [m]
rank_max = 30e3 # Maximum Range [m] (podría ser el Ru)
#rank_max = ru
rank_res = ts*c/2 # Range Step [m]
tmax = 2*rank_max/c # Maximum Simulation Time

gain = 4.0 # Gain for the CFAR window

radar_signal = pd.read_csv('signal_2.csv',index_col=None)
radar_signal = np.array(radar_signal['real']+1j*radar_signal['imag'])
radar_signal = radar_signal.reshape(Np,-1)

print(f'Pulse repetition Interval. PRI = {PRI*1e6:.2f} μs')
print(f'Unambiguous Range. Ru = {ru/1e3:.3f} km')
print(f'Unambiguous Velocity. Vu = {vu_ms:.2f} m/s')
print(f'Unambiguous Velocity. Vu = {vu_kmh:.2f} km/h')
print(f'Minimum Range. Rmin = {rank_min/1e3:.3f} km')
print(f'Maximum Range. Rmin = {rank_max/1e3:.3f} km')

#%% Signals

# Independant Variables

Npts = int(tmax/ts) # Simulation Points
t = np.linspace(-tmax/2,tmax/2,Npts)
ranks = np.linspace(rank_res,rank_max,Npts) # Range Vector
f = fftfreq(Npts,ts) # Freq Vector

# Tx Signal

tx_chirp = np.exp(1j*np.pi*BW/Tp * t**2) # Tx Linear Chiprs (t)
tx_rect = np.where(np.abs(t)<=Tp/2,1,0) # Rect Function
tx_chirp = tx_rect*tx_chirp # Tx Chirp Rectangular
tx_chirp_f = fft(tx_chirp,norm='ortho') # Tx Chirp (f)

# Matched Filter

matched_filter = np.conj(np.flip(tx_chirp))
matched_filter_f = fft(matched_filter,norm='ortho')

# Compress received signal
compressed_signal = np.zeros_like(radar_signal)
for row in range(Np):
    compressed_signal[row] = np.convolve(matched_filter, radar_signal[row], mode="same")

# Compressed signal through IFFT

print(f'Received Signal Shape: {radar_signal.shape}')
print(f'Compressed Signal Shape: {compressed_signal.shape}')

# %%
# Create CFAR Window 
number_of_zeros = 15
number_of_ones = 100
win = np.concatenate((np.ones(number_of_ones),np.zeros(number_of_zeros),np.ones(number_of_ones)))
win = win/(number_of_ones*2)

############################################ Compute canceladores ############################################

# MTI simple cancelador
compressed_signal_diff = np.abs(compressed_signal[1]- compressed_signal[0])
compressed_signal_window = gain*np.convolve(compressed_signal_diff,win,mode='same')
print(f'Compressed_signal_diff Shape: {compressed_signal_diff.shape}')

# MTI simple cancelador final result
final_signal = np.sign(np.abs(compressed_signal_diff)-np.abs(compressed_signal_window))
print(f'Final_signal Shape: {final_signal.shape}')
difference = np.concatenate((np.zeros(1),np.diff(final_signal,1)))
difference[np.where(difference<0)] = 0
print(f'Difference Shape: {difference.shape}')

# STI simple cancelador
compressed_signal_calc1 = compressed_signal[10]
compressed_signal_calc2 = compressed_signal[0]
compressed_signal_sum = np.abs(compressed_signal_calc1 + compressed_signal_calc2)
compressed_signal_sum_window = gain*np.convolve(compressed_signal_sum,win,mode='same')
print(f'Compressed_signal_sum Shape: {compressed_signal_sum.shape}')

# STI simple cancelador final result
final_signal_sum = np.sign(np.abs(compressed_signal_sum)-np.abs(compressed_signal_sum_window))
print(f'Final_signal Shape: {final_signal_sum.shape}')
sum = np.concatenate((np.zeros(1),np.diff(final_signal_sum,1)))
sum[np.where(sum<0)] = 0
print(f'Sum Shape: {sum.shape}')

# MTI doble cancelador
compressed_signal_double = np.abs(compressed_signal[0] - 2*compressed_signal[1] + compressed_signal[2])
compressed_signal_double_window = gain*np.convolve(compressed_signal_double,win,mode='same')

# MTI doble cancelador final result
final_signal_double = np.sign(np.abs(compressed_signal_double)-np.abs(compressed_signal_double_window))
print(f'Final_signal Shape: {final_signal_double.shape}')
double = np.concatenate((np.zeros(1),np.diff(final_signal_double,1)))
double[np.where(double<0)] = 0
print(f'Double Shape: {double.shape}')

#%% Doppler
############################################ Doppler processing ############################################
# Initialize the matrix with complex zeros
square_matrix = np.zeros((Np, Np), dtype=complex)

cols = np.arange(1,Np+1)
rows = np.arange(1,Np+1).reshape(Np,-1)
# Fill in the matrix with the complex exponential expression
square_matrix = np.exp(-1j * 2 * np.pi * (cols * rows) / Np)

# Compute MTI Simple Cancelador for all the compressed signals
MTI_simple = np.zeros_like(radar_signal)
for i in range(1, Np):  # Start from 1 to avoid index -1
    MTI_simple[i] = (compressed_signal[i] - compressed_signal[i - 1])

print(f'MTI simple Shape: {MTI_simple.shape}')
 
# Transpose MTI Simple Cancelador 
MTI_simple_T = MTI_simple.T
print(f'Compressed_signal_transpose Shape: {MTI_simple_T.shape}')

# Apply the matrix to the transposed compressed signal
doppler_signal = np.matmul(MTI_simple_T,square_matrix).T
#print(doppler_signal[:, 531])

#%% Plot Signals
############################################ Plots ############################################
# Compressed signal plots

# fig, axes = plt.subplots(2,1,figsize=(8,8),sharex=True)

# ax = axes[0]
# ax.plot(ranks/1e3,np.abs(radar_signal[0]))
# ax.set_ylabel('Abs value')
# ax.set_xlabel('Rx raw signal')
# ax.grid(True)

# ax = axes[1]
# ax.plot(ranks/1e3,np.abs(compressed_signal[0]))
# ax.set_ylabel('Abs value')
# ax.set_xlabel('Rx compressed signal')


# # Draw window with dots
# fig, axes = plt.subplots()
# axes.plot(win, marker='o')
# axes.set_title('CFAR window')
# axes.set_xlabel('Samples')
# axes.set_ylabel('Amplitude')
# axes.grid(True)
# # add legend
# axes.text(20, 0.001, "Reference Cells: 200\nGap Cells: 15",
#           bbox=dict(facecolor='lightyellow', edgecolor='black'))

# fig, axes = plt.subplots(4, 1, figsize=(8, 8), sharex=True)

# axes[0].plot(ranks / 1e3, np.abs(radar_signal[0]))
# axes[0].plot(ranks / 1e3, np.abs(radar_signal[1]))
# axes[0].set_ylabel('Value')
# axes[0].set_xlabel('Rx raw signals')
# axes[0].legend(['Rx t_0', 'Rx t_1'])
# axes[1].plot(ranks / 1e3, np.abs(compressed_signal[0][:2000]))
# axes[1].plot(ranks / 1e3, np.abs(compressed_signal[1][:2000]))
# axes[1].set_ylabel('Value')
# axes[1].set_xlabel('Rx compressed signals')
# axes[1].legend(['Comp t_0', 'Comp t_1'])
# axes[2].plot(ranks / 1e3, np.abs(compressed_signal_double))
# axes[2].plot(ranks / 1e3, np.abs(compressed_signal_double_window))
# axes[2].set_ylabel('Value')
# axes[2].set_xlabel('MTI DC')
# axes[2].legend(['MTI Signal', 'MTI Threshold'])
# axes[3].plot(ranks / 1e3, np.abs(double))
# axes[3].set_ylabel('Value')
# axes[3].set_xlabel('MTI DC signal')
# axes[0].grid(True)
# axes[1].grid(True)
# axes[2].grid(True)
# axes[3].grid(True)

print(f'Rangos de los blancos detectados por MTI DC (m): {ranks[np.where((np.abs(double))>0)]}')
rangos = np.where((np.abs(double))>0)
#print(f'indices de los blancos detectados por MTI DC (m): {rangos}')
vel_vect = np.linspace(-vu_ms / 2, vu_ms / 2, Np, endpoint=True)
velocities = [vel_vect[np.argmax(np.abs(doppler_signal[:, r]))] for r in rangos[0]]
print(f'Velocidades: {velocities}')
#print(ranks[np.where(np.abs(double)>0)])

# fig, axes = plt.subplots(4, 1, figsize=(8, 8), sharex=True)

# axes[0].plot(ranks / 1e3, np.abs(radar_signal[0]))
# axes[0].plot(ranks / 1e3, np.abs(radar_signal[1]))
# axes[0].set_ylabel('Value')
# axes[0].set_xlabel('Rx raw signals')
# axes[0].legend(['Rx t_0', 'Rx t_1'])
# axes[1].plot(ranks / 1e3, np.abs(compressed_signal[0][:2000]))
# axes[1].plot(ranks / 1e3, np.abs(compressed_signal[1][:2000]))
# axes[1].set_ylabel('Value')
# axes[1].set_xlabel('Rx compressed signals')
# axes[1].legend(['Comp t_0', 'Comp t_1'])
# axes[2].plot(ranks / 1e3, np.abs(compressed_signal_diff))
# axes[2].plot(ranks / 1e3, np.abs(compressed_signal_window))
# axes[2].set_ylabel('Value')
# axes[2].set_xlabel('MTI SC')
# axes[2].legend(['MTI Signal', 'MTI Threshold'])
# axes[3].plot(ranks / 1e3, np.abs(difference))
# axes[3].set_ylabel('Value')
# axes[3].set_xlabel('MTI SC signal')
# axes[0].grid(True)
# axes[1].grid(True)
# axes[2].grid(True)
# axes[3].grid(True)

print(f'Rangos de los blancos detectados por MTI SC (m): {ranks[np.where((np.abs(difference))>0)]}')
rangos = np.where((np.abs(difference))>0)
#print(rangos)
vel_vect = np.linspace(-vu_ms / 2, vu_ms / 2, Np, endpoint=True)
velocities = [vel_vect[np.argmax(np.abs(doppler_signal[:, r]))] for r in rangos[0]]
print(f'Velocidades: {velocities}')

# fig, axes = plt.subplots(4, 1, figsize=(8, 8), sharex=True)

# axes[0].plot(ranks / 1e3, np.abs(radar_signal[10]))
# axes[0].plot(ranks / 1e3, np.abs(radar_signal[0]))
# axes[0].set_ylabel('Value')
# axes[0].set_xlabel('Rx raw signals')
# axes[0].legend(['Rx t_10', 'Rx t_0'])
# axes[1].plot(ranks / 1e3, np.abs(compressed_signal_calc1[:2000]))
# axes[1].plot(ranks / 1e3, np.abs(compressed_signal_calc2[:2000]))
# axes[1].set_ylabel('Value')
# axes[1].set_xlabel('Rx compressed signals')
# axes[1].legend(['Comp t_10', 'Comp t_0'])
# axes[2].plot(ranks / 1e3, np.abs(compressed_signal_sum))
# axes[2].plot(ranks / 1e3, np.abs(compressed_signal_sum_window))
# axes[2].set_ylabel('Value')
# axes[2].set_xlabel('STI SC')
# axes[2].legend(['STI Signal', 'STI Threshold'])
# axes[3].plot(ranks / 1e3, np.abs(sum))
# axes[3].set_ylabel('Value')
# axes[3].set_xlabel('STI SC signal')
# axes[0].grid(True)
# axes[1].grid(True)
# axes[2].grid(True)
# axes[3].grid(True)

print(f'Rangos de los blancos detectados por STI SC (m): {ranks[np.where((np.abs(sum))>0)]}')
rangos = np.where((np.abs(sum))>0)
#print(f'indices de los blancos detectados por STI SC (m): {rangos}')
vel_vect = np.linspace(-vu_ms / 2, vu_ms / 2, Np, endpoint=True)

# velocities = []

# for r in rangos:
#     max_index = np.argmax(doppler_signal[:, r])  # Trouver l'indice maximum pour chaque colonne spécifiée par rangos
#     velocities.append(vel_vect[max_index])       # Extraire la valeur correspondante de vel_vect

# # Afficher toutes les valeurs extraites
# print(velocities)

velocities = [vel_vect[np.argmax(np.abs(doppler_signal[:, r]))] for r in rangos[0]]
print(f'Velocidades: {velocities}')

# Doppler 3D Plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X, Y = np.meshgrid(-vel_vect, ranks)
Z = np.abs(fftshift(doppler_signal, axes=0)).T
ax.plot_surface(X, Y, Z, cmap='coolwarm')
# Add title and labels
ax.set_title('Doppler processing')
ax.set_xlabel('Velocity [m/s]')
ax.set_ylabel('Range [m]')
# Add a color bar which maps values to colors.
cbar = fig.colorbar(ax.plot_surface(X, Y, Z, cmap='coolwarm'))

plt.show()