# Setting up all folders we can import from by adding them to python path
import sys, os, pdb

# Importing stuff from all folders in python path
import numpy as np
from propagate_polar import *
import scipy.io as sio
from scipy.signal import hilbert, freqz
from scipy.interpolate import griddata, interp1d, interpn
from scipy.optimize import fminbound
import matplotlib.pyplot as plt

# Load Channel Data
FocTxDataset = loadmat_hdf5('SiemensData5C1_Kidney2.mat');
t = FocTxDataset['t'][0]; # Time Axis [s]
elem_pos = FocTxDataset['elem_pos']; # Locations of Array Elements [m]
demodData = FocTxDataset['demodData']; # Actual Channel Data [Nt, NRx, NTx]
dt = np.mean(np.diff(t)); nt = t.size; # t = t(1)+(0:nt-1)*dt; 
fs = 1/dt; # Sampling Rate [Hz]
tx_apod = FocTxDataset['tx_apod']; # Transmit Apodization
tx_foci = FocTxDataset['tx_foci']; # Transmit Foci [m]
tx_orig = FocTxDataset['tx_orig']; # Transmit Origins [m]
c = FocTxDataset['c'][0][0]; # Sound Speed [m/s]
del FocTxDataset;

# Load File and Set Imaging Grid
dov = 135e-3; # Max Depth [m]
upsamp_theta = 2; # Upsampling in theta
upsamp_r = 1; # Upsampling in r
Nth0 = 128; # Number of Points Laterally in theta

# Select Subset of Transmit Elements
tx_evt = 31; rxdata_h = demodData[:,:,tx_evt]; del demodData;

# Find the Radius of the 5C1 Probe
lsfit = lambda R: np.mean((elem_pos[2,:]-np.sqrt(R**2-elem_pos[0,:]**2)+R)**2);
Rconvex = fminbound(lsfit,0.04,0.05,xtol=1e-10);

# Aperture Definition
thetapos = np.arctan2(elem_pos[0,:],elem_pos[2,:]+Rconvex)
dtheta = np.mean(np.diff(thetapos)) # angular spacing [rad]

# Transmit Pulse
fTx = 3.5e6; # Hz
lmbda = c/fTx; # m

# Simulation Space
theta = np.arange(-(upsamp_theta*Nth0-1)/2,1+(upsamp_theta*Nth0-1)/2)*(dtheta/upsamp_theta); # rad
Nu1 = np.round(dov/((lmbda/2)/upsamp_r));
r = Rconvex+(np.arange(Nu1)*(lmbda/2)/upsamp_r);

# Image Reconstruction Parameters and Anti-Aliasing Window
dBrange = np.array([-60, 0]); reg = 1e-1; ord = 50;
thetamax = (np.max(np.abs(thetapos))+np.max(np.abs(theta)))/2; # m
aawin = 1/np.sqrt(1+(theta/thetamax)**ord);

# Transmit Impulse Response in Frequency Domain
f = (fs/2)*np.arange(-1,1,2/nt); # [Hz]
P_Tx = lambda f: (np.abs(f-fTx)<(3.0e6)); # Pulse Spectrum
P_Tx_f = P_Tx(f); # Pulse Definition

# Get Receive Channel Data in the Frequency Domain
P_Rx_f = np.fft.fftshift(np.fft.fft(rxdata_h, n=nt, axis=0), axes=0);
T, F = np.meshgrid(np.arange(P_Rx_f.shape[1]), f);
P_Rx_f = P_Rx_f * np.exp(-1j*2*np.pi*F*t[0]); del T, F;
rxdata_f = interp1d(thetapos, np.transpose(P_Rx_f, (1,0)), \
    kind='nearest', axis=0, fill_value=0, bounds_error=False)(theta);
del rxdata_h, P_Rx_f; # Delete Receive Channel Data

# Only Keep Positive Frequencies within Passband
passband_f_idx = np.argwhere((P_Tx_f > reg) & (f > 0)).flatten();
rxdata_f = rxdata_f[:,passband_f_idx];
f = f[passband_f_idx]; P_Tx_f = P_Tx_f[passband_f_idx];
P_Tx_f = np.hanning(P_Tx_f.size); # Assume Flat Passband

# Construct Transmit Beamforming Delays
# transmit aperture locations
txAptPosRelToCtr = elem_pos - tx_orig[:,tx_evt][:,np.newaxis]*np.ones((1,elem_pos.shape[1]));
txFocRelToCtr = tx_foci[:,tx_evt] - tx_orig[:,tx_evt]
txFocRelToAptPos = txFocRelToCtr[:,np.newaxis]*np.ones((1,elem_pos.shape[1])) - txAptPosRelToCtr;
# positive value is time delay, negative is time advance
tx_delay = (np.sqrt(np.sum(txFocRelToCtr**2, axis=0)) - \
    np.sqrt(np.sum(txFocRelToAptPos**2, axis=0)))/c;

# Pulsed-Wave Frequency Response on Transmit
apod_theta = interp1d(thetapos, tx_apod[:,tx_evt], \
    kind='nearest', axis=0, fill_value=0, bounds_error=False)(theta);
delayIdeal = interp1d(thetapos, tx_delay, \
    kind='nearest', axis=0, fill_value=0, bounds_error=False)(theta);
txdata_f = np.outer(apod_theta,P_Tx_f) * np.exp(-1j*2*np.pi*np.outer(delayIdeal,f));

# Propagate Ultrasound Signals in Depth
rx_wf_th_r_f = np.zeros((r.size,)+rxdata_f.shape, dtype=np.dtype('complex64'));
tx_wf_th_r_f = np.zeros((r.size,)+txdata_f.shape, dtype=np.dtype('complex64'));
rx_wf_th_r_f[0,:,:] = rxdata_f; tx_wf_th_r_f[0,:,:] = txdata_f;
for r_idx in np.arange(r.size-1):
    # Propagate Signals in Depth
    rx_wf_th_r_f_out, tx_wf_th_r_f_out = propagate_polar(theta, 
        r[r_idx], r[r_idx+1], c, f, rx_wf_th_r_f[r_idx,:,:][:,:,np.newaxis], 
        tx_wf_th_r_f[r_idx,:,:][:,:,np.newaxis], aawin);
    rx_wf_th_r_f[r_idx+1,:,:], tx_wf_th_r_f[r_idx+1,:,:] = \
        rx_wf_th_r_f_out[:,:,0], tx_wf_th_r_f_out[:,:,0];
    print("r = "+str(r[r_idx]-Rconvex)+" m / "+str(dov)+" m");

# Compute Wavefield vs Time
tstart = 0; tend = 90e-6; Nt = 501;
t = np.linspace(tstart, tend, Nt);
ff, tt = np.meshgrid(f, t);
delays = np.exp(1j*2*np.pi*ff*tt);
Nth = theta.size; Nr = r.size;
tx_wf_th_r_t = np.transpose(np.reshape(delays.dot(np.reshape(np.transpose(tx_wf_th_r_f, \
    (2,0,1)), (f.size, Nr*Nth))), (Nt, Nr, Nth)), (1,2,0));
rx_wf_th_r_t = np.transpose(np.reshape(delays.dot(np.reshape(np.transpose(rx_wf_th_r_f, \
    (2,0,1)), (f.size, Nr*Nth))), (Nt, Nr, Nth)), (1,2,0));
img_th_r_t = np.cumsum(rx_wf_th_r_t*np.conj(tx_wf_th_r_t), axis=2);
del tx_wf_th_r_f, rx_wf_th_r_f;

# Perform Scan Conversion
theta_idx = np.logical_and(theta>=np.min(thetapos),theta<=np.max(thetapos));
theta = theta[theta_idx]; 
tx_wf_th_r_t = tx_wf_th_r_t[:,theta_idx,:]
rx_wf_th_r_t = rx_wf_th_r_t[:,theta_idx,:]
img_th_r_t = img_th_r_t[:,theta_idx,:];
THETA, R = np.meshgrid(theta, r);
X = R*np.sin(THETA); Z = R*np.cos(THETA)-Rconvex;
xpos = Rconvex*np.sin(thetapos); zpos = Rconvex*np.cos(thetapos)-Rconvex;
num_x = 400; num_z = 800;
x_img = np.linspace(np.min(X)/2,np.max(X)/2,num_x);
z_img = np.linspace(np.min(Z),np.max(Z),num_z);
X_IMG, Z_IMG = np.meshgrid(x_img, z_img);
tx_wf_x_z_t = np.zeros((num_z, num_x, Nt), dtype=np.dtype('complex64'));
rx_wf_x_z_t = np.zeros((num_z, num_x, Nt), dtype=np.dtype('complex64'));
img_x_z_t = np.zeros((num_z, num_x, Nt), dtype=np.dtype('complex64'));
for t_idx in np.arange(t.size):
    tx_wf_x_z_t[:, :, t_idx] = (interpn((r,theta), np.real(tx_wf_th_r_t[:,:,t_idx]), 
        np.vstack([np.sqrt(X_IMG**2+(Z_IMG+Rconvex)**2).flatten(),
        np.arctan2(X_IMG,Z_IMG+Rconvex).flatten()]).T, 
        bounds_error = False, method = 'splinef2d', fill_value = np.nan) 
        + 1j * interpn((r,theta), np.imag(tx_wf_th_r_t[:,:,t_idx]), 
        np.vstack([np.sqrt(X_IMG**2+(Z_IMG+Rconvex)**2).flatten(),
        np.arctan2(X_IMG,Z_IMG+Rconvex).flatten()]).T, 
        bounds_error = False, method = 'splinef2d', fill_value = np.nan)).reshape(num_z,num_x)
    rx_wf_x_z_t[:, :, t_idx] = (interpn((r,theta), np.real(rx_wf_th_r_t[:,:,t_idx]), 
        np.vstack([np.sqrt(X_IMG**2+(Z_IMG+Rconvex)**2).flatten(),
        np.arctan2(X_IMG,Z_IMG+Rconvex).flatten()]).T, 
        bounds_error = False, method = 'splinef2d', fill_value = np.nan) 
        + 1j * interpn((r,theta), np.imag(rx_wf_th_r_t[:,:,t_idx]), 
        np.vstack([np.sqrt(X_IMG**2+(Z_IMG+Rconvex)**2).flatten(),
        np.arctan2(X_IMG,Z_IMG+Rconvex).flatten()]).T, 
        bounds_error = False, method = 'splinef2d', fill_value = np.nan)).reshape(num_z,num_x)
    img_x_z_t[:, :, t_idx] = (interpn((r,theta), np.real(img_th_r_t[:,:,t_idx]), 
        np.vstack([np.sqrt(X_IMG**2+(Z_IMG+Rconvex)**2).flatten(),
        np.arctan2(X_IMG,Z_IMG+Rconvex).flatten()]).T, 
        bounds_error = False, method = 'splinef2d', fill_value = np.nan) 
        + 1j * interpn((r,theta), np.imag(img_th_r_t[:,:,t_idx]), 
        np.vstack([np.sqrt(X_IMG**2+(Z_IMG+Rconvex)**2).flatten(),
        np.arctan2(X_IMG,Z_IMG+Rconvex).flatten()]).T, 
        bounds_error = False, method = 'splinef2d', fill_value = np.nan)).reshape(num_z,num_x)
    print("t = "+str((1e6)*t[t_idx])+" microseconds");

## Plot Cross-Correlation of Tx and Rx Wavefields
plt.figure(); tpause = 1e-9;
while True:
    # Image Reconstructed at Each Time Step
    img = np.zeros((z_img.size, x_img.size),dtype=np.dtype('complex64'));
    for t_idx in np.arange(t.size):
        # Plot Transmit Wavefield
        plt.subplot(1,3,1); imagesc(1000*x_img, 1000*z_img, \
            np.real(tx_wf_x_z_t[:,:,t_idx]), \
            reg*np.max(np.abs(tx_wf_x_z_t[:,:,t_idx][~np.isnan(tx_wf_x_z_t[:,:,t_idx])]))*np.array([-1,1]));
        plt.xlabel('x Azimuthal Distance (mm)');
        plt.ylabel('z Axial Distance (mm)');
        plt.title('Transmit Wavefield');
        # Plot Receive Wavefield
        plt.subplot(1,3,2); imagesc(1000*x_img, 1000*z_img, \
            np.real(rx_wf_x_z_t[:,:,t_idx]), \
            reg*np.max(np.abs(rx_wf_x_z_t[:,:,t_idx][~np.isnan(rx_wf_x_z_t[:,:,t_idx])]))*np.array([-1,1]));
        plt.xlabel('x Azimuthal Distance (mm)');
        plt.ylabel('z Axial Distance (mm)');
        plt.title('Backpropagated Received Signals');
        # Accumulate Cross Corrleation
        img = img_x_z_t[:,:,t_idx];
        # Plot Accumulated Image
        plt.subplot(1,3,3); imagesc(1000*x_img, 1000*z_img, \
            20*np.log10(np.abs(img)/np.max(np.abs(img[~np.isnan(img)]))), dBrange);
        plt.xlabel('x Azimuthal Distance (mm)');
        plt.ylabel('z Axial Distance (mm)');
        plt.title('Time Domain Cross-Correlation');
        # Set Spacing between Subplots
        plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)
        # Animate
        plt.draw(); plt.pause(tpause); plt.clf();
