# Setting up all folders we can import from by adding them to python path
import sys, os, pdb

# Importing stuff from all folders in python path
import numpy as np
from propagate_polar import *
import scipy.io as sio
from scipy.signal import hilbert, freqz
from scipy.interpolate import griddata, interp1d, interp2d, interpn
from scipy.optimize import fminbound
import matplotlib.pyplot as plt

# Load Channel Data
FocTxDataset = loadmat_hdf5('SiemensData5C1_Kidney2.mat');
t = FocTxDataset['t'][0]; # Time Axis [s]
elem_pos = FocTxDataset['elem_pos']; # Locations of Array Elements [m]
demodData = FocTxDataset['demodData']; # Actual Channel Data [Nt, NRx, NTx]
dt = np.mean(np.diff(t)); nt = t.size; fs = 1/dt; # Sampling Rate [Hz]
tx_apod = FocTxDataset['tx_apod']; # Transmit Apodization
tx_foci = FocTxDataset['tx_foci']; # Transmit Foci [m]
tx_orig = FocTxDataset['tx_orig']; # Transmit Origins [m]
c = FocTxDataset['c'][0][0]; # Sound Speed [m/s]
del FocTxDataset;

# Load File and Set Imaging Grid
dov = 135e-3; # Max Depth [m]
upsamp_theta = 1; # Upsampling in theta
upsamp_r = 0.5; # Upsampling in r
Nth0 = 128; # Number of Points Laterally in theta

# Select Subset of Transmit Elements
tx_evts = np.arange(0,61,1);
rxdata_h = demodData[:,:,tx_evts];
del demodData;

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
T, F, N = np.meshgrid(np.arange(P_Rx_f.shape[1]), f, np.arange(P_Rx_f.shape[2]));
P_Rx_f = P_Rx_f * np.exp(-1j*2*np.pi*F*t[0]); del T, F, N;
rxdata_f = interp1d(thetapos, np.transpose(P_Rx_f, (1,0,2)), \
    kind='nearest', axis=0, fill_value=0, bounds_error=False)(theta);
del rxdata_h, P_Rx_f; # Delete Receive Channel Data

# Only Keep Positive Frequencies within Passband
passband_f_idx = np.argwhere((P_Tx_f > reg) & (f > 0)).flatten();
rxdata_f = rxdata_f[:,passband_f_idx,:];
f = f[passband_f_idx]; P_Tx_f = P_Tx_f[passband_f_idx];
P_Tx_f = np.hanning(P_Tx_f.size); # Assume Flat Passband

# Construct Transmit Beamforming Delays
tx_delay = np.zeros((elem_pos.shape[1],tx_orig.shape[1]));
for tx_idx in np.arange(tx_evts.size):
    # transmit aperture locations
    txAptPosRelToCtr = elem_pos - \
        tx_orig[:,tx_evts[tx_idx]][:,np.newaxis]*np.ones((1,elem_pos.shape[1]));
    txFocRelToCtr = tx_foci[:,tx_evts[tx_idx]] - tx_orig[:,tx_evts[tx_idx]]
    txFocRelToAptPos = txFocRelToCtr[:,np.newaxis]*np.ones((1,elem_pos.shape[1])) - txAptPosRelToCtr;
    # positive value is time delay, negative is time advance
    tx_delay[:,tx_evts[tx_idx]] = (np.sqrt(np.sum(txFocRelToCtr**2, axis=0)) - \
        np.sqrt(np.sum(txFocRelToAptPos**2, axis=0)))/c;

# Pulsed-Wave Frequency Response on Transmit
txdata_f = np.zeros((theta.size, f.size, tx_evts.size), dtype=np.dtype('complex64'));
for k in np.arange(tx_evts.size):
    # Construct Transmit Responses for Each Element
    apod_theta = interp1d(thetapos, tx_apod[:,tx_evts[k]], \
        kind='nearest', axis=0, fill_value=0, bounds_error=False)(theta);
    delayIdeal = interp1d(thetapos, tx_delay[:,tx_evts[k]], \
        kind='nearest', axis=0, fill_value=0, bounds_error=False)(theta);
    txdata_f[:,:,k] = np.outer(apod_theta,P_Tx_f) * \
        np.exp(-1j*2*np.pi*np.outer(delayIdeal,f));

# Create Image and Gain Compensation Maps
img = np.zeros((r.size, theta.size), dtype=np.dtype('complex64'));
tx_map = np.zeros((r.size, theta.size), dtype=np.dtype('complex64'));
img[0,:] = np.sum(np.sum(txdata_f*np.conj(rxdata_f),axis=1),axis=1);
tx_map[0,:] = np.sum(np.sum(txdata_f*np.conj(txdata_f),axis=1),axis=1);

# Propagate Ultrasound Signals Radially
rxdata_f_nxt = np.zeros(rxdata_f.shape);
txdata_f_nxt = np.zeros(txdata_f.shape);
for r_idx in np.arange(r.size-1):
    # Propagate Signals Radially
    rxdata_f_nxt, txdata_f_nxt = \
        propagate_polar(theta, r[r_idx], r[r_idx+1], c, f, rxdata_f, txdata_f, aawin);
    # Compute Image at this Radius
    img[r_idx+1,:] = np.sum(np.sum(txdata_f_nxt*np.conj(rxdata_f_nxt),axis=1),axis=1);
    tx_map[r_idx+1,:] = np.sum(txdata_f_nxt[:,int(np.round(f.size/2)),:] *
        np.conj(txdata_f_nxt[:,int(np.round(f.size/2)),:]),axis=1); # Based on Center Frequency
    # Setup Next Depth Step
    rxdata_f = rxdata_f_nxt; txdata_f = txdata_f_nxt;
    print("r = "+str(r[r_idx]-Rconvex)+" m / "+str(dov)+" m");

# Apply Time-Gain Compensation to Image
img_recon = img / (tx_map + reg*np.max(tx_map));

# Perform Scan Conversion
theta_idx = np.logical_and(theta>=np.min(thetapos),theta<=np.max(thetapos));
theta = theta[theta_idx]; img_recon = img_recon[:,theta_idx];
THETA, R = np.meshgrid(theta, r);
X = R*np.sin(THETA); Z = R*np.cos(THETA)-Rconvex;
num_x = 1000; num_z = 1000;
x_img = np.linspace(np.min(X),np.max(X),num_x);
z_img = np.linspace(np.min(Z),np.max(Z),num_z);
X_IMG, Z_IMG = np.meshgrid(x_img, z_img);
IMG_RECON = (interpn((r,theta), np.real(img_recon), 
    np.vstack([np.sqrt(X_IMG**2+(Z_IMG+Rconvex)**2).flatten(),
    np.arctan2(X_IMG,Z_IMG+Rconvex).flatten()]).T, 
    bounds_error = False, method = 'splinef2d', fill_value = 0) 
    + 1j * interpn((r,theta), np.imag(img_recon), 
    np.vstack([np.sqrt(X_IMG**2+(Z_IMG+Rconvex)**2).flatten(),
    np.arctan2(X_IMG,Z_IMG+Rconvex).flatten()]).T, 
    bounds_error = False, method = 'splinef2d', fill_value = 0)).reshape(num_z,num_x)

# Reconstruct and Plot Ultrasound Image
plt.figure(); imagesc(1000*x_img, 1000*z_img,
    20*np.log10(np.abs(IMG_RECON)/np.max(np.abs(IMG_RECON))), dBrange);
plt.xlabel('Lateral [mm]'); plt.ylabel('Axial [mm]');
plt.title('Multistatic Synthetic Aperture Reconstruction'); plt.colorbar(); plt.show();