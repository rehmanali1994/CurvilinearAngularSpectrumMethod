import numpy as np
from scipy import linalg
from scipy.interpolate import RectBivariateSpline, interpn
import pdb, warnings

def propagate_polar(theta, r1, r2, c, f, rxdata_r1_f, txdata_r1_f, aawin):
    '''rxdata_r2_f, txdata_r2_f = propagate_polar(theta, r1, r2, c, f, rxdata_r1_f, txdata_r1_f, aawin)

    PROPAGATE_POLAR - Angular Spectrum Propagation of TX/RX Signals into the 
    Medium for Curvilinear Arrays Based on a Log-Polar Conformal Mapping

    This function propagates transmit and receive wavefields at from one
    depth to another depth using the angular spectrum method

    INPUTS:
    theta              - 1 x T vector of theta-grid positions for wavefield
    r1                 - radial location for input TX and RX wavefields
    r2                 - radial location for output TX and RX wavefields
    c                  - speed of sound [m/s] between r1 and r2; default 1540 m/s
    f                  - 1 x F vector of pulse frequencies in spectrum
    rxdata_r1_f        - T x F x N array of input RX wavefields at r1 (N Transmit Events)
    txdata_r1_f        - T x F x N array of input TX wavefields at r1
    aawin              - 1 x T vector of lateral taper to prevent wraparound

    OUTPUT:
    rxdata_r2_f        - T x F x N array of output RX wavefields at r2
    txdata_r2_f        - T x F x N array of output TX wavefields at r2'''

    # Verify the Number of Common Shot Gathers
    ns = txdata_r1_f.shape[2]; assert(rxdata_r1_f.shape[2] == ns), \
        'Number of sources must equal to number of common-source gathers';
    AAwin = np.tile(aawin[:,np.newaxis,np.newaxis], (1, f.size, ns));

    # Forward and Inverse Fourier Transforms with Anti-Aliasing Windows
    ft = lambda sig: np.fft.fftshift(np.fft.fft(AAwin*sig, axis=0), axes=0);
    ift = lambda sig: AAwin*np.fft.ifft(np.fft.ifftshift(sig, axes=0), axis=0);

    # Spatial Grid
    dth = np.mean(np.diff(theta)); nth = theta.size;

    # FFT Axis for Lateral Spatial Frequency
    kth = np.mod(np.fft.fftshift(np.arange(nth)/(dth*nth))+1/(2*dth), 1/dth)-1/(2*dth);

    # Continuous Wave Response By Downward Angular Spectrum
    F, Kth = np.meshgrid(f,kth); # Create Grid in f-kth
    # Phase Shift For Propagator
    phs = lambda r: np.sqrt((F*r/c)**2-Kth**2)-np.abs(Kth)*np.arctan(np.sqrt((F*r/(c*Kth))**2-1));
    evanescent = lambda r: (((F*r/c)**2-Kth**2) <= 0);
    # Continuous Wave Response By Downward Angular Spectrum
    with warnings.catch_warnings():
        warnings.simplefilter("ignore"); # Ignore All Warnings From Next Two Lines
        H = np.exp(1j*2*np.pi*(phs(r2)-phs(r1)).astype('complex64')); # Propagation Filter
        H[evanescent(min(r1,r2))] = 0; # Remove Evanescent Components
    H = np.tile(H[:,:,np.newaxis], (1,1,ns)); # Replicate Across Shots
    # Apply Propagation Filter
    rxdata_r2_f = ift(H*ft(rxdata_r1_f));
    txdata_r2_f = ift(np.conj(H)*ft(txdata_r1_f));

    return rxdata_r2_f, txdata_r2_f;

# Define Loadmat Function for HDF5 Format ('-v7.3' in MATLAB)
import h5py
def loadmat_hdf5(filename):
    file = h5py.File(filename,'r')
    out_dict = {}
    for key in file.keys():
        out_dict[key] = np.ndarray.transpose(np.array(file[key]));
    file.close()
    return out_dict;

# Python-Equivalent Command for IMAGESC in MATLAB
import matplotlib.pyplot as plt
def imagesc(x, y, img, rng, cmap='gray', numticks=(3, 3), aspect='equal'):
    exts = (np.min(x)-np.mean(np.diff(x)), np.max(x)+np.mean(np.diff(x)), \
        np.min(y)-np.mean(np.diff(y)), np.max(y)+np.mean(np.diff(y)));
    plt.imshow(np.flipud(img), cmap=cmap, extent=exts, vmin=rng[0], vmax=rng[1], aspect=aspect);
    plt.xticks(np.linspace(np.min(x), np.max(x), numticks[0]));
    plt.yticks(np.linspace(np.min(y), np.max(y), numticks[1]));
    plt.gca().invert_yaxis();
