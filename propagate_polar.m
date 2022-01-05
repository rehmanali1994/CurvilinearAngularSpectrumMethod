function [rxdata_r2_f, txdata_r2_f] = ...
    propagate_polar(theta, r1, r2, c, f, rxdata_r1_f, txdata_r1_f, aawin)
% 
% PROPAGATE_POLAR - Angular Spectrum Propagation of TX/RX Signals into the 
% Medium for Curvilinear Arrays Based on a Log-Polar Conformal Mapping
%
% This function propagates transmit and receive wavefields at from one 
% radial section to another using the angular spectrum method
% 
% INPUTS:
% theta              - 1 x T vector of theta-grid positions for wavefield
% r1                 - radial location for input TX and RX wavefields
% r2                 - radial location for output TX and RX wavefields
% c                  - speed of sound [m/s] between r1 and r2; default 1540 m/s
% f                  - 1 x F vector of pulse frequencies in spectrum
% rxdata_r1_f        - T x F x N array of input RX wavefields at r1 (N Transmit Events)
% txdata_r1_f        - T x F x N array of input TX wavefields at r1
% aawin              - 1 x T vector of lateral taper to prevent wraparound
% 
% OUTPUT:
% rxdata_r2_f        - T x F x N array of output RX wavefields at r2
% txdata_r2_f        - T x F x N array of output TX wavefields at r2
% 

% Verify the Number of Common Shot Gathers
ns = size(txdata_r1_f, 3); assert(size(rxdata_r1_f, 3) == ns, ...
    'Number of sources must equal to number of common-source gathers');
AAwin = repmat(aawin(:), [1, numel(f), ns]);

% Forward and Inverse Fourier Transforms with Anti-Aliasing Windows
ft = @(sig) fftshift(fft(AAwin.*sig, [], 1), 1);
ift = @(sig) AAwin.*ifft(ifftshift(sig, 1), [], 1);

% Spatial Grid
dth = mean(diff(theta)); nth = numel(theta); 

% FFT Axis for Lateral Spatial Frequency
kth = mod(fftshift((0:nth-1)/(dth*nth))+1/(2*dth), 1/dth)-1/(2*dth);

% Continuous Wave Response By Downward Angular Spectrum
[F, Kth] = meshgrid(f,kth); % Create Grid in f-kx
% Phase Shift For Propagator
phs = @(r) sqrt((F*r/c).^2-Kth.^2) - ...
    abs(Kth).*atan(sqrt((F*r./(c*Kth)).^2-1));
evanescent = @(r) (((F*r/c).^2-Kth.^2) <= 0);
% Continuous Wave Response By Downward Angular Spectrum
H = exp(1i*2*pi*(phs(r2)-phs(r1))); % Propagation Filter
H(evanescent(min(r1,r2))) = 0; % Remove Evanescent Components
H = repmat(H, [1,1,ns]); % Replicate Across Shots
% Apply Propagation Filter
rxdata_r2_f = ift(H.*ft(rxdata_r1_f));
txdata_r2_f = ift(conj(H).*ft(txdata_r1_f)); 

end