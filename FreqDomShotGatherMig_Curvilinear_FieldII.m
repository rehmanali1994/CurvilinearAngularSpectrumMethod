clear
clc

% Load all Functions from Subdirectories
addpath(genpath(pwd));

% Load File and Set Imaging Grid
load FieldII_AnechoicLesionFullSynthData.mat; % Anechoic Lesion Phantom
dov = 45e-3; % Max Depth [m]
upsamp_theta = 2; % Upsampling in theta
upsamp_r = 1; % Upsampling in r 
Nth0 = 128; % Number of Points Laterally in theta

% Pick off Specific Transmitter
tx_elmts = 1:96;
rxdata_h = scat_h(:,:,tx_elmts);
clearvars scat_h;

% Aperture Definition
lambda = c/fTx; % m
dtheta = pitch/Rconvex; % angular spacing [rad]
thetapos = (-(no_elements-1)/2:(no_elements-1)/2)*dtheta; % rad

% Simulation Space
theta = (-(upsamp_theta*Nth0-1)/2:(upsamp_theta*Nth0-1)/2)*(dtheta/upsamp_theta); % rad
Nu1 = round(dov/((lambda/2)/upsamp_r)); 
r = Rconvex+(((0:Nu1-1))*(lambda/2)/upsamp_r);

% Image Reconstruction Parameters and Anti-Aliasing Window
dBrange = [-60, 0]; reg = 1e-3; ord = 50; 
thetamax = (max(abs(thetapos))+max(abs(theta)))/2;
aawin = 1./sqrt(1+(theta/thetamax).^ord);

% Transmit Impulse Response in Frequency Domain
nt = numel(time);
f = (fs/2)*(-1:2/nt:1-2/nt); % Hz
P_Tx = @(f) abs(freqz(impResp, 1, 2*pi*f/fs)) / ...
    max(abs(freqz(impResp, 1, 2*pi*f/fs)));
P_Tx_f = P_Tx(f); % Pulse Definition

% Only Keep Positive Frequencies within Passband
passband_f_idx = find((P_Tx_f > reg) & (f > 0));
f = f(passband_f_idx); P_Tx_f = P_Tx_f(passband_f_idx);
P_Tx_f = ones(size(P_Tx_f)); % Assume Flat Passband

% Get Receive Channel Data in the Frequency Domain
P_Rx_f = fftshift(fft(rxdata_h, nt, 1), 1); 
P_Rx_f = P_Rx_f(passband_f_idx,:,:); clearvars rxdata_h; 
[~, F, ~] = meshgrid(1:size(P_Rx_f,2), f, 1:size(P_Rx_f,3)); 
P_Rx_f = P_Rx_f .* exp(-1i*2*pi*F*time(1));
rxdata_f = interp1(thetapos, permute(P_Rx_f, [2,1,3]), theta, 'nearest', 0);

% Pulsed-Wave Frequency Response on Transmit
apod = eye(no_elements); delay = zeros(no_elements);
txdata_f = zeros(numel(theta), numel(f), numel(tx_elmts));
for k = 1:numel(tx_elmts) 
    % Construct Transmit Responses for Each Element
    apod_theta = interp1(thetapos, apod(:,tx_elmts(k)), theta, 'nearest', 0);
    delayIdeal = interp1(thetapos, delay(:,tx_elmts(k)), theta, 'nearest', 0);
    txdata_f(:,:,k) = (apod_theta'*P_Tx_f).*exp(-1i*2*pi*delayIdeal'*f);
end

% Create Image and Gain Compensation Maps
img = zeros(numel(r), numel(theta));
tx_map = zeros(numel(r), numel(theta));
img(1,:) = sum(sum(txdata_f .* conj(rxdata_f),3),2);
tx_map(1,:) = sum(sum(txdata_f .* conj(txdata_f),3),2);

% Propagate Ultrasound Signals Radially
rxdata_f_nxt = zeros(size(rxdata_f));
txdata_f_nxt = zeros(size(txdata_f));
for r_idx = 1:numel(r)-1
    % Propagate Signals Radially
    [rxdata_f_nxt, txdata_f_nxt] = ...
        propagate_polar(theta, r(r_idx), r(r_idx+1), c, f, rxdata_f, txdata_f, aawin);
    % Compute Image at this Radius
    img(r_idx+1,:) = sum(sum(txdata_f_nxt .* conj(rxdata_f_nxt),3),2);
    tx_map(r_idx+1,:) = sum(txdata_f_nxt(:,round(numel(f)/2),:) .* ...
        conj(txdata_f_nxt(:,round(numel(f)/2),:)),3); % Based on Center Frequency
    % Setup Next Depth Step
    rxdata_f = rxdata_f_nxt; txdata_f = txdata_f_nxt;
    disp(['r = ', num2str(r(r_idx)-Rconvex), ' m / ', num2str(dov), ' m']);
end

% Apply Time-Gain Compensation to Image
img_recon = img ./ (tx_map + reg*max(tx_map(:)));

% Perform Scan Conversion
theta_idx = (theta>=min(thetapos(:)) & theta<=max(thetapos(:)));
theta = theta(theta_idx); img_recon = img_recon(:,theta_idx);
[THETA, R] = meshgrid(theta, r);
X = R.*sin(THETA); Z = R.*cos(THETA)-Rconvex;
num_x = 1000; num_z = 1000;
x_img = linspace(min(X(:)),max(X(:)),num_x);
z_img = linspace(min(Z(:)),max(Z(:)),num_z);
[X_IMG, Z_IMG] = meshgrid(x_img, z_img);
IMG_RECON = interp2(THETA,R,abs(img_recon),...
    atan2(X_IMG,Z_IMG+Rconvex), sqrt(X_IMG.^2+(Z_IMG+Rconvex).^2),'spline',0);

% Reconstruct and Plot Ultrasound Image
figure; imagesc(1000*x_img, 1000*z_img, db(IMG_RECON/max(IMG_RECON(:))), dBrange);
xlabel('Lateral [mm]'); ylabel('Axial [mm]'); 
title('Multistatic Synthetic Aperture Reconstruction'); 
zoom on; axis equal; axis xy; axis image; 
colormap gray; colorbar(); set(gca, 'YDir', 'reverse'); 