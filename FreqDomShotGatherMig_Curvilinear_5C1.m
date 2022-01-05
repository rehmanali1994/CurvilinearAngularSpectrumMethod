clear
clc

% Load all Functions from Subdirectories
addpath(genpath(pwd));

% Load File
load('SiemensData5C1_Kidney2.mat');
dt = mean(diff(t)); fs = 1/dt; nt = numel(t);

% Set Imaging Grid
dov = 135e-3; % Max Depth [m]
upsamp_theta = 2; % Upsampling in theta 
upsamp_r = 1; % Upsampling in r
Nth0 = 128; % Number of Points Laterally in theta

% Pick off Specific Transmitter
tx_evts = 1:61;
rxdata_h = demodData(:,:,tx_evts); clearvars demodData;

% Find the Radius of the 5C1 Probe
lsfit = @(R) mean((elem_pos(3,:)-sqrt(R^2-elem_pos(1,:).^2)+R).^2);
Rconvex = fminbnd(lsfit,0.04,0.05,optimset('TolX',1e-10,'TolFun',1e-10));

% Aperture Definition
thetapos = atan2(elem_pos(1,:),elem_pos(3,:)+Rconvex); % rad
dtheta = mean(diff(thetapos)); % angular spacing [rad]

% Transmit Pulse
fTx = 3.5e6; % Hz
lambda = c/fTx; % m

% Simulation Space
theta = (-(upsamp_theta*Nth0-1)/2:(upsamp_theta*Nth0-1)/2)*(dtheta/upsamp_theta); % rad
Nu1 = round(dov/((lambda/2)/upsamp_r)); 
r = Rconvex+(((0:Nu1-1))*(lambda/2)/upsamp_r);

% Image Reconstruction Parameters and Anti-Aliasing Window
dBrange = [-60, 0]; reg = 1e-1; ord = 50; 
thetamax = (max(abs(thetapos))+max(abs(theta)))/2;
aawin = 1./sqrt(1+(theta/thetamax).^ord);

% Transmit Impulse Response in Frequency Domain
f = (fs/2)*(-1:2/nt:1-2/nt); % Hz
P_Tx = @(f) (abs(f-fTx)<(3.0e6));
P_Tx_f = P_Tx(f); % Pulse Definition

% Get Receive Channel Data in the Frequency Domain
P_Rx_f = fftshift(fft(rxdata_h, nt, 1), 1);
[~, F] = meshgrid(1:size(rxdata_h, 2), f); 
P_Rx_f = P_Rx_f .* exp(-1i*2*pi*F*t(1));
rxdata_f = interp1(thetapos, permute(P_Rx_f, [2,1,3]), theta, 'nearest', 0);

% Only Keep Positive Frequencies within Passband
passband_f_idx = find((P_Tx_f > reg) & (f > 0));
rxdata_f = rxdata_f(:,passband_f_idx,:);
f = f(passband_f_idx); P_Tx_f = P_Tx_f(passband_f_idx);
P_Tx_f = hanning(numel(P_Tx_f))';

% Construct Transmit Beamforming Delays
tx_delay = zeros(size(elem_pos,2),size(tx_orig,2));
for tx_idx = 1:numel(tx_evts)
    % Transmit Aperture Locations
    txAptPosRelToCtr = elem_pos - tx_orig(:,tx_evts(tx_idx))*ones(1,size(elem_pos,2));
    txFocRelToCtr = tx_foci(:,tx_evts(tx_idx)) - tx_orig(:,tx_evts(tx_idx));
    txFocRelToAptPos = txFocRelToCtr - txAptPosRelToCtr;
    % Positive Value is Time Delay, Negative is Time Advance
    tx_delay(:,tx_evts(tx_idx)) = (sqrt(sum(txFocRelToCtr.^2, 1)) - ...
        sqrt(sum(txFocRelToAptPos.^2, 1)))/c;
end

% Pulsed-Wave Frequency Response on Transmit
txdata_f = zeros(numel(theta), numel(f), numel(tx_evts));
for k = 1:numel(tx_evts) 
    % Construct Transmit Responses for Each Element
    apod_theta = interp1(thetapos, tx_apod(:,tx_evts(k)), theta, 'nearest', 0);
    delayIdeal = interp1(thetapos, tx_delay(:,tx_evts(k)), theta, 'nearest', 0);
    txdata_f(:,:,k) = (apod_theta'*P_Tx_f).*exp(-1i*2*pi*delayIdeal'*f);
end

% Create Image and Gain Compensation Maps
img = zeros(numel(r), numel(theta));
tx_map = zeros(numel(r), numel(theta));
img(1,:) = sum(sum(txdata_f .* conj(rxdata_f),3),2);
tx_map(1,:) = sum(sum(txdata_f .* conj(txdata_f),3),2);

% Propagate Ultrasound Signals in Depth
rxdata_f_nxt = zeros(size(rxdata_f));
txdata_f_nxt = zeros(size(txdata_f));
for r_idx = 1:numel(r)-1
    % Propagate Signals in Depth
    [rxdata_f_nxt, txdata_f_nxt] = ...
        propagate_polar(theta, r(r_idx), r(r_idx+1), c, f, rxdata_f, txdata_f, aawin);
    % Compute Image at this Depth
    img(r_idx+1,:) = sum(sum(txdata_f_nxt .* conj(rxdata_f_nxt),3),2);
    tx_map(r_idx+1,:) = sum(txdata_f_nxt(:,round(numel(f)/2),:) .* ...
        conj(txdata_f_nxt(:,round(numel(f)/2),:)),3); % Based on Center Frequency
    % Setup Next Depth Step
    rxdata_f = rxdata_f_nxt; txdata_f = txdata_f_nxt;
    disp(['r = ', num2str(r(r_idx)-Rconvex), ' m / ', num2str(dov), ' m']);
end

% Apply Time-Gain Compensation to Image
img_recon = img; % ./ (tx_map + reg*max(tx_map(:)));

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
title('Image Reconstruction'); zoom on; axis equal; axis xy; axis image; 
colormap gray; colorbar(); set(gca, 'YDir', 'reverse'); 