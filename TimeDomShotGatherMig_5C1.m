clear
clc

% Load all Functions from Subdirectories
addpath(genpath(pwd));

% Load File
load('SiemensData5C1_Kidney2.mat');

% Set Imaging Grid
dov = 135e-3; % Max Depth [m]
upsamp_theta = 2; % Upsampling in theta 
upsamp_r = 1; % Upsampling in r
Nth0 = 129; % Number of Points Laterally in theta

% Pick off Specific Transmitter
tx_evt = 31; rxdata_h = demodData(:,:,tx_evt); clearvars demodData;
dt = mean(diff(t)); fs = 1/dt; nt = numel(t); %t = t(1)+(0:nt-1)*dt; 

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
P_Tx = @(f) (abs(f-fTx)<(1.5e6));
P_Tx_f = P_Tx(f); % Pulse Definition

% Get Receive Channel Data in the Frequency Domain
P_Rx_f = fftshift(fft(rxdata_h, nt, 1), 1);
[~, F] = meshgrid(1:size(rxdata_h, 2), f); 
P_Rx_f = P_Rx_f .* exp(-1i*2*pi*F*t(1));
rxdata_f = interp1(thetapos, permute(P_Rx_f, [2,1]), theta, 'nearest', 0);

% Only Keep Positive Frequencies within Passband
passband_f_idx = find((P_Tx_f > reg) & (f > 0));
rxdata_f = rxdata_f(:,passband_f_idx,:);
f = f(passband_f_idx); P_Tx_f = P_Tx_f(passband_f_idx);
P_Tx_f = hanning(numel(P_Tx_f))';

% Construct Transmit Beamforming Delays
% Transmit Aperture Locations
txAptPosRelToCtr = elem_pos - tx_orig(:,tx_evt)*ones(1,size(elem_pos,2));
txFocRelToCtr = tx_foci(:,tx_evt) - tx_orig(:,tx_evt);
txFocRelToAptPos = txFocRelToCtr - txAptPosRelToCtr;
% Positive Value is Time Delay, Negative is Time Advance
tx_delay = (sqrt(sum(txFocRelToCtr.^2, 1)) - ...
    sqrt(sum(txFocRelToAptPos.^2, 1)))/c;

% Pulsed-Wave Frequency Response on Transmit
apod_theta = interp1(thetapos, tx_apod(:,tx_evt), theta, 'nearest', 0);
delayIdeal = interp1(thetapos, tx_delay, theta, 'nearest', 0);
txdata_f = (apod_theta'*P_Tx_f).*exp(-1i*2*pi*delayIdeal'*f);

% Construct the Transmit and Receive Wavefields
rx_wf_th_r_f = zeros(numel(r),numel(theta),numel(f));
tx_wf_th_r_f = zeros(numel(r),numel(theta),numel(f));
rx_wf_th_r_f(1,:,:) = rxdata_f; tx_wf_th_r_f(1,:,:) = txdata_f;
for r_idx = 1:numel(r)-1
    % Propagate Signals in Depth
    [rx_wf_th_r_f(r_idx+1,:,:), tx_wf_th_r_f(r_idx+1,:,:)] = ...
        propagate_polar(theta, r(r_idx), r(r_idx+1), c, f, ...
        squeeze(rx_wf_th_r_f(r_idx,:,:)), ...
        squeeze(tx_wf_th_r_f(r_idx,:,:)), aawin);    
end

% Compute Wavefield vs Time
tstart = 0; tend = 90e-6; Nt = 501;
t = linspace(tstart, tend, Nt);
[ff, tt] = meshgrid(f, t); 
delays = exp(1i*2*pi*ff.*tt);
Nth = numel(theta); Nr = numel(r);
tx_wf_th_r_t = permute(reshape(delays*reshape(permute(tx_wf_th_r_f, ...
    [3, 1, 2]), [numel(f), Nr*Nth]), [Nt, Nr, Nth]), [2, 3, 1]);
rx_wf_th_r_t = permute(reshape(delays*reshape(permute(rx_wf_th_r_f, ...
    [3, 1, 2]), [numel(f), Nr*Nth]), [Nt, Nr, Nth]), [2, 3, 1]);
img_th_r_t = cumsum(conj(tx_wf_th_r_t) .* rx_wf_th_r_t, 3);
clearvars tx_wf_th_r_f rx_wf_th_r_f;

% Perform Scan Conversion
theta_idx = (theta>=min(thetapos(:)) & theta<=max(thetapos(:)));
theta = theta(theta_idx); 
tx_wf_th_r_t = tx_wf_th_r_t(:,theta_idx,:);
rx_wf_th_r_t = rx_wf_th_r_t(:,theta_idx,:);
img_th_r_t = img_th_r_t(:,theta_idx,:);
[THETA, R] = meshgrid(theta, r);
X = R.*sin(THETA); Z = R.*cos(THETA)-Rconvex;
xpos = Rconvex*sin(thetapos); zpos = Rconvex*cos(thetapos)-Rconvex;
num_x = 400; num_z = 800;
x_img = linspace(min(X(:))/2,max(X(:))/2,num_x);
z_img = linspace(min(Z(:)),max(Z(:)),num_z);
[X_IMG, Z_IMG] = meshgrid(x_img, z_img);
tx_wf_x_z_t = zeros(num_z, num_x, Nt);
rx_wf_x_z_t = zeros(num_z, num_x, Nt);
img_x_z_t = zeros(num_z, num_x, Nt);
for t_idx = 1:numel(t)
    tx_wf_x_z_t(:, :, t_idx) = ...
        interp2(THETA, R, tx_wf_th_r_t(:,:,t_idx), ...
        atan2(X_IMG,Z_IMG+Rconvex), ...
        sqrt(X_IMG.^2+(Z_IMG+Rconvex).^2), 'spline', NaN);
    rx_wf_x_z_t(:, :, t_idx) = ...
        interp2(THETA, R, rx_wf_th_r_t(:,:,t_idx), ...
        atan2(X_IMG,Z_IMG+Rconvex), ...
        sqrt(X_IMG.^2+(Z_IMG+Rconvex).^2), 'spline', NaN);
    img_x_z_t(:, :, t_idx) = ...
        interp2(THETA, R, img_th_r_t(:,:,t_idx), ...
        atan2(X_IMG,Z_IMG+Rconvex), ...
        sqrt(X_IMG.^2+(Z_IMG+Rconvex).^2), 'spline', NaN);
end

%% Plot Cross-Correlation of Tx and Rx Wavefields
figure('Position',[0,0,1200,800])
M = moviein(ceil(numel(t)));
while true
    % Image Reconstructed at Each Time Step
    img = zeros(numel(z_img), numel(x_img));
    for t_idx = 1:numel(t)
        % Plot Transmit Wavefield
        subplot(1,3,1); imagesc(1000*x_img, 1000*z_img, real(tx_wf_x_z_t(:, :, t_idx)), ...
            reg*max(max(abs(tx_wf_x_z_t(:, :, t_idx))))*[-1,1]); 
        xlabel('Lateral [mm]'); ylabel('Axial [mm]'); 
        zoom on; axis equal; axis xy; axis image; set(gca, 'YDir', 'reverse'); 
        title('Transmit Wavefield');
        xlim(1000*[min(x_img),max(x_img)]); ylim(1000*[min(z_img),max(z_img)]); colormap gray; 
        % Plot Receive Wavefield
        subplot(1,3,2); imagesc(1000*x_img, 1000*z_img, real(rx_wf_x_z_t(:, :, t_idx)), ...
            reg*max(max(abs(rx_wf_x_z_t(:, :, t_idx))))*[-1,1]); 
        xlabel('Lateral [mm]'); ylabel('Axial [mm]'); 
        zoom on; axis equal; axis xy; axis image; set(gca, 'YDir', 'reverse'); 
        title('Backpropagated Received Signals');
        xlim(1000*[min(x_img),max(x_img)]); ylim(1000*[min(z_img),max(z_img)]); colormap gray; 
        % Accumulate Cross Corrleation
        img = img_x_z_t(:, :, t_idx);
        % Plot Accumulated Image
        subplot(1,3,3); imagesc(1000*x_img, 1000*z_img, ...
            db(abs(img)/max(abs(img(:)))), dBrange);
        xlabel('Lateral [mm]'); ylabel('Axial [mm]'); 
        title('Time Domain Cross-Correlation'); 
        zoom on; axis equal; axis xy; axis image; 
        colormap gray; xlim(1000*[min(x_img),max(x_img)]); ylim(1000*[min(z_img),max(z_img)]);  
        set(gca, 'YDir', 'reverse'); getframe(gca); 
        M(t_idx)= getframe(gcf); clf();
    end
end

