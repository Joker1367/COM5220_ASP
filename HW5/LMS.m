clear all;
clc;

% Parameter settings
total_trial = 100;
num_samples = 12000; % Number of samples
filter_length = 6;   % Length of the adaptive filter
alpha = 0.01;        % Step size

% Generate signals
rng(42); % Fix random seed for reproducibility
h = [0.227, 0.46, 0.688, 0.46, 0.227]';   % Channel impulse response

c_NLMS = 1e-3;

c_1 = 0.9;
c_2 = 1.1;
N_1 = 3;
N_2 = 3;
alpha_min = 1e-4;
alpha_max = 1e-2;

%% LMS Algorithm

e_LMS = zeros(num_samples, 1);                                          % Record squared error for learning curve
e_NLMS = zeros(num_samples, 1);                                         % Record squared error for learning curve
e_VSLMS = zeros(num_samples, 1);                                        % Record squared error for learning curve

ber_1_begin = 101;
ber_2_begin = 1001;
se_1_begin  = 101;
se_2_begin  = 1001;

BER_1_LMS   = zeros(total_trial, 1);
BER_1_NLMS  = zeros(total_trial, 1);
BER_1_VSLMS = zeros(total_trial, 1);
BER_2_LMS   = zeros(total_trial, 1);
BER_2_NLMS  = zeros(total_trial, 1);
BER_2_VSLMS = zeros(total_trial, 1);

SE_1_LMS    = zeros(total_trial, 1);
SE_1_NLMS   = zeros(total_trial, 1);
SE_1_VSLMS  = zeros(total_trial, 1);
SE_2_LMS    = zeros(total_trial, 1);
SE_2_NLMS   = zeros(total_trial, 1);
SE_2_VSLMS  
= zeros(total_trial, 1);

for trial = 1 : total_trial
    s = randi([0, 1], num_samples, 1) * 2 - 1;                          % Generate ±1 random signal (information signal)
    i = rand(num_samples, 1) * 2 - 1;                                   % Generate random interference signal
    x = conv(i, h, 'same');                                             % Convolve interference with channel
    d = s + i;                                                          % Desired signal (information + interference)

    % LMS
    f_LMS = zeros(filter_length, 1);                                    % Initial filter coefficients    
    
    for n = filter_length:num_samples
        x_n = flip(x(n-filter_length+1:n));                             % Current input vector (reversed order)
        y_n = f_LMS' * x_n;                                             % Filter output
        e_n = d(n) - y_n;                                               % Error signal
        f_LMS = f_LMS + alpha * e_n * x_n;                              % Update filter coefficients
        e_LMS(n) = e_LMS(n) + e_n^2;                                    % Record squared error

        if n >= ber_1_begin
            BER_1_LMS(trial) = BER_1_LMS(trial) + (sign(e_n) ~= s(n));
        end

        if n >= ber_2_begin
            BER_2_LMS(trial) = BER_2_LMS(trial) + (sign(e_n) ~= s(n));
        end

        if n >= se_1_begin
            SE_1_LMS(trial) = SE_1_LMS(trial) + (e_n - s(n))^2;
        end

        if n >= se_2_begin
            SE_2_LMS(trial) = SE_2_LMS(trial) + (e_n - s(n))^2;
        end
    end

    % NLMS
    f_NLMS = zeros(filter_length, 1);                                   % Initial filter coefficients

    for n = filter_length:num_samples
        x_n = flip(x(n-filter_length+1:n));                             % Current input vector (reversed order)
        y_n = f_NLMS' * x_n;                                            % Filter output
        e_n = d(n) - y_n;                                               % Error signal
        f_NLMS = f_NLMS + alpha * e_n * x_n / (c_NLMS + x_n' * x_n);    % Update filter coefficients
        e_NLMS(n) = e_NLMS(n) + e_n^2;                                  % Record squared error

        if n >= ber_1_begin
            BER_1_NLMS(trial) = BER_1_NLMS(trial) + (sign(e_n) ~= s(n));
        end

        if n >= ber_2_begin
            BER_2_NLMS(trial) = BER_2_NLMS(trial) + (sign(e_n) ~= s(n));
        end

        if n >= se_1_begin
            SE_1_NLMS(trial) = SE_1_NLMS(trial) + (e_n - s(n))^2;
        end

        if n >= se_2_begin
            SE_2_NLMS(trial) = SE_2_NLMS(trial) + (e_n - s(n))^2;
        end
    end

    % VS-LMS
    alpha_vslms = alpha_max * ones(filter_length, 1);
    f_VSLMS = zeros(filter_length, 1);                                  % Initial filter coefficients
    
    sign_changes = zeros(filter_length, 1); 
    no_sign_changes = zeros(filter_length, 1);
    prev_error_signs = zeros(filter_length, 1);
    
    for n = filter_length:num_samples
        x_n = flip(x(n-filter_length+1:n));                             % Current input vector (reversed order)
        y_n = f_VSLMS' * x_n;                                           % Filter output
        e_n = d(n) - y_n;                                               % Error signal
    
        f_VSLMS = f_VSLMS + diag(alpha_vslms) * x_n * e_n;              % Update filter coefficients
    
        current_error_signs = sign(e_n * x_n);
        sign_changes = sign_changes + (current_error_signs ~= prev_error_signs);
        no_sign_changes = no_sign_changes + (current_error_signs == prev_error_signs);
    
        for i = 1 : filter_length
            if sign_changes(i) >= N_1
                alpha_vslms(i) = max(alpha_vslms(i) * c_1, alpha_min);
                sign_changes(i) = 0;
            elseif no_sign_changes(i) >= N_2
                alpha_vslms(i) = min(alpha_vslms(i) * c_2, alpha_max);
                no_sign_changes(i) = 0;
            end
        end
    
        e_VSLMS(n) = e_VSLMS(n) + e_n^2;                                % Record squared error
        prev_error_signs = current_error_signs;

        if n >= ber_1_begin
            BER_1_VSLMS(trial) = BER_1_VSLMS(trial) + (sign(e_n) ~= s(n));
        end

        if n >= ber_2_begin
            BER_2_VSLMS(trial) = BER_2_VSLMS(trial) + (sign(e_n) ~= s(n));
        end

        if n >= se_1_begin
            SE_1_VSLMS(trial) = SE_1_VSLMS(trial) + (e_n - s(n))^2;
        end

        if n >= se_2_begin
            SE_2_VSLMS(trial) = SE_2_VSLMS(trial) + (e_n - s(n))^2;
        end
    end
end 

e_LMS = e_LMS / total_trial;
e_NLMS = e_NLMS / total_trial;
e_VSLMS = e_VSLMS / total_trial;

% Plot the learning curve
figure;
plot(e_LMS, 'LineWidth', 1.5);
xlabel('Iterations');
ylabel('Squared Error');
title('LMS Learning Curve');
grid on;

figure;
plot(e_NLMS, 'LineWidth', 1.5);
xlabel('Iterations');
ylabel('Squared Error');
title('NLMS Learning Curve');
grid on;

figure;
plot(e_VSLMS, 'LineWidth', 1.5);
xlabel('Iterations');
ylabel('Squared Error');
title('VS-LMS Learning Curve');
grid on;

%% BER
display(sprintf('BER 1 of LMS is %.2e', mean(BER_1_LMS) / (num_samples - ber_1_begin + 1)))
display(sprintf('BER 2 of LMS is %.2e', mean(BER_2_LMS) / (num_samples - ber_2_begin + 1)))
display(sprintf('BER 1 of NLMS is %.2e', mean(BER_1_NLMS) / (num_samples - ber_1_begin + 1)))
display(sprintf('BER 2 of NLMS is %.2e', mean(BER_2_NLMS) / (num_samples - ber_2_begin + 1)))
display(sprintf('BER 1 of VSLMS is %.2e', mean(BER_1_VSLMS) / (num_samples - ber_1_begin + 1)))
display(sprintf('BER 2 of VSLMS is %.2e', mean(BER_2_VSLMS) / (num_samples - ber_2_begin + 1)))

%% Squared Error
display(sprintf('Square Error 1 of LMS is %.2e', mean(SE_1_LMS) / (num_samples - se_1_begin + 1)))
display(sprintf('Square Error 2 of LMS is %.2e', mean(SE_2_LMS) / (num_samples - se_2_begin + 1)))
display(sprintf('Square Error 1 of NLMS is %.2e', mean(SE_1_NLMS) / (num_samples - se_1_begin + 1)))
display(sprintf('Square Error 2 of NLMS is %.2e', mean(SE_2_NLMS) / (num_samples - se_2_begin + 1)))
display(sprintf('Square Error 1 of VSLMS is %.2e', mean(SE_1_VSLMS) / (num_samples - se_1_begin + 1)))
display(sprintf('Square Error 2 of VSLMS is %.2e', mean(SE_2_VSLMS) / (num_samples - se_2_begin + 1)))