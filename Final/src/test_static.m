clear all;
clc;
close all;

load 'project_data2024.mat';

%% System Specification
train_length = 1000;
data_length = 200000;
total_length = length(data_static_1);

L_max = 20;             % filter length
alpha = 1e-2;       % step size
N_max = 20;             % MAF fileter length

BER_1 = zeros(N_max, L_max);
BER_2 = zeros(N_max, L_max);

for N = 1 : N_max
    for L = 1 : L_max
        f_1 = zeros(L, 1);
        f_2 = zeros(L, 1);
        b_1 = zeros(L,N);                                                   % Averaged LMS, LPF
        b_2 = zeros(L,N);
        e_1 = zeros(1, total_length);
        e_2 = zeros(1, total_length);
        x_1 = zeros(L, 1);
        x_2 = zeros(L, 1);
        result_1 = zeros(1, total_length);
        result_2 = zeros(1, total_length);
    
        for n = 1 : total_length
            x_1 = [data_static_1(n) ; x_1(1:L-1)];                          % col vector of input data with dimension L
            x_2 = [data_static_2(n) ; x_2(1:L-1)]; 
        
            y_1 = (f_1.') * x_1;                                            % output of adaptive filter
            y_2 = (f_2.') * x_2;
            if n <= train_length                                            % compute the error
                e_1(n) = trainseq_static_1(n) - y_1;                        % Training Mode
                e_2(n) = trainseq_static_2(n) - y_2;
            else
                e_1(n) = QPSK_decision(y_1) - y_1;                          % Decision Directed Mode
                e_2(n) = QPSK_decision(y_2) - y_2;
            end 
            b_1(:, 1:N-1) = b_1(:, 2:N); b_1(:, N) = e_1(n) * conj(x_1);    % Chap05 > ALMS > (31) 
            b_2(:, 1:N-1) = b_2(:, 2:N); b_2(:, N) = e_2(n) * conj(x_2);
            f_1 = f_1 + alpha * mean(b_1, 2);                               % Linear smoothing
            f_2 = f_2 + alpha * mean(b_2, 2);
            result_1(n) = QPSK_decision(y_1);
            result_2(n) = QPSK_decision(y_2);
        end
    
        %% ==================== BER for training mode ====================
        [train_result_1, train_result_2] = QPSK_symbol2bit(result_1(1:train_length), result_2(1:train_length));
        [train_ans_1, train_ans_2] = QPSK_symbol2bit(trainseq_static_1(1:train_length), trainseq_static_2(1:train_length));
        errorCount1 = sum(train_result_1 ~= train_ans_1);
        errorCount2 = sum(train_result_2 ~= train_ans_2);
    
        BER_1(N, L) = errorCount1/(2*train_length);
        BER_2(N, L) = errorCount2/(2*train_length);
    
    end
end

 
BER = BER_1 + BER_2 / 2;
save(['BER_static.mat'], 'BER_1', 'BER_2', 'BER');

%%
load BER_static.mat
[min_val, linear_idx] = min(BER(:));
[row, col] = ind2sub(size(BER), linear_idx);
disp([row col]);

figure;
hold on;
surf(BER);

view(45, 30); % 調整觀察角度 (方位角 45°, 仰角 30°)
xlabel('X-axis');
ylabel('Y-axis');
zlabel('Z-axis');
title('Static');