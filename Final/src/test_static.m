clear all;
clc;
close all;

load '../data/project_data2024.mat';

%% System Specification
train_length = 1000;
data_length = 200000;
total_length = train_length + data_length;
total_operation = length(data_static_1) / total_length;

L_max = 20;              % filter length
alpha = 9.45e-3;    % step size
N_max = 20;             % MAF fileter length

BER_1 = zeros(N_max, L_max);
BER_2 = zeros(N_max, L_max);
evm_1 = zeros(N_max, L_max);
evm_2 = zeros(N_max, L_max);

for L = 1 : L_max
    for N = 1 : N_max
        f_1 = zeros(L, 1);
        f_2 = zeros(L, 1);
        b_1 = zeros(L,N);                                                   % Averaged LMS, LPF
        b_2 = zeros(L,N);
        e_1_train = zeros(1, train_length * total_operation);
        e_2_train = zeros(1, train_length * total_operation);
        e_1 = zeros(1, total_length * total_operation);
        e_2 = zeros(1, total_length * total_operation);
        x_1 = zeros(L, 1);
        x_2 = zeros(L, 1);
        result_1 = zeros(1, data_length * total_operation);
        result_2 = zeros(1, data_length * total_operation);
        train_1  = zeros(1, train_length * total_operation);
        train_2  = zeros(1, train_length * total_operation);
        
        idx_total = 1;
        idx_train = 1;
        idx_data  = 1;

        evm_tmp_1 = 0;
        evm_tmp_2 = 0;
        
        for operation = 1 : total_operation
            x_1 = zeros(L, 1);
            x_2 = zeros(L, 1); 
            for n = 1 : total_length
                x_1 = [data_static_1(idx_total) ; x_1(1:L-1)];                          % col vector of input data with dimension L
                x_2 = [data_static_2(idx_total) ; x_2(1:L-1)]; 
            
                y_1 = (f_1.') * x_1;                                            % output of adaptive filter
                y_2 = (f_2.') * x_2;
                if n <= train_length                                            % compute the error
                    e_1(idx_total) = trainseq_static_1(n) - y_1;                        % Training Mode
                    e_2(idx_total) = trainseq_static_2(n) - y_2;
                    e_1_train(idx_train) = trainseq_static_1(n) - y_1;
                    e_2_train(idx_train) = trainseq_static_2(n) - y_2;
        
                    train_1(idx_train) = QPSK_decision(y_1);
                    train_2(idx_train) = QPSK_decision(y_2);
        
                    idx_train = idx_train + 1;
                else
                    e_1(idx_total) = QPSK_decision(y_1) - y_1;                          % Decision Directed Mode
                    e_2(idx_total) = QPSK_decision(y_2) - y_2;

                    evm_tmp_1 = evm_tmp_1 + abs(e_1(idx_total))^2;
                    evm_tmp_2 = evm_tmp_2 + abs(e_2(idx_total))^2;
        
                    result_1(idx_data) = QPSK_decision(y_1);
                    result_2(idx_data) = QPSK_decision(y_2);
        
                    idx_data = idx_data + 1;
                end 
                b_1(:, 1:N-1) = b_1(:, 2:N); b_1(:, N) = e_1(idx_total) * conj(x_1);    % Chap05 > ALMS > (31) 
                b_2(:, 1:N-1) = b_2(:, 2:N); b_2(:, N) = e_2(idx_total) * conj(x_2);
                f_1 = f_1 + alpha * mean(b_1, 2);                               % Linear smoothing
                f_2 = f_2 + alpha * mean(b_2, 2);
        
                idx_total = idx_total + 1;
            end
        end
        [train_result_1, train_result_2] = QPSK_symbol2bit(train_1, train_2);
        [train_ans_1, train_ans_2] = QPSK_symbol2bit(trainseq_static_1, trainseq_static_2);
        train_ans_1 = repmat(train_ans_1, 1, total_operation);
        train_ans_2 = repmat(train_ans_2, 1, total_operation);
        BER_1(N, L) = sum(train_result_1 ~= train_ans_1) / (2*train_length*total_operation);
        BER_2(N, L) = sum(train_result_2 ~= train_ans_2) / (2*train_length*total_operation);
        evm_1(N, L) = sqrt(evm_tmp_1 / (data_length * total_operation)) * 100;
        evm_2(N, L) = sqrt(evm_tmp_2 / (data_length * total_operation)) * 100;
    end
end
 
BER = BER_1 + BER_2 / 2;
save(['../performance/BER_static.mat'], 'BER_1', 'BER_2', 'BER');
save(['../performance/evm_static.mat'], 'evm_1', 'evm_2');

%% BER
load ../performance/BER_static.mat
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

%% EVM
load ../performance/evm_static.mat
[min_val, linear_idx] = min(evm_1(:));
[row, col] = ind2sub(size(evm_1), linear_idx);
disp([row col]);

figure;
hold on;
surf(evm_1);
colorbar;

%view(45, 30); % 調整觀察角度 (方位角 45°, 仰角 30°)
xlabel('N');
ylabel('L');
zlabel('EVM');
title('Static');

[min_val, linear_idx] = min(evm_2(:));
[row, col] = ind2sub(size(evm_2), linear_idx);
disp([row col]);

figure;
hold on;
surf(evm_2);
colorbar;

%view(45, 30); % 調整觀察角度 (方位角 45°, 仰角 30°)
xlabel('N');
ylabel('L');
zlabel('EVM');
title('Static');