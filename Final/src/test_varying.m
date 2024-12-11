clear all;
clc; 
close all;

load 'project_data2024.mat';

train_length = 50;
data_length = 400;
total_length = train_length + data_length;
total_operation = length(data_varying_1) / total_length;

L_max = 20;              % filter length
alpha = 9e-3;    % step size
N_max = 30;             % MAF fileter length

BER_1 = zeros(N_max, L_max);
BER_2 = zeros(N_max, L_max);

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
        
        for operation = 1 : total_operation
            x_1 = zeros(L, 1);
            x_2 = zeros(L, 1); 
            for n = 1 : total_length
                x_1 = [data_varying_1(idx_total) ; x_1(1:L-1)];                          % col vector of input data with dimension L
                x_2 = [data_varying_2(idx_total) ; x_2(1:L-1)]; 
            
                y_1 = (f_1.') * x_1;                                            % output of adaptive filter
                y_2 = (f_2.') * x_2;
                if n <= train_length                                            % compute the error
                    e_1(idx_total) = trainseq_varying_1(n) - y_1;                        % Training Mode
                    e_2(idx_total) = trainseq_varying_2(n) - y_2;
                    e_1_train(idx_train) = trainseq_varying_1(n) - y_1;
                    e_2_train(idx_train) = trainseq_varying_2(n) - y_2;
        
                    train_1(idx_train) = QPSK_decision(y_1);
                    train_2(idx_train) = QPSK_decision(y_2);
        
                    idx_train = idx_train + 1;
                else
                    e_1(idx_total) = QPSK_decision(y_1) - y_1;                          % Decision Directed Mode
                    e_2(idx_total) = QPSK_decision(y_2) - y_2;
        
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
        [train_ans_1, train_ans_2] = QPSK_symbol2bit(trainseq_varying_1, trainseq_varying_2);
        train_ans_1 = repmat(train_ans_1, 1, total_operation);
        train_ans_2 = repmat(train_ans_2, 1, total_operation);
        BER_1(N, L) = sum(train_result_1 ~= train_ans_1) / (2*train_length*total_operation);
        BER_2(N, L) = sum(train_result_2 ~= train_ans_2) / (2*train_length*total_operation);
    end
end

BER = BER_1 + BER_2 / 2;
save(['BER_varying.mat'], 'BER_1', 'BER_2', 'BER');

%%
load BER_varying.mat
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
title('varying');