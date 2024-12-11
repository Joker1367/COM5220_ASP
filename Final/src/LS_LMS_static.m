clear all;
clc;
close all;

load '../data/project_data2024.mat';

%% System Specification
train_length = 1000;
data_length = 200000;
total_length = length(data_static_1);

L = 9;              % filter length
alpha = 9.45e-3;    % step size
N = 15;             % MAF fileter length

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

%% ==================== plot ====================
J1_training = abs(e_1(1, 1:train_length)).*abs(e_1(1, 1:train_length));  % |e(n)|^2
J2_training = abs(e_2(1, 1:train_length)).*abs(e_2(1, 1:train_length));
J1 = abs(e_1).*abs(e_1);
J2 = abs(e_2).*abs(e_2);

intervalAmount = 200;
interval = ceil(train_length / intervalAmount);
figure
plot(1:interval:train_length, J1_training(1:interval:train_length))
title("Training error1 for Static Channel Case");
xlabel("bits");
ylabel("|e|^2");
figure
plot(1:interval:train_length, J2_training(1:interval:train_length))
title("Training error2 for Static Channel Case");
xlabel("bits");
ylabel("|e|^2");

interval = ceil(total_length / intervalAmount);
figure
subplot(2,1,1); 
plot(1:interval:total_length, J1(1:interval:total_length))
title('Static Channel Case 1 :')
xlabel('index of bit (n)')
ylabel('|e(n)^2|')
subplot(2,1,2); 
plot(1:interval:total_length, J2(1:interval:total_length))
title('Static Channel Case 2 :')
xlabel('index of bit (n)')
ylabel('|e(n)^2|')

%% ==================== BER for training mode ====================
[train_result_1, train_result_2] = QPSK_symbol2bit(result_1(1:train_length), result_2(1:train_length));
[train_ans_1, train_ans_2] = QPSK_symbol2bit(trainseq_static_1(1:train_length), trainseq_static_2(1:train_length));
errorCount1 = sum(train_result_1 ~= train_ans_1);
errorCount2 = sum(train_result_2 ~= train_ans_2);

fprintf('Static Channel Case BER1 is %.5e \n',errorCount1/(2*train_length));
fprintf('Static Channel Case BER2 is %.5e \n',errorCount2/(2*train_length));

%% ==================== Save the result ====================
[ans_static_1, ans_static_2] = QPSK_symbol2bit(result_1(train_length+1:total_length), result_2(train_length+1:total_length));
save(['../ans/ans_static.mat'], 'ans_static_1', 'ans_static_2');