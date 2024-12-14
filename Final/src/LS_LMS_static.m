clear all;
clc;
close all;

load '../data/project_data2024.mat';

%% System Specification
train_length = 1000;
data_length = 200000;
total_length = train_length + data_length;
total_operation = length(data_static_1) / total_length;

L = 26;              % filter length
alpha = 9e-3;    % step size
N = 20;             % MAF fileter length

f_1 = zeros(L, 1);
f_2 = zeros(L, 1);
b_1 = zeros(L,N);                                                   % Averaged LMS, LPF
b_2 = zeros(L,N);
e_1_train = zeros(1, train_length * total_operation);
e_2_train = zeros(1, train_length * total_operation);
e_1_data = zeros(1, data_length * total_operation);
e_2_data = zeros(1, data_length * total_operation);
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
            e_1_data(idx_total) = QPSK_decision(y_1) - y_1;             % Decision Directed Mode
            e_2_data(idx_total) = QPSK_decision(y_2) - y_2;
            e_1(idx_total) = QPSK_decision(y_1) - y_1;                          % Decision Directed Mode
            e_2(idx_total) = QPSK_decision(y_2) - y_2;

            result_1(idx_data) = QPSK_decision(y_1);
            result_2(idx_data) = QPSK_decision(y_2);

            idx_data = idx_data + 1;
        end 
        b_1(:, 1:N-1) = b_1(:, 2:N); b_1(:, N) = e_1(idx_total) * conj(x_1);     
        b_2(:, 1:N-1) = b_2(:, 2:N); b_2(:, N) = e_2(idx_total) * conj(x_2);
        f_1 = f_1 + alpha * mean(b_1, 2);                                       % Linear smoothing
        f_2 = f_2 + alpha * mean(b_2, 2);

        idx_total = idx_total + 1;
    end
end

%% ==================== plot ====================
J1_training = abs(e_1_train).*abs(e_1_train);  % |e(n)|^2
J2_training = abs(e_2_train).*abs(e_2_train);
J1_data = abs(e_1_data).*abs(e_1_data);
J2_data = abs(e_2_data).*abs(e_2_data);
J1 = abs(e_1).*abs(e_1);
J2 = abs(e_2).*abs(e_2);


n = 1:length(J1);          
evm_1 = sqrt(cumsum(J1) ./ n) * 100;
evm_2 = sqrt(cumsum(J2) ./ n) * 100;

% training square error
% intervalAmount = 200;
% interval = ceil(train_length * total_operation / intervalAmount);
% figure
% plot(1:interval:train_length * total_operation, J1_training(1:interval:train_length * total_operation))
% title("Training error for Static Channel Case");
% xlabel("bits");
% ylabel("|e|^2");
% figure
% plot(1:interval:train_length * total_operation, J2_training(1:interval:train_length * total_operation))
% title("Training error for Static Channel Case");
% xlabel("bits");
% ylabel("|e|^2");

% data square error
intervalAmount = 200;
interval = ceil(data_length * total_operation / intervalAmount);
figure
plot(1:interval:data_length * total_operation, J1_data(1:interval:data_length * total_operation))
title("Data Square error for Static Channel Case 1");
xlabel("symbols");
ylabel("|e|^2");
figure
plot(1:interval:data_length * total_operation, J2_data(1:interval:data_length * total_operation))
title("Data Square error for Static Channel Case 2");
xlabel("symbols");
ylabel("|e|^2");

% total square error
% interval = ceil(total_length * total_operation / intervalAmount);
% figure
% subplot(2,1,1); 
% plot(1:interval:total_length * total_operation, J1(1:interval:total_length * total_operation))
% title('Square Error Static Channel Case 1 :')
% xlabel('index of bit (n)')
% ylabel('evm')
% subplot(2,1,2); 
% plot(1:interval:total_length * total_operation, J2(1:interval:total_length * total_operation))
% title('Square Error Static Channel Case 2 :')
% xlabel('index of bit (n)')
% ylabel('evm')
% 
% interval = ceil(total_length * total_operation / intervalAmount);
% figure
% subplot(2,1,1); 
% plot(1:interval:total_length * total_operation, evm_1(1:interval:total_length * total_operation))
% title('EVM Static Channel Case 1 :')
% xlabel('index of bit (n)')
% ylabel('evm')
% subplot(2,1,2); 
% plot(1:interval:total_length * total_operation, evm_2(1:interval:total_length * total_operation))
% title('EVM Static Channel Case 2 :')
% xlabel('index of bit (n)')
% ylabel('evm')

%% ==================== BER for training mode ====================
[train_result_1, train_result_2] = QPSK_symbol2bit(train_1, train_2);
[train_ans_1, train_ans_2] = QPSK_symbol2bit(trainseq_static_1, trainseq_static_2);
train_ans_1 = repmat(train_ans_1, 1, total_operation);
train_ans_2 = repmat(train_ans_2, 1, total_operation);
errorCount1 = sum(train_result_1 ~= train_ans_1);
errorCount2 = sum(train_result_2 ~= train_ans_2);

fprintf('Static Channel Case 1 EVM is %.2f %% \n',evm_1(total_length*total_operation));
fprintf('Static Channel Case 2 EVM is %.2f %% \n',evm_2(total_length*total_operation));
fprintf('Static Channel Case 1 BER is %.5e \n',errorCount1/(2*train_length*total_operation));
fprintf('Static Channel Case 2 BER is %.5e \n',errorCount2/(2*train_length*total_operation));

%% ==================== Save the result ====================
[ans_static_1, ans_static_2] = QPSK_symbol2bit(result_1, result_2);
save(['../ans/ans_static_1.mat'], 'ans_static_1');
%save(['../ans/ans_static_2.mat'], 'ans_static_2');