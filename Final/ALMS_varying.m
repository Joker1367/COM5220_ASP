% ==================== revised sections ====================
% errorCount1 = errorCount1 + (d1 > 1.4) + (d1 > 1.9)
% === === === === === === === === === === === === === === ===
clc;
clear;
close all;
load project_data2024.mat;
%% ==================== Time-Varying Channel Case ====================
trainLength = 50;
dataLength = 400;
totalLength = trainLength + dataLength;
set = length(data_varying_1) / totalLength;
% ==================== parameter setup ====================
L = 10;                                                         % filter length
alpha = 0.004;                                                   % Averaged LMS paremeter, step size
N = 25;                                                          % Averaged LMS paremeter
% === === === === === === === === === === === === === === ===
b1 = zeros(L,N);                                                % Averaged LMS, LPF
b2 = zeros(L,N);
f1 = zeros(L,1);                                                % filter coefficients
f2 = zeros(L,1);
y1_training = zeros(1, trainLength * set);                      % the training sequence of y(n)
y2_training = zeros(1, trainLength * set);
ans_varying_1 = zeros(1, dataLength * set);                     % recovered data, information for verifying answers with the TA
ans_varying_2 = zeros(1, dataLength * set);

data_varying_1_tmp = reshape(data_varying_1, totalLength, set).';
data_varying_2_tmp = reshape(data_varying_2, totalLength, set).';
e1_train = zeros(1, trainLength * set);
e2_train = zeros(1, trainLength * set);
e1 = zeros(1, totalLength * set);
e2 = zeros(1, totalLength * set);
for m = 1:set
    for n = 1:totalLength
        % ==================== Training Mode ====================
        if n <= trainLength
            if n<L
                x1 = [data_varying_1_tmp(m, n:-1:1) zeros(1, L-n)].';   % col vector of input data with dimension L
                x2 = [data_varying_2_tmp(m, n:-1:1) zeros(1, L-n)].';
            else
                x1 = (data_varying_1_tmp(m, n:-1:n-L+1)).';             % col vector of input data with dimension L
                x2 = (data_varying_2_tmp(m, n:-1:n-L+1)).';
            end
            y1 = (f1.') * x1;                                           % Chap05 > Complex LMS > (46) 
            y2 = (f2.') * x2;
            e1(1, (m-1)*totalLength + n) = trainseq_varying_1(1,n) - y1;
            e2(1, (m-1)*totalLength + n) = trainseq_varying_2(1,n) - y2;
            b1(:, 1:N-1) = b1(:, 2:N); b1(:, N) = e1(1, (m-1)*totalLength + n) * conj(x1);      % Chap05 > ALMS > (31)
            b2(:, 1:N-1) = b2(:, 2:N); b2(:, N) = e2(1, (m-1)*totalLength + n) * conj(x2);
            f1 = f1 + (alpha * mean(b1, 2));                                                    % Chap05 > ALMS > (37)
            f2 = f2 + (alpha * mean(b2, 2));

            y1_training(1, (m-1)*trainLength + n) = y1;                 % ???
            y2_training(1, (m-1)*trainLength + n) = y2;
            e1_train(1, (m-1)*trainLength + n) = trainseq_varying_1(1,n) - y1;
            e2_train(1, (m-1)*trainLength + n) = trainseq_varying_2(1,n) - y2;
        % ==================== Decision-Directed Mode ====================
        else
            x1 = (data_varying_1_tmp(m, n:-1:n-L+1)).';                 % col vector of input data with dimension L
            x2 = (data_varying_2_tmp(m, n:-1:n-L+1)).';
            y1 = (f1.') * x1;                                           % Chap05 > Complex LMS > (46) 
            y2 = (f2.') * x2;
            ans_varying_1(1, (m-1)*dataLength + n - trainLength) = QPSK_decision(y1);           % Decision Device
            ans_varying_2(1, (m-1)*dataLength + n - trainLength) = QPSK_decision(y2);
            e1(1, (m-1)*totalLength + n) = ans_varying_1(1, (m-1)*dataLength + n - trainLength) - y1;
            e2(1, (m-1)*totalLength + n) = ans_varying_2(1, (m-1)*dataLength + n - trainLength) - y2;
            b1(:, 1:N-1) = b1(:, 2:N); b1(:, N) = e1(1, (m-1)*totalLength + n) * conj(x1);      % Chap05 > ALMS > (31)
            b2(:, 1:N-1) = b2(:, 2:N); b2(:, N) = e2(1, (m-1)*totalLength + n) * conj(x2);
            f1 = f1 + (alpha * mean(b1, 2));                                                    % Chap05 > ALMS > (37)
            f2 = f2 + (alpha * mean(b2, 2));
        end
        % ==================== ==================== ====================
    end
end

% calculate the cost function J = e(n)^2
J1_training = abs(e1_train).*abs(e1_train);
J2_training = abs(e2_train).*abs(e2_train);
J1 = abs(e1).*abs(e1);
J2 = abs(e2).*abs(e2);

% ==================== BER for training mode ====================
errorCount1 = 0;
errorCount2 = 0;
for m = 1:set
    for n = 1:trainLength
        d1 = abs(QPSK_decision(y1_training((m-1)*trainLength + n)) - trainseq_varying_1(1,n));
        d2 = abs(QPSK_decision(y2_training((m-1)*trainLength + n)) - trainseq_varying_2(1,n));
        errorCount1 = errorCount1 + (d1 > 1.4) + (d1 > 1.9);
        errorCount2 = errorCount2 + (d2 > 1.4) + (d2 > 1.9);
    end
end

fprintf(' Time-Varying Channel Case BER1 is %s \n',errorCount1/2/(trainLength * set));
fprintf(' Time-Varying Channel Case BER2 is %s \n',errorCount2/2/(trainLength * set));

% ==================== plot ====================
intervalAmount = 100;
interval = ceil(trainLength * set / intervalAmount);
figure
plot(1:interval:(trainLength * set), J1_training(1:interval:(trainLength * set)))
title("Training error1 for Time-Varying Channel Case");
xlabel("bits");
ylabel("|e|^2");
figure
plot(1:interval:(trainLength * set), J2_training(1:interval:(trainLength * set)))
title("Training error2 for Time-Varying Channel Case");
xlabel("bits");
ylabel("|e|^2");

interval = ceil(totalLength * set / intervalAmount);
figure
subplot(2,1,1); 
plot(1:interval:(totalLength * set), J1(1:interval:(totalLength * set)))
title('Time-Varying Channel Case 1 :')
xlabel('index of bit (n)')
ylabel('|e(n)^2|')
subplot(2,1,2); 
plot(1:interval:(totalLength * set), J2(1:interval:(totalLength * set)))
title('Time-Varying Channel Case 2 :')
xlabel('index of bit (n)')
ylabel('|e(n)^2|')

% saved recovered data as ans_varying.mat
save(['ans_varying.mat'], 'ans_varying_1', 'ans_varying_2');