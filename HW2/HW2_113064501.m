clear all
clc;

%% (a)
M = 100;
x = randn(1, M);

r_unbiased = [];
r_biased = [];

for m = -M+1 : M-1
    cur = 0;
    for n = 1 : M - abs(m)
        cur = cur + x(n) * x(n + abs(m));
    end
    r_unbiased = [r_unbiased cur / (M - abs(m))];
    r_biased = [r_biased cur / M];
end

delta = linspace(-M+1, M-1, 2*M-1);

figure(1)
subplot(2, 1, 1); 
grid on; 
hold on;
stem(delta, r_unbiased, 'b')
xlabel('m');
ylabel('autocorrelation');
title('Unbiased');

subplot(2, 1, 2); 
grid on; 
hold on;
stem(delta, r_biased, 'r')
xlabel('m');
ylabel('autocorrelation');
title('Biased');

%% (b)
R_unbiased_25 = toeplitz(r_unbiased(M:M+24));
R_unbiased_100 = toeplitz(r_unbiased(M: 2*M-1));

R_biased_25 = toeplitz(r_biased(M:M+24));
R_biased_100 = toeplitz(r_biased(M: 2*M-1));


eigen_R_unbiased_25 = eig(R_unbiased_25);
eigen_R_unbiased_100 = eig(R_unbiased_100);
eigen_R_biased_25 = eig(R_biased_25);
eigen_R_biased_100 = eig(R_biased_100);

figure(2)
subplot(2, 2, 1); 
stem(eigen_R_unbiased_25, 'filled');
title('Unbiased 25');
xlabel('Index');
ylabel('Eigenvalue');
grid on;

subplot(2, 2, 2);
stem(eigen_R_unbiased_100, 'filled');
title('Unbiased 100');
xlabel('Index');
ylabel('Eigenvalue');
grid on;

subplot(2, 2, 3);
stem(eigen_R_biased_25, 'filled');
title('Biased 25');
xlabel('Index');
ylabel('Eigenvalue');
grid on;

subplot(2, 2, 4);
stem(eigen_R_biased_100, 'filled');
title('Biased 100');
xlabel('Index');
ylabel('Eigenvalue');
grid on;
