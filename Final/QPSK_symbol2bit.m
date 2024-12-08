function [ans1, ans2] = QPSK_symbol2bit(input1, input2)

ans1 = zeros(1, 2*length(input1));
ans2 = zeros(1, 2*length(input2));

for i = 1 : length(input1)
    ans1(2 * i - 1) = (real(input1(i)) < 0);
    ans1(2 * i)     = (imag(input1(i)) < 0);
    ans2(2 * i - 1) = (real(input2(i)) < 0);
    ans2(2 * i)     = (imag(input2(i)) < 0);
end