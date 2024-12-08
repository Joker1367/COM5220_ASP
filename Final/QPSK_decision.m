function result = QPSK_decision(y)

if (real(y) >= 0 && imag(y) >= 0)
    result = 1 + 1j;               % 1st Quadrant
elseif (real(y) < 0 && imag(y) >= 0)
    result = -1 + 1j;              % 2nd
elseif (real(y) >= 0 && imag(y) < 0)
    result = 1 - 1j;              % 3rd
else
    result = -1 - 1j;               % 4th
end
result = result / sqrt(2);
