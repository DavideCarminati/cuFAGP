%% Example
% The script replicates the results in the paper
clear
clc

% N = [ 10000, 100, 21, 10 ];
N = [ 10000, 100, 10 ];
% N = 5;
N_test = 1;
% n = [10, 11, 12];%[ 2, 3, 4, 5, 6, 7, 8, 9 ];%, 6, 7, 8, 9, 10 ];
n = [ 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 ];
% p = [ 1, 2, 3, 4 ];
p = [ 1, 2, 4 ];

% generateDataset(['input_matrices_', num2str(69), '/'], p(2), n(1), N(2), N_test, rand(N(2), p(2)), rand(N_test, p(2)));

count = 0;
for kk = 1:length(p)
    for ii = 1:length(n)
        mkdir(['input_matrices_', num2str(count),  '/']);
        generateDataset(['input_matrices_', num2str(count), '/'], p(kk), n(ii), N(kk), N_test);
%         generateDataset(['input_matrices_', num2str(count), '/'], p(kk), n(ii), N, N_test);
        count = count + 1;
    end
end
