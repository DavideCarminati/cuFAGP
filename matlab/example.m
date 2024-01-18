%% Example script
% ========================================================================
% | The script replicates the train and test datasets used in the paper. |
% ========================================================================
%
% It generates a dataset for each number of eigenvalues n and number of
% sample dimensions, along with the matrix of all the eigenvalues 
% combinations. Each dataset is saved in a different folder.
close all
clear
clc

N = [ 10000, 100, 10 ];                     % Number of samples used for training
N_test = 1;                                 % Number of samples used for testing
n = [ 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 ];     % Number of considered eigenvalues
p = [ 1, 2, 4 ];                            % Number of sample dimensions

count = 0;
for kk = 1:length(p)
    for ii = 1:length(n)
        mkdir(['../input_matrices_', num2str(count),  '/']);
        generateDataset(['../input_matrices_', num2str(count), '/'], p(kk), n(ii), N(kk), N_test);
        count = count + 1;
    end
end
