%% Plots
% Aggregate results and plot CPU and GPU mean execution times for each sample
% dimensions when varying the number of eigenvalues.

clear
clc
set(0,'defaultTextInterpreter','latex');
set(0, 'defaultAxesTickLabelInterpreter','latex');

%% Import .csv log file

if exist('__octave_config_info__', 'builtin')
    % If I'm running in Octave
    fileID = fopen("../log_cpu.csv");
    C = textscan(fileID,'%f %f %f %f %f %s',...
    'Delimiter',',','EmptyValue',NaN, 'HeaderLines', 1);
    fclose(fileID);
    log_cpu = cell2mat(C(1:5));

    fileID = fopen("../log_gpu.csv");
    C = textscan(fileID,'%f %f %f %f %f %s',...
    'Delimiter',',','EmptyValue',NaN, 'HeaderLines', 1);
    fclose(fileID);
    log_gpu = cell2mat(C(1:5));
else
    % If I'm running in Matlab
    set(0, 'defaultLegendInterpreter','latex');

    log_cpu = readmatrix("../log_cpu.csv", 'FileType', 'text', 'Delimiter', ',');
    log_cpu = log_cpu(:, 1:5);
    log_gpu = readmatrix("../log_gpu.csv", 'FileType', 'text', 'Delimiter', ',');
    log_gpu = log_gpu(:, 1:5);
end

%% Aggregating and plotting

% Averaging the runtimes for each number of considered eigenvalues
for dims = 1:max(log_cpu(:, 4))
    % For each considered sample dimension, average the runtime of each
    % number of considered eigenvalues
    dims_mask = log_cpu(:, 4) == dims;
    if sum(dims_mask) == 0 
        continue; 
    end
    count = 1;
    mean_time_cpu = [];
    n_eig_cpu = [];
    for n_eig = 1:max(log_cpu(dims_mask, 5))
        % For each number of considered eigenvalues, average the CPU runtimes
        if sum(log_cpu(dims_mask, 5) == n_eig) > 0
            eig_mask = log_cpu(:, 5) == n_eig;
            mean_time_cpu(count) = mean(log_cpu(eig_mask & dims_mask), 1);
            n_eig_cpu(count) = n_eig;
            count = count + 1;
        end
    end
    count = 1;
    mean_time_gpu = [];
    n_eig_gpu = [];
    for n_eig = 1:max(log_gpu(dims_mask, 5))
        % For each number of considered eigenvalues, average the GPU runtimes
        if sum(log_gpu(dims_mask, 5) == n_eig) > 0
            eig_mask = log_gpu(:, 5) == n_eig;
            mean_time_gpu(count) = mean(log_gpu(eig_mask & dims_mask), 1);
            n_eig_gpu(count) = n_eig;
            count = count + 1;
        end
    end

    figure
    plot(n_eig_cpu, mean_time_cpu, 'o-')
    hold on, grid on
    plot(n_eig_gpu, mean_time_gpu, '-o')
    legend('CPU', 'GPU', 'Location','best')
    xlabel("Number of eigenvalues $n$")
    ylabel("Time [ms]")
    xlim([min(min(n_eig_cpu), min(n_eig_gpu)), max(max(n_eig_cpu), max(n_eig_gpu))])
    title([ num2str(dims), " dimensions"])
end

disp("Press any key to quit.")
pause;
