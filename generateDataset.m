function status = generateDataset(rel_input_path, dims, n, N, N_test, varargin)

    % Generate a train and test dataset and the matrix of the n^dims 
    % combinations of eigenvalues, and save them in at the location
    % specified in rel_input_path.
    % Input arguments:
    %       - rel_input_path            [string]    Folder in which the output is
    %                                               saved.
    %       - dims                      [int]       Number of dimensions of the
    %                                               problem.
    %       - n                         [int]       Number of eigenvalues
    %                                               considered.
    %       - N                         [int]       Number of train samples.
    %       - N_test                    [int]       Number of test samples.
    %       - x_train_1D (optional)     [double]    Vector with train 
    %                                               samples in one dimension.
    %       - x_test_1D (optional)      [double]    Vector with test 
    %                                               samples in one dimension.
    % Output:
    %       - status                    [int]       0 if the operation
    %                                               succeded.

    % Creating train dataset
    if nargin > 5
        x_train_1D = varargin{1};
        if size(x_train_1D, 2) ~= dims
            error("The number of columns of x_train_1D must be equal to dims");
        end
        cell_x_train = mat2cell(x_train_1D, N, ones(1, dims));
    else
        x_train_1D = linspace(-pi/2, pi/2, N);
        [cell_x_train{1:dims}] = deal(x_train_1D);
    end
    [x_train_tmp{1:dims}] = ndgrid(cell_x_train{1:dims});
    x_train = reshape(cat(dims+1, x_train_tmp{:}), [], dims);
    y_train = sum(cos(x_train), 2);
    
    % Creating test dataset
    if nargin > 6
        x_test_1D = varargin{2};
        if size(x_train_1D, 2) ~= dims
            error("The number of columns of x_train_1D must be equal to dims");
        end
        cell_x_test = mat2cell(x_test_1D, N_test, ones(1, dims));
    else
        x_test_1D = linspace(-pi/2, pi/2, N_test);
        [cell_x_test{1:dims}] = deal(x_test_1D);
    end
    [x_test_tmp{1:dims}] = ndgrid(cell_x_test{1:dims});
    x_test = reshape(cat(dims+1, x_test_tmp{:}), [], dims);
    y_test= sum(cos(x_test), 2);
    
    % Compute all the possible combinations of eigenvalues
    eigen_comb = combinationRepeated(1:n, dims);
    
    % Save matrices at the given location
    if exist('__octave_config_info__', 'builtin')
        csvwrite([rel_input_path, "x_train.csv"], x_train)
        csvwrite([rel_input_path, "y_train.csv"], y_train)
        csvwrite([rel_input_path, "x_test.csv"], x_test)
        csvwrite([rel_input_path, "y_test.csv"], y_test)
        csvwrite([rel_input_path, "eigen_comb.csv"], eigen_comb)
    else
        writematrix(x_train, rel_input_path + "x_train.csv")
        writematrix(y_train, rel_input_path + "y_train.csv")
        writematrix(x_test, rel_input_path + "x_test.csv")
        writematrix(y_test, rel_input_path + "y_test.csv")
        writematrix(eigen_comb, rel_input_path + "eigen_comb.csv")
    end
    
    status = 0;

end
