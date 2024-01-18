function out_matrix = combinationRepeated(vect_in, dims)

    % Generate a matrix containing dims combination with repetitions of the
    % elements of vect_in. The matrix looks like:
    %       ⌈ 1  1  1 ⌉
    %       │ 1  1  2 │
    %       │ 1  2  1 │
    %       │ :  :  : │
    %       ⌊ 2  2  2 ⌋
    % Input arguments:
    %       - vect_in       [double]    Vector whose elements will be
    %                                   combined dims times.
    %       - dims          [int]       Number of combinations
    % Output:
    %       - out_matrix    [double]    Matrix whose rows contain the
    %                                   combinations.

    idx = dims:-1:1;
    vect_cell = cell(dims, 1);
    for ii = 1:dims
        vect_cell(ii) = {vect_in};
    end
    [out_matrix{idx}] = ndgrid(vect_cell{idx});
    out_matrix = reshape(cat(dims+1, out_matrix{:}), [], dims);

end