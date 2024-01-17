function out = combinationRepeated(vect_in, dims)

    % Generate a matrix containing dims combination with repetitions of the
    % elements of vect_in.
    % Input arguments:
    %       - vect_in       [double]    Vector whose elements will be
    %                                   combined dims times.
    %       - dims          [int]       Number of combinations
    
    idx = dims:-1:1;
    vect_cell = cell(dims, 1);
    for ii = 1:dims
        vect_cell(ii) = {vect_in};
    end
    [out{idx}] = ndgrid(vect_cell{idx});
    out = reshape(cat(dims+1, out{:}), [], dims);

end