function ocp = ocp_qp_broadcast_bounds(ocp)
% Bounds given as a single column (or a scalar) are held over the horizon: lb, ub to nu x N, lbx, ubx to nx x (N+1), lg, ug to ng x (N+1)
    [nx, nu, N] = size(ocp.B);
    ocp = expand(ocp, {'lb', 'ub'}, nu, N);
    ocp = expand(ocp, {'lbx', 'ubx'}, nx, N+1);
    if isfield(ocp, 'Cg') && ~isempty(ocp.Cg), ocp = expand(ocp, {'lg', 'ug'}, size(ocp.Cg, 1), N+1); end
end

function ocp = expand(ocp, names, n, K)
    for k = 1:numel(names)
        if isfield(ocp, names{k}) && ~isempty(ocp.(names{k})), ocp.(names{k}) = ocp.(names{k}) .* ones(n, K); end
    end
end
