function [dtl, dtu, dll, dlu] = ipm_bound_steps(dv, rl, ru, rcl, rcu, tl, tu, ll, lu, hl, hu)
% Slack and multiplier steps of a two-sided bound set, given the primal step dv (interior-point QP solvers)
    dtl = ( dv + rl) .* hl;
    dtu = (-dv + ru) .* hu;
    dll = (rcl - ll .* dtl) ./ tl;
    dlu = (rcu - lu .* dtu) ./ tu;
end
