function ok = ipm_acceptable(res_stat, res_eq, res_comp, opts)
% IPOPT's acceptable point: every residual within TolAcceptable of its tolerance
    k = opts.TolAcceptable;
    ok = res_stat <= k * opts.TolStat && res_eq <= k * opts.TolEq && res_comp <= k * opts.TolComp;
end
