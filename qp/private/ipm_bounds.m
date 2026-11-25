function [lb, ub, hl, hu] = ipm_bounds(lb, ub)
% Masks of the finite bounds; +-Inf entries set to 0 (they are masked out of the interior-point machinery)
    hl = isfinite(lb);  hu = isfinite(ub);
    lb(~hl) = 0;        ub(~hu) = 0;
end
