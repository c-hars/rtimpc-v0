function res = ocp_nlp_res_compute(qp, dual, uscale, xscaled)
%OCP_NLP_RES_COMPUTE Compute the KKT residuals at an SQP iterate, from its OCP-structured QP.
%   res = OCP_NLP_RES_COMPUTE(qp, dual) evaluates the KKT conditions of the QP qp (in OCP_QP_RICCATI_IPM's format) at the zero step, x_1 = x0 with all other states and inputs 0, for the multipliers in dual. For the QP of an SQP iteration, built at the current iterate, these are the residuals of the NLP at that iterate, as acados's ocp_nlp_res_compute.
%
%   res is a structure with
%     stat   inf-norm of the Lagrangian gradient (states of nodes 2..N+1, inputs, soft slacks)
%     eq     inf-norm of the dynamics gaps (and of the equality rows Eq1)
%     ineq   largest violation of a hard bound or constraint (soft ones are absorbed by their slacks)
%     comp   inf-norm of multiplier x margin
%
%   dual holds pi (nx,N+1), lam_lb, lam_ub (nu,N) and, as present, lam_lbx, lam_ubx (nx,N+1), lam_lg, lam_ug (ng,N+1) and nu_eq1 (ne,1); missing fields are zero.
%
%   res = OCP_NLP_RES_COMPUTE(qp, dual, uscale, xscaled) gives the residuals in physical units when the QP's inputs are scaled, u = uscale .* u_qp, and so are the states with row indices xscaled.
%
%   A few notes:
%     - x_1 is fixed and eliminated. acados keeps it, with an equality bound whose multiplier absorbs its stationarity row; the conditions are the same.
%     - The cost is the QP's: rtimpc's J/2, as acados's.
%
%   See also OCP_QP_RICCATI_IPM.

    [nx, nu, N] = size(qp.B);
    if nargin < 3 || isempty(uscale), uscale = ones(nu, 1); end
    if nargin < 4, xscaled = []; end
    ng = 0;  if isfield(qp, 'Cg') && ~isempty(qp.Cg), ng = size(qp.Cg, 1); end
    get = @(f, sz) field_or_zero(dual, f, sz);
    pi_ = get('pi', [nx, N+1]);
    lam_lb = get('lam_lb', [nu, N]);  lam_ub = get('lam_ub', [nu, N]);
    lam_lbx = get('lam_lbx', [nx, N+1]);  lam_ubx = get('lam_ubx', [nx, N+1]);
    x0 = qp.x0;
    S = zeros(nu, nx, N);  if isfield(qp, 'S') && ~isempty(qp.S), S = qp.S; end

    % stationarity (inputs; states of nodes 2..N+1)
    res_u = qp.r - lam_lb + lam_ub;
    res_u(:,1) = res_u(:,1) + S(:,:,1) * x0;
    res_u = res_u + reshape(pagemtimes(qp.B, 'transpose', reshape(pi_(:,2:N+1), nx, 1, N), 'none'), nu, N);
    res_x = qp.q(:,2:N+1) - pi_(:,2:N+1) - lam_lbx(:,2:N+1) + lam_ubx(:,2:N+1);
    if N > 1
        res_x(:,1:N-1) = res_x(:,1:N-1) + reshape(pagemtimes(qp.A(:,:,2:N), 'transpose', reshape(pi_(:,3:N+1), nx, 1, N-1), 'none'), nx, N-1);
    end
    res_e = zeros(0, 1);
    if isfield(qp, 'Eq1') && ~isempty(qp.Eq1)
        res_u(:,1) = res_u(:,1) + qp.Eq1' * get('nu_eq1', [size(qp.Eq1, 1), 1]);
        res_e = qp.eq1;   % Eq1 * 0 - eq1, sign irrelevant for the norm
    end

    % row values at the zero step: inputs 0, states 0 except x_1 = x0
    xv = zeros(nx, N+1);  xv(:,1) = x0;
    res_s = zeros(0, 1);  viol = zeros(0, 1);  comp = zeros(0, 1);

    % input bounds (always hard)
    [v, c] = bound_terms(zeros(nu, N), qp.lb, qp.ub, lam_lb, lam_ub, Inf, Inf, 0, 0, uscale);
    viol = [viol; v];  comp = [comp; c];
    % state bounds, nodes 2..N+1 (column 1 ignored, as the solver)
    if isfield(qp, 'lbx') && ~isempty(qp.lbx)
        sc = ones(nx, 1);  if ~isempty(xscaled), sc(xscaled) = uscale; end
        [zl, zu, Zl, Zu] = soft_weights(qp, 'zlx', 'zux', 'Zlx', 'Zux', nx, N+1);
        k = 2:N+1;
        [v, c, s] = bound_terms(xv(:,k), qp.lbx(:,k), qp.ubx(:,k), lam_lbx(:,k), lam_ubx(:,k), zl(:,k), zu(:,k), Zl(:,k), Zu(:,k), sc);
        viol = [viol; v];  comp = [comp; c];  res_s = [res_s; s];
    end
    % general rows, nodes 1..N+1 (node N+1: Cg x only)
    if ng > 0
        lam_lg = get('lam_lg', [ng, N+1]);  lam_ug = get('lam_ug', [ng, N+1]);
        dl = lam_ug - lam_lg;
        gv = reshape(pagemtimes(qp.Cg, reshape(xv, nx, 1, N+1)), ng, N+1);   % inputs are 0
        res_u = res_u + reshape(pagemtimes(qp.Dg(:,:,1:N), 'transpose', reshape(dl(:,1:N), ng, 1, N), 'none'), nu, N);
        res_x = res_x + reshape(pagemtimes(qp.Cg(:,:,2:N+1), 'transpose', reshape(dl(:,2:N+1), ng, 1, N), 'none'), nx, N);
        [zl, zu, Zl, Zu] = soft_weights(qp, 'zlg', 'zug', 'Zlg', 'Zug', ng, N+1);
        [v, c, s] = bound_terms(gv, qp.lg, qp.ug, lam_lg, lam_ug, zl, zu, Zl, Zu, ones(ng, 1));
        viol = [viol; v];  comp = [comp; c];  res_s = [res_s; s];
    end

    res_u = res_u ./ uscale;
    if ~isempty(xscaled), res_x(xscaled, :) = res_x(xscaled, :) ./ uscale; end
    res_d = qp.c;  res_d(:,1) = res_d(:,1) + qp.A(:,:,1) * x0;   % x_2 = 0
    if ~isempty(xscaled), res_d(xscaled, :) = res_d(xscaled, :) .* uscale; end

    res = struct();
    res.stat = max([0; abs(res_u(:)); abs(res_x(:)); abs(res_s(:))]);
    res.eq   = max([0; abs(res_d(:)); abs(res_e(:))]);
    res.ineq = max([0; viol(:)]);
    res.comp = max([0; abs(comp(:))]);
end

function v = field_or_zero(s, f, sz)
    if isstruct(s) && isfield(s, f) && ~isempty(s.(f)), v = s.(f); else, v = zeros(sz); end
end

function [zl, zu, Zl, Zu] = soft_weights(qp, fzl, fzu, fZl, fZu, n, m)
    zl = inf(n, m);  zu = zl;  Zl = zeros(n, m);  Zu = Zl;
    if isfield(qp, fzl) && ~isempty(qp.(fzl)), zl = qp.(fzl) .* ones(n, m);  zu = qp.(fzu) .* ones(n, m); end
    if isfield(qp, fZl) && ~isempty(qp.(fZl)), Zl = qp.(fZl) .* ones(n, m);  Zu = qp.(fZu) .* ones(n, m); end
end

function [viol, comp, rs] = bound_terms(v, lo, hi, lam_lo, lam_hi, zlo, zhi, Zlo, Zhi, scale)
% Violation (hard sides), complementarity and slack stationarity of lo <= v <= hi with multipliers lam_lo, lam_hi.
%   Soft side (finite L1 weight): relaxed by its slack s = max(0, violation), the smallest feasible (with a quadratic weight Z > 0, at least (lam - z)/Z, see below), with the slack's own conditions z + Z s - lam - n = 0, n >= 0, n s = 0. With n = max(0, z + Z s - lam), the closest admissible, the stationarity residual is max(0, lam - z - Z s) and n s joins the complementarity. This is continuous in s: a round-off slack does not switch to |z + Z s - lam|.
%   Hard equalities (lo = hi, both sides hard) carry no complementarity (acados zeroes them: the two multipliers act as one free one). A soft side keeps its product: it is an inequality with its own slack, lo - s <= v (or v <= hi + s), whether or not lo = hi.
%   scale: per row, physical = scale x QP units.
    m = size(v, 2);  scale = scale .* ones(1, m);
    flo = isfinite(lo);  fhi = isfinite(hi);
    dlo = zeros(size(v));  dhi = dlo;
    dlo(flo) = lo(flo) - v(flo);   % > 0: violated
    dhi(fhi) = v(fhi) - hi(fhi);
    softlo = flo & isfinite(zlo);  softhi = fhi & isfinite(zhi);
    slo = max(dlo, 0) .* softlo;  shi = max(dhi, 0) .* softhi;
    % Quadratic soft sides (Z > 0): the slack the multiplier implies, (lam - z)/Z, where it exceeds the smallest feasible one. The slack is a variable of the QP (acados keeps it in its iterate). This one zeroes its own stationarity, and the row's margin it leaves, s - violation, is in the row's complementarity, lam (s - violation), as acados's. (Using the violation itself would turn an IPM margin t ~ comp/lam into a stationarity residual Z t.)
    qlo = softlo & Zlo > 0;  qhi = softhi & Zhi > 0;
    slo(qlo) = max(slo(qlo), (lam_lo(qlo) - zlo(qlo)) ./ Zlo(qlo));
    shi(qhi) = max(shi(qhi), (lam_hi(qhi) - zhi(qhi)) ./ Zhi(qhi));
    viol = [reshape(max(dlo, 0) .* (flo & ~softlo) .* scale, [], 1); reshape(max(dhi, 0) .* (fhi & ~softhi) .* scale, [], 1)];
    eqs = flo & fhi & lo == hi & ~softlo & ~softhi;   % hard equalities only
    clo = zeros(size(v));  chi = clo;
    clo(flo) = lam_lo(flo) .* (slo(flo) - dlo(flo));   % lam x (v - lo + s)
    chi(fhi) = lam_hi(fhi) .* (shi(fhi) - dhi(fhi));   % lam x (hi - v + s)
    clo(eqs) = 0;  chi(eqs) = 0;
    [rlo, nlo] = slack_terms(slo, softlo, zlo, Zlo, lam_lo);
    [rhi, nhi] = slack_terms(shi, softhi, zhi, Zhi, lam_hi);
    comp = [clo(:); chi(:); nlo; nhi];
    rs = [rlo; rhi];
end

function [r, ns] = slack_terms(s, soft, z, Z, lam)
% Slack stationarity residual max(0, lam - z - Z s) and the slack complementarity n s, soft sides only
    r = zeros(size(s));  ns = r;
    t = z(soft) + Z(soft) .* s(soft) - lam(soft);
    r(soft) = max(0, -t);
    ns(soft) = max(0, t) .* s(soft);
    r = r(:);  ns = ns(:);
end
