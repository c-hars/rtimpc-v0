function [sol, info] = ocp_qp_riccati_ipm(ocp, opts)
% OCP_QP_RICCATI_IPM  Structure-exploiting interior-point solver for OCP-structured QPs.
%
%   min  sum_{i=1}^{N} ( 1/2 x_i'Q_i x_i + x_i'S_i'u_i + 1/2 u_i'R_i u_i + q_i'x_i + r_i'u_i )
%        + 1/2 x_{N+1}'Q_{N+1} x_{N+1} + q_{N+1}'x_{N+1}
%   s.t. x_{i+1} = A_i x_i + B_i u_i + c_i,   x_1 = x0,
%        lb_i <= u_i <= ub_i,   lbx_i - sl_i <= x_i <= ubx_i + su_i  (i >= 2)
%        lg_i - sgl_i <= Cg_i x_i + Dg_i u_i <= ug_i + sgu_i   (general rows, optional; i = N+1: Cg x only)
%   with optional soft state bounds: slacks sl, su >= 0 add  zl'sl + 1/2 sl'diag(Zl)sl  (and the same for su) to the cost; a bound with zl = Inf (the default) is hard (sl = 0). General rows are soft in the same way (zlg, zug, Zlg, Zug), as HPIPM's general constraints.
%
% The algorithm of HPIPM's OCP-QP IPM (Frison & Diehl, 2020), in plain MATLAB: primal-dual interior point with Mehrotra predictor-corrector. Each iteration eliminates the bound slacks/multipliers into modified stage Hessians
%   R~_i = R_i + diag(lam_l/t_l + lam_u/t_u)      (input bounds)
%   Q~_i = Q_i + diag(lam_xl/t_xl + lam_xu/t_xu)  (state bounds)
% (for a soft bound, the eliminated slack puts its own curvature in series with the barrier:
%   1/D~ = 1/D + 1/(Z + nu/s), which reduces to D~ = D as Z -> Inf); general rows add Cg'D~Cg to Q~, Dg'D~Dg to R~ and Dg'D~Cg to the cross term
% and solves the Newton system -- an equality-constrained LQ problem in the step, with the current KKT residuals as its linear terms -- by one backward Riccati recursion, O(N). The factorisation is shared by the predictor and corrector, which each only repeat the (cheap) vector pass.
%
% ocp fields (N stages, nx states, nu inputs):
%   A (nx,nx,N)  B (nx,nu,N)  c (nx,N)  x0 (nx,1)
%   Q (nx,nx,N+1)  q (nx,N+1)     Q(:,:,1), q(:,1) unused (x_1 is fixed)
%   R (nu,nu,N)  r (nu,N)         R must be positive definite
%   S (nu,nx,N)                   optional cross term (default 0)
%   lb, ub (nu,N)                 input bounds, may contain -Inf / +Inf
%   lbx, ubx (nx,N+1)             optional state bounds (column 1 ignored), +-Inf allowed
%   zlx, zux (nx,N+1 or nx,1)     optional L1 slack weights of the lower/upper state bounds (Inf = hard, the default; finite = soft)
%   Zlx, Zux (nx,N+1 or nx,1)     optional L2 slack weights (default 0)
%   Cg (ng,nx,N+1)  Dg (ng,nu,N)  optional general rows (Dg may have N+1 pages: the last is unused)
%   lg, ug (ng,N+1)               their bounds, +-Inf allowed (column 1: x_1 is fixed, so a finite bound there constrains u_1 alone)
%   zlg, zug, Zlg, Zug            L1/L2 slack weights of the general rows (ng,N+1 or ng,1; default Inf / 0: hard)
%   Eq1 (ne,nu), eq1 (ne,1)       optional equality rows on the first input, Eq1 u_1 = eq1 (x_1 is fixed, so rows of the first node pinned to one value constrain u_1 alone; e.g. scenario inputs tied together). Enforced exactly in the Newton step, without slacks or barrier (see Equalities)
%   u_init (nu,N)                 optional primal initial guess (default 0)
%   lam_init                      optional dual initial guess (warm start): struct with lam_lb, lam_ub (nu,N) and lam_lbx, lam_ubx (nx,N+1) (and lam_lg, lam_ug (ng,N+1) with general rows), e.g. the previous solution's multipliers shifted by one stage. The bound slacks still start from the primal point; each multiplier starts at max(lam_init, WarmFloor * Mu0 / t), so none starts at zero
%
% Returns sol.x (nx,N+1), sol.u (nu,N), sol.pi (nx,N+1) (pi(:,i+1): multiplier of the dynamics x_{i+1} = ...), sol.lam_lb, sol.lam_ub (nu,N), sol.lam_lbx, sol.lam_ubx (nx,N+1), sol.slack_lbx, sol.slack_ubx (nx,N+1) (0 for hard bounds), and with general rows sol.lam_lg, sol.lam_ug, sol.slack_lg, sol.slack_ug (ng,N+1), and with Eq1 their multipliers sol.nu_eq1 (ne,1) (stationarity in u_1 includes + Eq1'nu_eq1); info.status ('solved' | 'max_iter' | 'stalled': numerical breakdown, the last good iterate is returned; a breakdown at an iterate within TolAcceptable of every tolerance counts as 'solved', with info.acceptable = true, as IPOPT's acceptable point; so do AcceptableIter consecutive iterates within it, and, after a later stall or MaxIter, the best acceptable iterate passed), info.iter (iterations done; info.iter_selected: the returned best iterate's index, when one was), info.res_stat, info.res_eq, info.res_comp (largest t*lam), info.mu (mean t*lam), info.no_step (stalled before the first iteration: no usable step). TolStat is absolute, as HPIPM's (stationarity stuck at round-off ends at the acceptable point).
%
% Refine: iterative refinement of each Newton step (as HPIPM's balance/robust modes): the residual of the barrier-modified linear system is re-solved with the same factorisation (vector passes only) and the correction added. Near convergence lam/t on active bounds reaches 1e16-1e29 and the plain solve loses accuracy, so stationarity degrades and the IPM wanders off a solution it had all but reached (mass-spring M = 30, starts in +-3: 2 of 14 QPs unsolved with Refine = 0). Refinement costs about a third more per iteration, so it only runs once the largest lam/t exceeds RefineAbove (there the accuracy was intact up to ~1e14 and lost from ~1e16).
%
% Equalities (Eq1): an equality posed as a row with equal lower and upper bounds leaves the IPM no interior (both slacks must vanish together), and it stalls or creeps (24 such rows tying the 9 scenarios of a robust MPC: every QP unsolved in 500 iterations). HPIPM likewise does not treat equality-flagged general rows in its IPM; it only removes variables pinned by equality bounds before solving (ocp_qp_red: input bounds, and state bounds at the first stage). Here rows on the first input are kept as equalities in the Newton system: the Riccati recursion reduces it to an LQ problem in du_1 alone, min 1/2 du'H du + g'du with H = R~_1 + B_1'P_2B_1, so the rows add one small KKT system, (Eq1 H^-1 Eq1') dnu = r_e + Eq1 du_free, du_1 = du_free - H^-1 Eq1'dnu (du_free: the unconstrained step, r_e = Eq1 u_1 - eq1). Eq1 must have full row rank.
%
% InertiaCorrection: with x_1 fixed, the QP is strictly convex (its reduced Hessian positive definite) exactly when every stage's R~_i + B_i'P_{i+1}B_i of the Riccati recursion is, barrier terms included. An indefinite QP Hessian (an exact or general-cost Hessian) shows up as a stage whose Cholesky factorisation fails beyond round-off. The QP is then convexified, as IPOPT does for its Newton systems (Waechter & Biegler 2006, Sec. 3.1, with its default constants): delta I is added to every Q_i and R_i, in the factorisation and the residuals alike (the QP solved is the convexified one, so its step is a descent direction for an SQP merit), and the iteration is repeated. delta starts at 1e-4, or at max(1e-20, delta_hint/3) given the previous QP's (InertiaDeltaHint), and grows 100x (no hint) or 8x per failure, up to delta_max = 2 max_i ||H_i||_inf over the stage blocks H_i = [Q_i S_i'; S_i R_i] (Gershgorin: H_i + delta_max I has smallest eigenvalue >= ||H_i||_inf > 0, positive definite at the Hessian's own scale, and with the barrier terms the reduced Hessians are then positive definite in exact arithmetic). A factorisation that fails at delta_max is round-off, handled as for positive semidefinite blocks below; if that fails too, the QP is reported stalled. It only grows within a solve. (Without this cap, round-off failures under extreme barrier terms near convergence can drive delta far beyond it, e.g. 1e35; the next QPs, started from it, return huge multipliers and the SQP stalls.) info.inertia_delta is the final delta (0: none needed), info.inertia_corrections the count. Off: a failed factorisation gets increasing diagonal shifts at that stage alone, up to 1e6 x its diagonal (the earlier behaviour). The same holds, with the correction on, for a QP whose Hessian blocks are all positive semidefinite: its reduced Hessians are then PSD in exact arithmetic, so a failure is round-off, not inertia, and delta I cannot remove it. (Flying-robot planner, first QP: an LP, with hard terminal rows whose barrier terms reach 1e39; the Riccati Schur complement then loses definiteness by eps |P| ~ 1e23, and delta grew to 1e40 while the IPM diverged.)

    arguments
        ocp struct
        opts.MaxIter   (1,1) double = 50
        opts.TolStat   (1,1) double = 1e-8   % stationarity (inf-norm)
        opts.TolEq     (1,1) double = 1e-8   % dynamics + bound equality residuals (inf-norm)
        opts.TolComp   (1,1) double = 1e-8   % complementarity: largest t*lam (inf-norm)
        opts.TolAcceptable (1,1) double = 100 % a breakdown at an iterate within this factor of every tolerance counts as solved (IPOPT's acceptable point, acceptable_tol / tol = 1e-6 / 1e-8)
        opts.AcceptableIter (1,1) double = 15 % this many consecutive iterates within TolAcceptable of every tolerance count as solved (IPOPT's acceptable_iter)
        opts.StepFrac  (1,1) double = 0.995  % fraction-to-boundary
        opts.Mu0       (1,1) double = 1      % initial complementarity scale
        opts.WarmFloor (1,1) double = 1e-2   % lam_init floor, relative to the cold start Mu0/t
        opts.Refine    (1,1) double = 1      % iterative refinement steps per Newton solve, see below
        opts.RefineAbove (1,1) double = 1e12 % ... only when the largest lam/t exceeds this
        opts.InertiaCorrection (1,1) logical = true   % convexify an indefinite QP (see below)
        opts.InertiaDeltaHint (1,1) double = 0        % the previous QP's final delta (0: none)
    end

    [nx, nu, N] = size(ocp.B);
    c = ocp.c; q = ocp.q; r = ocp.r;
    % Stage matrices as cells: slicing a 3-D array, A(:,:,i), copies; cell access does not
    A = pages(ocp.A); B = pages(ocp.B); Q = pages(ocp.Q); R = pages(ocp.R);
    if isfield(ocp, 'S'), S = pages(ocp.S); else, S = repmat({zeros(nu, nx)}, N, 1); end
    % ... and as 3-D arrays, for the residuals, which are computed for all stages at once
    A3 = ocp.A;  B3 = ocp.B;  R3 = ocp.R;  Q3 = ocp.Q(:,:,2:N);  Q3N = ocp.Q(:,:,N+1);
    if isfield(ocp, 'S'), S3 = ocp.S; else, S3 = zeros(nu, nx, N); end
    A3i = A3(:,:,2:N);  S3i = S3(:,:,2:N);
    pg = @(v) reshape(v, size(v, 1), 1, []);   % columns -> pages
    Q0 = Q;  R0 = R;  Q30 = Q3;  R30 = R3;  Q3N0 = Q3N;   % the Hessian before any inertia correction
    delta = 0;  n_corr = 0;
    % Inertia correction only for a QP Hessian that is not positive semidefinite, decided at the first failed factorisation (inertia_setup): most QPs never fail one, and the test costs N + 1 eigendecompositions
    inertia_on = [];  delta_max = 0;

    % Bounds: +-Inf entries are masked out of the interior point machinery
    [lb, ub, hl, hu] = ipm_bounds(ocp.lb, ocp.ub);
    if isfield(ocp, 'lbx')
        [lbx, ubx, hxl, hxu] = ipm_bounds(ocp.lbx, ocp.ubx);
        hxl(:,1) = false; hxu(:,1) = false;   % x_1 is fixed
    else
        [lbx, ubx] = deal(zeros(nx, N+1));
        [hxl, hxu] = deal(false(nx, N+1));
    end
    % Soft state bounds: masks and weights (zero outside the mask)
    [zl, zu, Zl, Zu] = deal(inf(nx, N+1), inf(nx, N+1), zeros(nx, N+1), zeros(nx, N+1));
    if isfield(ocp, 'zlx'), zl = ocp.zlx .* ones(nx, N+1); zu = ocp.zux .* ones(nx, N+1); end
    if isfield(ocp, 'Zlx'), Zl = ocp.Zlx .* ones(nx, N+1); Zu = ocp.Zux .* ones(nx, N+1); end
    hsl = hxl & isfinite(zl);  hsu = hxu & isfinite(zu);
    zl(~hsl) = 0;  zu(~hsu) = 0;  Zl(~hsl) = 0;  Zu(~hsu) = 0;
    % General rows lg <= Cg x + Dg u <= ug (none: ng = 0, and the terms below vanish)
    hasg = isfield(ocp, 'Cg') && ~isempty(ocp.Cg) && size(ocp.Cg, 1) > 0;
    if hasg
        ng = size(ocp.Cg, 1);
        Cg3 = ocp.Cg;  Dg3 = ocp.Dg(:,:,1:N);
        Cg = pages(Cg3);  Dg = pages(Dg3);
        [lg, ug, hgl, hgu] = ipm_bounds(ocp.lg, ocp.ug);
        [zlg, zug, Zlg, Zug] = deal(inf(ng, N+1), inf(ng, N+1), zeros(ng, N+1), zeros(ng, N+1));
        if isfield(ocp, 'zlg'), zlg = ocp.zlg .* ones(ng, N+1); zug = ocp.zug .* ones(ng, N+1); end
        if isfield(ocp, 'Zlg'), Zlg = ocp.Zlg .* ones(ng, N+1); Zug = ocp.Zug .* ones(ng, N+1); end
    else
        ng = 0;
        [lg, ug, zlg, zug, Zlg, Zug] = deal(zeros(0, N+1));
        [hgl, hgu] = deal(false(0, N+1));
    end
    hsgl = hgl & isfinite(zlg);  hsgu = hgu & isfinite(zug);
    % Equality rows on the first input (none: ne = 0)
    haseq = isfield(ocp, 'Eq1') && ~isempty(ocp.Eq1);
    if haseq, E1 = ocp.Eq1;  e1 = ocp.eq1(:); else, E1 = zeros(0, nu);  e1 = zeros(0, 1); end
    nu_e = zeros(size(E1, 1), 1);  res_e = zeros(size(E1, 1), 1);
    zlg(~hsgl) = 0;  zug(~hsgu) = 0;  Zlg(~hsgl) = 0;  Zug(~hsgu) = 0;
    m = nnz(hl) + nnz(hu) + nnz(hxl) + nnz(hxu) + nnz(hsl) + nnz(hsu) ...
        + nnz(hgl) + nnz(hgu) + nnz(hsgl) + nnz(hsgu);

    % --- Initial point: u inside the box, x = 0 after the fixed x_1 (HPIPM's cold start; the Newton steps close the dynamics), t, lam > 0. (Not the rollout of the dynamics from x_1: on unstable models it grows without bound, and the IPM stalls.)
    if isfield(ocp, 'u_init'), u = ocp.u_init; else, u = zeros(nu, N); end
    both  = hl & hu;
    width = ub - lb;
    u(both) = min(max(u(both), lb(both) + 0.05*width(both)), ub(both) - 0.05*width(both));
    x = zeros(nx, N+1);
    x(:,1) = ocp.x0;
    pi_ = zeros(nx, N+1);
    warm = isfield(ocp, 'lam_init') && ~isempty(ocp.lam_init);
    [tl, tu, lam_l, lam_u]     = init_slacks(u, lb, ub, hl, hu, opts.Mu0, warm);
    [txl, txu, lam_xl, lam_xu] = init_slacks(x, lbx, ubx, hxl, hxu, opts.Mu0, warm);
    [tgl, tgu, lam_gl, lam_gu] = init_slacks(gfun(x, u), lg, ug, hgl, hgu, opts.Mu0, warm);
    if warm
        w = ocp.lam_init;  f = opts.WarmFloor;
        lam_l  = max(w.lam_lb,  f * lam_l)  .* hl;   lam_u  = max(w.lam_ub,  f * lam_u)  .* hu;
        lam_xl = max(w.lam_lbx, f * lam_xl) .* hxl;  lam_xu = max(w.lam_ubx, f * lam_xu) .* hxu;
        if hasg && isfield(w, 'lam_lg') && isequal(size(w.lam_lg), [ng, N+1])
            lam_gl = max(w.lam_lg, f * lam_gl) .* hgl;  lam_gu = max(w.lam_ug, f * lam_gu) .* hgu;
        end
    end
    % Soft-bound slacks s (kept at 1 outside the soft mask, where they are unused) and their multipliers n (for s >= 0). Start on the slack stationarity z + Z s = lam + n by giving the weight to n, with a small slack s = Mu0/n so that s*n = Mu0 like the other pairs: the bound starts out (nearly) hard and its multiplier lam is free to grow only if it is active. (With n = Mu0, a large z would leave a huge residual that steps to the boundary remove very slowly.)
    nxl = max(zl + Zl - lam_xl, opts.Mu0) .* hsl;
    nxu = max(zu + Zu - lam_xu, opts.Mu0) .* hsu;
    sxl = ones(nx, N+1);  sxl(hsl) = opts.Mu0 ./ nxl(hsl);
    sxu = ones(nx, N+1);  sxu(hsu) = opts.Mu0 ./ nxu(hsu);
    [rsl, rsu, rcsl, rcsu, gsl, gsu] = deal(zeros(nx, N+1));
    [El, Eu] = deal(ones(nx, N+1));
    % ... and the same for the general rows
    ngl = max(zlg + Zlg - lam_gl, opts.Mu0) .* hsgl;
    ngu = max(zug + Zug - lam_gu, opts.Mu0) .* hsgu;
    sgl = ones(ng, N+1);  sgl(hsgl) = opts.Mu0 ./ ngl(hsgl);
    sgu = ones(ng, N+1);  sgu(hsgu) = opts.Mu0 ./ ngu(hsgu);
    [rgl, rgu, rsgl, rsgu, rcgl, rcgu, rcsgl, rcsgu, gsgl, gsgu, Dgl, Dgu] = deal(zeros(ng, N+1));
    [Egl, Egu] = deal(ones(ng, N+1));
    Dg_eff = zeros(ng, N+1);

    % Riccati storage
    Li  = cell(N, 1);         % inverse of the upper Cholesky factor of R~ + B'PB: (R~ + B'PB)^-1 = Li*Li'
    Inu = eye(nu);
    K   = cell(N, 1);
    Rux = cell(N, 1);         % S + B'PA
    P   = cell(N+1, 1);

    % KKT residuals
    res_u = zeros(nu, N);     % stationarity wrt u_i
    res_x = zeros(nx, N+1);   % stationarity wrt x_i (i >= 2)
    res_d = zeros(nx, N);     % dynamics: A x_i + B u_i + c_i - x_{i+1}

    % Near convergence lam./t spans many decades, so the (exact) Cholesky factor is badly scaled and triangular solves raise spurious near-singularity warnings
    ws = ipm_silence_warnings();
    restore = onCleanup(@() warning(ws));


    refine_now = false;   % set per iteration (see Refine)
    info.status = 'max_iter';
    iter = 0;
    n_acc = 0;   % consecutive acceptable iterates
    best = [];  best_score = Inf;   % the best acceptable iterate (returned if the IPM later stalls or hits MaxIter)
    for iter = 1:opts.MaxIter

        % --- Residuals / termination
        compute_residuals();
        rl  = (u - lb - tl) .* hl;     ru  = (ub - u - tu) .* hu;
        rxl = (x - lbx + sxl.*hsl - txl) .* hxl;  rxu = (ubx - x + sxu.*hsu - txu) .* hxu;
        rsl = (zl + Zl.*sxl - lam_xl - nxl) .* hsl;  rsu = (zu + Zu.*sxu - lam_xu - nxu) .* hsu;
        if hasg
            gv = gfun(x, u);
            rgl = (gv - lg + sgl.*hsgl - tgl) .* hgl;  rgu = (ug - gv + sgu.*hsgu - tgu) .* hgu;
            rsgl = (zlg + Zlg.*sgl - lam_gl - ngl) .* hsgl;  rsgu = (zug + Zug.*sgu - lam_gu - ngu) .* hsgu;
        end
        mu = complementarity(tl, tu, lam_l, lam_u, txl, txu, lam_xl, lam_xu, sxl, sxu, nxl, nxu, ...
                             tgl, tgu, lam_gl, lam_gu, sgl, sgu, ngl, ngu);
        res_stat = max([max(abs(res_u), [], 'all'), max(abs(res_x), [], 'all'), max(abs([rsl; rsu]), [], 'all'), ...
                        max(abs([rsgl; rsgu]), [], 'all')]);
        res_eq   = max([max(abs(res_d), [], 'all'), max(abs([rl; ru]), [], 'all'), max(abs([rxl; rxu]), [], 'all'), ...
                        max(abs([rgl; rgu]), [], 'all'), max(abs(res_e))]);
        res_comp = max_complementarity(tl, tu, lam_l, lam_u, txl, txu, lam_xl, lam_xu, sxl, sxu, nxl, nxu, ...
                                       tgl, tgu, lam_gl, lam_gu, sgl, sgu, ngl, ngu);
        if res_stat <= opts.TolStat && res_eq <= opts.TolEq && res_comp <= opts.TolComp
            info.status = 'solved';
            iter = iter - 1;
            break
        end
        % IPOPT's acceptable termination: AcceptableIter consecutive iterates within TolAcceptable of every tolerance (e.g. stationarity stuck at round-off above an absolute TolStat when the gradients are ~1e8: Hock-Schittkowski 99)
        if ipm_acceptable(res_stat, res_eq, res_comp, opts)
            n_acc = n_acc + 1;
            score = max([res_stat / opts.TolStat, res_eq / opts.TolEq, res_comp / opts.TolComp]);
            if score < best_score
                best_score = score;
                best = {x, u, pi_, nu_e, tl, tu, lam_l, lam_u, txl, txu, lam_xl, lam_xu, sxl, sxu, nxl, nxu, ...
                        tgl, tgu, lam_gl, lam_gu, sgl, sgu, ngl, ngu, res_stat, res_eq, res_comp, iter - 1, mu};
            end
        else
            n_acc = 0;
        end
        if n_acc >= opts.AcceptableIter
            info.status = 'solved';  info.acceptable = true;
            iter = iter - 1;
            break
        end

        try   % numerical breakdown (extreme barrier terms) -> return the last good iterate
        % --- Riccati factorisation of the barrier-modified KKT system
        Dl = lam_l ./ tl;    Du = lam_u ./ tu;
        Dxl = lam_xl ./ txl; Dxu = lam_xu ./ txu;
        % Soft bounds: eliminate the slack; its pivot E = Z + D + n/s is in series with D
        El(hsl) = Zl(hsl) + Dxl(hsl) + nxl(hsl) ./ sxl(hsl);
        Eu(hsu) = Zu(hsu) + Dxu(hsu) + nxu(hsu) ./ sxu(hsu);
        Dxl_eff = Dxl;  Dxl_eff(hsl) = Dxl(hsl) .* (Zl(hsl) + nxl(hsl) ./ sxl(hsl)) ./ El(hsl);
        Dxu_eff = Dxu;  Dxu_eff(hsu) = Dxu(hsu) .* (Zu(hsu) + nxu(hsu) ./ sxu(hsu)) ./ Eu(hsu);
        if hasg   % general rows: the same elimination on v = Cg x + Dg u
            Dgl = lam_gl ./ tgl;  Dgu = lam_gu ./ tgu;
            Egl(hsgl) = Zlg(hsgl) + Dgl(hsgl) + ngl(hsgl) ./ sgl(hsgl);
            Egu(hsgu) = Zug(hsgu) + Dgu(hsgu) + ngu(hsgu) ./ sgu(hsgu);
            Dgl_eff = Dgl;  Dgl_eff(hsgl) = Dgl(hsgl) .* (Zlg(hsgl) + ngl(hsgl) ./ sgl(hsgl)) ./ Egl(hsgl);
            Dgu_eff = Dgu;  Dgu_eff(hsgu) = Dgu(hsgu) .* (Zug(hsgu) + ngu(hsgu) ./ sgu(hsgu)) ./ Egu(hsgu);
            Dg_eff = (Dgl_eff .* hgl + Dgu_eff .* hgu);
        end
        refine_now = opts.Refine > 0 && max([Dl(:); Du(:); Dxl_eff(:); Dxu_eff(:); Dg_eff(:)]) > opts.RefineAbove;
        P{N+1} = Q{N+1} + diag(Dxl_eff(:,N+1) + Dxu_eff(:,N+1));
        if hasg, P{N+1} = P{N+1} + Cg{N+1}' * (Dg_eff(:,N+1) .* Cg{N+1}); end
        for i = N:-1:1
            Pn = P{i+1};
            BP = B{i}' * Pn;
            % Inverse factor once per node: the gain and both back-substitutions then multiply instead of solving (cheaper at these sizes, where per-operation overhead dominates)
            Ri = R{i} + diag(Dl(:,i) + Du(:,i)) + BP * B{i};
            Si = S{i};  Qi = Q{i};
            if hasg
                DC = Dg_eff(:,i) .* Cg{i};
                Ri = Ri + Dg{i}' * (Dg_eff(:,i) .* Dg{i});
                Si = Si + Dg{i}' * DC;
                Qi = Qi + Cg{i}' * DC;
            end
            [Lci, pd] = chol(Ri);
            if pd
                inertia_setup();
                Lci = chol_reg(Ri, ternary(inertia_on && delta < delta_max, 3, 10));   % round-off only, or (off, or at delta_max) the old shifts
                if isempty(Lci), error('rtimpc:inertia', 'Indefinite reduced Hessian at stage %d.', i); end
            end
            Li{i}  = Lci \ Inu;
            Rux{i} = Si + BP * A{i};
            K{i}   = -(Li{i} * (Li{i}' * Rux{i}));
            if i > 1
                Pi = Qi + diag(Dxl_eff(:,i) + Dxu_eff(:,i)) + A{i}' * Pn * A{i} + Rux{i}' * K{i};
                P{i} = 0.5 * (Pi + Pi');
            end
        end

        % --- Predictor (affine scaling)
        rcl  = -tl .* lam_l;    rcu  = -tu .* lam_u;
        rcxl = -txl .* lam_xl;  rcxu = -txu .* lam_xu;
        rcsl = -sxl .* nxl .* hsl;  rcsu = -sxu .* nxu .* hsu;
        rcgl = -tgl .* lam_gl;  rcgu = -tgu .* lam_gu;
        rcsgl = -sgl .* ngl .* hsgl;  rcsgu = -sgu .* ngu .* hsgu;
        [dx, du, dpi, dnue] = newton_step();
        [dtl, dtu, dll, dlu] = ipm_bound_steps(du, rl, ru, rcl, rcu, tl, tu, lam_l, lam_u, hl, hu);
        [dtxl, dtxu, dlxl, dlxu, dsl, dsu, dnl, dnu] = state_bound_steps(dx);
        [dtgl, dtgu, dlgl, dlgu, dsgl, dsgu, dngl, dngu] = general_row_steps(dx, du);
        alpha = max_step(1);
        if m > 0
            mu_aff = complementarity(tl + alpha*dtl, tu + alpha*dtu, lam_l + alpha*dll, lam_u + alpha*dlu, ...
                txl + alpha*dtxl, txu + alpha*dtxu, lam_xl + alpha*dlxl, lam_xu + alpha*dlxu, ...
                sxl + alpha*dsl, sxu + alpha*dsu, nxl + alpha*dnl, nxu + alpha*dnu, ...
                tgl + alpha*dtgl, tgu + alpha*dtgu, lam_gl + alpha*dlgl, lam_gu + alpha*dlgu, ...
                sgl + alpha*dsgl, sgu + alpha*dsgu, ngl + alpha*dngl, ngu + alpha*dngu);
            sigma = (mu_aff / mu)^3;
        else
            sigma = 0;
        end

        % --- Corrector (centring + second-order term)
        rcl  = (sigma*mu - tl .* lam_l - dtl .* dll) .* hl;
        rcu  = (sigma*mu - tu .* lam_u - dtu .* dlu) .* hu;
        rcxl = (sigma*mu - txl .* lam_xl - dtxl .* dlxl) .* hxl;
        rcxu = (sigma*mu - txu .* lam_xu - dtxu .* dlxu) .* hxu;
        rcsl = (sigma*mu - sxl .* nxl - dsl .* dnl) .* hsl;
        rcsu = (sigma*mu - sxu .* nxu - dsu .* dnu) .* hsu;
        rcgl = (sigma*mu - tgl .* lam_gl - dtgl .* dlgl) .* hgl;
        rcgu = (sigma*mu - tgu .* lam_gu - dtgu .* dlgu) .* hgu;
        rcsgl = (sigma*mu - sgl .* ngl - dsgl .* dngl) .* hsgl;
        rcsgu = (sigma*mu - sgu .* ngu - dsgu .* dngu) .* hsgu;
        [dx, du, dpi, dnue] = newton_step();
        [dtl, dtu, dll, dlu] = ipm_bound_steps(du, rl, ru, rcl, rcu, tl, tu, lam_l, lam_u, hl, hu);
        [dtxl, dtxu, dlxl, dlxu, dsl, dsu, dnl, dnu] = state_bound_steps(dx);
        [dtgl, dtgu, dlgl, dlgu, dsgl, dsgu, dngl, dngu] = general_row_steps(dx, du);
        alpha = max_step(opts.StepFrac);
        if ~all(isfinite([alpha; dx(:); du(:); dpi(:); dnue; dlxl(:); dlxu(:); dsl(:); dsu(:); dnl(:); dnu(:); ...
                          dlgl(:); dlgu(:); dsgl(:); dsgu(:); dngl(:); dngu(:)]))
            error('rtimpc:chol', 'Non-finite IPM step.')
        end
        catch err
            if strcmp(err.identifier, 'rtimpc:inertia') && isequal(inertia_on, true) && next_delta()
                continue   % the iteration again, on the convexified QP (residuals recomputed)
            end
            if ~any(strcmp(err.identifier, {'rtimpc:chol', 'rtimpc:inertia'})), rethrow(err), end
            % IPOPT's acceptable point: when the step computation fails at an iterate that meets every tolerance within TolAcceptable, return it as solved (near the solution the barrier terms are extreme and the factorisation can break down although the iterate is converged to round-off; e.g. stationarity 1.02e-8 with mu ~1e-15)
            if ipm_acceptable(res_stat, res_eq, res_comp, opts)
                info.status = 'solved';  info.acceptable = true;
            else
                info.status = 'stalled';
            end
            iter = iter - 1;
            break
        end

        % --- Update (common primal/dual step length)
        x      = x      + alpha * dx;
        u      = u      + alpha * du;
        pi_    = pi_    + alpha * dpi;
        nu_e   = nu_e   + alpha * dnue;
        tl     = tl     + alpha * dtl;   tu     = tu     + alpha * dtu;
        lam_l  = lam_l  + alpha * dll;   lam_u  = lam_u  + alpha * dlu;
        txl    = txl    + alpha * dtxl;  txu    = txu    + alpha * dtxu;
        lam_xl = lam_xl + alpha * dlxl;  lam_xu = lam_xu + alpha * dlxu;
        sxl    = sxl    + alpha * dsl;   sxu    = sxu    + alpha * dsu;
        nxl    = nxl    + alpha * dnl;   nxu    = nxu    + alpha * dnu;
        if hasg
            tgl    = tgl    + alpha * dtgl;  tgu    = tgu    + alpha * dtgu;
            lam_gl = lam_gl + alpha * dlgl;  lam_gu = lam_gu + alpha * dlgu;
            sgl    = sgl    + alpha * dsgl;  sgu    = sgu    + alpha * dsgu;
            ngl    = ngl    + alpha * dngl;  ngu    = ngu    + alpha * dngu;
        end
    end

    if ~strcmp(info.status, 'solved') && ~isempty(best)
        % An acceptable iterate (every residual within TolAcceptable of its tolerance) was passed, and the IPM then stalled or ran to MaxIter (near the solution the iterates can deteriorate as the barrier terms become extreme): return that iterate as IPOPT's acceptable point
        [x, u, pi_, nu_e, tl, tu, lam_l, lam_u, txl, txu, lam_xl, lam_xu, sxl, sxu, nxl, nxu, ...
         tgl, tgu, lam_gl, lam_gu, sgl, sgu, ngl, ngu, res_stat, res_eq, res_comp, info.iter_selected, mu] = best{:};
        info.status = 'solved';  info.acceptable = true;   % iter stays the work done (all iterations); iter_selected: the iterate returned
    end
    sol.x = x; sol.u = u; sol.pi = pi_;
    sol.lam_lb = lam_l;   sol.lam_ub = lam_u;
    sol.lam_lbx = lam_xl; sol.lam_ubx = lam_xu;
    sol.slack_lbx = sxl .* hsl;  sol.slack_ubx = sxu .* hsu;
    if hasg
        sol.lam_lg = lam_gl;  sol.lam_ug = lam_gu;
        sol.slack_lg = sgl .* hsgl;  sol.slack_ug = sgu .* hsgu;
    end
    if haseq, sol.nu_eq1 = nu_e; end
    info.iter = iter;
    info.no_step = strcmp(info.status, 'stalled') && iter == 0;   % broke down before its first iterate: sol is the starting point, not a step
    info.res_stat = res_stat;
    info.res_eq = res_eq;
    info.res_comp = res_comp;
    info.mu = mu;
    info.inertia_delta = delta;
    info.inertia_capped = delta_max > 0 && delta >= delta_max;   % delta reached delta_max
    info.inertia_corrections = n_corr;


    % ---- nested functions ----

    function inertia_setup()
        % With PSD blocks the barrier-modified reduced Hessians are PSD in exact arithmetic, so a failed factorisation is round-off (e.g. Riccati cancellation under huge barrier terms), not inertia. A symmetric eigensolver is backward stable (eigenvalues within ~n eps |H|), hence the tolerance. delta_max: see InertiaCorrection above.
        if ~isempty(inertia_on), return, end
        hess_psd = true;
        for k = 1:N+1
            if k == 1, Hk = R0{1}; elseif k == N+1, Hk = Q0{N+1}; else, Hk = [Q0{k} S{k}'; S{k} R0{k}]; end
            Hk = (Hk + Hk') / 2;
            if min(eig(Hk)) < -10 * size(Hk, 1) * eps * norm(Hk, 1), hess_psd = false; break, end
        end
        inertia_on = opts.InertiaCorrection && ~hess_psd;
        if inertia_on
            for k = 1:N+1
                if k == 1, Hk = R0{1}; elseif k == N+1, Hk = Q0{N+1}; else, Hk = [Q0{k} S{k}'; S{k} R0{k}]; end
                delta_max = max(delta_max, 2 * norm(Hk, inf));
            end
        end
    end

    function ok = next_delta()
        % IPOPT's inertia-correction update (Waechter & Biegler 2006, Sec. 3.1, default constants), up to delta_max
        ok = delta < delta_max;   % at delta_max a failure is round-off (see InertiaCorrection)
        if ~ok, return, end
        if delta == 0
            if opts.InertiaDeltaHint > 0, delta = max(1e-20, opts.InertiaDeltaHint / 3); else, delta = 1e-4; end
        elseif opts.InertiaDeltaHint > 0
            delta = 8 * delta;
        else
            delta = 100 * delta;
        end
        delta = min(delta, delta_max);
        n_corr = n_corr + 1;
        Iq = delta * eye(nx);  Iu = delta * eye(nu);
        Q = cellfun(@(M) M + Iq, Q0, 'UniformOutput', false);
        R = cellfun(@(M) M + Iu, R0, 'UniformOutput', false);
        Q3 = Q30 + Iq;  R3 = R30 + Iu;  Q3N = Q3N0 + Iq;
    end

    function compute_residuals()
        % All stages at once (pagemtimes): a loop over the stages costs more in per-operation overhead than in arithmetic at these sizes
        xs = pg(x(:,1:N));  us = pg(u);  ps = pg(pi_(:,2:N+1));
        res_u = reshape(pagemtimes(R3, us) + pagemtimes(S3, xs) + pagemtimes(B3, 'transpose', ps, 'none'), nu, N) ...
                + r - lam_l + lam_u;
        res_d = reshape(pagemtimes(A3, xs) + pagemtimes(B3, us), nx, N) + c - x(:,2:N+1);
        if N > 1
            res_x(:,2:N) = reshape(pagemtimes(Q3, pg(x(:,2:N))) + pagemtimes(S3i, 'transpose', pg(u(:,2:N)), 'none') ...
                                   + pagemtimes(A3i, 'transpose', pg(pi_(:,3:N+1)), 'none'), nx, N-1) ...
                           + q(:,2:N) - pi_(:,2:N) - lam_xl(:,2:N) + lam_xu(:,2:N);
        end
        res_x(:,N+1) = Q3N * x(:,N+1) + q(:,N+1) - pi_(:,N+1) - lam_xl(:,N+1) + lam_xu(:,N+1);
        if hasg   % general rows: + Cg'(lam_ug - lam_lg), Dg'(lam_ug - lam_lg)
            dl = lam_gu - lam_gl;
            res_u = res_u + reshape(pagemtimes(Dg3, 'transpose', pg(dl(:,1:N)), 'none'), nu, N);
            res_x(:,2:N+1) = res_x(:,2:N+1) + reshape(pagemtimes(Cg3(:,:,2:N+1), 'transpose', pg(dl(:,2:N+1)), 'none'), nx, N);
        end
        if haseq   % equality rows on u_1: + Eq1'nu_e, and their residual
            res_u(:,1) = res_u(:,1) + E1' * nu_e;
            res_e = E1 * u(:,1) - e1;
        end
    end

    function v = gfun(x, u)
        % General rows' values Cg x + Dg u at every node (node N+1: Cg x)
        if ~hasg, v = zeros(0, N+1); return, end
        v = reshape(pagemtimes(Cg3, pg(x)), ng, N+1);
        v(:,1:N) = v(:,1:N) + reshape(pagemtimes(Dg3, pg(u)), ng, N);
    end

    function [dx, du, dpi, dnue] = newton_step()
        % LQ problem in the step:  min 1/2 [dx;du]'[Q~ S';S R~][dx;du] + gx'dx + gu'du
        %                          s.t. dx_{i+1} = A dx_i + B du_i + res_d_i,  dx_1 = 0
        % gu, gx fold the bound residuals and complementarity RHS into the gradients.
        gu = res_u - rcl./tl + Dl.*rl + rcu./tu - Du.*ru;
        % Soft bounds: the slack step is ds = g_s -/+ (D/E) dx; its constant part enters gx
        gsl = (-rsl + rcxl./txl - Dxl.*rxl + rcsl./sxl) ./ El .* hsl;
        gsu = (-rsu + rcxu./txu - Dxu.*rxu + rcsu./sxu) ./ Eu .* hsu;
        gx = res_x - rcxl./txl + Dxl.*(rxl + gsl) + rcxu./txu - Dxu.*(rxu + gsu);
        if hasg   % general rows: their eliminated terms hg enter through Cg' and Dg'
            gsgl = (-rsgl + rcgl./tgl - Dgl.*rgl + rcsgl./sgl) ./ Egl .* hsgl;
            gsgu = (-rsgu + rcgu./tgu - Dgu.*rgu + rcsgu./sgu) ./ Egu .* hsgu;
            hg = (-rcgl./tgl + Dgl.*(rgl + gsgl)) .* hgl + (rcgu./tgu - Dgu.*(rgu + gsgu)) .* hgu;
            gu = gu + reshape(pagemtimes(Dg3, 'transpose', pg(hg(:,1:N)), 'none'), nu, N);
            gx = gx + reshape(pagemtimes(Cg3, 'transpose', pg(hg), 'none'), nx, N+1);
        end

        [dx, du, dpi, dnue] = lq_solve(gu, gx, res_d, res_e);
        if ~refine_now, return, end
        for rf = 1:opts.Refine   % iterative refinement: solve again for the residual and correct
            [eu, ex, ed, ee] = lq_residual(dx, du, dpi, dnue, gu, gx, res_d, res_e);
            [cx, cu, cp, cn] = lq_solve(eu, ex, ed, ee);
            dx = dx + cx;  du = du + cu;  dpi = dpi + cp;  dnue = dnue + cn;
        end
    end

    function [dx, du, dpi, dnue] = lq_solve(gu, gx, rd, re)
        % The LQ problem by the factorised Riccati recursion: backward pass, forward rollout. With equality rows on u_1 (E1 du_1 + re = 0), the first stage's step is the constrained one.
        kff = zeros(nu, N);
        p   = zeros(nx, N+1);
        p(:,N+1) = gx(:,N+1);
        for j = N:-1:1
            w  = p(:,j+1) + P{j+1} * rd(:,j);
            kff(:,j) = -(Li{j} * (Li{j}' * (gu(:,j) + B{j}' * w)));
            if j > 1
                p(:,j) = gx(:,j) + A{j}' * w + Rux{j}' * kff(:,j);
            end
        end
        dnue = zeros(size(E1, 1), 1);
        if haseq   % dx_1 = 0, so stage 1's step is du_1 = kff_1 unless constrained
            HiE = Li{1} * (Li{1}' * E1');
            dnue = (E1 * HiE) \ (re + E1 * kff(:,1));
            kff(:,1) = kff(:,1) - HiE * dnue;
        end
        dx  = zeros(nx, N+1);
        du  = zeros(nu, N);
        dpi = zeros(nx, N+1);
        for j = 1:N
            du(:,j)    = K{j} * dx(:,j) + kff(:,j);
            dx(:,j+1)  = A{j} * dx(:,j) + B{j} * du(:,j) + rd(:,j);
            dpi(:,j+1) = P{j+1} * dx(:,j+1) + p(:,j+1);
        end
    end

    function [eu, ex, ed, ee] = lq_residual(dx, du, dpi, dnue, gu, gx, rd, re)
        % Residual of the LQ problem's optimality conditions at (dx, du, dpi): stationarity wrt du and dx (barrier-modified Hessians), and the dynamics. All stages at once, as compute_residuals.
        eu = reshape(pagemtimes(R3, pg(du)) + pagemtimes(S3, pg(dx(:,1:N))) ...
                     + pagemtimes(B3, 'transpose', pg(dpi(:,2:N+1)), 'none'), nu, N) ...
             + (Dl + Du) .* du + gu;
        ed = reshape(pagemtimes(A3, pg(dx(:,1:N))) + pagemtimes(B3, pg(du)), nx, N) + rd - dx(:,2:N+1);
        ex = zeros(nx, N+1);
        if N > 1
            ex(:,2:N) = reshape(pagemtimes(Q3, pg(dx(:,2:N))) + pagemtimes(S3i, 'transpose', pg(du(:,2:N)), 'none') ...
                                + pagemtimes(A3i, 'transpose', pg(dpi(:,3:N+1)), 'none'), nx, N-1) ...
                        + (Dxl_eff(:,2:N) + Dxu_eff(:,2:N)) .* dx(:,2:N) - dpi(:,2:N) + gx(:,2:N);
        end
        ex(:,N+1) = Q3N * dx(:,N+1) + (Dxl_eff(:,N+1) + Dxu_eff(:,N+1)) .* dx(:,N+1) - dpi(:,N+1) + gx(:,N+1);
        if hasg
            hv = Dg_eff .* gfun(dx, du);
            eu = eu + reshape(pagemtimes(Dg3, 'transpose', pg(hv(:,1:N)), 'none'), nu, N);
            ex(:,2:N+1) = ex(:,2:N+1) + reshape(pagemtimes(Cg3(:,:,2:N+1), 'transpose', pg(hv(:,2:N+1)), 'none'), nx, N);
        end
        ee = zeros(size(E1, 1), 1);
        if haseq, eu(:,1) = eu(:,1) + E1' * dnue;  ee = E1 * du(:,1) + re; end
    end

    function alpha = max_step(frac)
        % Largest alpha in (0,1] keeping all slacks and multipliers positive (times frac)
        v = [tl(:); tu(:); lam_l(:); lam_u(:); txl(:); txu(:); lam_xl(:); lam_xu(:); sxl(:); sxu(:); nxl(:); nxu(:); ...
             tgl(:); tgu(:); lam_gl(:); lam_gu(:); sgl(:); sgu(:); ngl(:); ngu(:)];
        d = [dtl(:); dtu(:); dll(:); dlu(:); dtxl(:); dtxu(:); dlxl(:); dlxu(:); dsl(:); dsu(:); dnl(:); dnu(:); ...
             dtgl(:); dtgu(:); dlgl(:); dlgu(:); dsgl(:); dsgu(:); dngl(:); dngu(:)];
        neg = d < 0;
        if any(neg), alpha = min(1, frac * min(-v(neg) ./ d(neg))); else, alpha = 1; end
    end

    function mu = complementarity(tl, tu, ll, lu, txl, txu, lxl, lxu, sl, su, nl, nu, tgl, tgu, lgl, lgu, sgl, sgu, ngl, ngu)
        if m == 0, mu = 0; return, end
        mu = (sum(tl .* ll .* hl, 'all') + sum(tu .* lu .* hu, 'all') + ...
              sum(txl .* lxl .* hxl, 'all') + sum(txu .* lxu .* hxu, 'all') + ...
              sum(sl .* nl .* hsl, 'all') + sum(su .* nu .* hsu, 'all') + ...
              sum(tgl .* lgl .* hgl, 'all') + sum(tgu .* lgu .* hgu, 'all') + ...
              sum(sgl .* ngl .* hsgl, 'all') + sum(sgu .* ngu .* hsgu, 'all')) / m;
    end

    function c = max_complementarity(tl, tu, ll, lu, txl, txu, lxl, lxu, sl, su, nl, nu, tgl, tgu, lgl, lgu, sgl, sgu, ngl, ngu)
        c = max([0, max(tl .* ll .* hl, [], 'all'), max(tu .* lu .* hu, [], 'all'), ...
                    max(txl .* lxl .* hxl, [], 'all'), max(txu .* lxu .* hxu, [], 'all'), ...
                    max(sl .* nl .* hsl, [], 'all'), max(su .* nu .* hsu, [], 'all'), ...
                    max(tgl .* lgl .* hgl, [], 'all'), max(tgu .* lgu .* hgu, [], 'all'), ...
                    max(sgl .* ngl .* hsgl, [], 'all'), max(sgu .* ngu .* hsgu, [], 'all')]);
    end

    function [dtxl, dtxu, dlxl, dlxu, dsl, dsu, dnl, dnu] = state_bound_steps(dx)
        % State-bound slack/multiplier steps (and soft-bound slack steps) given the primal step dx
        dsl  = (gsl - (Dxl ./ El) .* dx) .* hsl;
        dsu  = (gsu + (Dxu ./ Eu) .* dx) .* hsu;
        dtxl = ( dx + dsl + rxl) .* hxl;
        dtxu = (-dx + dsu + rxu) .* hxu;
        dlxl = (rcxl - lam_xl .* dtxl) ./ txl;
        dlxu = (rcxu - lam_xu .* dtxu) ./ txu;
        dnl  = (rcsl - nxl .* dsl) ./ sxl .* hsl;
        dnu  = (rcsu - nxu .* dsu) ./ sxu .* hsu;
    end

    function [dtgl, dtgu, dlgl, dlgu, dsgl, dsgu, dngl, dngu] = general_row_steps(dx, du)
        % As state_bound_steps, for the general rows with v = Cg dx + Dg du
        if ~hasg, [dtgl, dtgu, dlgl, dlgu, dsgl, dsgu, dngl, dngu] = deal(zeros(0, N+1)); return, end
        v = gfun(dx, du);
        dsgl = (gsgl - (Dgl ./ Egl) .* v) .* hsgl;
        dsgu = (gsgu + (Dgu ./ Egu) .* v) .* hsgu;
        dtgl = ( v + dsgl + rgl) .* hgl;
        dtgu = (-v + dsgu + rgu) .* hgu;
        dlgl = (rcgl - lam_gl .* dtgl) ./ tgl;
        dlgu = (rcgu - lam_gu .* dtgu) ./ tgu;
        dngl = (rcsgl - ngl .* dsgl) ./ sgl .* hsgl;
        dngu = (rcsgu - ngu .* dsgu) ./ sgu .* hsgu;
    end
end


function [tl, tu, ll, lu] = init_slacks(v, lb, ub, hl, hu, mu0, warm)
% Slacks at the current primal point, floored (infeasible starts are allowed). Cold start: floor 1 and lam = mu0, a start well inside whatever the primal point, as in dense_qp_ipm (a floor of 1e-2 with lam = mu0/t put pairs at t = 1e-2, lam = 100: on the mass-spring benchmark 14-16 iterations instead of 10-11, and far from the solution after 10). Warm start: floor 1e-2 and lam = mu0/t, the reference for the WarmFloor of the multipliers passed in, so that a bound active in the guess keeps its small slack.
    floor = 1;  if warm, floor = 1e-2; end
    tl = ones(size(v));  tl(hl) = max(v(hl) - lb(hl), floor);
    tu = ones(size(v));  tu(hu) = max(ub(hu) - v(hu), floor);
    if warm
        ll = (mu0 ./ tl) .* hl;
        lu = (mu0 ./ tu) .* hu;
    else
        ll = mu0 * hl;
        lu = mu0 * hu;
    end
end

function C = pages(M)
% 3-D array -> column cell array of its pages
    C = squeeze(num2cell(M, [1 2]));
    if isrow(C), C = C'; end
end

function L = chol_reg(M, tries)
% Cholesky factor of a (nominally) positive definite matrix. If rounding has made it numerically indefinite (extreme barrier terms near convergence), add the smallest diagonal regularisation that restores definiteness; the residual-based Newton steps absorb the perturbation. Shifts 1e-12, 1e-10, ... x max(1, max|diag|), up to tries of them (3: round-off only, to 1e-8; 10: to 1e6). Returns [] if none succeeds.
    [L, p] = chol(M);
    if p == 0, return, end
    d = 1e-12 * max(1, max(abs(diag(M))));
    I = eye(size(M, 1));
    for k = 1:tries
        [L, p] = chol(M + d*I);
        if p == 0, return, end
        d = 100 * d;
    end
    L = [];
end

function v = ternary(c, a, b)
    if c, v = a; else, v = b; end
end


