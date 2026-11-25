function [z, info] = dense_qp_ipm(H, g, lb, ub, G, cl, cu, opts)
% DENSE_QP_IPM  Dense primal-dual interior-point solver for QPs with bounds and linear constraints.
%
%   min 1/2 z'H z + g'z   s.t.  lb <= z <= ub,   cl <= G z <= cu
%
% H positive (semi)definite; +-Inf bounds allowed; G may be empty (box constraints only). Rows
% with cl == cu are equality constraints, handled as such (an equality multiplier), not as two
% inequalities whose slacks would both vanish and wreck the conditioning.
% Mehrotra predictor-corrector. Per iteration, one factorisation shared by the predictor and
% corrector: Cholesky of M = H + diag(Db) + Gi' diag(Dg) Gi (Gi: the inequality rows) without
% equalities, else LU of the augmented system [M Ge'; Ge 0] (lightly regularised). Stops when
% the infinity norms of the stationarity, primal (slack) and complementarity residuals are all
% within their tolerances.
% Returns z and info with fields status ('solved' | 'max_iter' | 'stalled': numerical breakdown,
% the last good iterate is returned; within TolAcceptable of every tolerance it counts as 'solved',
% with info.acceptable, as IPOPT's acceptable point; so do AcceptableIter consecutive iterates within it, and, after a later stall or MaxIter, the best acceptable iterate passed), iter (iterations done; iter_selected: the returned best iterate's index, when one was), res_stat, res_eq, res_comp (largest t*lam),
% mu (mean t*lam), lam_lb, lam_ub, lam_cl, lam_cu.
%
% Factorisation and refinement follow HPIPM's dense QP IPM (Frison & Diehl, 2020; its modes in
% brackets):
%   Factor 'normal'  Cholesky of M = H + diag(Db) + G'diag(Dg)G (HPIPM lq_fact = 0, speed modes).
%                    Near convergence Dg = lam/t reaches 1e16+ on active rows and M, formed
%                    explicitly, loses the accuracy of the Newton step.
%          'auto'    as 'normal', but after each predictor solve the residual of the linear
%                    system is checked; above 1e-5 (or NaN) the step is refactorised with LQ and
%                    LQ is used for the rest of the solve (lq_fact = 1, balance mode)
%          'lq'      LQ from the start (lq_fact = 2, robust mode)
%   LQ: with L_H = chol(H + 1e-15 I) (computed once), the lower factor L of
%   [L_H, diag(sqrt(Db)), G'diag(sqrt(Dg))] = L Q gives L L' = M without forming G'diag(Dg)G.
%   Equalities by the Schur complement Ge M^-1 Ge', itself factorised by LQ of [1e-15 I, Ge L^-T].
%   If H is not positive definite the LQ path is not available and 'normal' is used (as HPIPM).
%   Refine   iterative refinement steps on the corrector (itref_corr_max: 2 in balance, 8 in
%            robust mode): the residual of the linear system is solved again with the same
%            factorisation, until it is below the tolerances or 1e-3 of the IPM's residuals; a
%            step that does not reduce it is undone.
%   Proximal  false (default) | true | struct overriding any of Rho0 (1e-6), Delta0 (1e-4), Min
%            (1e-10): PIQP's proximal interior-point method instead (extras/pipqp: qp_pipm with the kkt_dense backend). Its barrier
%            terms are bounded by 1/delta and equalities enter through Ge'Ge/delta, so the Newton
%            matrix stays well conditioned on degenerate or LP-like QPs. Factor and Refine do not
%            apply there (PIQP's own safeguards are used). It also stops on PIQP's duality gap:
%            TolGap (1e-8) absolute or TolGapRel (1e-9) relative to its largest term.
%   Scale    Ruiz equilibration passes (0, the default: off; PIQP uses 10; extras/pipqp/ruiz_equilibrate), in either mode. The
%            QP is solved in scaled variables and mapped back; the stopping tests use residuals
%            in the original units. Variable bounds are scaled directly (PIQP instead scales them
%            as rows of the KKT matrix).
%   SoftL1, SoftL2   soft inequality rows (as HPIPM's soft constraints): row i becomes
%            cl_i - sl_i <= G_i z <= cu_i + su_i with slacks sl, su >= 0 that add
%            SoftL1_i (sl_i + su_i) + 1/2 SoftL2_i (sl_i^2 + su_i^2) to the objective. Each is
%            rows x 1, or rows x 2 for [lower upper]; SoftL1 = Inf (the default) keeps a row hard.
%            The slacks are eliminated per row, as in ocp_qp_riccati_ipm: a soft row's barrier
%            term D = lam/t is replaced by D in series with the slack's curvature,
%            1/D~ = 1/D + 1/(Z + nu/s), so the Newton matrix keeps its size. Equality rows stay
%            hard. Not with Proximal or Scale. Returns info.slack_cl, info.slack_cu (0 on hard
%            rows). TolStat is absolute, as HPIPM's.
%   Defaults: HPIPM's balance mode ('auto', Refine = 2). On the hardest mass-spring QPs (condensed,
%   M = 30, starts in +-3) it solves 14 of 14, against 12 for 'normal' without refinement and 13
%   for 'normal' with it; LQ from the start ('lq') solves 12, as its accuracy also runs out once
%   lam/t passes ~1e20, so refinement and the late switch are what matter. Maros-Meszaros: 38 of
%   42 either way (with TolEq at its default 1e-8, HPIPM's res_b; at 1e-6 'auto' stops QADLITTL and
%   QSHARE2B where an independent KKT check fails them). Cost per iteration: none to +30% while
%   'normal' is accurate; once LQ engages, about 3-4x (it stays on for the rest of the solve).

    arguments
        H; g; lb; ub
        G  = zeros(0, numel(g))
        cl = zeros(0, 1)
        cu = zeros(0, 1)
        opts.MaxIter   (1,1) double = 50
        opts.TolStat   (1,1) double = 1e-8
        opts.TolEq     (1,1) double = 1e-8
        opts.TolComp   (1,1) double = 1e-8
        opts.TolAcceptable (1,1) double = 100   % a breakdown within this factor of every tolerance counts as solved (IPOPT's acceptable point; as ocp_qp_riccati_ipm)
        opts.AcceptableIter (1,1) double = 15 % this many consecutive iterates within TolAcceptable of every tolerance count as solved (IPOPT's acceptable_iter)
        opts.TolGap    (1,1) double = 1e-8     % duality gap, absolute or relative (as PIQP; qp_pipm only)
        opts.TolGapRel (1,1) double = 1e-9
        opts.StepFrac  (1,1) double = 0.995
        opts.Mu0       (1,1) double = 1      % initial multipliers (slacks start at >= 1)
        opts.Factor    (1,:) char {mustBeMember(opts.Factor, {'normal', 'auto', 'lq'})} = 'auto'   % see below
        opts.Refine    (1,1) double = 2      % iterative refinement steps on the corrector (see below)
        opts.Proximal = false                % false | true | struct with Rho0, Delta0, Min (see below)
        opts.Scale    (1,1) double = 0       % Ruiz equilibration passes (0: off; PIQP uses 10)
        opts.SoftL1 = []                     % rows x 1 or rows x 2 ([lower upper]): L1 slack weights (Inf: hard)
        opts.SoftL2 = []                     % rows x 1 or rows x 2: L2 slack weights (default 0)
        opts.z_init = []
        opts.UnscaleZ = []                   % (internal: Ruiz factors, residual tests in original units)
        opts.UnscaleRows = []
    end

    prox_off = (islogical(opts.Proximal) || isnumeric(opts.Proximal)) && isscalar(opts.Proximal) && ~opts.Proximal;
    soft = ~isempty(opts.SoftL1) && any(isfinite(opts.SoftL1(:)));
    if soft && (opts.Scale > 0 || ~prox_off)
        error('rtimpc:SoftRows', 'Soft rows (SoftL1) are not supported with Proximal or Scale.')
    end
    if (opts.Scale > 0 || ~prox_off) && ~exist('qp_pipm', 'file')
        error('rtimpc:pipqp', 'Proximal and Scale need extras/pipqp on the path (addpath rtimpc/extras/pipqp).')
    end

    % --- Ruiz equilibration (as PIQP's preconditioner): solve the scaled QP, map the result back
    if opts.Scale > 0
        [Hs, gs, lbs, ubs, Gs, cls, cus, Dz, Er] = ruiz_equilibrate(H, g, lb, ub, G, cl, cu, opts.Scale);
        Hs = full(Hs);  Gs = full(Gs);
        o = opts;  o.Scale = 0;  o.UnscaleZ = Dz;  o.UnscaleRows = Er;
        if ~isempty(o.z_init), o.z_init = o.z_init(:) ./ Dz; end
        args = namedargs2cell(o);
        [z, info] = dense_qp_ipm(Hs, gs, lbs, ubs, Gs, cls, cus, args{:});
        z = Dz .* z;
        info.lam_lb = info.lam_lb ./ Dz;  info.lam_ub = info.lam_ub ./ Dz;
        info.lam_cl = Er .* info.lam_cl;  info.lam_cu = Er .* info.lam_cu;
        info.scaled = true;
        return
    end
    % --- Proximal mode: PIQP's algorithm (extras/pipqp/qp_pipm, dense backend)
    if ~prox_off
        P = struct('Rho0', 1e-6, 'Delta0', 1e-4, 'Min', 1e-10);   % PIQP's rho_init, delta_init, reg_lower_limit
        if isstruct(opts.Proximal)
            f = fieldnames(opts.Proximal);
            for k = 1:numel(f)
                if ~isfield(P, f{k}), error('rtimpc:Proximal', 'Unknown Proximal field ''%s''.', f{k}); end
                P.(f{k}) = opts.Proximal.(f{k});
            end
        end
        P.MaxIter = opts.MaxIter;  P.TolStat = opts.TolStat;  P.TolEq = opts.TolEq;  P.TolComp = opts.TolComp;
        P.TolGap = opts.TolGap;  P.TolGapRel = opts.TolGapRel;
        P.UnscaleZ = opts.UnscaleZ;  P.UnscaleRows = opts.UnscaleRows;
        [z, info] = qp_pipm(H, g, lb, ub, G, cl, cu, P, @kkt_dense);
        return
    end

    n = numel(g);
    cl = cl(:);  cu = cu(:);
    eq = isfinite(cl) & cl == cu;                        % equality rows: Ge z = ce
    rows = numel(cl);  Ge = G(eq, :);  ce = cl(eq);  neq = nnz(eq);
    % Soft rows: L1/L2 weights per inequality row (zero outside the soft masks hsl, hsu)
    [zgl, zgu] = deal(inf(rows, 1));  [Zgl, Zgu] = deal(zeros(rows, 1));
    if soft
        W1 = opts.SoftL1 .* ones(rows, 2);  zgl = W1(:,1);  zgu = W1(:,2);
        if ~isempty(opts.SoftL2), W2 = opts.SoftL2 .* ones(rows, 2);  Zgl = W2(:,1);  Zgu = W2(:,2); end
        if any(eq & (isfinite(zgl) | isfinite(zgu)))
            error('rtimpc:SoftRows', 'Equality rows (cl == cu) cannot be soft.')
        end
    end
    zgl = zgl(~eq);  zgu = zgu(~eq);  Zgl = Zgl(~eq);  Zgu = Zgu(~eq);
    G = G(~eq, :);  cl = reshape(cl(~eq), [], 1);  cu = reshape(cu(~eq), [], 1);
    y = zeros(neq, 1);                                   % equality multipliers
    Dz = opts.UnscaleZ;  Er = opts.UnscaleRows;          % Ruiz factors (residual tests in original units)
    if isempty(Dz), Dz = ones(n, 1); end
    if isempty(Er), Er = ones(rows, 1); end
    Ee = reshape(Er(eq), [], 1);  Ei = reshape(Er(~eq), [], 1);
    [lb, ub, hl, hu]  = ipm_bounds(lb, ub);
    [cl, cu, hgl, hgu] = ipm_bounds(cl, cu);
    hsl = hgl & isfinite(zgl);  hsu = hgu & isfinite(zgu);
    zgl(~hsl) = 0;  zgu(~hsu) = 0;  Zgl(~hsl) = 0;  Zgu(~hsu) = 0;
    m = nnz(hl) + nnz(hu) + nnz(hgl) + nnz(hgu) + nnz(hsl) + nnz(hsu);

    if isempty(opts.z_init), z = zeros(n, 1); else, z = opts.z_init; end
    both = hl & hu;
    w = ub - lb;
    z(both) = min(max(z(both), lb(both) + 0.05*w(both)), ub(both) - 0.05*w(both));
    Gz = G*z;
    [tl, tu, ll, lu]     = init_slacks(z,  lb, ub, hl, hu, opts.Mu0);
    [tgl, tgu, lgl, lgu] = init_slacks(Gz, cl, cu, hgl, hgu, opts.Mu0);
    % Soft-row slacks s and their multipliers n, started as in ocp_qp_riccati_ipm: the weight on n,
    % s = Mu0/n (the row starts out nearly hard)
    ngl = max(zgl + Zgl - lgl, opts.Mu0) .* hsl;  ngu = max(zgu + Zgu - lgu, opts.Mu0) .* hsu;
    sgl = ones(size(cl));  sgl(hsl) = opts.Mu0 ./ ngl(hsl);
    sgu = ones(size(cl));  sgu(hsu) = opts.Mu0 ./ ngu(hsu);
    [Egl, Egu] = deal(ones(size(cl)));  [rsl, rsu, rcsl, rcsu, gsl, gsu] = deal(zeros(size(cl)));

    % Near convergence lam./t spans many decades: silence spurious near-singularity warnings
    ws = ipm_silence_warnings();
    restore = onCleanup(@() warning(ws));

    force_lq = strcmp(opts.Factor, 'lq');               % LQ for the rest of the solve
    npd = false;  Lh = [];                               % H not positive definite; its factor (LQ path)
    use_lq = false;  Ll = [];  Re = [];  mi = size(G, 1);
    [L, Lk, Uk, Pk, M] = deal([]);
    info.lq = false;                                     % did the solve switch to LQ

    info.status = 'max_iter';
    iter = 0;
    n_acc = 0;   % consecutive acceptable iterates
    best = [];  best_score = Inf;   % the best acceptable iterate (returned if the IPM later stalls or hits MaxIter)
    for iter = 1:opts.MaxIter
        Gz  = G*z;
        rd  = H*z + g - ll + lu - G'*(lgl - lgu) - Ge'*y;
        re  = Ge*z - ce;
        rl  = (z - lb - tl) .* hl;    ru  = (ub - z - tu) .* hu;
        rgl = (Gz - cl + sgl.*hsl - tgl) .* hgl; rgu = (cu - Gz + sgu.*hsu - tgu) .* hgu;
        rsl = (zgl + Zgl.*sgl - lgl - ngl) .* hsl;  rsu = (zgu + Zgu.*sgu - lgu - ngu) .* hsu;
        mu = comp(tl, tu, ll, lu, tgl, tgu, lgl, lgu, sgl, sgu, ngl, ngu);
        res_stat = max([abs(rd ./ Dz); abs(rsl); abs(rsu)]);
        res_eq   = max([abs(rl .* Dz); abs(ru .* Dz); abs(rgl ./ Ei); abs(rgu ./ Ei); abs(re ./ Ee); 0]);
        res_comp = max([tl.*ll.*hl; tu.*lu.*hu; tgl.*lgl.*hgl; tgu.*lgu.*hgu; sgl.*ngl.*hsl; sgu.*ngu.*hsu; 0]);
        if res_stat <= opts.TolStat && res_eq <= opts.TolEq && res_comp <= opts.TolComp
            info.status = 'solved';
            iter = iter - 1;
            break
        end
        % IPOPT's acceptable termination: AcceptableIter consecutive iterates within TolAcceptable of
        % every tolerance (e.g. stationarity stuck at round-off above an absolute TolStat when the
        % gradients are ~1e8: Hock-Schittkowski 99)
        if ipm_acceptable(res_stat, res_eq, res_comp, opts)
            n_acc = n_acc + 1;
            score = max([res_stat / opts.TolStat, res_eq / opts.TolEq, res_comp / opts.TolComp]);
            if score < best_score
                best_score = score;
                best = {z, y, tl, tu, ll, lu, tgl, tgu, lgl, lgu, sgl, sgu, ngl, ngu, res_stat, res_eq, res_comp, iter - 1, mu};
            end
        else
            n_acc = 0;
        end
        if n_acc >= opts.AcceptableIter
            info.status = 'solved';  info.acceptable = true;
            iter = iter - 1;
            break
        end

        Db = ll ./ tl + lu ./ tu;
        Dgl = lgl ./ tgl;  Dgu = lgu ./ tgu;
        % Soft rows: eliminate the slack; its pivot E = Z + D + n/s is in series with D
        Egl(hsl) = Zgl(hsl) + Dgl(hsl) + ngl(hsl) ./ sgl(hsl);
        Egu(hsu) = Zgu(hsu) + Dgu(hsu) + ngu(hsu) ./ sgu(hsu);
        Dg = Dgl + Dgu;
        if soft
            Dgl_eff = Dgl;  Dgl_eff(hsl) = Dgl(hsl) .* (Zgl(hsl) + ngl(hsl) ./ sgl(hsl)) ./ Egl(hsl);
            Dgu_eff = Dgu;  Dgu_eff(hsu) = Dgu(hsu) .* (Zgu(hsu) + ngu(hsu) ./ sgu(hsu)) ./ Egu(hsu);
            Dg = Dgl_eff + Dgu_eff;
        end
        try
            if force_lq, factor_lq(); else, factor_normal(); end
        catch err                                        % numerical breakdown: keep the last iterate
            if ~strcmp(err.identifier, 'rtimpc:chol'), rethrow(err), end
            info.status = breakdown_status();
            iter = iter - 1;
            break
        end

        % Predictor
        rcl = -tl .* ll;    rcu = -tu .* lu;
        rcgl = -tgl .* lgl; rcgu = -tgu .* lgu;
        rcsl = -sgl .* ngl .* hsl;  rcsu = -sgu .* ngu .* hsu;
        [dz, dy, dtl, dtu, dll, dlu, dtgl, dtgu, dlgl, dlgu, dsl, dsu, dnl, dnu] = newton(false);
        if strcmp(opts.Factor, 'auto') && ~use_lq && ~npd
            % HPIPM's balance mode: an inaccurate factorisation shows in the linear system's residual
            [es, ee] = lin_residual(dz, dy);
            r = max([abs(es); abs(ee); 0]);
            if ~(r <= 1e-5)
                try
                    factor_lq();
                catch err
                    if ~strcmp(err.identifier, 'rtimpc:chol'), rethrow(err), end
                    info.status = breakdown_status();
                    iter = iter - 1;
                    break
                end
                force_lq = use_lq;
                if use_lq, [dz, dy, dtl, dtu, dll, dlu, dtgl, dtgu, dlgl, dlgu, dsl, dsu, dnl, dnu] = newton(false); end
            end
        end
        a = max_step(1);
        if m > 0
            sigma = (comp(tl + a*dtl, tu + a*dtu, ll + a*dll, lu + a*dlu, ...
                          tgl + a*dtgl, tgu + a*dtgu, lgl + a*dlgl, lgu + a*dlgu, ...
                          sgl + a*dsl, sgu + a*dsu, ngl + a*dnl, ngu + a*dnu) / mu)^3;
        else
            sigma = 0;
        end

        % Corrector
        rcl  = (sigma*mu - tl .* ll - dtl .* dll) .* hl;
        rcu  = (sigma*mu - tu .* lu - dtu .* dlu) .* hu;
        rcgl = (sigma*mu - tgl .* lgl - dtgl .* dlgl) .* hgl;
        rcgu = (sigma*mu - tgu .* lgu - dtgu .* dlgu) .* hgu;
        rcsl = (sigma*mu - sgl .* ngl - dsl .* dnl) .* hsl;
        rcsu = (sigma*mu - sgu .* ngu - dsu .* dnu) .* hsu;
        [dz, dy, dtl, dtu, dll, dlu, dtgl, dtgu, dlgl, dlgu, dsl, dsu, dnl, dnu] = newton(true);
        a = max_step(opts.StepFrac);

        z   = z   + a*dz;   y = y + a*dy;
        tl  = tl  + a*dtl;   tu  = tu  + a*dtu;   ll  = ll  + a*dll;   lu  = lu  + a*dlu;
        tgl = tgl + a*dtgl;  tgu = tgu + a*dtgu;  lgl = lgl + a*dlgl;  lgu = lgu + a*dlgu;
        sgl = sgl + a*dsl;   sgu = sgu + a*dsu;   ngl = ngl + a*dnl;   ngu = ngu + a*dnu;
    end

    if ~strcmp(info.status, 'solved') && ~isempty(best)   % an acceptable iterate was passed, then a stall or MaxIter (as ocp_qp_riccati_ipm)
        [z, y, tl, tu, ll, lu, tgl, tgu, lgl, lgu, sgl, sgu, ngl, ngu, res_stat, res_eq, res_comp, info.iter_selected, mu] = best{:};
        info.status = 'solved';  info.acceptable = true;   % iter stays the work done; iter_selected: the iterate returned
    end
    info.iter = iter;
    info.res_stat = res_stat;
    info.res_eq = res_eq;
    info.res_comp = res_comp;
    info.mu = mu;
    info.lq = force_lq;
    info.lam_lb = ll;  info.lam_ub = lu;
    info.lam_cl = zeros(rows, 1);  info.lam_cl(~eq) = lgl;  info.lam_cl(eq) = max(y, 0);
    info.lam_cu = zeros(rows, 1);  info.lam_cu(~eq) = lgu;  info.lam_cu(eq) = max(-y, 0);
    if soft
        info.slack_cl = zeros(rows, 1);  info.slack_cl(~eq) = sgl .* hsl;
        info.slack_cu = zeros(rows, 1);  info.slack_cu(~eq) = sgu .* hsu;
    end


    function [dz, dy, dtl, dtu, dll, dlu, dtgl, dtgu, dlgl, dlgu, dsl, dsu, dnl, dnu] = newton(corrector)
        r0 = rhs();
        [dz, dy] = kkt_solve(r0, re);
        if corrector && opts.Refine > 0
            % Iterative refinement (HPIPM's itref on the corrector)
            [es, ee] = corr_residual(dz, dy);
            nr = max([abs(es); abs(ee); 0]);
            for rf = 1:opts.Refine
                if (max([abs(es); 0]) < opts.TolStat || max([abs(es); 0]) < 1e-3 * res_stat) && ...
                   (max([abs(ee); 0]) < opts.TolEq   || max([abs(ee); 0]) < 1e-3 * res_eq)
                    break
                end
                [cz, cy] = kkt_solve(es, ee);
                [es1, ee1] = corr_residual(dz + cz, dy + cy);
                nr1 = max([abs(es1); abs(ee1); 0]);
                if ~(nr1 < nr), break, end               % did not help: keep the previous step
                dz = dz + cz;  dy = dy + cy;  es = es1;  ee = ee1;  nr = nr1;
            end
        end
        [dtl, dtu, dll, dlu]     = ipm_bound_steps(dz,   rl,  ru,  rcl,  rcu,  tl,  tu,  ll,  lu,  hl,  hu);
        Gdz = G*dz;
        % Soft rows: the slack step ds = g_s -/+ (D/E) G dz (ocp_qp_riccati_ipm's state_bound_steps)
        dsl = (gsl - (Dgl ./ Egl) .* Gdz) .* hsl;
        dsu = (gsu + (Dgu ./ Egu) .* Gdz) .* hsu;
        [dtgl, dtgu, dlgl, dlgu] = ipm_bound_steps(Gdz, rgl + dsl, rgu + dsu, rcgl, rcgu, tgl, tgu, lgl, lgu, hgl, hgu);
        dnl = (rcsl - ngl .* dsl) ./ sgl .* hsl;
        dnu = (rcsu - ngu .* dsu) ./ sgu .* hsu;

        function [es, ee] = corr_residual(dz, dy)
            % Residual of [M Ge'; Ge 0] [dz; -dy] = -[r0; re], M applied without forming G'DG
            es = H*dz + Db.*dz + G'*(Dg.*(G*dz)) - Ge'*dy + r0;
            ee = Ge*dz + re;
        end
    end

    function [es, ee] = lin_residual(dz, dy)
        % (predictor) residual of the linear system just solved, for the 'auto' switch
        es = H*dz + Db.*dz + G'*(Dg.*(G*dz)) - Ge'*dy + rhs();
        ee = Ge*dz + re;
    end

    function r0 = rhs()
        % Right-hand side of the reduced Newton system: bound and row residuals folded into rd.
        % Soft rows: the slack step's constant part g_s enters the row's residual.
        gb = -rcl./tl + (ll./tl).*rl + rcu./tu - (lu./tu).*ru;
        gsl = (-rsl + rcgl./tgl - Dgl.*rgl + rcsl./sgl) ./ Egl .* hsl;
        gsu = (-rsu + rcgu./tgu - Dgu.*rgu + rcsu./sgu) ./ Egu .* hsu;
        gg = -rcgl./tgl + Dgl.*(rgl + gsl) + rcgu./tgu - Dgu.*(rgu + gsu);
        r0 = rd + gb + G'*gg;
    end

    function factor_normal()
        use_lq = false;
        M = H + diag(Db) + G' * (Dg .* G);
        if neq == 0
            L = chol_reg(M);
        else                                             % augmented system, quasi-definite
            reg = 1e-10;
            [Lk, Uk, Pk] = builtin('lu', [M + reg*eye(n), Ge'; Ge, -reg*eye(neq)]);   % (lu: a multiplier here)
        end
    end

    function factor_lq()
        % HPIPM's LQ factorisation (Schur-complement method); the normal one if H is not PD
        if isempty(Lh) && ~npd
            [U, p] = chol(H + 1e-15 * eye(n));
            if p > 0, npd = true; else, Lh = U'; end
        end
        if npd, factor_normal(); return, end
        [~, R] = qr([Lh, diag(sqrt(Db)), G' .* sqrt(Dg)']', 0);   % R'R = M
        Ll = R';                                         % lower, Ll Ll' = M
        if any(~isfinite(Ll(:))) || any(diag(Ll) == 0)
            error('rtimpc:chol', 'LQ factorisation broke down.')
        end
        if neq > 0
            AL = (Ll \ Ge')';                            % Ge M^-1 Ge' = AL AL'
            [~, Re] = qr([1e-15 * eye(neq), AL]', 0);    % Re'Re = AL AL' + 1e-30 I
        end
        use_lq = true;
    end

    function [dz, dy] = kkt_solve(r0, re)
        % [M Ge'; Ge 0] [dz; -dy] = -[r0; re] with the current factorisation
        if use_lq
            Mi = @(v) Ll' \ (Ll \ v);
            if neq == 0
                dz = -Mi(r0);  dy = zeros(0, 1);
            else                                         % w = -dy: (Ge M^-1 Ge') w = re - Ge M^-1 r0
                wq = Re \ (Re' \ (re - Ge * Mi(r0)));
                dz = -Mi(r0 + Ge' * wq);  dy = -wq;
            end
        elseif neq == 0
            dz = -(L \ (L' \ r0));  dy = zeros(0, 1);
        else
            sk = Uk \ (Lk \ (Pk * -[r0; re]));
            dz = sk(1:n);  dy = -sk(n+1:end);
        end
    end

    function st = breakdown_status()
    % IPOPT's acceptable point: a breakdown at an iterate within TolAcceptable of every tolerance
    % counts as solved (info.acceptable); else 'stalled'
        if ipm_acceptable(res_stat, res_eq, res_comp, opts)
            st = 'solved';  info.acceptable = true;
        else
            st = 'stalled';
        end
    end

    function a = max_step(frac)
        v = [tl; tu; ll; lu; tgl; tgu; lgl; lgu; sgl; sgu; ngl; ngu];
        d = [dtl; dtu; dll; dlu; dtgl; dtgu; dlgl; dlgu; dsl; dsu; dnl; dnu];
        neg = d < 0;
        if any(neg), a = min(1, frac * min(-v(neg) ./ d(neg))); else, a = 1; end
    end

    function mu = comp(tl, tu, ll, lu, tgl, tgu, lgl, lgu, sgl, sgu, ngl, ngu)
        if m == 0, mu = 0; return, end
        mu = (sum(tl.*ll.*hl) + sum(tu.*lu.*hu) + sum(tgl.*lgl.*hgl) + sum(tgu.*lgu.*hgu) ...
              + sum(sgl.*ngl.*hsl) + sum(sgu.*ngu.*hsu)) / m;
    end
end


function [tl, tu, ll, lu] = init_slacks(v, lb, ub, hl, hu, mu0)
% Slacks at least 1 and multipliers mu0: a start well inside, whatever the initial z. (A slack
% floor of 1e-2 with lam = mu0/t put a pair at t = 1e-2, lam = 100 and held the first steps to
% ~1e-6 on badly scaled problems; on the Maros-Meszaros set this start solves 3 more.)
    tl = ones(size(v));  tl(hl) = max(v(hl) - lb(hl), 1);
    tu = ones(size(v));  tu(hu) = max(ub(hu) - v(hu), 1);
    ll = mu0 * hl;
    lu = mu0 * hu;
end

function L = chol_reg(M)
% Cholesky factor of a (nominally) positive definite matrix. If rounding has made it numerically
% indefinite (extreme barrier terms near convergence), add the smallest diagonal regularisation
% that restores definiteness; the residual-based Newton steps absorb the perturbation.
    [L, p] = chol(M);
    if p == 0, return, end
    d = 1e-12 * max(1, max(abs(diag(M))));
    I = eye(size(M, 1));
    for k = 1:10
        [L, p] = chol(M + d*I);
        if p == 0, return, end
        d = 100 * d;
    end
    error('rtimpc:chol', 'Matrix is not positive definite, even after regularisation.')
end


