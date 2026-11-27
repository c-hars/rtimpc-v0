function [sol, info] = ocp_qp_condensed_ipm(ocp, opts)
% OCP_QP_CONDENSED_IPM  OCP-QP solver by full condensing + dense interior point.
%
% Same problem and interface as ocp_qp_riccati_ipm, soft state bounds and general rows included. The states are eliminated using the
% dynamics, x = Theta*u + w, leaving a dense QP in the N*nu inputs:
%
%   min 1/2 z'H z + g'z  s.t. lb <= z <= ub,  lbx - w <= Theta z <= ubx - w (bounded rows only),
%        and each general row lg <= Cg x + Dg u <= ug with x = Theta z + w
%   H = Theta'Qb Theta + Rb + Sb Theta + Theta'Sb',   g = Theta'(Qb w + qb) + r + Sb w
%
% (the classic ACADO/qpOASES pipeline). Condensing costs O(N^2), each IPM iteration a dense
% O((N nu)^3) Cholesky -- more flops than the O(N) Riccati recursion, but executed as a few
% large BLAS calls rather than many small matrix operations.

    arguments
        ocp struct
        opts.MaxIter   (1,1) double = 50
        opts.TolStat   (1,1) double = 1e-8
        opts.TolEq     (1,1) double = 1e-8
        opts.TolComp   (1,1) double = 1e-8
        opts.StepFrac  (1,1) double = 0.995
        opts.Mu0       (1,1) double = 1
        opts.Factor    (1,:) char = 'auto'               % 'normal' | 'auto' | 'lq' (see dense_qp_ipm)
        opts.Refine    (1,1) double = 2      % iterative refinement steps on the corrector (see dense_qp_ipm)
        opts.Proximal = false                            % PIQP's proximal IPM (see dense_qp_ipm)
        opts.Scale    (1,1) double = 0                   % Ruiz passes (see dense_qp_ipm)
    end

    A = ocp.A; B = ocp.B; c = ocp.c;
    [nx, nu, N] = size(B);
    nz = nu*N;
    if isfield(ocp, 'S'), S = ocp.S; else, S = zeros(nu, nx, N); end

    % --- Condense: Theta (block lower-triangular) and the free response w, nodes 2..N+1
    Th = zeros(nx*N, nz);
    w  = zeros(nx*N, 1);
    xi = ocp.x0;
    for i = 1:N
        rows = (i-1)*nx + (1:nx);
        cols = 1:(i-1)*nu;
        if i > 1
            Th(rows, cols) = A(:,:,i) * Th(rows - nx, cols);
        end
        Th(rows, (i-1)*nu + (1:nu)) = B(:,:,i);
        xi = A(:,:,i) * xi + c(:,i);
        w(rows) = xi;
    end

    % Block-diagonal weights (nodes 2..N+1), applied blockwise
    QTh = zeros(nx*N, nz);
    qw  = zeros(nx*N, 1);
    for i = 1:N
        rows = (i-1)*nx + (1:nx);
        cols = 1:i*nu;
        QTh(rows, cols) = ocp.Q(:,:,i+1) * Th(rows, cols);
        qw(rows) = ocp.Q(:,:,i+1) * w(rows) + ocp.q(:,i+1);
    end
    H = Th' * QTh;
    g = Th' * qw + ocp.r(:);

    % Stage terms: R_i on the diagonal; S_i couples u_i with x_i (x_1 = x0 fixed, x_i = node i)
    for i = 1:N
        iu = (i-1)*nu + (1:nu);
        H(iu, iu) = H(iu, iu) + ocp.R(:,:,i);
        if i == 1
            g(iu) = g(iu) + S(:,:,1) * ocp.x0;
        elseif any(S(:,:,i), 'all')
            rows = (i-2)*nx + (1:nx);
            cols = 1:(i-1)*nu;
            SThi = S(:,:,i) * Th(rows, cols);
            H(iu, cols) = H(iu, cols) + SThi;
            H(cols, iu) = H(cols, iu) + SThi';
            g(iu) = g(iu) + S(:,:,i) * w(rows);
        end
    end
    H = 0.5*(H + H');

    % --- State bounds become general linear constraints on z (bounded rows of Theta only);
    % soft ones become soft rows, their slacks eliminated inside dense_qp_ipm
    if isfield(ocp, 'lbx')
        lbx = ocp.lbx(:,2:end);  ubx = ocp.ubx(:,2:end);
        bnd = isfinite(lbx(:)) | isfinite(ubx(:));
    else
        lbx = zeros(nx, N);  ubx = lbx;  bnd = false(nx*N, 1);
    end
    G  = Th(bnd,:);
    cl = lbx(bnd) - w(bnd);
    cu = ubx(bnd) - w(bnd);
    row = @(v) reshape(v(:, 2:end), [], 1);
    ex = @(v) v .* ones(nx, N+1);
    W1 = inf(nnz(bnd), 2);  W2 = zeros(nnz(bnd), 2);
    if isfield(ocp, 'zlx'), W1 = [row(ex(ocp.zlx)), row(ex(ocp.zux))];  W1 = W1(bnd, :); end
    if isfield(ocp, 'Zlx'), W2 = [row(ex(ocp.Zlx)), row(ex(ocp.Zux))];  W2 = W2(bnd, :); end
    % General rows, node by node (column i: x_i = x0 for i = 1, else rows of Theta z + w; u_i for i <= N)
    ng = 0;
    if isfield(ocp, 'Cg') && ~isempty(ocp.Cg), ng = size(ocp.Cg, 1); end
    gbnd = false(ng, N+1);
    if ng > 0
        exg = @(v) v .* ones(ng, N+1);
        gbnd = isfinite(ocp.lg) | isfinite(ocp.ug);
        for i = 1:N+1
            k = gbnd(:, i);
            if ~any(k), continue, end
            C = ocp.Cg(k,:,i);
            Gi = zeros(nnz(k), nz);
            if i == 1, off = C * ocp.x0;
            else
                rows = (i-2)*nx + (1:nx);
                Gi = C * Th(rows, :);  off = C * w(rows);
            end
            if i <= N, Gi(:, (i-1)*nu + (1:nu)) = Gi(:, (i-1)*nu + (1:nu)) + ocp.Dg(k,:,i); end
            G = [G; Gi];  cl = [cl; ocp.lg(k,i) - off];  cu = [cu; ocp.ug(k,i) - off]; %#ok<AGROW>
            z1 = inf(nnz(k), 2);  z2 = zeros(nnz(k), 2);
            if isfield(ocp, 'zlg'), zl = exg(ocp.zlg);  zu = exg(ocp.zug);  z1 = [zl(k,i), zu(k,i)]; end
            if isfield(ocp, 'Zlg'), Zl = exg(ocp.Zlg);  Zu = exg(ocp.Zug);  z2 = [Zl(k,i), Zu(k,i)]; end
            W1 = [W1; z1];  W2 = [W2; z2]; %#ok<AGROW>
        end
    end
    soft_args = {};
    if any(isfinite(W1(:)))
        soft_args = {'PenaltyL1', W1};
        if any(W2(:)), soft_args = [soft_args, {'PenaltyL2', W2}]; end
    end

    % --- Dense QP
    z_init = [];
    if isfield(ocp, 'u_init'), z_init = ocp.u_init(:); end
    % Stationarity in the OCP's units: the condensed gradient residual is Theta'*r_x + r_u, with r_x,
    % r_u the per-stage residuals the Riccati IPM tests. For unstable dynamics over long horizons
    % Theta grows exponentially, and an absolute 1e-8 on the condensed residual lies below
    % round-off (cart-pole, 4 s: H spans 1e3..1e10, and solved QPs ran to MaxIter).
    o = opts;  o.TolStat = opts.TolStat * max(1, norm(Th, 1));
    args = namedargs2cell(o);
    [z, qinfo] = dense_qp_ipm(H, g, ocp.lb(:), ocp.ub(:), G, cl, cu, args{:}, soft_args{:}, 'z_init', z_init);

    % --- Expand: states, bound multipliers, then dynamics multipliers by the adjoint recursion
    u = reshape(z, nu, N);
    x = [ocp.x0, reshape(Th*z + w, nx, N)];
    lam_lbx = zeros(nx, N+1);  lam_ubx = zeros(nx, N+1);
    ns = nnz(bnd);  sr = 1:ns;  gr = ns+1 : numel(cl);    % state-bound rows, then general rows
    tmp = zeros(nx*N, 1); tmp(bnd) = qinfo.lam_cl(sr); lam_lbx(:,2:end) = reshape(tmp, nx, N);
    tmp = zeros(nx*N, 1); tmp(bnd) = qinfo.lam_cu(sr); lam_ubx(:,2:end) = reshape(tmp, nx, N);
    [lam_lg, lam_ug] = deal(zeros(ng, N+1));
    lam_lg(gbnd) = qinfo.lam_cl(gr);  lam_ug(gbnd) = qinfo.lam_cu(gr);   % column-major: node by node
    dl = lam_ug - lam_lg;
    pi_ = zeros(nx, N+1);
    pi_(:,N+1) = ocp.Q(:,:,N+1) * x(:,N+1) + ocp.q(:,N+1) - lam_lbx(:,N+1) + lam_ubx(:,N+1);
    if ng > 0, pi_(:,N+1) = pi_(:,N+1) + ocp.Cg(:,:,N+1)' * dl(:,N+1); end
    for i = N:-1:2
        pi_(:,i) = ocp.Q(:,:,i) * x(:,i) + S(:,:,i)' * u(:,i) + ocp.q(:,i) + A(:,:,i)' * pi_(:,i+1) ...
                   - lam_lbx(:,i) + lam_ubx(:,i);
        if ng > 0, pi_(:,i) = pi_(:,i) + ocp.Cg(:,:,i)' * dl(:,i); end
    end

    sol.x = x; sol.u = u; sol.pi = pi_;
    sol.lam_lb = reshape(qinfo.lam_lb, nu, N);
    sol.lam_ub = reshape(qinfo.lam_ub, nu, N);
    sol.lam_lbx = lam_lbx;  sol.lam_ubx = lam_ubx;
    sol.slack_lbx = zeros(nx, N+1);  sol.slack_ubx = zeros(nx, N+1);
    if ng > 0, sol.lam_lg = lam_lg;  sol.lam_ug = lam_ug;  [sol.slack_lg, sol.slack_ug] = deal(zeros(ng, N+1)); end
    if isfield(qinfo, 'slack_cl')
        tmp = zeros(nx*N, 1); tmp(bnd) = qinfo.slack_cl(sr); sol.slack_lbx(:,2:end) = reshape(tmp, nx, N);
        tmp = zeros(nx*N, 1); tmp(bnd) = qinfo.slack_cu(sr); sol.slack_ubx(:,2:end) = reshape(tmp, nx, N);
        if ng > 0, sol.slack_lg(gbnd) = qinfo.slack_cl(gr);  sol.slack_ug(gbnd) = qinfo.slack_cu(gr); end
        qinfo = rmfield(qinfo, {'slack_cl', 'slack_cu'});
    end
    info = rmfield(qinfo, {'lam_lb', 'lam_ub', 'lam_cl', 'lam_cu'});
end
