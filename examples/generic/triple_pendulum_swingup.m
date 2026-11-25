%% rtimpc example: triple pendulum swing-up at 100 Hz
%
% The identified triple pendulum on a cart of Glück, Eder & Kugi ("Swing-up control of a triple
% pendulum on a cart with experimental validation", Automatica 49(3), 2013): three arms swung from
% all hanging at rest to all upright, under the rig's limits (rail |s| <= 0.7 m, cart speed
% |s'| <= 3 m/s, acceleration |s''| <= 22 m/s^2; the input is the cart acceleration). The paper uses
% an offline trajectory and a time-variant Riccati controller; here one NMPC does it all, from a
% cold start: no offline plan, the first guess is the rig left hanging with zero input.
%
% Set-up:
%   - 100 Hz feedback, 4 s horizon on a two-tier grid: 0.08 s at 10 ms, then 40 ms intervals
%     (DecimationFactors [1 4], StepsPerTier 8; N = 106 instead of 400). Its closed loops stay
%     close to the full 10 ms grid's, at roughly a quarter of its step time.
%   - Cost as the NMPC suite's triple_pendulum, per 10 ms sample (its 20 Hz weights x 0.2): arm angles,
%     cart, velocities and input; terminal weight 10x. The rail and cart-speed limits are soft.
%   - Input-rate weight RateR = 10 per 10 ms change (rtimpc applies it per sample of the fine grid,
%     so a 40 ms interval's change gets 10/4: the same smoothness penalty on any grid). Without it
%     the input chatters at the bounds.
%   - Cold first step: up to 300 SQP iterations, unconverged but enough to set up the swing (run to
%     convergence, the first plan swings faster but pushes the cart over the soft rail limit). Then
%     2 SQP iterations per step with the merit line search. A single iteration per step fails from
%     this start: the swing is chaotic, and plain RTI's full steps overshoot from the unconverged
%     first plan.
%   - After a failed step the next one keeps the plan (OnFailure 'keep-primal'): no cold restart,
%     which would cost a 300-iteration step inside a 10 ms budget.
% Analytic Jacobians (checked by validate); all shooting intervals integrated in one vectorised call
% (Model.IntervalFcn: RK4 with sensitivities, as rtimpc's own). Step times are plain MATLAB, far from the 10 ms
% budget: the example shows the controller's behaviour, not real-time execution.

clear
here = fileparts(mfilename('fullpath'));
root = fullfile(here, '..', '..');  addpath(root, fullfile(root, 'qp'));

par = rig_parameters();
Ts = 0.01;  Horizon = 4;                                 % 100 Hz, 4 s horizon
FineTime = 0.08;  Coarse = 4;                            % 10 ms intervals for 0.08 s, then 40 ms
SQPIter = 2;  FirstIter = 300;                           % SQP iterations per step; on the cold first step
RateR = 10;                                              % input-rate weight per 10 ms change
T = 6;                                                   % closed-loop run [s]
SaveRun = '';                                            % e.g. 'triple_pendulum_run.mat' to keep X, U
ColdStart = '';                                          % e.g. 'triple_pendulum_cold.mat': saves the first plan, then reuses it
                                                         % as the first step's initial guess (skips the cold solve)

N = round(Horizon / Ts);  nfine = round(FineTime / Ts);
w = 0.2;                                                 % per-sample weights: the suite's (per 50 ms) x 10/50
Q = diag([100, 100, 100, 10, 0.1, 0.1, 0.1, 1]);         % phi1..3, s, phi1..3', s'

nmpc = rtimpc(8, 1, 8, Ts, N);                           % x = [phi1; phi2; phi3; s; phi1'; phi2'; phi3'; s'], u = s''
nmpc.Model.StateFcn = @(x, u) dynamics(x, u, par);
nmpc.Jacobian.StateFcn = @(x, u) dynamics_jacobian(x, u, par);
nmpc.Weights.Qy = w * Q;  nmpc.Weights.R = w * 1e-3;  nmpc.Weights.Qyf = 10 * Q;
nmpc.Weights.RateR = RateR;
nmpc.Constraints.InputMin = -22;  nmpc.Constraints.InputMax = 22;
nmpc.Constraints.StateMin = [-inf(3,1); -0.7; -inf(3,1); -3];
nmpc.Constraints.StateMax = [ inf(3,1);  0.7;  inf(3,1);  3];
nmpc.Constraints.StateSoftL1 = 1e4;                      % soft rail and speed limits
op = nmpc.Optimisation;
op.IntegratorSubsteps = 2;                               % RK4, 2 steps per interval (20 ms on the coarse ones)
op.DecimationFactors = [1 Coarse];  op.StepsPerTier = nfine;
op.FirstStepSQPMaxIter = FirstIter;  op.SQPMaxIter = SQPIter;
op.OnFailure = 'keep-primal';
nmpc.Optimisation = op;
nmpc.Model.IntervalFcn = @(X, U, d) rk4_intervals(X, U, d, Ts, op.IntegratorSubsteps, par);   % all intervals in one call

jac = nmpc.validate([0.3; 2.9; -1.2; 0.1; 0.5; -0.8; 1.1; 0.4], 3.0);   % Jacobians vs central differences

%% Closed loop (plant: the same model, integrated at 1 ms)

steps = round(T / Ts);
x = [pi; pi; pi; 0; 0; 0; 0; 0];                         % all arms hanging, at rest, cart centred
X = zeros(8, steps+1);  U = zeros(1, steps);  X(:,1) = x;
tc = zeros(1, steps);  sqpit = zeros(1, steps);  flag = zeros(1, steps);
live = ~batchStartupOptionUsed;
if live, status = status_line('Closed loop: cold-start SQP for the first plan (up to a few minutes)...'); end
guess = {};
if ~isempty(ColdStart) && isfile(ColdStart)                % the saved first plan: a warm step, not another cold solve
    G = load(ColdStart);  guess = {'InitialGuess', struct('X', G.X, 'U', G.U)};
    nmpc.Optimisation.FirstStepSQPMaxIter = SQPIter;
end
for k = 1:steps
    if k == 1, [u, info] = nmpc.step(x, zeros(8, 1), guess{:}); else, [u, info] = nmpc.step(x, zeros(8, 1)); end
    if k == 1 && ~isempty(ColdStart) && isempty(guess), [C.X, C.U] = nmpc.plan();  save(ColdStart, '-struct', 'C'); end
    tc(k) = info.TimePreparation + info.TimeFeedback + info.TimeLineSearch;
    sqpit(k) = info.SQPIterations;  flag(k) = info.ExitFlag;
    for s = 1:10, x = rk4(@(z) dynamics(z, u, par), x, Ts/10); end
    X(:,k+1) = x;  U(k) = u;
    if live
        th = rad2deg(atan2(sin(x(1:3)), cos(x(1:3))));
        status = status_line(sprintf('Closed loop: step %3d/%d  t = %4.2f s | arms %+6.1f %+6.1f %+6.1f deg  s = %+.3f m  s'''' = %+5.1f | %5.0f ms', ...
            k, steps, k*Ts, th, x(4), u, 1e3*tc(k)), status);
    end
end
if live, fprintf('\n'); end

print_report(X, U, tc, sqpit, flag, jac, nmpc, Ts, nfine + (N - nfine) / Coarse, 1e3 * Coarse * Ts)
if ~isempty(SaveRun), save(SaveRun, 'X', 'U', 'Ts', 'tc', 'flag'); fprintf('  Saved %s\n\n', SaveRun); end

if usejava('desktop')
    t = (0:steps) * Ts;
    subplot(3,1,1); plot(t, rad2deg(atan2(sin(X(1:3,:)), cos(X(1:3,:))))); ylabel('\phi [deg]'); legend('\phi_1', '\phi_2', '\phi_3'); grid on
    subplot(3,1,2); plot(t, X(4,:)); hold on; yline([-0.7 0.7], 'k--'); hold off; ylabel('s [m]'); grid on
    subplot(3,1,3); stairs(t(1:end-1), U); hold on; yline([-22 22], 'k--'); hold off; ylabel('s'''' [m/s^2]'); xlabel('t [s]'); grid on
end


%% ---- Report ----

function print_report(X, U, tc, sqpit, flag, jac, nmpc, Ts, M, coarse_ms)
    steps = numel(U);  th = atan2(sin(X(1:3,:)), cos(X(1:3,:)));
    ms = 1e3 * tc;  rest = ms(2:end);  jerr = max(cell2mat(struct2cell(jac)));
    up = all(abs(th) < deg2rad(5), 1);  k_up = find(~up, 1, 'last') + 1;
    flips = nnz(U(1:end-1) .* U(2:end) < 0);  jump = max(abs(diff(U)));  sat = nnz(abs(U) > 21.9);

    fprintf('\n=== rtimpc: triple pendulum swing-up (Glück, Eder & Kugi rig) ===\n')
    fprintf('  Grid       Ts = %g ms, %.1f s horizon, %d shooting intervals (two-tier: 10 ms, then %d ms)\n', 1e3*Ts, nmpc.PredictionHorizon*Ts, M, coarse_ms)
    fprintf('  Solver     cold start: SQP (<= %d its), then %d SQP its/step with the line search\n', ...
        nmpc.Optimisation.FirstStepSQPMaxIter, nmpc.Optimisation.SQPMaxIter)
    fprintf('  Jacobians  max rel. error %.1e vs central differences  [%s]\n', jerr, ternary(jerr < 1e-6, 'ok', 'FAIL'))
    fprintf('\n  Closed loop: %.1f s, %d steps\n', steps*Ts, steps)
    if k_up <= size(th, 2)
        fprintf('    Swing-up     all three arms within 5 deg of upright from t = %.2f s onwards\n', (k_up-1)*Ts)
    else
        fprintf('    Swing-up     NOT upright at the end of the run\n')
    end
    fprintf('    Final state  arms %+.2f %+.2f %+.2f deg, cart %+.3f m\n', rad2deg(th(:,end)), X(4,end))
    fprintf('    Limits       max |s| = %.3f m (0.7), max |s''| = %.2f m/s (3), input at +-22 on %d of %d steps\n', ...
        max(abs(X(4,:))), max(abs(X(8,:))), sat, steps)
    fprintf('    Input        %d sign changes, largest change between samples %.1f m/s^2\n', flips, jump)
    fprintf('    Solver       %s; SQP its per step median %d\n', ternary(nnz(flag < 0) == 0, 'no failed steps', ...
        sprintf('%d FAILED steps', nnz(flag < 0))), round(median(sqpit(2:end))))
    fprintf('\n  Compute time per step [ms]  (budget %g ms; plain MATLAB)\n', 1e3*Ts)
    fprintf('    first step                      %8.0f   cold SQP: %d its\n', ms(1), sqpit(1))
    fprintf('    later steps median / p95 / max  %5.0f / %.0f / %.0f\n\n', median(rest), prctile_(rest, 95), max(rest))
end

function n = status_line(msg, n_prev)
    if nargin < 2, n_prev = 0; end
    fprintf('%s%s', repmat(sprintf('\b'), 1, n_prev), msg);
    if numel(msg) < n_prev, fprintf('%s%s', blanks(n_prev - numel(msg)), repmat(sprintf('\b'), 1, n_prev - numel(msg))); end
    n = numel(msg);
end
function s = ternary(b, a, c), if b, s = a; else, s = c; end, end
function q = prctile_(v, p), v = sort(v(:));  q = v(max(1, ceil(p/100 * numel(v)))); end


%% ---- Model ----

function par = rig_parameters()
% Identified parameters (Glück, Eder & Kugi 2013, Section 3)
    par = struct('m', [0.876, 0.938, 0.553], 'l', [0.323, 0.419, 0.484], 'a', [0.215, 0.269, 0.226], ...
                 'J', [0.013, 0.024, 0.018], 'd', [0.215, 0.002, 0.002], 'g', 9.81);
end

function dx = dynamics(x, u, par)
% Absolute arm angles from the upright vertical; the cart acceleration is the input (s'' = u).
% From the Lagrangian of the paper's eq. (2):
%   sum_k M_jk cos(phi_j - phi_k) phi_k'' + sum_k M_jk sin(phi_j - phi_k) phi_k'^2
%     - b_j cos(phi_j) u - g b_j sin(phi_j) + dR/dphi_j' = 0,
% M_jk = sum_i m_i lam_ij lam_ik + J_j delta_jk, b_j = sum_i m_i lam_ij, lam_ik the lever of arm k
% in the position of arm i's centre of gravity (l_k for k < i, a_i for k = i, else 0).
    m = par.m;  l = par.l;  a = par.a;  J = par.J;  d = par.d;  g = par.g;
    lam = [a(1), 0, 0; l(1), a(2), 0; l(1), l(2), a(3)];
    Mc = lam' * diag(m) * lam + diag(J);  b = (m * lam)';
    phi = x(1:3);  w = x(5:7);
    c12 = cos(phi(1) - phi(2));  c13 = cos(phi(1) - phi(3));  c23 = cos(phi(2) - phi(3));
    s12 = sin(phi(1) - phi(2));  s13 = sin(phi(1) - phi(3));  s23 = sin(phi(2) - phi(3));
    M  = [Mc(1,1), Mc(1,2)*c12, Mc(1,3)*c13;  Mc(2,1)*c12, Mc(2,2), Mc(2,3)*c23;  Mc(3,1)*c13, Mc(3,2)*c23, Mc(3,3)];
    MS = [0, Mc(1,2)*s12, Mc(1,3)*s13;  -Mc(2,1)*s12, 0, Mc(2,3)*s23;  -Mc(3,1)*s13, -Mc(3,2)*s23, 0];
    fr = [d(1)*w(1) - d(2)*(w(2) - w(1)); d(2)*(w(2) - w(1)) - d(3)*(w(3) - w(2)); d(3)*(w(3) - w(2))];
    rhs = b .* cos(phi) * u + g * b .* sin(phi) - MS * (w.^2) - fr;
    dx = [w; x(8); M \ rhs; u];
end

function [A, B] = dynamics_jacobian(x, u, par)
% df/dx and df/du of dynamics. With M_jk = Mc_jk cos(phi_j - phi_k) and the arm accelerations
% acc = M \ rhs:  d acc = M \ (d rhs - dM acc).
    m = par.m;  l = par.l;  a = par.a;  J = par.J;  d = par.d;  g = par.g;
    lam = [a(1), 0, 0; l(1), a(2), 0; l(1), l(2), a(3)];
    Mc = lam' * diag(m) * lam + diag(J);  b = (m * lam)';
    phi = x(1:3);  w = x(5:7);
    P = phi - phi.';  C = cos(P);  S = sin(P);               % C(j,k) = cos(phi_j - phi_k)
    M = Mc .* C;  MS = Mc .* S;
    Dfr = [d(1) + d(2), -d(2), 0; -d(2), d(2) + d(3), -d(3); 0, -d(3), d(3)];
    rhs = b .* cos(phi) * u + g * b .* sin(phi) - MS * (w.^2) - Dfr * w;
    acc = M \ rhs;
    Dq = -Mc .* C .* (w.^2).';  Dq = Dq - diag(sum(Dq, 2));  % d(MS w.^2)/dphi
    Dm = MS .* acc.';           Dm = Dm - diag(sum(Dm, 2));  % d(M acc)/dphi, acc held
    dphi = M \ (diag(g * b .* cos(phi) - b .* sin(phi) * u) - Dq - Dm);
    dw   = M \ (-2 * MS .* w.' - Dfr);
    du   = M \ (b .* cos(phi));
    A = zeros(8);  A(1:3, 5:7) = eye(3);  A(4, 8) = 1;
    A(5:7, 1:3) = dphi;  A(5:7, 5:7) = dw;
    B = [zeros(4, 1); du; 1];
end

function [Xn, Ad, Bd] = rk4_intervals(X, U, d, Ts, ns, par)
% All shooting intervals at once (columns): ns RK4 steps of d(i)*Ts/ns each, with forward
% sensitivities, the same scheme as rtimpc's own RK4. Xn 8 x K, Ad 8 x 8 x K, Bd 8 x 1 x K.
    K = size(X, 2);  h = reshape(d, 1, K) * Ts / ns;  hp = reshape(h, 1, 1, K);  I = eye(8);
    sens = nargout > 1;  Xn = X;
    if sens, Ad = repmat(I, 1, 1, K);  Bd = zeros(8, 1, K); end
    for s = 1:ns
        if ~sens
            k1 = dynamics_batch(Xn, U, par);              k2 = dynamics_batch(Xn + h/2 .* k1, U, par);
            k3 = dynamics_batch(Xn + h/2 .* k2, U, par);  k4 = dynamics_batch(Xn + h .* k3, U, par);
            Xn = Xn + h/6 .* (k1 + 2*k2 + 2*k3 + k4);
            continue
        end
        [k1, A1, B1] = dynamics_batch(Xn, U, par);
        [k2, A2, B2] = dynamics_batch(Xn + h/2 .* k1, U, par);
        K2x = pmul(A2, I + hp/2 .* A1);  K2u = pmul(A2, hp/2 .* B1) + B2;
        [k3, A3, B3] = dynamics_batch(Xn + h/2 .* k2, U, par);
        K3x = pmul(A3, I + hp/2 .* K2x); K3u = pmul(A3, hp/2 .* K2u) + B3;
        [k4, A4, B4] = dynamics_batch(Xn + h .* k3, U, par);
        K4x = pmul(A4, I + hp .* K3x);   K4u = pmul(A4, hp .* K3u) + B4;
        Xn = Xn + h/6 .* (k1 + 2*k2 + 2*k3 + k4);
        Ax = I + hp/6 .* (A1 + 2*K2x + 2*K3x + K4x);
        Bu = hp/6 .* (B1 + 2*K2u + 2*K3u + K4u);
        Bd = pmul(Ax, Bd) + Bu;  Ad = pmul(Ax, Ad);
    end
end

function [F, A, B] = dynamics_batch(X, U, par)
% dynamics and dynamics_jacobian for all columns at once (3 x 3 x K pages for the arm terms)
    K = size(X, 2);
    m = par.m;  l = par.l;  a = par.a;  J = par.J;  d = par.d;  g = par.g;
    lam = [a(1), 0, 0; l(1), a(2), 0; l(1), l(2), a(3)];
    Mc = lam' * diag(m) * lam + diag(J);  b = (m * lam)';
    Dfr = [d(1) + d(2), -d(2), 0; -d(2), d(2) + d(3), -d(3); 0, -d(3), d(3)];
    phi = X(1:3,:);  w = X(5:7,:);
    P = reshape(phi, 3, 1, K) - reshape(phi, 1, 3, K);  C = cos(P);  S = sin(P);
    M = Mc .* C;  MS = Mc .* S;  Mi = inv3(M);
    rhs = b .* cos(phi) .* U + g * b .* sin(phi) - pmulv(MS, w.^2) - Dfr * w;
    acc = pmulv(Mi, rhs);
    F = [w; X(8,:); acc; U];
    if nargout < 2, return, end
    Dq = -Mc .* C .* reshape(w.^2, 1, 3, K);  Dq = Dq - eye(3) .* sum(Dq, 2);
    Dm = MS .* reshape(acc, 1, 3, K);         Dm = Dm - eye(3) .* sum(Dm, 2);
    Dg = eye(3) .* reshape(g * b .* cos(phi) - b .* sin(phi) .* U, 3, 1, K);
    A = zeros(8, 8, K);  A(1:3, 5:7, :) = repmat(eye(3), 1, 1, K);  A(4, 8, :) = 1;
    A(5:7, 1:3, :) = pmul(Mi, Dg - Dq - Dm);
    A(5:7, 5:7, :) = pmul(Mi, -2 * MS .* reshape(w, 1, 3, K) - Dfr);
    B = zeros(8, 1, K);  B(5:7, 1, :) = reshape(pmulv(Mi, b .* cos(phi)), 3, 1, K);  B(8, 1, :) = 1;
end

function Mi = inv3(M)
% Inverses of 3 x 3 pages by the adjugate
    a = M(1,1,:); b = M(1,2,:); c = M(1,3,:); d = M(2,1,:); e = M(2,2,:); f = M(2,3,:); g = M(3,1,:); h = M(3,2,:); k = M(3,3,:);
    A = e.*k - f.*h;  B = f.*g - d.*k;  C = d.*h - e.*g;
    det = a.*A + b.*B + c.*C;
    Mi = [A, c.*h - b.*k, b.*f - c.*e;  B, a.*k - c.*g, c.*d - a.*f;  C, b.*g - a.*h, a.*e - b.*d] ./ det;
end

function C = pmul(A, B)
% Page-wise A * B (pagemtimes where available)
    if exist('pagemtimes', 'builtin') || exist('pagemtimes', 'file'), C = pagemtimes(A, B); return, end
    C = permute(sum(permute(A, [1 2 4 3]) .* permute(B, [4 1 2 3]), 2), [1 3 4 2]);
end

function y = pmulv(A, x)
% Page-wise A * x(:,i): A n x n x K, x n x K
    y = reshape(sum(A .* reshape(x, 1, size(x, 1), []), 2), size(A, 1), []);
end

function x = rk4(f, x, h)
    k1 = f(x); k2 = f(x + h/2*k1); k3 = f(x + h/2*k2); k4 = f(x + h*k3);
    x = x + h/6*(k1 + 2*k2 + 2*k3 + k4);
end
