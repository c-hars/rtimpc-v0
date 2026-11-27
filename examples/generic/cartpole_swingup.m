%% rtimpc example: cart-pole swing-up on a short track
%
% Classic NMPC benchmark: swing a pole up from hanging (theta = pi) to upright (theta = 0) and bring the cart back to x = 0, with a bounded force, a smoothness cost on force changes, and a track only 0.2 m long (a state constraint on the cart position).
%
% Nothing in the cost says how to swing up. With a track this short, the controller finds energy pumping by itself, rocking the cart within +-0.1 m until the pole has enough energy to be caught upright.
%
% Shows the full rtimpc workflow: model and analytic Jacobians, Jacobian validation, a cold-start SQP, then real-time iterations.
%
% Comparison with nlmpc (Controller = 'nlmpc'; needs the Model Predictive Control Toolbox): nlmpc solves the same OCP (the same RK4 discretisation, cost and bounds) to convergence with fmincon at every step. The one difference is the track limit: hard in nlmpc, an exact L1 penalty in rtimpc.
%
% nlmpcMaxIter caps nlmpc's iterations per step after the first, to get as close to RTI as nlmpc allows. Two settings make the cap usable: UseSuboptimalSolution, so the capped iterate is applied (nlmpc would otherwise hold the last input), and ConstraintTolerance 1e3, so an iterate that does not yet satisfy the dynamics is not rejected. Capped this way, nlmpc fails the swing-up. The loose tolerance also loosens its hard track limit, but that is not the cause: capped nlmpc fails with the track removed too.

clear
here = fileparts(mfilename('fullpath'));
root = fullfile(here, '..', '..');  addpath(root, fullfile(root, 'qp'));

p = struct('mc', 1.0, 'mp', 0.2, 'l', 0.5, 'g', 9.81);   % cart mass, pole mass, pole length, gravity
Ts = 0.05;  N = 80;                                      % 20 Hz, 4 s horizon
Fmax = 15;                                               % |force| <= 15 N
Xmax = 0.1;                                              % |cart position| <= 0.1 m (track length 0.2 m)
Animate = true;                                          % ASCII stop-motion replay in the terminal
Controller = 'rtimpc';                                   % 'rtimpc' | 'nlmpc' (needs the MPC Toolbox)
nlmpcMaxIter = Inf;                                      % nlmpc SQP iterations per step after the first (Inf: to convergence; see the header)
SQPIter = 1;                                             % rtimpc SQP iterations per step after the cold start: 1 = RTI; 50 = to convergence, as nlmpc

nmpc = rtimpc(4, 1, 2, Ts, N);                           % nx = 4 [x; xdot; th; thdot], nu = 1, ny = 2
nmpc.Model.StateFcn     = @(x,u) cartpole(x, u, p);
nmpc.Jacobian.StateFcn  = @(x,u) cartpole_jac(x, u, p);
nmpc.Model.OutputFcn    = @(x,u) x([1 3]);               % track cart position and pole angle
nmpc.Jacobian.OutputFcn = @(x,u) [1 0 0 0; 0 0 1 0];
nmpc.Weights.Qy  = diag([5, 20]);
nmpc.Weights.Qyf = diag([50, 200]);
nmpc.Weights.R   = 0.1;
nmpc.Weights.RateR = 4;                                  % cost on force changes: smooth, not twitchy
nmpc.Constraints.InputMin = -Fmax;
nmpc.Constraints.InputMax =  Fmax;
nmpc.Constraints.StateMin = [-Xmax; -inf; -inf; -inf];   % only the cart position is bounded
nmpc.Constraints.StateMax = [ Xmax;  inf;  inf;  inf];
nmpc.Constraints.StatePenaltyL1 = 1e5;                   % soft track limit: cost per metre of violation (Inf = hard)
nmpc.Optimisation.SQPMaxIter = SQPIter;                  % after the (full SQP) cold start
nmpc.Optimisation.FirstStepSQPMaxIter = 100;             % the cold swing-up plan needs ~70 SQP its

jac = nmpc.validate([0.3; -0.2; 2.5; 1.0], 3.0);         % Jacobians vs central differences
if strcmp(Controller, 'nlmpc'), nl = build_nlmpc(nmpc, p, Fmax, Xmax);  mvopt = nlmpcmoveopt; end

%% Closed loop (plant: the same model, integrated finely)

T = 10;  steps = round(T/Ts);
x = [0; 0; pi; 0];
X = zeros(4, steps+1); U = zeros(1, steps);
tc = zeros(1, steps); qpit = zeros(1, steps); sqpit = zeros(1, steps); flag = zeros(1, steps);
qpok = true(1, steps); xviol = zeros(1, steps); xmult = zeros(1, steps);
X(:,1) = x;  u = 0;                                      % u_{-1} = 0 for the rate cost
live = ~batchStartupOptionUsed;                          % one-line live status (not in -batch logs)
if live, status = status_line('Closed loop: cold-start solve for the first plan...'); end
for k = 1:steps
    if strcmp(Controller, 'nlmpc')
        c = tic;  [u, mvopt, info] = nlmpcmove(nl, x, u, [0 0], [], mvopt);  tc(k) = toc(c);
        sqpit(k) = info.Iterations;  flag(k) = info.ExitFlag;
        if k == 1 && isfinite(nlmpcMaxIter)              % capped from the second step on
            so = nl.Optimization.SolverOptions;  so.MaxIterations = nlmpcMaxIter;
            so.ConstraintTolerance = 1e3;  nl.Optimization.SolverOptions = so;
            nl.Optimization.UseSuboptimalSolution = true;
        end
    else
        c = tic;  [u, info] = nmpc.step(x, [0; 0], 'InputPrevious', u);  tc(k) = toc(c);
        qpit(k) = info.QPIterations;  sqpit(k) = info.SQPIterations;  flag(k) = info.ExitFlag;
        qpok(k) = strcmp(info.QPStatus, 'solved');
        xviol(k) = info.StateViolation;  xmult(k) = info.StateBoundMultiplier;
    end
    for s = 1:10                                         % plant: RK4 at Ts/10
        x = rk4(@(z) cartpole(z, u, p), x, Ts/10);
    end
    X(:,k+1) = x; U(k) = u;
    if live
        status = status_line(sprintf(['Closed loop: step %3d/%d  t = %5.2f s | theta = %+7.1f deg  ' ...
            'x = %+.3f m  F = %+5.1f N | %4.1f ms, %2d SQP its%s'], k, steps, k*Ts, ...
            rad2deg(atan2(sin(x(3)), cos(x(3)))), x(1), u, 1e3*tc(k), sqpit(k), ...
            ternary(flag(k) < 0, '  FAILED', '')), status);
    end
end
if live, fprintf('\n'); end

if Animate
    ascii_cartpole(X, U, Ts, 'TrackLimit', Xmax, 'PoleLength', p.l);
end
print_report(X, U, tc, qpit, sqpit, flag, qpok, xviol, xmult, jac, nmpc, Fmax, Xmax, Controller, nlmpcMaxIter)

if usejava('desktop') || ~exist('OCTAVE_VERSION', 'builtin')
    t = (0:steps)*Ts;
    subplot(3,1,1); plot(t, X(1,:)); hold on
    yline([-Xmax Xmax], 'k--'); hold off; ylabel('x [m]'); grid on
    subplot(3,1,2); plot(t, X(3,:)); ylabel('\theta [rad]'); grid on
    subplot(3,1,3); stairs(t(1:end-1), U); ylabel('F [N]'); xlabel('t [s]'); grid on
end


%% ---- Report ----

function print_report(X, U, tc, qpit, sqpit, flag, qpok, xviol, xmult, jac, nmpc, Fmax, Xmax, ctrl, nlcap)
    Ts = nmpc.Ts;  N = nmpc.PredictionHorizon;  steps = numel(U);
    th = atan2(sin(X(3,:)), cos(X(3,:)));                % wrap pole angle to (-pi, pi]
    ms = 1e3 * tc;  rti = ms(2:end);
    jerr = max(cell2mat(struct2cell(jac)));

    up = abs(th) < deg2rad(5);                           % "upright": within 5 deg
    k_up = find(~up, 1, 'last') + 1;                     % first sample after which it stays up
    act_x = abs(X(1,2:end)) > Xmax - 1e-3;               % track bound (numerically) active
    sat_u = abs(U) > Fmax - 1e-6;

    nl = strcmp(ctrl, 'nlmpc');
    fprintf('\n=== %s: cart-pole swing-up ===\n', ctrl)
    fprintf('  Problem    nx = %d, nu = %d, ny = %d | Ts = %g ms, N = %d (%.1f s horizon), %s\n', ...
        nmpc.nx, nmpc.nu, nmpc.ny, 1e3*Ts, N, N*Ts, upper(nmpc.Optimisation.Integrator))
    L1 = nmpc.Constraints.StatePenaltyL1(1);
    fprintf('  Weights    Qy = diag(%s), Qyf = diag(%s), R = %g, RateR = %g\n', mat2str(diag(nmpc.Weights.Qy)'), ...
        mat2str(diag(nmpc.Weights.Qyf)'), nmpc.Weights.R, nmpc.Weights.RateR)
    fprintf('  Bounds     |F| <= %g N (hard),  |x| <= %g m (%s)\n', Fmax, Xmax, ...
        ternary(isinf(L1) || nl, 'hard', sprintf('soft, L1 = %g per m', L1)))
    if nl
        if isfinite(nlcap), fprintf('  Solver     nlmpc: fmincon SQP, capped at %d it(s)/step after the first, warm-started\n', nlcap)
        else, fprintf('  Solver     nlmpc: fmincon SQP to convergence every step, warm-started\n')
        end
    else
        fprintf('  Solver     cold start: SQP (<= %d its), then %d SQP it(s)/step, %s IPM\n', ...
            nmpc.Optimisation.FirstStepSQPMaxIter, nmpc.Optimisation.SQPMaxIter, nmpc.Optimisation.QPSolver)
    end
    fprintf('  Jacobians  max rel. error %.1e vs central differences  [%s]\n', jerr, okfail(jerr < 1e-6))

    fprintf('\n  Closed loop: %.1f s, %d steps\n', steps*Ts, steps)
    if k_up <= numel(th)
        fprintf('    Swing-up     upright (|theta| < 5 deg) from t = %.2f s onwards\n', (k_up-1)*Ts)
    else
        fprintf('    Swing-up     NOT upright at the end of the run\n')
    end
    fprintf('    Final state  x = %+.3f m, theta = %+.2f deg\n', X(1,end), rad2deg(th(end)))
    fprintf('    Track        max |x| = %.3f m of %.3f m, bound active %d steps, violation %.1e m\n', ...
        max(abs(X(1,:))), Xmax, nnz(act_x), max([0, abs(X(1,:)) - Xmax]))
    if ~isinf(L1) && ~nl
        bought = xmult(2:end) > 0.99 * L1;               % multiplier at its cap: violating is cheaper
        fprintf('                 bound multiplier max %.0f (L1 = %g): price reached in %d steps, planned violation max %.1e m\n', ...
            max(xmult(2:end)), L1, nnz(bought), max(xviol(2:end)))
    end
    fprintf('    Input        saturated %d/%d steps\n', nnz(sat_u), steps)
    W = nmpc.Weights;  E = X([1 3], 2:end);              % closed-loop cost: the stage cost summed over the run
    J = sum(sum(E .* (W.Qy * E))) + W.R * sum(U.^2) + W.RateR * sum(diff([0, U]).^2);
    fprintf('    Cost         closed-loop %.4g (stage cost summed over the run)\n', J)
    nbad = nnz(flag < 0);
    nunc = nnz(~qpok(2:end));
    if nl                                                % fmincon exit flags: > 0 converged, 0 at max its
        fprintf('    Solver       %s, %d step(s) at the iteration cap\n', ternary(nbad == 0, 'no failed steps', ...
            sprintf('%d FAILED steps', nbad)), nnz(flag == 0))
    else
        fprintf('    Solver       %s, %s\n', ternary(nbad == 0, 'no failed steps', sprintf('%d FAILED steps', nbad)), ...
            ternary(nunc == 0, 'all QPs converged', sprintf('%d QP(s) hit the IPM iteration cap', nunc)))
        fprintf('                 IPM iterations per step median %d, max %d\n', round(median(qpit(2:end))), max(qpit(2:end)))
    end
    fprintf('                 SQP iterations per step median %g, max %d\n', median(sqpit(2:end)), max(sqpit(2:end)))

    fprintf('\n  Compute time per step [ms]  (budget Ts = %g ms; wall time of the solver call)\n', 1e3*Ts)
    fprintf('    first step                   %7.1f   cold solve: %d SQP its, %s\n', ms(1), ...
        sqpit(1), ternary(flag(1) == 1, 'converged', ternary(flag(1) == 0, 'hit max its', ternary(flag(1) > 0, 'converged', 'FAILED'))))
    fprintf('                                           (also includes MATLAB JIT warm-up)\n')
    fprintf('    later steps median / p95        %5.1f / %.1f\n\n', median(rti), prctile_(rti, 95))
end

function s = okfail(b), s = ternary(b, 'ok', 'FAIL'); end
function n = status_line(msg, n_prev)
% Rewrite the current command-window line: back over the previous message, then print this one
    if nargin < 2, n_prev = 0; end
    fprintf('%s%s', repmat(sprintf('\b'), 1, n_prev), msg);
    if numel(msg) < n_prev, fprintf('%s%s', blanks(n_prev - numel(msg)), repmat(sprintf('\b'), 1, n_prev - numel(msg))); end
    n = numel(msg);
end
function s = ternary(b, a, c), if b, s = a; else, s = c; end, end
function q = prctile_(v, p)                              % prctile without the Statistics Toolbox
    v = sort(v(:));  q = v(max(1, ceil(p/100 * numel(v))));
end


%% ---- Model: continuous-time dynamics and analytic Jacobians ----

function dx = cartpole(x, F, p)
% theta = 0 upright; pole as a point mass at distance l
    s = sin(x(3)); c = cos(x(3));
    D = p.mc + p.mp*s^2;
    xdd = (F - p.mp*p.g*s*c + p.mp*p.l*x(4)^2*s) / D;
    thdd = (p.g*s - xdd*c) / p.l;
    dx = [x(2); xdd; x(4); thdd];
end

function [A, B] = cartpole_jac(x, F, p)
    s = sin(x(3)); c = cos(x(3)); w = x(4);
    D   = p.mc + p.mp*s^2;
    N1  = F - p.mp*p.g*s*c + p.mp*p.l*w^2*s;
    xdd = N1 / D;
    dxdd_dth = ((-p.mp*p.g*(c^2 - s^2) + p.mp*p.l*w^2*c)*D - N1*2*p.mp*s*c) / D^2;
    dxdd_dw  = 2*p.mp*p.l*w*s / D;
    dxdd_dF  = 1 / D;
    A = [0, 1, 0, 0;
         0, 0, dxdd_dth, dxdd_dw;
         0, 0, 0, 1;
         0, 0, (p.g*c - dxdd_dth*c + xdd*s)/p.l, -c*dxdd_dw/p.l];
    B = [0; dxdd_dF; 0; -c*dxdd_dF/p.l];
end

%% ---- nlmpc on the same OCP (comparison; needs the MPC Toolbox) ----

function nl = build_nlmpc(nmpc, p, Fmax, Xmax)
% Same discrete model (one RK4 step per sample), cost and bounds as the rtimpc controller
    Ts = nmpc.Ts;  N = nmpc.PredictionHorizon;  W = nmpc.Weights;
    nl = nlmpc(4, 2, 1);
    nl.Ts = Ts;  nl.PredictionHorizon = N;  nl.ControlHorizon = N;
    nl.Model.IsContinuousTime = false;
    nl.Model.StateFcn = @(x, u) rk4_sens(x, u, p, Ts);
    nl.Jacobian.StateFcn = @(x, u) rk4_jac(x, u, p, Ts);
    nl.Model.OutputFcn = @(x, u) x([1 3]);
    nl.Jacobian.OutputFcn = @(x, u) [1 0 0 0; 0 0 1 0];
    nl.Optimization.ReplaceStandardCost = true;            % no slack in the cost: all constraints hard
    mpcverbosity('off');                                   % (nlmpc otherwise notes that at every step)
    nl.Optimization.CustomCostFcn = @(X, U, e, data) ocp_cost(X, U, data, W);
    nl.Jacobian.CustomCostFcn = @(X, U, e, data) ocp_cost_grad(X, U, data, W);
    nl.MV.Min = -Fmax;  nl.MV.Max = Fmax;
    nl.States(1).Min = -Xmax;  nl.States(1).Max = Xmax;  % hard
end

function J = ocp_cost(X, U, data, W)
% rtimpc's cost: Qy on nodes 1..N-1, Qyf on node N, R and RateR on inputs 0..N-1 (u_{-1} = LastMV)
    Y = X(2:end, [1 3]);  u = U(1:end-1);  du = diff([data.LastMV; u]);
    J = sum(sum((Y(1:end-1,:) * W.Qy) .* Y(1:end-1,:))) + Y(end,:) * W.Qyf * Y(end,:)' ...
        + W.R * sum(u.^2) + W.RateR * sum(du.^2);
end

function [Gx, Gmv, Ge] = ocp_cost_grad(X, U, data, W)
    Y = X(2:end, [1 3]);  u = U(1:end-1);  du = diff([data.LastMV; u]);
    Gx = zeros(numel(u), 4);
    Gx(:, [1 3]) = 2 * [Y(1:end-1,:) * W.Qy; Y(end,:) * W.Qyf];
    Gmv = 2 * W.R * u + 2 * W.RateR * (du - [du(2:end); 0]);
    Ge = 0;
end

function [xn, A, B] = rk4_sens(x, u, p, h)
% One RK4 step of the continuous model, with its sensitivities
    f = @(z) cartpole(z, u, p);  J = @(z) cartpole_jac(z, u, p);
    k1 = f(x);            [A1, B1] = J(x);
    x2 = x + h/2*k1;      k2 = f(x2);  [A2, B2] = J(x2);
    x3 = x + h/2*k2;      k3 = f(x3);  [A3, B3] = J(x3);
    x4 = x + h*k3;        k4 = f(x4);  [A4, B4] = J(x4);
    xn = x + h/6*(k1 + 2*k2 + 2*k3 + k4);
    if nargout > 1
        I = eye(4);
        K1x = A1;                    K1u = B1;
        K2x = A2*(I + h/2*K1x);      K2u = A2*(h/2*K1u) + B2;
        K3x = A3*(I + h/2*K2x);      K3u = A3*(h/2*K2u) + B3;
        K4x = A4*(I + h*K3x);        K4u = A4*(h*K3u) + B4;
        A = I + h/6*(K1x + 2*K2x + 2*K3x + K4x);
        B = h/6*(K1u + 2*K2u + 2*K3u + K4u);
    end
end

function [A, B] = rk4_jac(x, u, p, h)
    [~, A, B] = rk4_sens(x, u, p, h);
end

function x = rk4(f, x, h)
    k1 = f(x); k2 = f(x + h/2*k1); k3 = f(x + h/2*k2); k4 = f(x + h*k3);
    x = x + h/6*(k1 + 2*k2 + 2*k3 + k4);
end
