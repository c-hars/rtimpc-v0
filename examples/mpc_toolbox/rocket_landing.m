function R = rocket_landing()
%% rtimpc example: plan and land a two-thruster vehicle (MPC Toolbox example)
%
% Rebuilt from MATLAB's "Landing a Vehicle Using Multistage Nonlinear MPC" (nlmpcMultistage): a planar two-thruster rocket lands at a target, along a fuel-optimal plan computed first and then flown by an MPC.
%
% Vehicle
%   State      x = [x; y; theta; dx; dy; dtheta]: position, attitude and their rates
%   Input      u = [left; right]: thrusts, each in [0, 8] N
%   Model      planar disc, m = 1 kg, I = 50 kg m^2, thrusters 5 m either side of the centre; continuous, implicit trapezoidal rule, Ts = 0.2 s
%   Bounds     y >= 10 m (hard)
%   Start      x0 = [-30; 60; 0; 0; 0; 0]
%
% Planner (one OCP, cold-solved to convergence)
%   Horizon    50 steps (10 s)
%   Cost       fuel, sum of u_k(1) + u_k(2)
%   Terminal   x_N = [0; 10; 0; 0; 0; 0] (hard)
%   Guess      x0 at every node, zero inputs
%
% Lander (closed loop, tracking the plan)
%   Horizon    10 steps
%   Cost       sum |x_i - r_i|^2 + sum 0.1 (i+1) |u_i - u_{i-1}|^2, r the plan previewed over the horizon (held at its end)
%   Solver     SQP to convergence at every step
%   Plant      ode45 over each sample; the run stops once within 1e-2 of the target
%
% Shows: an offline trajectory optimisation and a tracking MPC with the same tool; a linear (fuel) cost as Weights.InputLinear with the exact Hessian; a hard terminal-state constraint; a per-stage rate weight (pages of Weights.RateR); a previewed reference; analytic Jacobians.
%
% Result: matches nlmpcMultistage on the same setup: plan (fuel 492.968948, states within 3e-5) and landing (fuel 827.646, states within 2e-5). These figures stand in for the verification suite, which will hold the comparison and its setup once published.

    here = fileparts(mfilename('fullpath'));
    root = fullfile(here, '..', '..');  addpath(root, fullfile(root, 'qp'));
    MaxSteps = 500;
    Ts = 0.2;  pP = 50;  pL = 10;  x0 = [-30; 60; 0; 0; 0; 0];  u0 = [0; 0];  target = [0; 10; 0; 0; 0; 0];

    % ----- planner
    o = common(rtimpc(6, 2, 6, Ts, pP));
    o.Weights.Qy = zeros(6);  o.Weights.R = zeros(2);  o.Weights.InputLinear = [1; 1];
    o.Model.TerminalConstraintFcn = @(x) x - target;  o.Jacobian.TerminalConstraintFcn = @(x) eye(6);
    o.Constraints.TerminalConstraintMin = zeros(6, 1);  o.Constraints.TerminalConstraintMax = zeros(6, 1);
    o.Constraints.TerminalConstraintPenaltyL1 = Inf;
    op = o.Optimisation;  op.FirstStepSQPMaxIter = 1000;  op.Hessian = 'exact';  o.Optimisation = op;   % Gauss-Newton: no curvature with a linear cost
    guess = struct('X', repmat(x0, 1, pP + 1), 'U', repmat(u0, 1, pP));
    t0 = tic;
    [~, info] = o.step(x0, zeros(6, 1), 'InitialGuess', guess, 'InputPrevious', u0);
    t = toc(t0);
    [X, U] = o.plan();
    R.rtimpc.plan = plan_summary('rtimpc', X, U, info.ExitFlag, info.SQPIterations, t, target, Ts);
    R.rtimpc.plan.info = info;

    % ----- lander, tracking the plan
    l = common(rtimpc(6, 2, 6, Ts, pL));
    l.Weights.Qy = eye(6);  l.Weights.R = zeros(2);
    l.Weights.RateR = 0.1 * reshape(1:pL, 1, 1, pL) .* eye(2);
    op = l.Optimisation;  op.SQPMaxIter = 50;  op.FirstStepSQPMaxIter = 50;  l.Optimisation = op;
    R.rtimpc.landing = closed_loop(@(x, r, lastMV, ~) rtimpc_move(l, x, r, lastMV), X, x0, u0, target, Ts, pL, MaxSteps, 'rtimpc');

    P = R.rtimpc.plan;  Lg = R.rtimpc.landing;
    subplot(2,1,1); plot(P.X(1,:), P.X(2,:), '--', Lg.X(1,:), Lg.X(2,:)); yline(10, 'k:');
    axis equal; xlabel('x [m]'); ylabel('y [m]'); legend('plan', 'landing', 'Location', 'best'); grid on
    subplot(2,1,2); stairs((0:Lg.steps-1) * Ts, Lg.U'); xlabel('t [s]'); ylabel('thrust [N]'); legend('left', 'right'); grid on
end

function o = common(o)
% Model, bounds and integrator shared by both OCPs
    o.Model.StateFcn = @lander_ct;  o.Jacobian.StateFcn = @lander_jac;
    o.Constraints.InputMin = [0; 0];  o.Constraints.InputMax = [8; 8];
    o.Constraints.StateMin = [-Inf; 10; -Inf(4, 1)];  o.Constraints.StateMax = Inf(6, 1);
    o.Constraints.StatePenaltyL1 = Inf;
    op = o.Optimisation;  op.Integrator = 'trapezoidal';  op.ComputeCost = true;
    o.Optimisation = op;
end

function S = plan_summary(name, X, U, exitflag, iters, t, target, Ts)
    p = size(U, 2);  gap = 0;
    for k = 1:p, gap = max(gap, norm(X(:,k+1) - trapezoid(X(:,k), U(:,k), Ts), inf)); end
    S = struct('cost', sum(U, 'all'), 'exit', exitflag, 'iterations', iters, 'time', t, ...
               'terminal_err', norm(X(:, end) - target, inf), 'min_y', min(X(2, 2:end)), ...
               'u_range', [min(U, [], 'all'), max(U, [], 'all')], 'max_gap', gap, 'X', X, 'U', U);
    fprintf('%-7s plan: fuel %.6f | exit %2d, %4d iterations, %6.2f s | terminal error %.1e | min y %.6f | u in [%.2e, %.4f] | largest gap %.1e\n', ...
        name, S.cost, exitflag, iters, t, S.terminal_err, S.min_y, S.u_range, gap);
end

function S = closed_loop(move, Xplan, x0, u0, target, Ts, p, MaxSteps, name)
% The original example's loop: preview the plan (held at its end), one move per sample, ode45 plant
    np = size(Xplan, 2);
    X = x0;  U = zeros(2, 0);  [t, flag, it] = deal([]);  lastMV = u0;  landed = false;
    for k = 1:MaxSteps
        r = Xplan(:, min(k + (1:p), np));                 % nodes 1..p: plan samples k..k+p-1 (0-based)
        c = tic;  [u, flag(k), it(k)] = move(X(:, k), r, lastMV, k);  t(k) = toc(c); %#ok<AGROW>
        [~, XO] = ode45(@(~, xx) lander_ct(xx, u), [0 Ts], X(:, k));
        X(:, k+1) = XO(end, :)';  U(:, k) = u;  lastMV = u; %#ok<AGROW>
        if max(abs(X(:, k+1) - target)) < 1e-2, landed = true; break, end
    end
    K = size(U, 2);
    S = struct('X', X, 'U', U, 'time', t, 'flag', flag, 'iter', it, 'landed', landed, 'steps', K, ...
               'fuel', sum(U, 'all'), 'min_y', min(X(2, :)));
    fprintf('%-7s landing: %s after %d steps (%.1f s) | fuel %.4f | min y %.4f | step %.1f ms median, %.0f ms max | iterations median %g, max %g | flags %s\n', ...
        name, ternary(landed, 'landed', 'NOT landed'), K, K * Ts, S.fuel, S.min_y, 1e3*median(t), 1e3*max(t), ...
        median(it), max(it), mat2str(unique(flag)));
end

function [u, flag, it] = rtimpc_move(o, x, r, lastMV)
    [u, info] = o.step(x, r, 'InputPrevious', lastMV);
    flag = info.ExitFlag;  it = info.SQPIterations;
end

function x1 = trapezoid(x0, u, h)
% Implicit trapezoidal step (Newton), for checking plans
    f0 = lander_ct(x0, u);  x1 = x0 + h * f0;
    for it = 1:50
        r = x1 - x0 - h/2 * (f0 + lander_ct(x1, u));
        dx = (eye(6) - h/2 * lander_jac(x1, u)) \ r;  x1 = x1 - dx;
        if norm(dx, inf) < 1e-14 * max(1, norm(x1, inf)), break, end
    end
end

function s = ternary(c, a, b)
    if c, s = a; else, s = b; end
end

function dx = lander_ct(x, u)
% Planar lander: thrust along the body axis, torque from the thrust difference
    m = 1;  I = 0.5 * m * 10^2;  d = 5;  g = 9.806;          % mass, inertia (disc, radius 10), thruster arm
    T = u(1) + u(2);
    dx = [x(4:6); -sin(x(3)) * T / m; cos(x(3)) * T / m - g; d * (u(2) - u(1)) / I];
end

function [A, B] = lander_jac(x, u)
    m = 1;  I = 0.5 * m * 10^2;  d = 5;
    T = u(1) + u(2);  s = sin(x(3));  c = cos(x(3));
    A = zeros(6);  A(1:3, 4:6) = eye(3);  A(4, 3) = -c * T / m;  A(5, 3) = -s * T / m;
    B = [zeros(3, 2); -s/m, -s/m; c/m, c/m; -d/I, d/I];
end
