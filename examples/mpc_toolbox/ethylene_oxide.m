function S = ethylene_oxide()
%% rtimpc example: economic MPC of an ethylene oxide reactor (MPC Toolbox example)
%
% Rebuilt from MATLAB's "Economic MPC Control of Ethylene Oxide Production" (nlmpc with a custom cost, in Simulink), its closed loop as a script: economic MPC of a chemical reactor, maximising its product output while two disturbances step.
%
% Problem
%   Reactor    an ethylene oxidation CSTR (Durand, Ellis, Christofides 2016); all quantities dimensionless
%   State      x = [gas density; C2H4; C2H4O; temperature]
%   Input      u = [C2H4 feed concentration (MV); jacket temperature Tc; C2H4 feed rate]; the last two are measured disturbances
%   Cost       maximise the C2H4O production rate at the end of the horizon, i.e. minimise V = -(u3/u1) x3 x4 at node p
%   Horizon    2 steps of Ts = 25 s; prediction model: 2500 forward-Euler steps of 0.01 s
%   Bounds     0.1 <= u1 <= 3, |du1| <= 0.5 per sample, x >= 0 (hard)
%   Scenario   300 s from the steady state at u = [0.5 1.1 0.175]; Tc steps 1.1 -> 1.15 at t = 100 s, the feed rate 0.175 -> 0.25 at 200 s
%   Plant      the continuous model by ode15s
%
% Shows: an economic (non-tracking) cost as Weights.TerminalCostFcn with its gradient and Hessian, Hessian 'exact', input rate limits, a discrete model returning its own Jacobians, and time-varying disturbances read by a nested model function.
%
% Result: the production rate at the end of each phase (just before the next disturbance step), against the steady-state maximum given in the original example:
%   t        maximum   rtimpc   nlmpc
%   100 s    0.0156    0.0153   0.0153
%   200 s    0.0135    0.0135   0.0131
%   300 s    0.0195    0.0193   0.0193
% Both make the same first move, 1.0 (rate-limited from 0.5). Each step's OCP, solved by rtimpc at the states nlmpc's closed loop visited, reaches the same optimal cost as nlmpc's solve or a lower one (by at most 8e-7). These figures stand in for the verification suite, which will hold the comparison and its setup once published.

    here = fileparts(mfilename('fullpath'));
    root = fullfile(here, '..', '..');  addpath(root, fullfile(root, 'qp'));

    Ts = 25;  p = 2;  K = 12;  u0 = 0.5;
    md = [1.1; 0.175];                       % measured disturbances of the step in progress (nested functions share it)
    o = rtimpc(4, 1, 4, Ts, p);
    o.Model.IsContinuousTime = false;  o.Model.StateFcnReturnsJacobian = true;
    o.Model.StateFcn = @model;
    o.Weights.Qy = zeros(4);  o.Weights.Qyf = zeros(4);  o.Weights.R = 0;
    o.Weights.TerminalCostFcn = @cost;
    o.Constraints.InputMin = 0.1;  o.Constraints.InputMax = 3;
    o.Constraints.RateMin = -0.02 * Ts;  o.Constraints.RateMax = 0.02 * Ts;
    o.Constraints.StateMin = zeros(4, 1);  o.Constraints.StateMax = inf(4, 1);  o.Constraints.StatePenaltyL1 = Inf;
    op = o.Optimisation;  op.Hessian = 'exact';  op.ComputeCost = true;  op.SQPMaxIter = 50;
    % QP complementarity: an active bound is left t = TolComp / lam short of its value. The economic cost is ~0.02, and the rate limits' multipliers are ~3e-4, so the IPM's absolute default 1e-8 leaves t ~ 3e-5 (first move 0.99997 for 1.0) and the plan 3e-4 off the optimum at an active second-move limit; 1e-12 gives t ~ 3e-9 (the plans then match fmincon to 3e-8).
    op.QPSolverOptions = struct('TolComp', 1e-12);  op.FirstStepSQPMaxIter = 50;
    o.Optimisation = op;

    x0 = steady_state([u0; md], [0.998; 0.42; 0.03; 1.0]);
    S = closed_loop(@rtimpc_move, 'rtimpc');
    ts = (0:K) * Ts;                         % (not t: nested functions share the parent's variables)
    subplot(3,1,1); stairs(ts(1:end-1), S.U); ylabel('u_1 (C_2H_4 feed)'); grid on
    subplot(3,1,2); plot(ts(2:end), S.rate, 'o-'); ylabel('production rate'); grid on
    subplot(3,1,3); plot(ts, S.X(3:4,:)); ylabel('x_3, x_4'); xlabel('t [s]'); grid on

    function [x, A, B] = model(x, u)   % nested: reads the step's md (an anonymous function would freeze it)
        if nargout > 1, [x, A, B] = euler_map(x, [u; md]); else, x = euler_map(x, [u; md]); end
    end

    function [V, g, H] = cost(x, u, ~)
        [V, g, H] = production_cost(x, u, md(2));
    end

    function [u, flag, it, J] = rtimpc_move(xk, lastMV, k)
        if k == 1, o.reset(); end
        [u, info] = o.step(xk, zeros(4, 1), 'InputPrevious', lastMV);
        flag = info.ExitFlag;  it = info.SQPIterations;  J = info.Cost;
    end

    function S = closed_loop(move, name)
        xp = x0;  lastMV = u0;
        X = zeros(4, K + 1);  X(:, 1) = xp;
        [U, P, t, flag, it, J] = deal(zeros(1, K));
        for k = 1:K
            tk = (k - 1) * Ts;
            md = md_at(tk);
            c = tic;  [u, flag(k), it(k), J(k)] = move(xp, lastMV, k);  t(k) = toc(c);
            [~, XO] = ode15s(@(~, z) oxidation_ct(z, [u; md]), [0 Ts], xp, odeset('RelTol', 1e-8, 'AbsTol', 1e-10));
            P(k) = md(2) / u * XO(end, 3) * XO(end, 4);   % production rate at the end of the sample
            xp = XO(end, :)';
            X(:, k+1) = xp;  U(k) = u;  lastMV = u;
        end
        S = struct('X', X, 'U', U, 'rate', P, 'production', sum(P), 'time', t, 'flag', flag, 'iter', it, 'cost', J);
        fprintf('%-7s first move %.7f | u at 75/175/275 s %.4f %.4f %.4f | rate at 100/200/300 s %.5f %.5f %.5f (maxima 0.0156 0.0135 0.0195) | sum %.6f | step %.0f ms median, %.0f ms max | iterations median %g, max %g | flags %s\n', ...
            name, U(1), U(4), U(8), U(12), P(4), P(8), P(12), sum(P), 1e3*median(t), 1e3*max(t), median(it), max(it), mat2str(unique(flag)));
    end
end

function md = md_at(t)
% The measured disturbances [Tc; availability] at time t (the model's step blocks)
    md = [ternary(t >= 100, 1.15, 1.1); ternary(t >= 200, 0.25, 0.175)];
end

function [V, g, H] = production_cost(x, u, u3)
% -(u3/u) x3 x4 and its gradient and Hessian in [x; u]
    a = u3 / u;
    V = -a * x(3) * x(4);
    g = [0; 0; -a * x(4); -a * x(3); a / u * x(3) * x(4)];
    H = zeros(5);
    H(3, 4) = -a;  H(3, 5) = a / u * x(4);  H(4, 5) = a / u * x(3);  H(5, 5) = -2 * a / u^2 * x(3) * x(4);
    H = H + triu(H, 1)';
end

function [x, A, B] = euler_map(x, u)
% The prediction model (2500 Euler steps of 0.01 s) and its sensitivities in x and in the MV u(1)
    h = 0.01;  A = eye(4);  B = zeros(4, 1);
    for k = 1:2500
        if nargout > 1
            [f, Jx, Ju] = oxidation_ct(x, u);
            B = B + h * (Jx * B + Ju);
            A = A + h * (Jx * A);
        else
            f = oxidation_ct(x, u);
        end
        x = x + h * f;
    end
end

function [f, Jx, Ju] = oxidation_ct(x, u)
% The continuous model and its Jacobians in x and in u(1) (hand-derived)
    g1 = -8.13;  g2 = -7.12;  g3 = -11.07;  A1 = 92.80;  A2 = 12.66;  A3 = 2412.71;
    B1 = 7.32;  B2 = 10.39;  B3 = 2170.57;  B4 = 7.02;
    r1 = exp(g1 / x(4)) * (x(2) * x(4))^0.5;
    r2 = exp(g2 / x(4)) * (x(2) * x(4))^0.25;
    r3 = exp(g3 / x(4)) * (x(3) * x(4))^0.5;
    V = u(3) / u(1);
    num = V * (1 - x(4)) + B1 * r1 + B2 * r2 + B3 * r3 - B4 * (x(4) - u(2));
    f = [V * (1 - x(1) * x(4));
         V * (u(1) - x(2) * x(4)) - A1 * r1 - A2 * r2;
         -V * x(3) * x(4) + A1 * r1 - A3 * r3;
         num / x(1)];
    if nargout < 2, return, end
    r1_2 = 0.5 * r1 / x(2);   r1_4 = r1 * (-g1 / x(4)^2 + 0.5 / x(4));
    r2_2 = 0.25 * r2 / x(2);  r2_4 = r2 * (-g2 / x(4)^2 + 0.25 / x(4));
    r3_3 = 0.5 * r3 / x(3);   r3_4 = r3 * (-g3 / x(4)^2 + 0.5 / x(4));
    dV = -V / u(1);
    Jx = [-V * x(4), 0, 0, -V * x(1);
          0, -V * x(4) - A1 * r1_2 - A2 * r2_2, 0, -V * x(2) - A1 * r1_4 - A2 * r2_4;
          0, A1 * r1_2, -V * x(4) - A3 * r3_3, -V * x(3) + A1 * r1_4 - A3 * r3_4;
          -num / x(1)^2, (B1 * r1_2 + B2 * r2_2) / x(1), B3 * r3_3 / x(1), (-V + B1 * r1_4 + B2 * r2_4 + B3 * r3_4 - B4) / x(1)];
    Ju = [dV * (1 - x(1) * x(4));
          dV * (u(1) - x(2) * x(4)) + V;
          -dV * x(3) * x(4);
          dV * (1 - x(4)) / x(1)];
end

function x = steady_state(u, x)
% Newton on the continuous model = 0 (the example uses fsolve from a random start)
    for it = 1:50
        [f, J] = oxidation_ct(x, u);
        dx = J \ f;  x = x - dx;
        if norm(dx, inf) < 1e-14, break, end
    end
    assert(norm(oxidation_ct(x, u), inf) < 1e-12, 'ethylene_oxide:steady', 'Steady-state Newton did not converge.')
end

function s = ternary(c, a, b)
    if c, s = a; else, s = b; end
end
