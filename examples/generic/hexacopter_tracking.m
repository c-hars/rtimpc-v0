%% rtimpc example: hexacopter tracking a 3D figure-8
%
% A hexacopter (13 states: position, body-frame velocity, attitude quaternion, body rates; 6 rotor thrusts as inputs) flies a 3D figure-8 (8 m x 4 m x 8 m) in one 6 s lap, starting on the path at its velocity. The plant is open-loop unstable; NMPC runs at 20 Hz with a 2 s horizon and tracks position and yaw (via the quaternion's z part), and damps the roll and pitch rates. Weights by Bryson's rule. The quaternion's unit norm is not constrained: the dynamics preserve it (up to integration error, reported), and the plant renormalises after each step. Shows: a larger model (nx = 13, nu = 6) with analytic Jacobians, a reference that varies along the horizon, an input reference (hover thrust), real-time iterations.
%
% Controller = 'nlmpc' runs MATLAB's nlmpc (Model Predictive Control Toolbox) on the same OCP for comparison: the same RK4 discretisation, cost (custom cost function with its gradient) and bounds, solved to convergence by fmincon at every step. It takes seconds per step. nlmpcMaxIter caps nlmpc's SQP iterations from the second step on, as close to RTI as nlmpc allows: the capped solution is applied (UseSuboptimalSolution), with ConstraintTolerance 1e3 so that an iterate with open gaps is not rejected (nlmpc would otherwise hold the last input).

clear
here = fileparts(mfilename('fullpath'));
root = fullfile(here, '..', '..');  addpath(root, fullfile(root, 'qp'));

P = hexa_params();
Ts = 1/20;  N = 40;                                      % 20 Hz, 2 s horizon
Lap = 6;                                                 % figure-8 lap time [s]
T = Lap;                                                 % simulated time [s]: one lap
Controller = 'rtimpc';                                   % 'rtimpc' | 'nlmpc' (needs the MPC Toolbox)
nlmpcMaxIter = Inf;                                      % nlmpc SQP iterations per step after the first (Inf: to convergence; see the header)
SQPIter = 1;                                             % rtimpc SQP iterations per step after the cold start: 1 = RTI; 50 = to convergence, as nlmpc
out = [1:3, 10:12];                                      % outputs: position, q_z (yaw), roll and pitch rates
uh = P.m * P.g / 6 * ones(6, 1);                         % hover thrust per rotor

nmpc = rtimpc(13, 6, 6, Ts, N);
nmpc.Model.StateFcn     = @(x, u) hexa(x, u, P);
nmpc.Jacobian.StateFcn  = @(x, u) hexa_jac(x, u, P);
nmpc.Model.OutputFcn    = @(x, u) x(out);
nmpc.Jacobian.OutputFcn = @(x, u) sel(out, 13);
nmpc.Weights.Qy  = Ts * diag(1 ./ [0.1 0.1 0.1 sin(deg2rad(5)/2) deg2rad([50 50])].^2);   % Bryson: 10 cm,
                                                         % 5 deg yaw (q_z = sin(yaw/2)), 50 deg/s
nmpc.Weights.Qyf = diag([50.70 50.66 46.01 130.7 0.0767 0.0776]);   % from the DARE at hover (ZOH,
                                                         % diagonal kept)
dw = P.wmax - P.wh;                                      % rotor-speed headroom above hover
nmpc.Weights.R   = 10 * Ts / (2*P.kF*P.wh*dw)^2 * eye(6);  % 10 Ts / headroom^2 on rotor speed, as thrust (thrust change = 2 kF w dw near hover)
nmpc.Constraints.InputMin = zeros(6, 1);
nmpc.Constraints.InputMax = P.fmax * ones(6, 1);
nmpc.Optimisation.SQPMaxIter = SQPIter;

q = [0.9; 0.2; -0.3; 0.25];  q = q / norm(q);
jac = nmpc.validate([0.1; -0.2; 0.9; 0.3; 0.2; -0.1; q; 0.3; -0.2; 0.1], uh + 0.3*(-1).^(1:6)');
if strcmp(Controller, 'nlmpc'), nl = build_nlmpc(nmpc, P, out, uh);  mvopt = nlmpcmoveopt; end

%% Closed loop (plant: the same model, integrated finely)

steps = round(T/Ts);
w = 2*pi/Lap;  x = [0; 0; 4; 4*w; -4*w; 0; 1; 0; 0; 0; zeros(3, 1)];   % on the path at its velocity, level
u = uh;                                                  % at hover thrust
Ug = repmat(uh, 1, N);  Xg = [x, zeros(13, N)];          % cold-start guess for both controllers: the
for i = 1:N, Xg(:,i+1) = rk4(@(z) hexa(z, uh, P), Xg(:,i), Ts); end   % model rolled out at hover thrust
X = zeros(13, steps+1);  X(:,1) = x;  U = zeros(6, steps);
[tc, sqpit, flag, qplan, qplant] = deal(zeros(1, steps));
for k = 1:steps
    r = figure8((k - 1 + (1:N)) * Ts, Lap);              % references at nodes 1..N
    if strcmp(Controller, 'nlmpc')
        if k == 1, mvopt.X0 = Xg(:, 2:end)';  mvopt.MV0 = Ug'; end
        c = tic;  [u, mvopt, info] = nlmpcmove(nl, x, u, r', [], mvopt);  tc(k) = toc(c);
        sqpit(k) = info.Iterations;  flag(k) = info.ExitFlag;  Xp = info.Xopt';
        if k == 1 && isfinite(nlmpcMaxIter)              % capped from the second step on
            so = nl.Optimization.SolverOptions;  so.MaxIterations = nlmpcMaxIter;
            so.ConstraintTolerance = 1e3;  nl.Optimization.SolverOptions = so;
            nl.Optimization.UseSuboptimalSolution = true;
        end
    else
        guess = {};  if k == 1, guess = {'InitialGuess', struct('X', Xg, 'U', Ug)}; end
        c = tic;  [u, info] = nmpc.step(x, r, 'InputReference', uh, 'InputPrevious', u, guess{:});  tc(k) = toc(c);
        sqpit(k) = info.SQPIterations;  flag(k) = info.ExitFlag;  Xp = nmpc.Xbar;
    end
    u = min(max(u, 0), P.fmax);                          % actuator limits (the plant saturates)
    if isempty(Xp), qplan(k) = NaN;                      % no plan (the controller reset)
    else, qplan(k) = max(abs(vecnorm(Xp(7:10,:)) - 1)); end   % unit-norm error along the plan
    for s = 1:10, x = rk4(@(z) hexa(z, u, P), x, Ts/10); end
    qplant(k) = abs(norm(x(7:10)) - 1);  x(7:10) = x(7:10) / norm(x(7:10));   % renormalise
    X(:,k+1) = x;  U(:,k) = u;
end

%% Report

t = (0:steps) * Ts;  Rf = figure8(t, Lap);
e = vecnorm(X(1:3,:) - Rf(1:3,:));                       % position error
E = X(out, 2:end) - Rf(:, 2:end);  W = nmpc.Weights;
J = sum(sum(E .* (W.Qy * E))) + sum(sum((U - uh) .* (W.R * (U - uh))));
ms = 1e3 * tc;  later = ms(2:end);
fprintf('\n=== %s: hexacopter 3D figure-8, %g s lap ===\n', Controller, Lap)
fprintf('  Problem    nx = 13, nu = 6, ny = 6 | Ts = %g ms, N = %d (%.1f s horizon), RK4\n', 1e3*Ts, N, N*Ts)
if strcmp(Controller, 'nlmpc')
    if isfinite(nlmpcMaxIter), fprintf('  Solver     nlmpc: fmincon SQP, capped at %d it(s)/step after the first, warm-started\n', nlmpcMaxIter)
    else, fprintf('  Solver     nlmpc: fmincon SQP to convergence every step, warm-started\n')
    end
else
    fprintf('  Solver     cold start: SQP (<= %d its), then %d SQP it(s)/step\n', nmpc.Optimisation.FirstStepSQPMaxIter, SQPIter)
end
fprintf('  Jacobians  max rel. error %.1e vs central differences\n', max(cell2mat(struct2cell(jac))))
fprintf('  Tracking   position error RMS %.3f m, max %.3f m | final %.3f m\n', sqrt(mean(e.^2)), max(e), e(end))
q0 = X(7,:); q1 = X(8,:); q2 = X(9,:); q3 = X(10,:);
tilt = acos(max(-1, min(1, 1 - 2*(q1.^2 + q2.^2))));    % angle between body z and world z
yaw = atan2(2*(q0.*q3 + q1.*q2), 1 - 2*(q2.^2 + q3.^2));
fprintf('  Attitude   max tilt %.1f deg, max |yaw| %.1f deg | thrust in [%.2f, %.2f] N (bounds 0, %.1f)\n', ...
    rad2deg(max(tilt)), rad2deg(max(abs(yaw))), min(U(:)), max(U(:)), P.fmax)
fprintf('  Unit norm  max ||q| - 1| along the plans %.1e, plant per step (before renormalising) %.1e\n', ...
    max(qplan), max(qplant))
sat = any(U <= 1e-6 | U >= P.fmax - 1e-6, 1);
fprintf('  Saturation a rotor at a thrust bound in %d/%d steps\n', nnz(sat), steps)
fprintf('  Cost       closed-loop %.4g (stage cost summed over the run)\n', J)
fprintf('  Solver     %s | SQP iterations per step median %g, max %d\n', ...
    ternary(all(flag >= 0), 'no failed steps', sprintf('%d FAILED steps', nnz(flag < 0))), median(sqpit(2:end)), max(sqpit(2:end)))
fprintf('  Time/step  first %.0f ms (cold, incl. JIT) | later median %.1f ms, p95 %.1f ms (wall time; budget %g ms)\n\n', ...
    ms(1), median(later), prctile_(later, 95), 1e3*Ts)

figure(1); clf
subplot(2,2,[1 3]); plot3(Rf(1,:), Rf(2,:), Rf(3,:), 'k--', X(1,:), X(2,:), X(3,:));
axis equal; grid on; xlabel('x [m]'); ylabel('y [m]'); zlabel('z [m]'); legend('reference', Controller); view(145, 20)
subplot(2,2,2); plot(t, X(1:3,:), t, Rf(1:3,:), 'k--'); ylabel('position [m]'); grid on
legend('x', 'y', 'z', 'reference')
subplot(2,2,4); stairs(t(1:end-1), U'); ylabel('thrust [N]'); xlabel('t [s]'); grid on
legend(compose('rotor %d', 1:6))


%% ---- Reference ----

function r = figure8(t, Lap)
% 3D figure-8, one lap per Lap seconds; references for the outputs (position, yaw, roll and pitch rates), the last three zero
    w = 2*pi/Lap;
    r = [4*sin(w*t); -2*sin(2*w*t); 4*cos(w*t); zeros(3, numel(t))];
end


%% ---- Model: continuous-time dynamics and analytic Jacobians ----

function P = hexa_params()
    P.m = 5.0;  P.g = 9.81;  P.J = [0.008; 0.009; 0.015];        % mass, inertia (diagonal)
    P.l = 0.20;  P.kF = 1.5e-5;  P.kM = 0.01 * P.kF;              % arm; thrust, drag torque per (rad/s)^2
    P.wmax = 10000 * 2*pi/60;                                     % max rotor speed (10,000 RPM)
    P.fmax = P.kF * P.wmax^2;                                     % max thrust per rotor (16.4 N)
    P.wh = sqrt(P.m * P.g / 6 / P.kF);                            % hover rotor speed
    a = (0:5) * pi/3 + pi/6;  dirs = (-1).^(1:6);                 % rotor angles; spin directions
    P.M = [P.l*sin(a); -P.l*cos(a); -P.kM/P.kF * dirs];           % body torques from the six thrusts
end

function dx = hexa(x, f, P)
% x = [p; v; q; w]: world position, body-frame velocity, attitude quaternion [q0; qv] (body to world), body rates
    v = x(4:6);  q = x(7:10);  w = x(11:13);  J = P.J;
    dx = [rotate(q, v, 1);
          -cross(w, v) + [0; 0; sum(f) / P.m] - P.g * rotate(q, [0; 0; 1], -1);
          0.5 * [-q(2:4)' * w; q(1) * w + cross(q(2:4), w)];
          (P.M * f - cross(w, J .* w)) ./ J];
end

function [A, B] = hexa_jac(x, f, P)
    v = x(4:6);  q = x(7:10);  w = x(11:13);  J = P.J;
    [~, dRv] = rotate(q, v, 1);
    [~, dg] = rotate(q, [0; 0; 1], -1);
    A = zeros(13);
    A(1:3, 4:6)     = rot(q);
    A(1:3, 7:10)    = dRv;
    A(4:6, 4:6)     = -skew(w);
    A(4:6, 7:10)    = -P.g * dg;
    A(4:6, 11:13)   = skew(v);
    A(7:10, 7:10)   = 0.5 * [0, -w'; w, -skew(w)];
    A(7:10, 11:13)  = 0.5 * [-q(2:4)'; q(1) * eye(3) + skew(q(2:4))];
    A(11:13, 11:13) = -(skew(w) * diag(J) - skew(J .* w)) ./ J;
    B = [zeros(3, 6); [0; 0; 1/P.m] * ones(1, 6); zeros(4, 6); P.M ./ J];
end

function [y, dy] = rotate(q, a, s)
% y = R(q) a (s = 1) or R(q)' a (s = -1), with R(q) = (q0^2 - |qv|^2) I + 2 qv qv' + 2 q0 [qv]x, the rotation matrix for unit q; dy = dy/dq (3 x 4)
    q0 = q(1);  qv = q(2:4);
    y = (q0^2 - qv'*qv) * a + 2 * qv * (qv'*a) + 2 * s * q0 * cross(qv, a);
    if nargout > 1
        dy = [2*q0*a + 2*s*cross(qv, a), -2*a*qv' + 2*(qv'*a)*eye(3) + 2*qv*a' - 2*s*q0*skew(a)];
    end
end

function R = rot(q)
    q0 = q(1);  qv = q(2:4);
    R = (q0^2 - qv'*qv) * eye(3) + 2 * (qv * qv') + 2 * q0 * skew(qv);
end

function S = skew(a)
    S = [0 -a(3) a(2); a(3) 0 -a(1); -a(2) a(1) 0];
end

function C = sel(idx, n)
    C = zeros(numel(idx), n);  C(sub2ind(size(C), 1:numel(idx), idx)) = 1;
end

function x = rk4(f, x, h)
    k1 = f(x); k2 = f(x + h/2*k1); k3 = f(x + h/2*k2); k4 = f(x + h*k3);
    x = x + h/6*(k1 + 2*k2 + 2*k3 + k4);
end


%% ---- nlmpc on the same OCP (comparison; needs the MPC Toolbox) ----

function nl = build_nlmpc(nmpc, P, out, uh)
% Same discrete model (one RK4 step per sample), cost and bounds as the rtimpc controller
    Ts = nmpc.Ts;  N = nmpc.PredictionHorizon;  W = nmpc.Weights;
    nl = nlmpc(13, 6, 6);
    nl.Ts = Ts;  nl.PredictionHorizon = N;  nl.ControlHorizon = N;
    nl.Model.IsContinuousTime = false;
    nl.Model.StateFcn = @(x, u) rk4_sens(x, u, P, Ts);
    nl.Jacobian.StateFcn = @(x, u) rk4_jac(x, u, P, Ts);
    nl.Model.OutputFcn = @(x, u) x(out);
    nl.Jacobian.OutputFcn = @(x, u) sel(out, 13);
    nl.Optimization.ReplaceStandardCost = true;            % no slack in the cost: all constraints hard
    mpcverbosity('off');                                   % (nlmpc otherwise notes that at every step)
    nl.Optimization.CustomCostFcn = @(X, U, e, data) ocp_cost(X, U, data, W, out, uh);
    nl.Jacobian.CustomCostFcn = @(X, U, e, data) ocp_cost_grad(X, U, data, W, out, uh);
    for i = 1:6, nl.MV(i).Min = 0;  nl.MV(i).Max = P.fmax; end
end

function J = ocp_cost(X, U, data, W, out, uh)
% rtimpc's cost: Qy on nodes 1..N-1, Qyf on node N, R on (u - hover) for inputs 0..N-1
    E = X(2:end, out) - data.References;  D = U(1:end-1, :) - uh';
    J = sum(sum((E(1:end-1,:) * W.Qy) .* E(1:end-1,:))) + E(end,:) * W.Qyf * E(end,:)' + sum(sum((D * W.R) .* D));
end

function [Gx, Gmv, Ge] = ocp_cost_grad(X, U, data, W, out, uh)
    E = X(2:end, out) - data.References;  D = U(1:end-1, :) - uh';
    Gx = zeros(size(E, 1), 13);
    Gx(:, out) = 2 * [E(1:end-1,:) * W.Qy; E(end,:) * W.Qyf];
    Gmv = 2 * D * W.R;
    Ge = 0;
end

function [xn, A, B] = rk4_sens(x, u, P, h)
% One RK4 step of the continuous model, with its sensitivities
    f = @(z) hexa(z, u, P);  Jf = @(z) hexa_jac(z, u, P);
    k1 = f(x);            [A1, B1] = Jf(x);
    x2 = x + h/2*k1;      k2 = f(x2);  [A2, B2] = Jf(x2);
    x3 = x + h/2*k2;      k3 = f(x3);  [A3, B3] = Jf(x3);
    x4 = x + h*k3;        k4 = f(x4);  [A4, B4] = Jf(x4);
    xn = x + h/6*(k1 + 2*k2 + 2*k3 + k4);
    if nargout > 1
        I = eye(13);
        K1x = A1;                    K1u = B1;
        K2x = A2*(I + h/2*K1x);      K2u = A2*(h/2*K1u) + B2;
        K3x = A3*(I + h/2*K2x);      K3u = A3*(h/2*K2u) + B3;
        K4x = A4*(I + h*K3x);        K4u = A4*(h*K3u) + B4;
        A = I + h/6*(K1x + 2*K2x + 2*K3x + K4x);
        B = h/6*(K1u + 2*K2u + 2*K3u + K4u);
    end
end

function [A, B] = rk4_jac(x, u, P, h)
    [~, A, B] = rk4_sens(x, u, P, h);
end

function s = ternary(b, a, c), if b, s = a; else, s = c; end, end
function q = prctile_(v, p)                              % prctile without the Statistics Toolbox
    v = sort(v(:));  q = v(max(1, ceil(p/100 * numel(v))));
end
