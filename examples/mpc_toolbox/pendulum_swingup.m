%% rtimpc example: pendulum swing-up and cart step (MPC Toolbox example)
%
% Rebuilt from MATLAB's "Swing-Up Control of Pendulum Using Nonlinear Model Predictive Control"
% (nlmpc). The model is written here from the same equations; no MathWorks files are needed.
%
% Cart-pole: cart 1 kg with viscous damping 10, pendulum 1 kg, l = 0.5 m; state [z; dz; theta;
% dtheta], theta = 0 upright. Discrete model: 10 forward-Euler steps per sample, Ts = 0.1 s, used
% as the prediction model and as the plant. Swing up from hanging, hold the cart at z = 0, then
% step the cart to z = 5 m at t = 10 s with the pendulum kept upright.
%
% Shows: a discrete-time model with complex-step Jacobians, a control horizon as a decimated
% shooting grid (DecimationFactors), a soft output bound, a cost on input changes.
%
% Differences from the original: exact state feedback (the original uses an EKF on noisy
% measurements); the bound |z| <= 10 is soft with a quadratic penalty per node (nlmpc: one slack
% over the horizon; inactive in this run).
%
% Parity with nlmpc (same OCP, state feedback): both swing up (upright from t = 1.5 s vs 1.4 s) and
% settle at z = 5 m after the step; the swing paths differ, closed-loop cost 2000 vs 1709.

clear
here = fileparts(mfilename('fullpath'));
root = fullfile(here, '..', '..');  addpath(root, fullfile(root, 'qp'));

Ts = 0.1;  p = 10;                                       % 10 Hz, 1 s horizon
f = @(x, u) pendulum_dt(x, u, Ts);

nmpc = rtimpc(4, 1, 2, Ts, p);                           % nx = 4, nu = 1, ny = 2 (z, theta)
nmpc.Model.IsContinuousTime = false;
nmpc.Model.StateFcn = f;
nmpc.Jacobian.StateFcn = @(x, u) cs_jacobian(f, x, u);
nmpc.Model.OutputFcn = @(x, u) x([1 3]);
nmpc.Jacobian.OutputFcn = @(x, u) [1 0 0 0; 0 0 1 0];
nmpc.Weights.Qy = 9 * eye(2);  nmpc.Weights.Qyf = 9 * eye(2);
nmpc.Weights.R = 0;  nmpc.Weights.RateR = 0.01;
nmpc.Constraints.InputMin = -100;  nmpc.Constraints.InputMax = 100;
nmpc.Constraints.OutputMin = [-10; -Inf];  nmpc.Constraints.OutputMax = [10; Inf];
nmpc.Constraints.OutputPenaltyL1 = 0;  nmpc.Constraints.OutputPenaltyL2 = 1e5;
op = nmpc.Optimisation;
op.DecimationFactors = [1 6];  op.StepsPerTier = 4;      % control horizon 5: moves 5..10 held
op.CostDiscretisation = 'integrator';                    % outputs still costed at every sample
op.SQPMaxIter = 50;                                      % converged SQP each step, as nlmpc
nmpc.Optimisation = op;

%% Closed loop: 20 s from hanging, reference [0; 0], then [5; 0] from t = 10 s

K = 200;  x = [0; 0; -pi; 0];  u = 0;
X = zeros(4, K+1);  X(:,1) = x;  U = zeros(1, K);  [tc, it, flag] = deal(zeros(1, K));
for k = 1:K
    r = [5 * (k*Ts >= 10); 0];
    c = tic;  [u, info] = nmpc.step(x, r, 'InputPrevious', u);  tc(k) = toc(c);
    it(k) = info.SQPIterations;  flag(k) = info.ExitFlag;
    x = f(x, u);
    X(:,k+1) = x;  U(k) = u;
end

t = (0:K) * Ts;  th = abs(X(3,:));  pre = t < 10;
up = find(th(pre) > 0.1, 1, 'last') + 1;
fprintf('\n=== rtimpc: pendulum swing-up (MPC Toolbox example) ===\n')
fprintf('  upright (|theta| <= 0.1 rad) from t = %.1f s | largest |theta| after the step %.3f rad\n', t(up), max(th(~pre)))
Yr = [5 * ((1:K)*Ts >= 10); zeros(1, K)];
J = 9 * sum((X([1 3], 2:end) - Yr).^2, 'all') + 0.01 * sum(diff([0, U]).^2);
fprintf('  final z = %.4f m, theta = %.1e rad | max |F| = %.1f N | closed-loop cost %.6g\n', X(1,end), X(3,end), max(abs(U)), J)
fprintf('  SQP iterations median %g, max %g | %s\n', median(it), max(it), ...
    ternary(all(flag >= 0), 'no failed steps', sprintf('%d FAILED steps', nnz(flag < 0))))
fprintf('  step time median %.1f ms (plain MATLAB)\n\n', 1e3 * median(tc))

subplot(3,1,1); plot(t, X(1,:)); ylabel('z [m]'); grid on
subplot(3,1,2); plot(t, X(3,:)); ylabel('\theta [rad]'); grid on
subplot(3,1,3); stairs(t(1:end-1), U); ylabel('F [N]'); xlabel('t [s]'); grid on


%% ---- Model ----

function x = pendulum_dt(x, F, Ts)
% 10 forward-Euler steps of the continuous model over one sample
    for i = 1:10, x = x + Ts/10 * pendulum_ct(x, F); end
end

function dx = pendulum_ct(x, F)
% Cart-pole with cart damping; theta = 0 upright, anticlockwise positive
    M = 1;  m = 1;  l = 0.5;  g = 9.81;  Kd = 10;
    s = sin(x(3));  c = cos(x(3));
    a = F - Kd*x(2) - m*l*x(4)^2*s;                      % force on the cart, less the pendulum's reaction
    dx = [x(2);
          (a + m*g*s*c) / (M + m*s^2);
          x(4);
          (g*s + a*c/(M + m)) / (l - m*l*c^2/(M + m))];
end

function [A, B] = cs_jacobian(f, x, u)
% Jacobians of a discrete map by complex-step differentiation (exact to round-off)
    h = 1e-30;  n = numel(x);  m = numel(u);  [A, B] = deal(zeros(n), zeros(n, m));
    for j = 1:n, e = zeros(n, 1); e(j) = 1i*h; A(:,j) = imag(f(x + e, u)) / h; end
    for j = 1:m, e = zeros(m, 1); e(j) = 1i*h; B(:,j) = imag(f(x, u + e)) / h; end
end

function s = ternary(b, a, c), if b, s = a; else, s = c; end, end
