%% rtimpc example: double inverted pendulum swing-up
%
% A harder benchmark than the cart-pole: two links on a cart, swung from hanging (both angles pi)
% to upright (both 0) with a bounded force on a 1 m track. The cost is a plain quadratic on the
% cart position and both link angles; nothing in it says how to swing up. The dynamics are
% strongly nonlinear and the swing-up takes most of the 2.5 s horizon, so one QP per step (RTI)
% is not enough here: two SQP iterations per step are (see the note at the end of the report).
% Model in mass-matrix form with analytic Jacobians.

clear
here = fileparts(mfilename('fullpath'));
root = fullfile(here, '..', '..');  addpath(root, fullfile(root, 'qp'));

p = dip_params();                                        % Bogdanov (2004): 1.5 kg cart, two uniform links
Ts = 0.025;  N = 100;                                    % 40 Hz, 2.5 s horizon
Fmax = 40;                                               % |force| <= 40 N
Xmax = 0.5;                                              % |cart position| <= 0.5 m (track length 1 m)
SQPIter = 2;                                             % SQP iterations per step (1 = RTI)
Animate = true;                                          % ASCII stop-motion replay in the terminal

nmpc = rtimpc(6, 1, 3, Ts, N);                           % x = [x; th1; th2; xdot; w1; w2], nu = 1, ny = 3
nmpc.Model.StateFcn = @(x,u) dip(x, u, p);               % returns [dx/dt, A, B] in one call
nmpc.Model.StateFcnReturnsJacobian = true;
nmpc.Model.OutputFcn    = @(x,u) x(1:3);                 % track cart position and both link angles
nmpc.Jacobian.OutputFcn = @(x,u) [eye(3), zeros(3)];
nmpc.Weights.Qy  = diag([5, 20, 20]);
nmpc.Weights.Qyf = diag([50, 200, 200]);
nmpc.Weights.R   = 0.01;
nmpc.Constraints.InputMin = -Fmax;
nmpc.Constraints.InputMax =  Fmax;
nmpc.Constraints.StateMin = [-Xmax; -inf(5,1)];          % only the cart position is bounded
nmpc.Constraints.StateMax = [ Xmax;  inf(5,1)];
nmpc.Constraints.StateSoftL1 = 1e5;                      % soft track limit: cost per metre of violation (Inf = hard)
nmpc.Optimisation.SQPMaxIter = SQPIter;

jac = nmpc.validate([0.2; 2.5; -1.3; 0.4; 1.1; -0.8], 7.0);   % Jacobians vs central differences

%% Closed loop (plant: the same model, integrated finely)

T = 8;  steps = round(T/Ts);
x = [0; pi; pi; 0; 0; 0];
X = zeros(6, steps+1); U = zeros(1, steps);
tc = zeros(1, steps); qpit = zeros(1, steps); sqpit = zeros(1, steps); flag = zeros(1, steps);
rebuilt = false(1, steps);
X(:,1) = x;
live = ~batchStartupOptionUsed;                          % one-line live status (not in -batch logs)
if live, status = status_line('Closed loop: cold-start SQP for the first plan...'); end
for k = 1:steps
    [u, info] = nmpc.step(x, zeros(3,1));
    tc(k) = info.TimePreparation + info.TimeFeedback + info.TimeLineSearch;
    qpit(k) = info.QPIterations;  sqpit(k) = info.SQPIterations;  flag(k) = info.ExitFlag;
    rebuilt(k) = info.PlanRebuilt;
    for s = 1:10                                         % plant: RK4 at Ts/10
        x = rk4(@(z) dip(z, u, p), x, Ts/10);
    end
    X(:,k+1) = x; U(k) = u;
    if live
        th = rad2deg(atan2(sin(x(2:3)), cos(x(2:3))));
        status = status_line(sprintf(['Closed loop: step %3d/%d  t = %5.2f s | theta = %+6.1f, %+6.1f deg  ' ...
            'x = %+.3f m  F = %+5.1f N | %5.1f ms, %d SQP its%s'], k, steps, k*Ts, th, x(1), u, ...
            1e3*tc(k), sqpit(k), ternary(flag(k) < 0, '  FAILED', '')), status);
    end
end
if live, fprintf('\n'); end

if Animate
    ascii_cartpole(X, U, Ts, 'TrackLimit', Xmax, 'PoleLength', [p.L1 p.L2], 'Angles', X(2:3,:), 'Width', 76);
end
print_report(X, U, tc, qpit, sqpit, flag, rebuilt, jac, nmpc, Fmax, Xmax)

if usejava('desktop') || ~exist('OCTAVE_VERSION', 'builtin')
    t = (0:steps)*Ts;
    subplot(3,1,1); plot(t, X(1,:)); hold on
    yline([-Xmax Xmax], 'k--'); hold off; ylabel('x [m]'); grid on
    subplot(3,1,2); plot(t, X(2:3,:)); ylabel('\theta [rad]'); legend('\theta_1', '\theta_2'); grid on
    subplot(3,1,3); stairs(t(1:end-1), U); ylabel('F [N]'); xlabel('t [s]'); grid on
end


%% ---- Report ----

function print_report(X, U, tc, qpit, sqpit, flag, rebuilt, jac, nmpc, Fmax, Xmax)
    Ts = nmpc.Ts;  N = nmpc.PredictionHorizon;  steps = numel(U);
    th = atan2(sin(X(2:3,:)), cos(X(2:3,:)));            % wrap link angles to (-pi, pi]
    ms = 1e3 * tc;  rest = ms(2:end);
    jerr = max(cell2mat(struct2cell(jac)));

    up = all(abs(th) < deg2rad(5), 1);                   % "upright": both links within 5 deg
    k_up = find(~up, 1, 'last') + 1;                     % first sample after which it stays up
    sat_u = abs(U) > 0.99 * Fmax;

    fprintf('\n=== rtimpc: double inverted pendulum swing-up ===\n')
    fprintf('  Problem    nx = %d, nu = %d, ny = %d | Ts = %g ms, N = %d (%.1f s horizon), %s\n', ...
        nmpc.nx, nmpc.nu, nmpc.ny, 1e3*Ts, N, N*Ts, upper(nmpc.Optimisation.Integrator))
    fprintf('  Weights    Qy = diag(%s), Qyf = diag(%s), R = %g\n', mat2str(diag(nmpc.Weights.Qy)'), ...
        mat2str(diag(nmpc.Weights.Qyf)'), nmpc.Weights.R)
    fprintf('  Bounds     |F| <= %g N (hard),  |x| <= %g m (soft, L1 = %g per m)\n', Fmax, Xmax, ...
        nmpc.Constraints.StateSoftL1(1))
    fprintf('  Solver     cold start: SQP (<= %d its), then <= %d SQP its/step, %s IPM\n', ...
        nmpc.Optimisation.FirstStepSQPMaxIter, nmpc.Optimisation.SQPMaxIter, nmpc.Optimisation.QPSolver)
    fprintf('  Jacobians  max rel. error %.1e vs central differences  [%s]\n', jerr, okfail(jerr < 1e-6))

    fprintf('\n  Closed loop: %.1f s, %d steps\n', steps*Ts, steps)
    if k_up <= size(th, 2)
        fprintf('    Swing-up     both links upright (|theta| < 5 deg) from t = %.2f s onwards\n', (k_up-1)*Ts)
    else
        fprintf('    Swing-up     NOT upright at the end of the run\n')
    end
    fprintf('    Final state  x = %+.3f m, theta = %+.2f, %+.2f deg\n', X(1,end), rad2deg(th(:,end)))
    fprintf('    Track        max |x| = %.3f m of %.3f m, violation %.1e m\n', ...
        max(abs(X(1,:))), Xmax, max([0, abs(X(1,:)) - Xmax]))
    fprintf('    Input        saturated %d/%d steps\n', nnz(sat_u), steps)
    fprintf('    Solver       %s, %d rebuilt plan(s); SQP its per step median %d\n', ...
        ternary(nnz(flag < 0) == 0, 'no failed steps', sprintf('%d FAILED steps', nnz(flag < 0))), ...
        nnz(rebuilt), round(median(sqpit(2:end))))
    fprintf('                 IPM iterations per step median %d, max %d\n', round(median(qpit(2:end))), max(qpit(2:end)))

    fprintf('\n  Compute time per step [ms]  (budget Ts = %g ms; plain MATLAB, not indicative of compiled code)\n', 1e3*Ts)
    fprintf('    first step                   %7.1f   cold SQP: %d its, %s\n', ms(1), sqpit(1), ...
        ternary(flag(1) == 1, 'converged', ternary(flag(1) == 0, 'hit max its', 'FAILED')))
    fprintf('    later steps median / p95 / max  %5.1f / %.1f / %.1f\n\n', median(rest), prctile_(rest, 95), max(rest))
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

function p = dip_params()
% Bogdanov (2004): cart 1.5 kg; links 0.5 kg / 0.5 m and 0.75 kg / 0.75 m, uniform rods
    m0 = 1.5; m1 = 0.5; m2 = 0.75; L1 = 0.5; L2 = 0.75;
    l1 = L1/2; l2 = L2/2; I1 = m1*L1^2/12; I2 = m2*L2^2/12;  % centres of mass, inertias about them
    p.g = 9.81;  p.L1 = L1;  p.L2 = L2;
    p.d = [m0 + m1 + m2, m1*l1 + m2*L1, m2*l2, m1*l1^2 + m2*L1^2 + I1, m2*L1*l2, m2*l2^2 + I2];
end

function [dx, A, B] = dip(x, F, p)
% x = [cart position; theta1; theta2; cart velocity; omega1; omega2], absolute link angles,
% theta = 0 upright. Mass-matrix form M(q) qdd = f(q, qd, F), from the Lagrangian; the
% Jacobians follow from differentiating it: d(qdd) = M \ (df - dM qdd).
    s1 = sin(x(2)); c1 = cos(x(2)); s2 = sin(x(3)); c2 = cos(x(3));
    s12 = sin(x(2) - x(3)); c12 = cos(x(2) - x(3));
    w1 = x(5); w2 = x(6);
    d = p.d;
    M = [d(1),      d(2)*c1,   d(3)*c2
         d(2)*c1,   d(4),      d(5)*c12
         d(3)*c2,   d(5)*c12,  d(6)];
    f = [F + d(2)*s1*w1^2 + d(3)*s2*w2^2
         -d(5)*s12*w2^2 + p.g*d(2)*s1
          d(5)*s12*w1^2 + p.g*d(3)*s2];
    qdd = M \ f;
    dx = [x(4:6); qdd];
    if nargout < 2, return, end

    dM1 = [0, -d(2)*s1, 0; -d(2)*s1, 0, -d(5)*s12; 0, -d(5)*s12, 0];    % dM/dtheta1
    dM2 = [0, 0, -d(3)*s2; 0, 0, d(5)*s12; -d(3)*s2, d(5)*s12, 0];      % dM/dtheta2
    df_th = [d(2)*c1*w1^2,                  d(3)*c2*w2^2                 % df/d(theta1, theta2)
             -d(5)*c12*w2^2 + p.g*d(2)*c1,  d(5)*c12*w2^2
              d(5)*c12*w1^2,                -d(5)*c12*w1^2 + p.g*d(3)*c2];
    df_w  = [2*d(2)*s1*w1,   2*d(3)*s2*w2                                % df/d(omega1, omega2)
             0,              -2*d(5)*s12*w2
             2*d(5)*s12*w1,  0];
    J_q = M \ [zeros(3,1), df_th - [dM1*qdd, dM2*qdd]];  % d(qdd)/d(x, theta1, theta2)
    J_w = M \ [zeros(3,1), df_w];                        % d(qdd)/d(xdot, omega1, omega2)
    A = [zeros(3), eye(3); J_q, J_w];
    B = [zeros(3,1); M \ [1; 0; 0]];
end

function x = rk4(f, x, h)
    k1 = f(x); k2 = f(x + h/2*k1); k3 = f(x + h/2*k2); k4 = f(x + h*k3);
    x = x + h/6*(k1 + 2*k2 + 2*k3 + k4);
end
