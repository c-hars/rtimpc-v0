function R = truck_trailer()
%% rtimpc example: truck-and-trailer reverse parking between obstacles (MPC Toolbox example)
%
% Rebuilt from MATLAB's "Truck and Trailer Automatic Parking Using Multistage Nonlinear MPC"
% (nlmpcMultistage). One trajectory-planning OCP, solved cold by SQP (no closed loop). The model
% and the obstacle distances are written here; no MathWorks files are needed.
%
% x = [x; y; theta (trailer heading); beta (truck relative to trailer)], u = [alpha (steering);
% v (truck speed)], hitch M = 1, truck L1 = 6 x W1 = 2.5, trailer L2 = 10 x W2 = 2.5 (m);
% p = 40 steps of Ts = 0.5 s (20 s), implicit trapezoidal rule, cost sum u_k'u_k,
% |alpha| <= pi/4, |v| <= 10, |beta| <= pi/2 (hard), terminal state [0 -25 pi/2 0] (hard), and
% truck and trailer keep 1 m from two rectangular obstacles (17.5 x 20 m, centred at
% (+-11.25, -20)) at nodes 1..p-1 (hard). Initial guess: the original's waypoint heuristic.
%
% Shows: nonlinear path constraints (Model.PathConstraintFcn, Jacobian by complex step through a
% polygon-distance function), a hard terminal-state constraint, an initial guess for a cold solve.
%
% Parity with nlmpcMultistage (same OCP and guess): both park feasibly with the clearance held;
% rtimpc finds a different local optimum with a lower cost (200.7 vs 251.8).

    here = fileparts(mfilename('fullpath'));
    root = fullfile(here, '..', '..');  addpath(root, fullfile(root, 'qp'));

    par = struct('M', 1, 'L1', 6, 'W1', 2.5, 'L2', 10, 'W2', 2.5);
    p = 40;  Ts = 0.5;  x0 = [0; 0; 0; 0];  target = [0; -25; pi/2; 0];
    f = @(x, u) dynamics(x, u, par);
    o = rtimpc(4, 2, 4, Ts, p);
    o.Model.StateFcn = f;
    o.Jacobian.StateFcn = @(x, u) cs_jacobian(f, x, u);
    o.Weights.Qy = zeros(4);  o.Weights.R = eye(2);
    o.Constraints.InputMin = [-pi/4; -10];  o.Constraints.InputMax = [pi/4; 10];
    o.Constraints.StateMin = [-Inf; -Inf; -Inf; -pi/2];  o.Constraints.StateMax = [Inf; Inf; Inf; pi/2];
    o.Constraints.StatePenaltyL1 = Inf;
    o.Model.PathConstraintFcn = @(x, u) collision_rows(x, par);      % <= 0: 1 m clearance
    o.Constraints.PathConstraintMin = -inf(4, 1);  o.Constraints.PathConstraintMax = zeros(4, 1);
    o.Constraints.PathConstraintPenaltyL1 = Inf;
    o.Model.TerminalConstraintFcn = @(x) x - target;
    o.Jacobian.TerminalConstraintFcn = @(x) eye(4);
    o.Constraints.TerminalConstraintMin = zeros(4, 1);  o.Constraints.TerminalConstraintMax = zeros(4, 1);
    o.Constraints.TerminalConstraintPenaltyL1 = Inf;
    op = o.Optimisation;
    op.Integrator = 'trapezoidal';  op.FirstStepSQPMaxIter = 200;  op.ComputeCost = true;
    o.Optimisation = op;

    Xg = initial_guess(x0, target, p);
    Xg(:, p+1) = Xg(:, p);   % as the example's guess: its last block repeats node p-1's state
    Ug = zeros(2, p);
    guess = struct('X', Xg, 'U', Ug);
    t0 = tic;
    [~, info] = o.step(x0, zeros(4, 1), 'InitialGuess', guess, 'InputPrevious', zeros(2, 1));
    t = toc(t0);
    [X, U] = o.plan();
    R.rtimpc = summary('rtimpc', X, U, info.ExitFlag, info.SQPIterations, t, target, par, f, Ts);
    R.rtimpc.info = info;

    obs = {rect(-11.25, -20, 0, 17.5, 20), rect(11.25, -20, 0, 17.5, 20)};
    for i = 1:2, fill(obs{i}(1,:), obs{i}(2,:), [0.8 0.8 0.8]); hold on, end
    plot(X(1,:), X(2,:), 'o-');  hold off;  axis equal;  grid on
    xlabel('x [m]'); ylabel('y [m]'); title('trailer axle path')
end

function S = summary(name, X, U, exitflag, iters, t, target, par, f, Ts)
    p = size(U, 2);
    J = sum(U .* U, 'all');
    clear_ = zeros(4, p - 1);
    for k = 2:p, clear_(:, k-1) = collision_rows(X(:, k), par); end
    gap = 0;
    for k = 1:p, gap = max(gap, norm(X(:,k+1) - trapezoid(f, X(:,k), U(:,k), Ts), inf)); end
    S = struct('cost', J, 'exit', exitflag, 'iterations', iters, 'time', t, ...
               'terminal_err', norm(X(:, end) - target, inf), 'max_row', max(clear_, [], 'all'), ...
               'max_gap', gap, 'X', X, 'U', U);
    fprintf('%-7s cost %10.6g | exit %2d, %3d iterations, %6.2f s | terminal error %.1e | largest collision row %.1e (<= 0) | largest gap %.1e\n', ...
        name, J, exitflag, iters, t, S.terminal_err, S.max_row, gap);
end

function dx = dynamics(x, u, par)
    theta = x(3);  beta = x(4);  alpha = u(1);  v = u(2);
    M = par.M;  L1 = par.L1;  L2 = par.L2;
    c = v * cos(beta) * (1 + M/L1 * tan(beta) * tan(alpha));
    dx = [c * cos(theta);
          c * sin(theta);
          v * (sin(beta)/L2 - M/(L1*L2) * cos(beta) * tan(alpha));
          v * (tan(alpha)/L1 - sin(beta)/L2 + M/(L1*L2) * cos(beta) * tan(alpha))];
end

function g = collision_rows(x, par)
% 1 m - distance from [trailer; truck] to each obstacle, ordered [trailer-obs1; trailer-obs2;
% truck-obs1; truck-obs2] (the example's order); -10 distance when overlapping
    th = x(3);  th1 = x(3) + x(4);
    trailer = rect(x(1) + par.L2/2*cos(th), x(2) + par.L2/2*sin(th), th, par.L2, par.W2);
    truck   = rect(x(1) + par.L2*cos(th) + (par.M + par.L1/2)*cos(th1), ...
                   x(2) + par.L2*sin(th) + (par.M + par.L1/2)*sin(th1), th1, par.L1, par.W1);
    obs = {rect(-11.25, -20, 0, 17.5, 20), rect(11.25, -20, 0, 17.5, 20)};
    d = zeros(4, 1);  bodies = {trailer, trailer, truck, truck};  which = [1 2 1 2];
    for i = 1:4, d(i) = polygon_distance(bodies{i}, obs{which(i)}); end
    g = 1 - d;
end

function P = rect(cx, cy, th, L, W)
% Corners (2 x 4, counter-clockwise) of an L x W rectangle centred at (cx, cy), heading th
    c = cos(th);  s = sin(th);
    loc = [L/2, -L/2, -L/2, L/2; W/2, W/2, -W/2, -W/2];
    P = [c, -s; s, c] * loc + [cx; cy];
end

function d = polygon_distance(A, B)
% Euclidean distance between convex polygons (corner columns); -10 if they overlap
    if overlap(A, B), d = -10; return, end
    d = Inf;
    for k = 1:size(A, 2)
        for j = 1:size(B, 2)
            d = min([d, point_segment(A(:,k), B(:,j), B(:,mod(j, size(B,2)) + 1)), ...
                        point_segment(B(:,j), A(:,k), A(:,mod(k, size(A,2)) + 1))]);
        end
    end
end

function o = overlap(A, B)
% Separating-axis test on the edge normals of both polygons
    o = true;
    for P = {A, B}
        Q = P{1};
        for k = 1:size(Q, 2)
            e = Q(:, mod(k, size(Q,2)) + 1) - Q(:, k);  n = [-e(2); e(1)];
            a = n' * A;  b = n' * B;
            if max(a) < min(b) || max(b) < min(a), o = false; return, end
        end
    end
end

function d = point_segment(q, a, b)
    ab = b - a;  t = min(max((q - a)' * ab / (ab' * ab), 0), 1);
    d = norm(q - (a + t * ab));
end

function X = initial_guess(x0, target, p)
% The example's waypoint heuristic: straight-line pieces through one waypoint (0, 10, pi/4) when
% starting at y >= -5, else two ((x0 +- 10, 10, pi/6), (0, 10, pi/3)); beta 0
    if x0(2) >= -5
        p1 = round(p/2);  p2 = p - p1 + 1;
        W = [0; 10; pi/4];
        xyz = [lin(x0(1:3), W, p1), lin(W, target(1:3), p2)];
    else
        p1 = round(p/3);  p2 = round(p/3);  p3 = p - p1 - p2 + 1;
        W1 = [x0(1) + sign(x0(1))*10; 10; pi/6];  W2 = [0; 10; pi/3];
        xyz = [lin(x0(1:3), W1, p1), lin(W1, W2, p2), lin(W2, target(1:3), p3)];
    end
    X = [xyz; zeros(1, p + 1)];
end

function M = lin(a, b, n)
    M = a + (b - a) * linspace(0, 1, n);
end

function x1 = trapezoid(f, x0, u, h)
% Implicit trapezoidal step (Newton), for checking plans
    x1 = x0 + h * f(x0, u);
    for it = 1:50
        r = x1 - x0 - h/2 * (f(x0, u) + f(x1, u));
        [A] = cs_jacobian(f, x1, u);
        dx = (eye(numel(x0)) - h/2 * A) \ r;  x1 = x1 - dx;
        if norm(dx, inf) < 1e-14, break, end
    end
end

function [A, B] = cs_jacobian(f, x, u)
    h = 1e-30;  n = numel(x);  m = numel(u);
    f0 = f(x, u);  A = zeros(numel(f0), n);  B = zeros(numel(f0), m);
    for j = 1:n, e = zeros(n, 1); e(j) = 1i*h; A(:,j) = imag(f(x + e, u)) / h; end
    for j = 1:m, e = zeros(m, 1); e(j) = 1i*h; B(:,j) = imag(f(x, u + e)) / h; end
end
