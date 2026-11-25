classdef rtimpc < handle
% RTIMPC  Real-time nonlinear MPC: multiple-shooting SQP-RTI with a Riccati interior-point QP.
%
% Each call to step() solves:
%
%   min  sum_{i=0}^{N-1} (u_i - u_ref)'R(u_i - u_ref) + sum_{i=1}^{N} e_i'W_i e_i  [+ V_f(x_N)]
%   s.t. x_{i+1} = F(x_i, u_i),   x_0 = x (measured),   InputMin <= u_i <= InputMax,
%        StateMin - sl_i <= x_i <= StateMax + su_i,  sl_i, su_i >= 0  (i = 1..N)
%   e_i = h(x_i, u_i) - ref_i,    W_i = Qy (i < N), Qyf (i = N)
%   [+ sum_i g_i'u_i  (InputLinear)]
%   with the state-bound slacks penalised by  StateSoftL1'(sl_i + su_i) + StateSoftL2'(sl_i.^2 + su_i.^2)
%   (soft bounds; StateSoftL1 = Inf makes a bound hard); optionally path constraints (i = 1..N)
%        OutputMin <= h(x_i, u_i) <= OutputMax  (i = 1..N; node N uses u_{N-1}),
%        ConstraintMin <= g(x_i, u_i) <= ConstraintMax  (i = 1..N-1; node 0: InitialConstraintFcn, as acados),
%        TerminalConstraintMin <= g_N(x_N) <= TerminalConstraintMax,
%        InitialConstraintMin <= g_0(x_0, u_0) <= InitialConstraintMax  (node 0: x_0 = x is fixed, so
%        these constrain u_0, e.g. input-only rows; acados's con_h_expr_0)
%   soft in the same way (OutputSoftL1/L2, ConstraintSoftL1/L2, TerminalConstraintSoftL1/L2,
%   InitialConstraintSoftL1/L2); and input-rate terms
%   sum_i (u_i - u_{i-1})'RateR(u_i - u_{i-1}),  RateMin <= u_i - u_{i-1} <= RateMax  (u_{-1}: last input)
%
% using the standard real-time NMPC machinery (Diehl et al. 2002; acados):
%   * direct multiple shooting; F from RK4 or implicit trapezoidal integration of a
%     continuous model, with exact sensitivities (or a user-supplied discrete model)
%   * Gauss-Newton Hessian of the least-squares output tracking cost (or the exact Hessian of
%     the Lagrangian, Hessian = 'exact', e.g. for linear costs)
%   * QP solved by a structure-exploiting Riccati interior-point method (HPIPM-style), O(N)
%   * real-time iteration: one QP per control step, warm-started from the shifted plan;
%     more SQP iterations (with a merit line search) for converged NMPC
%
% Usage:
%   obj = rtimpc(nx, nu, ny, Ts, N);
%   obj.Model.StateFcn     = @(x,u) ...;        % dx/dt (or x+ if IsContinuousTime = false)
%   obj.Jacobian.StateFcn  = @(x,u) ...;        % [A, B] = d(StateFcn)/d(x,u)
%   obj.Model.OutputFcn    = @(x,u) ...;        % y   (default: y = x)
%   obj.Jacobian.OutputFcn = @(x,u) ...;        % [C, D] = d(OutputFcn)/d(x,u)
%   obj.Weights.Qy = ...; obj.Weights.Qyf = ...; obj.Weights.R = ...;
%   obj.Constraints.InputMin = ...; obj.Constraints.InputMax = ...;
%   [u, info] = obj.step(x, ref);                % ref: ny x N (nodes 1..N) or ny x 1
%
% Options and their defaults: the property blocks below (Model, Jacobian, Weights, Constraints, Optimisation).

    properties (SetAccess = private)
        nx
        nu
        ny
    end

    properties
        Ts                    % sample time [s]
        PredictionHorizon     % N [steps]

        Model = struct( ...
            'StateFcn',         [], ...   % @(x,u) dx/dt, or x_{k+1} when IsContinuousTime = false; @(x,u,p) with step's Parameters
            'OutputFcn',        [], ...   % @(x,u) y; empty -> y = x
            'TerminalOutputFcn', [], ...  % @(x) y_N at node N (ny outputs, as acados's terminal output); empty -> OutputFcn, which must then not depend on u (D = 0 at node N)
            'IntervalFcn',      [], ...   % optional @(X,U,d) [Xn, A, B]: all intervals at once, replaces integrating StateFcn; @(X,U,d,P) with Parameters
            'ConstraintFcn',    [], ...   % @(x,u) g: path constraints ConstraintMin <= g(x_i, u_i) <= ConstraintMax, nodes 1..N-1 (node 0: InitialConstraintFcn; node N: TerminalConstraintFcn; as acados)
            'TerminalConstraintFcn', [], ... % @(x) gN: TerminalConstraintMin <= gN(x_N) <= TerminalConstraintMax
            'InitialConstraintFcn', [], ...  % @(x,u) g0: InitialConstraintMin <= g0(x_0, u_0) <= InitialConstraintMax, node 0
            'ConstraintHessianFcn', [], ...  % @(x,u) H (nx+nu x nx+nu x ng): row curvature added under Gauss-Newton x the row's upper multiplier (see ggn_curvature)
            'TerminalConstraintHessianFcn', [], ... % @(x) H (nx x nx x ngt), as ConstraintHessianFcn
            'InitialConstraintHessianFcn', [], ...  % @(x,u) H (nx+nu x nx+nu x ng0), as ConstraintHessianFcn
            'IntervalHessianFcn', [], ...    % Hessian 'exact': optional @(X,U,d,Pi) H (nx+nu x nx+nu x M): Hessian of Pi(:,i)'F(x_i, u_i) in (x_i, u_i), all intervals at once (@(X,U,d,Pi,P) with Parameters); replaces lagrangian_curvature's finite differences of the dynamics
            'ConstraintLagrangianHessianFcn', [], ... % Hessian 'exact': optional @(X,U,L) H (nx+nu x nx+nu x K): Hessian of L(:,k)'g(x_k, u_k) (ConstraintFcn rows), K inner nodes at once; replaces their finite differences
            'IsContinuousTime', true, ...
            'StateFcnReturnsJacobian', false)  % true: [f, A, B] = StateFcn(x,u) (one call, no
                                               % separate Jacobian.StateFcn -- avoids recomputing f)

        Jacobian = struct( ...
            'StateFcn',  [], ...          % @(x,u) [A, B] (continuous or discrete, matching the model)
            'OutputFcn', [], ...          % @(x,u) [C, D] (D may be omitted -> 0)
            'TerminalOutputFcn', [], ...  % @(x) C_N = dy_N/dx; empty -> central differences
            'ConstraintFcn', [], ...      % @(x,u) [Cg, Dg] = dg/d(x,u); empty -> central differences
            'TerminalConstraintFcn', [], ... % @(x) dgN/dx; empty -> central differences
            'InitialConstraintFcn', [])   % @(x,u) [C0, D0] = dg0/d(x,u); empty -> central differences

        Weights = struct( ...
            'Qy',  [], ...                % ny x ny stage output weight (nodes 1..N-1)
            'Qyf', [], ...                % ny x ny terminal output weight (node N); empty -> Qy
            'R',   [], ...                % nu x nu input weight
            'InputLinear', [], ...        % nu x 1 or nu x N linear input cost g_i'u_i per sample (e.g. fuel; empty -> none)
            'RateR', [], ...              % nu x nu (or nu x nu x N, per change i = 0..N-1) weight on input changes u_i - u_{i-1} (empty -> none)
            'StageCostFcn', [], ...       % optional @(x_i, u_i, ref_i) [l, grad, Hess] (in [x_i; u_i]) added at nodes 0..N-1, x the interval length (ref_i: ref column i+1, as InputReference's for u_i)
            'StageCostVectorised', false, ... % true: StageCostFcn takes all nodes at once, @(X, U, Ref) (columns i = nodes 0..M-1), returning l 1 x M, grad (nx+nu) x M, Hess (nx+nu) x (nx+nu) x M (as IntervalFcn)
            'TerminalCostFcn', [], ...    % optional @(xN, u_last, ref_N) [V, grad, Hess] added at node N (grad, Hess in x_N, or in [x_N; u_last])
            'TerminalLQR', 'off', ...     % 'off' | 'fixed' | 'adaptive': LQR cost-to-go tail at node N, from the DARE at the plan's end ('fixed': once; 'adaptive': every step); needs TerminalTarget
            'TerminalTarget', [])         % @(ref_N) xbar: state target of the LQR tail (required with TerminalLQR)

        Constraints = struct( ...
            'InputMin', [], ...           % nu x 1 (-Inf allowed); empty -> unbounded
            'InputMax', [], ...           % nu x 1 (+Inf allowed)
            'StateMin', [], ...           % nx x 1, bounds on predicted states x_1..x_N
            'StateMax', [], ...           % (the measured x_0 is not constrained)
            'StateSoftL1', [], ...        % scalar, nx x 1 or nx x N (per node): cost per unit bound violation per node (Inf = hard; empty -> 1e4)
            'StateSoftL2', [], ...        % scalar, nx x 1 or nx x N: cost per unit violation squared (empty -> 0)
            'RateMin', [], ...            % nu x 1 hard limit on u_i - u_{i-1} per sample (-Inf allowed)
            'RateMax', [], ...            % nu x 1 (+Inf allowed); empty -> no rate limit
            'OutputMin', [], ...          % ny x 1 bounds on y = OutputFcn(x_i, u_i), nodes 1..N (+-Inf allowed)
            'OutputMax', [], ...
            'OutputSoftL1', [], ...       % scalar or ny x 1, as StateSoftL1 (Inf = hard; empty -> 1e4)
            'OutputSoftL2', [], ...       % scalar or ny x 1 (empty -> 0)
            'ConstraintMin', [], ...      % ng x 1 bounds on g = Model.ConstraintFcn(x_i, u_i), nodes 1..N-1
            'ConstraintMax', [], ...
            'ConstraintSoftL1', [], ...   % scalar or ng x 1 (Inf = hard; empty -> 1e4)
            'ConstraintSoftL2', [], ...   % scalar or ng x 1 (empty -> 0)
            'TerminalConstraintMin', [], ... % bounds on Model.TerminalConstraintFcn(x_N)
            'TerminalConstraintMax', [], ...
            'TerminalConstraintSoftL1', [], ... % as ConstraintSoftL1 (not scaled by interval length)
            'TerminalConstraintSoftL2', [], ...
            'InitialConstraintMin', [], ...  % bounds on Model.InitialConstraintFcn(x_0, u_0) (+-Inf allowed)
            'InitialConstraintMax', [], ...
            'InitialConstraintSoftL1', [], ... % scalar or ng0 x 1, as ConstraintSoftL1 (Inf = hard; empty -> 1e4)
            'InitialConstraintSoftL2', [])     % scalar or ng0 x 1 (empty -> 0)

        Optimisation = struct( ...
            'Integrator',          'rk4', ... % 'rk4' | 'trapezoidal' | 'irk' (continuous models; 'irk': implicit Runge-Kutta, see integrate)
            'IntegratorSubsteps',  1, ...     % integration steps per shooting interval (coarse intervals: larger steps)
            'IRKStages',           4, ...     % 'irk': collocation stages (acados's default 4: order 8 Gauss-Legendre, 7 Radau IIA)
            'IRKScheme',           'gauss-legendre', ... % 'irk': 'gauss-legendre' (A-stable, as acados's default) | 'radau' (Radau IIA: L-stable, damps fast modes)
            'IRKNewtonIterations', [], ...    % 'irk': [] -> Newton to convergence; n -> exactly n iterations per step (acados: 3)
            'SQPMaxIter',          1, ...     % 1 = RTI
            'Hessian',             'gauss-newton', ... % QP Hessian: 'gauss-newton' | 'exact' (Lagrangian, see lagrangian_curvature)
            'Regularise',          'none', ... % QP Hessian blocks made positive definite: 'none' | 'project' | 'mirror' | 'gershgorin-lm' (see regularise_blocks) | 'project-reduced' (project_reduced_hessian)
            'RegulariseEpsilon',   1e-4, ...  % their smallest eigenvalue (acados's reg_epsilon)
            'RegulariseRate',      true, ...  % with input-rate terms: regularise the rate-augmented blocks over (x, u_prev, u), rate term included, as acados's augmented state (regularise_rate_blocks); false: the [x u] blocks before the rate term
            'FirstStepSQPMaxIter', 50, ...    % iterations on a cold start
            'SQPStopping',         'kkt', ... % SQP iterations stop on: 'kkt' (acados's NLP residuals, see kkt_converged) | 'step' (SQPTol, SQPTolCost)
            'SQPTolStat',          1e-6, ...  % 'kkt': inf-norm of the Lagrangian gradient (cost J/2, as acados; inputs in their own units)
            'SQPTolStatScaling',   'none', ... % 'kkt': 'none' (absolute, as acados) | 'ipopt': stationarity and complementarity divided by IPOPT's s_d, s_c (ipopt_stat_scale)
            'SQPTolEq',            1e-6, ...  % 'kkt': largest shooting gap
            'SQPTolIneq',          1e-6, ...  % 'kkt': largest violation of a hard bound or row
            'SQPTolComp',          1e-6, ...  % 'kkt': largest multiplier x margin (acados's defaults, all four)
            'SQPTol',              1e-6, ...  % 'step': inf-norm of the (scaled) SQP step ('kkt': a step below 1e-12 also stops, acados's tol_min_step_norm)
            'SQPTolCost',          1e-9, ...  % 'step': or predicted merit decrease <= SQPTolCost * |J| at a gap-free plan
            'LineSearch',          true, ...  % merit backtracking when iterating (never in single-iteration RTI)
            'RTILineSearch',       false, ... % true: the line search also in single-iteration steps (RTI with globalisation); false: full steps, as acados's SQP_RTI
            'LineSearchMaxSteps',  20, ...
            'LineSearchArmijo',    0, ...     % merit: sufficient-decrease factor (0: any decrease, as acados's default)
            'LineSearchFactor',    0.7, ...   % backtracking factor, as acados's alpha_reduction
            'LineSearchRoundoff',  0, ...     % Armijo test allowance, in eps(phi0): 0, a strict decrease, as acados's (10: IPOPT's Compare_le)
            'LineSearchMinStep',   0.05, ...  % stop backtracking below this alpha and take the step (as acados's MINSTEP), except at a converged point (see merit_line_search); 0: reject it
            'Globalisation',       'merit', ... % line search: 'merit' (the l1 merit, Armijo) | 'funnel' (acados's FUNNEL_L1PEN_LINESEARCH, Kiessling, Leyffer & Vanaret 2025)
            'QPAccuracyTest',      true, ...  % line search: re-solve the QP with tolerances x1e-2 (to 1e-14) while its step's merit model reduction is below its curvature (Curtis, Johnson, Robinson & Waechter 2014)
            'MeritWeights',        'elementwise', ... % 'elementwise': per gap and bound (Leineweber 1999, as acados) | 'component': one per state component | 'scalar': one weight
            'MeritWeightMargin',   1, ...     % merit weights = this x |multiplier|: 1, the exact penalty's bound itself, as acados
            'SecondOrderCorrection', false, ... % on a rejected full step, re-solve the QP for the gaps it opens
            'QPWarmStart',         true, ...  % start the QP multipliers from the previous step's, shifted (riccati, undecimated); a warm-started QP that fails is retried cold
            'WarmStart',           'shift', ... % next step's guess: 'shift' (the plan one sample on) | 'previous' (the previous plan and multipliers as they are, MATMPC's shifting 'no')
            'BarrierHomotopy',     0, ...     % mu0 > 0: each warm-started step first solves the OCP plus mu0 x relaxed log barriers on its constraint rows, then the OCP itself from that plan (an interior-point-like start against local minima on active rows); 0: off
            'BarrierRelax',        1e-2, ...  % the relaxed barrier's switch point delta: -log z for z > delta, its quadratic extension below (Feller & Ebenbauer)
            'BarrierMaxIter',      [], ...    % SQP iteration cap of the barrier phase ([]: SQPMaxIter); the phase only has to move the plan, not converge
            'CandidateGuess',      [], ...    % [] | 'cold' | @(x) [X, U] on the shooting grid: each warm-started step also solves the OCP from this guess (FirstStepSQPMaxIter) and keeps the converged plan with the lower NLP objective (a two-start step against local minima)
            'QPSolver',            'riccati', ... % 'riccati' | 'condensed'
            'QPSolverOptions',     struct(), ...
            'InputScale',          [], ...    % nu x 1; empty -> half the input range (conditioning)
            'ComputeCost',         false, ... % report the NLP objective of the returned plan
            'Display',             'off', ... % 'iter': print one line per SQP iteration (diagnostics)
            'DelayCompensation',   false, ... % true: input applied one sample after measuring (plan from the predicted state)
            'DecimationFactors',   [], ...    % e.g. [1 5]: coarse shooting intervals of d*Ts further out ([] -> none)
            'StepsPerTier',        [], ...    % blocks per near tier ([] -> largest factor); grid as SD-OPT's
            'IntervalSamples',     [], ...    % e.g. [1 2 2 2]: samples per shooting interval, nearest first, summing to the horizon (exclusive with DecimationFactors)
            'CostDiscretisation',  'euler', ...% coarse intervals: 'euler' (cost at the node x its length) | 'integrator' (at each integration step)
            'InfeasibleQP',        'byrd-omojokun', ... % after an unsolved QP: feasibility QP + relaxed QP (acados's SQP_WITH_FEASIBLE_QP direction) | 'none'
            'InfeasibleQPSticky',  false, ... % false: the plain QP first at every iteration, the pair only when it fails; true: two QPs per iteration for the rest of the step (as acados)
            'OnFailure',           'auto')    % after a failed step (ExitFlag < 0), the next step:
                                              % 'auto'  -> cold-starts if this step was warm-started,
                                              %            else warm-starts from this step's plan
                                              % 'reset' -> always cold-starts
                                              % 'keep'  -> always warm-starts from this step's plan
                                              % 'keep-primal' -> as 'keep', without its multipliers

        UserData = struct()       % free for the caller (e.g. application-specific memory)
    end

    properties (Dependent)
        NodeTimes                 % times of the plan's nodes 0..M relative to now [s] (decimated grid: M < N)
    end

    properties (SetAccess = private)
        Xbar = []                 % current plan: states  (nx, N+1)
        Ubar = []                 % current plan: inputs  (nu, N)
        Uprev = []                % last input returned by step() (u_{-1} for input-rate terms)
        RateNow = []              % input-rate data of the step in progress (for nlp_cost)
        GridNow = []              % shooting grid and cost points of the step in progress
        PlanTimes = []            % node times of the current plan [fine samples from now]
        HasDCache = []            % output_has_D result for the current Jacobian.OutputFcn
        PathNow = []              % path-constraint rows of the step in progress (path_setup)
        BarrierNow = 0            % Optimisation.BarrierHomotopy: the barrier weight of the solve in progress (0: the OCP itself)
        TerminalP = []            % TerminalLQR: the cost-to-go matrix in use ('fixed': computed once)
        TailNow = []              % TerminalLQR tail of the step in progress (P, xbar), for nlp_cost
        TailFrozen = false        % CandidateGuess: the candidate solve keeps TerminalP (the warm solve's), also 'adaptive'
        QPDual = []               % multipliers of the last QP solved (as passed to the solver, i.e. rate-augmented)
        QPDualInit = []           % multipliers to start the next QP from (QPWarmStart)
        InertiaDelta = 0          % the last nonzero inertia-correction delta of a solved Riccati QP (the next QP's hint; IPOPT keeps delta_last from successes only)
        FunnelNow = []            % Globalisation 'funnel': this step's funnel width tau, penalty parameter mu, penalty mode
        Multipliers = []          % Hessian 'exact' or row curvature: the plan's multiplier estimates (pi: dynamics, lg: rows' net, ug: rows' upper)
        NLPDual = []              % SQPStopping 'kkt': the plan's multipliers in the QP's layout (rate-augmented), carried as acados's
        ParamNow = []             % model parameters of the step in progress, per node (np x M; [] -> none)
        IntegratorUnconverged = 0 % implicit integration steps of the step in progress whose Newton iteration hit its cap
    end


    methods

        function obj = rtimpc(nx, nu, ny, Ts, N)
            obj.nx = nx;
            obj.nu = nu;
            obj.ny = ny;
            obj.Ts = Ts;
            obj.PredictionHorizon = N;
        end

        function reset(obj)
        % Forget the warm start: the next step() is a cold start.
            obj.Xbar = [];
            obj.Ubar = [];
            obj.PlanTimes = [];
            obj.QPDual = [];
            obj.Multipliers = [];
            obj.NLPDual = [];
            obj.TerminalP = [];   % TerminalLQR 'fixed': recomputed at the next step
            obj.InertiaDelta = 0;
        end

        function set.Model(obj, v),        obj.Model        = rtimpc.check_fields('Model', v); end
        function set.Jacobian(obj, v),     obj.Jacobian     = rtimpc.check_fields('Jacobian', v); end
        function set.Weights(obj, v),      obj.Weights      = rtimpc.check_fields('Weights', v); end
        function set.Constraints(obj, v),  obj.Constraints  = rtimpc.check_fields('Constraints', v); end
        function set.Optimisation(obj, v), obj.Optimisation = rtimpc.check_fields('Optimisation', v); end

        function t = get.NodeTimes(obj)
            if isempty(obj.PlanTimes), t = obj.Ts * [0, cumsum(obj.horizon_grid(obj.PredictionHorizon))];
            else, t = obj.Ts * obj.PlanTimes;
            end
        end

        function [X, U] = plan(obj)
        % Current predicted trajectory (nodes 0..N), valid after step().
            X = obj.Xbar;
            U = obj.Ubar;
        end

        function [u, info] = step(obj, x, ref, opts)
        % Compute the control input for measured state x.
        %   ref            : ny x N output references for nodes 1..N (or ny x 1, held constant)
        %   InputReference : nu x 1 or nu x N input target (default 0)
        %   OutputWeights  : ny x ny x N per-node output weights (overriding Qy/Qyf)
        %   InitialGuess   : struct with X (nx,M+1), U (nu,M) on the shooting grid (M = N without
        %                    decimation; see NodeTimes); replaces the warm/cold start
        %   InputPrevious  : nu x 1 input applied over the last sample (u_{-1} for RateR/RateMin/
        %                    RateMax, and the input assumed until the next sample with
        %                    Optimisation.DelayCompensation); default: the last input this object returned (0, clipped to
        %                    the input bounds, before the first step)
        %   RateReference  : nu x N reference input changes r_i (change i = u_i - u_{i-1}, i = 0..N-1,
        %                    per sample of the fine grid); RateR then weighs (u_i - u_{i-1}) - r_i,
        %                    e.g. the plan's own input changes, so only departures from them cost
        %                    (default 0). On a decimated grid a node change gets the sum of its samples' r_i.
        %   HorizonLength  : plan over only the first n <= N samples (a horizon shrinking onto the
        %                    end of a task); with decimation the grid is rebuilt over them, so the
        %                    last node lands exactly on sample n. The terminal weight Qyf, terminal
        %                    output and terminal constraints apply at sample n (OutputWeights, when
        %                    given, are used as given)
        %   Parameters     : np x N model parameters per sample 0..N-1 (or np x 1, held), e.g. a
        %                    disturbance preview; the model is then called as StateFcn(x,u,p) (and
        %                    Jacobian.StateFcn(x,u,p), IntervalFcn(X,U,d,P)), each interval with the
        %                    parameter of its first sample. They enter the model only, not the QP
            arguments
                obj
                x   (:,1) double
                ref (:,:) double
                opts.InputReference = []
                opts.OutputWeights  = []
                opts.InitialGuess   = []
                opts.InputPrevious  = []
                opts.RateReference  = []
                opts.HorizonLength  = []
                opts.Parameters     = []
            end
            mu0 = obj.Optimisation.BarrierHomotopy;
            if mu0 > 0 && ~isempty(obj.Optimisation.CandidateGuess)
                error('rtimpc:Optimisation', 'BarrierHomotopy and CandidateGuess cannot be combined: set one.')
            end
            if mu0 > 0 && ~isempty(obj.Xbar) && isempty(opts.InitialGuess) && obj.BarrierNow == 0
                [u, info] = obj.barrier_step(x, ref, opts, mu0);
                return
            end
            cg = obj.Optimisation.CandidateGuess;
            if isempty(cg) || isempty(obj.Xbar) || ~isempty(opts.InitialGuess)
                [u, info] = obj.solve_step(x, ref, opts);
                return
            end
            % CandidateGuess: the warm-started solve, then the same OCP from the candidate guess with
            % the controller's state as before the step; the converged plan with the lower NLP
            % objective is kept, with its state (plan, multipliers, QP warm start)
            cost_on = obj.Optimisation.ComputeCost;
            obj.Optimisation.ComputeCost = true;
            restore = onCleanup(@() set_compute_cost(obj, cost_on));
            before = obj.state_snapshot();
            [u, info] = obj.solve_step(x, ref, opts);
            warm = obj.state_snapshot();
            obj.state_restore(before);
            if ischar(cg) || isstring(cg)
                if ~strcmpi(cg, 'cold'), error('rtimpc:CandidateGuess', 'Unknown CandidateGuess ''%s'' (''cold'' or a function handle).', cg), end
                obj.reset();   % rtimpc's own cold start
            else
                [opts.InitialGuess.X, opts.InitialGuess.U] = cg(x);
            end
            % TerminalLQR: the same OCP for both solves, so their costs compare: the warm solve's tail
            % (reset() clears it; 'adaptive' would relinearise it at the candidate's own terminal point)
            obj.TerminalP = warm.TerminalP;  obj.TailFrozen = ~isempty(warm.TerminalP);
            unfreeze = onCleanup(@() set_tail_frozen(obj, false));
            [uc, ic] = obj.solve_step(x, ref, opts);
            clear unfreeze
            keep = ic.ExitFlag == 1 && (info.ExitFlag ~= 1 || ic.Cost < info.Cost);
            if keep
                [u, info, ic] = deal(uc, ic, info);
            else
                obj.state_restore(warm);
            end
            info.Candidate = struct('Kept', keep, 'ExitFlag', ic.ExitFlag, 'Cost', ic.Cost, ...
                'SQPIterations', ic.SQPIterations, 'QPIterations', ic.QPIterations);   % the other solve's (warm's if Kept)
            for f = ["TimePreparation", "TimeFeedback", "TimeLineSearch", "SQPIterations", "QPIterations"]
                info.(f) = info.(f) + ic.(f);   % both solves: what the step cost
            end
            if ~cost_on, info.Cost = NaN; end
        end
    end

    methods (Access = private)

        function [u, info] = barrier_step(obj, x, ref, opts, mu0)
        % Optimisation.BarrierHomotopy: the warm-started step on the OCP plus mu0 x relaxed log
        % barriers on the constraint rows (barrier_cost), then the OCP itself from that plan and its
        % multipliers, unshifted (WarmStart 'previous' for the second solve). The barrier phase
        % pulls the plan off active rows, as an interior-point method's start does; the second
        % solve converges to a KKT point of the OCP as posed.
            ws = obj.Optimisation.WarmStart;  cap = obj.Optimisation.SQPMaxIter;
            restore = onCleanup(@() barrier_done(obj, ws, cap));
            obj.BarrierNow = mu0;
            if isempty(opts.InputPrevious), opts.InputPrevious = obj.Uprev; end   % both solves: the input applied, not the barrier phase's
            if ~isempty(obj.Optimisation.BarrierMaxIter), obj.Optimisation.SQPMaxIter = obj.Optimisation.BarrierMaxIter; end
            [~, ib] = obj.solve_step(x, ref, opts);
            obj.BarrierNow = 0;  obj.Optimisation.WarmStart = 'previous';  obj.Optimisation.SQPMaxIter = cap;
            [u, info] = obj.solve_step(x, ref, opts);
            info.Barrier = struct('ExitFlag', ib.ExitFlag, 'SQPIterations', ib.SQPIterations, 'QPIterations', ib.QPIterations);
            for f = ["TimePreparation", "TimeFeedback", "TimeLineSearch", "SQPIterations", "QPIterations"]
                info.(f) = info.(f) + ib.(f);   % both solves: what the step cost
            end
        end

        function [u, info] = solve_step(obj, x, ref, opts)
        % step() for one initial guess (see step for opts)
            N  = obj.PredictionHorizon;
            nx = obj.nx; nu = obj.nu;
            op = obj.Optimisation;
            [umin, umax, su] = input_bounds(obj);
            Rw = obj.Weights.R;

            n_eff = N;   % samples planned over (HorizonLength)
            if ~isempty(opts.HorizonLength), n_eff = min(max(round(opts.HorizonLength), 1), N); end
            obj.check_step_inputs(x, ref, opts, n_eff);
            if size(ref, 2) == 1, ref = repmat(ref, 1, N); end
            if isempty(opts.InputReference), uref = zeros(nu, N);
            elseif size(opts.InputReference, 2) == 1, uref = repmat(opts.InputReference, 1, N);
            else, uref = opts.InputReference;
            end
            W = stage_weights(obj, opts.OutputWeights);
            if isempty(opts.OutputWeights) && n_eff < N   % a shorter horizon ends at sample n_eff:
                W(:,:,n_eff) = W(:,:,N);                  % the terminal weight goes there
            end
            if ~strcmpi(obj.Weights.TerminalLQR, 'off')   % the LQR tail replaces the terminal output weight
                W(:,:,n_eff) = 0;
            end
            % Shooting grid: M intervals of d(i) fine samples (all 1 without decimation). Costs are
            % given per fine sample; aggregate them onto the nodes (held-x) or keep the samples
            % inside coarse intervals for the exact cost.
            d = obj.horizon_grid(n_eff);
            G = obj.node_costs(d, W(:,:,1:n_eff), ref(:,1:n_eff), uref(:,1:n_eff));
            G.ulin = obj.node_linear_cost(G);   % nu x M (node level) or [] (none)
            obj.GridNow = G;
            obj.ParamNow = node_parameters(opts.Parameters, G.tau, G.M, n_eff);
            M = G.M;
            W = G.W;  ref = G.ref;  uref = G.uref;   % node level from here on
            rate = obj.rate_setup(opts.InputPrevious, umin, umax, G.tau, opts.RateReference);
            x_predicted = [];
            if op.DelayCompensation
                % The input returned now takes effect at the next sample; until then the plant runs
                % on the input already committed (u_{-1}). Plan from where that leaves it.
                x = obj.integrate(x, rate.u_prev, 1, obj.node_param(1));
                x_predicted = x;
            end
            obj.RateNow = rate;

            % --- Initial guess: explicit, cold (hold 0 / clipped input), or shifted warm start
            cold = isempty(obj.Xbar);
            warm = ~cold && isempty(opts.InitialGuess);
            if ~ismember(lower(string(op.WarmStart)), ["shift", "previous"])
                error('rtimpc:WarmStart', 'Unknown WarmStart ''%s'' (''shift'' or ''previous'').', op.WarmStart)
            end
            unshifted = strcmpi(op.WarmStart, 'previous');
            sh_idx = @(L) L(:, [2:end, end]);   % multipliers carried to the next step, shifted like the plan
            if unshifted, sh_idx = @(L) L; end
            if ~isempty(opts.InitialGuess)
                Xb = opts.InitialGuess.X;  Ub = opts.InitialGuess.U;
                max_sqp = op.FirstStepSQPMaxIter;
            elseif cold
                Ub = repmat(min(max(zeros(nu,1), umin), umax), 1, M);
                Xb = obj.rollout(x, Ub);
                max_sqp = op.FirstStepSQPMaxIter;
            elseif unshifted && size(obj.Ubar, 2) == M
                Ub = obj.Ubar;  Xb = obj.Xbar;
                max_sqp = op.SQPMaxIter;
            elseif M == N && size(obj.Ubar, 2) == N
                Ub = [obj.Ubar(:,2:end), obj.Ubar(:,end)];
                Xb = [obj.Xbar(:,2:end), obj.integrate(obj.Xbar(:,end), Ub(:,end), 1, obj.node_param(M))];
                max_sqp = op.SQPMaxIter;
            else
                [Xb, Ub] = obj.shift_plan(obj.Xbar, obj.Ubar, obj.PlanTimes, G.tau);
                max_sqp = op.SQPMaxIter;
            end

            % LQR terminal cost: P at the plan's terminal point (once, or every step), target from ref_N
            obj.TailNow = [];  dare_ok = true;
            lqr_mode = lower(string(obj.Weights.TerminalLQR));
            if lqr_mode ~= "off"
                if isempty(obj.Weights.TerminalTarget)
                    error('rtimpc:TerminalTarget', 'Weights.TerminalLQR needs Weights.TerminalTarget = @(ref_N) xbar.')
                end
                if (lqr_mode == "adaptive" && ~obj.TailFrozen) || isempty(obj.TerminalP)
                    [obj.TerminalP, dare_ok] = obj.terminal_lqr(Xb(:,end), Ub(:,end));
                end
                obj.TailNow = struct('P', obj.TerminalP, 'xbar', obj.Weights.TerminalTarget(ref(:,end)));
            end

            % QP warm start: the previous step's multipliers, shifted one node like the plan
            obj.QPDualInit = [];
            if op.QPWarmStart && warm && M == N && ~isempty(obj.QPDual) && strcmp(op.QPSolver, 'riccati') ...
                    && size(obj.QPDual.lam_lb, 2) == N
                obj.QPDualInit = structfun(sh_idx, obj.QPDual, 'UniformOutput', false);
            end

            ocp.A = zeros(nx, nx, M); ocp.B = zeros(nx, nu, M); ocp.c = zeros(nx, M);
            ocp.Q = zeros(nx, nx, M+1); ocp.q = zeros(nx, M+1);
            ocp.S = zeros(nu, nx, M);
            Rs = su .* Rw .* su';
            ocp.R = Rs .* reshape(d, 1, 1, M);
            hasD = output_has_D(obj, x, Ub(:,1));
            [xmin, xmax, xL1, xL2] = state_bounds(obj, M);
            has_xb = any(isfinite([xmin; xmax]));
            if has_xb   % node-level slack weights in QP scaling (the QP objective is J/2); node 0 unused
                ocp.zlx = [xL1(:,1), xL1] / 2;  ocp.zux = ocp.zlx;
                ocp.Zlx = [xL2(:,1), xL2];      ocp.Zux = ocp.Zlx;
            end
            pc = obj.path_setup(x, Ub(:,1), hasD, d);   % path constraints (output bounds, ConstraintFcn)
            g0min = pc.gmin(:, 1);  g0max = pc.gmax(:, 1);   % node 0's row bounds as posed (see the linearisation)
            if pc.on   % general QP rows at nodes 0..M (column 1, node 0: InitialConstraintFcn rows only)
                ng = pc.ng;
                ocp.Cg = zeros(ng, nx, M+1);  ocp.Dg = zeros(ng, nu, M+1);   % page M+1: node M's D (rate_augment)
                ocp.lg = -inf(ng, M+1);  ocp.ug = inf(ng, M+1);
                ocp.zlg = pc.L1 / 2;  ocp.zug = ocp.zlg;
                ocp.Zlg = pc.L2;      ocp.Zug = ocp.Zlg;
            end

            info = struct('ExitFlag', 0, 'SQPIterations', 0, 'SQPStepNorm', NaN, ...
                'QPIterations', 0, 'QPStatus', '', 'QPSwitchedToLQ', false, 'LineSearchSteps', 0, 'SOCAttempts', 0, 'SOCAccepted', 0, 'PlanRebuilt', false, ...
                'TimePreparation', 0, 'TimeFeedback', 0, 'TimeLineSearch', 0, 'Cost', NaN, 'PlanGap', NaN, ...
                'StateViolation', 0, 'StateBoundMultiplier', 0, 'ConstraintViolation', 0, 'ColdStart', cold && isempty(opts.InitialGuess), ...
                'PredictedState', x_predicted);
            nu_merit = 0;
            obj.FunnelNow = [];   % Globalisation 'funnel': initialised at the step's first line search
            step_norm = NaN;
            % Hessian 'exact': multiplier estimates, from the previous plan's (shifted like it) or zero
            % (the first iteration then has the Gauss-Newton Hessian)
            exact = strcmpi(op.Hessian, 'exact');
            mdl = obj.Model;   % Gauss-Newton with row curvature (generalised Gauss-Newton, ggn_curvature)
            ggn = ~exact && (~isempty(mdl.ConstraintHessianFcn) || ~isempty(mdl.TerminalConstraintHessianFcn) || ~isempty(mdl.InitialConstraintHessianFcn));
            if exact && ~isempty(G.pts)
                error('rtimpc:Hessian', 'Hessian ''exact'' does not support CostDiscretisation ''integrator''.')
            end
            stage = ~isempty(obj.Weights.StageCostFcn);
            if stage && ~isempty(G.pts)
                error('rtimpc:StageCostFcn', 'Weights.StageCostFcn does not support CostDiscretisation ''integrator''.')
            end
            if ~ismember(lower(string(op.Globalisation)), ["merit", "funnel"])
                error('rtimpc:Globalisation', 'Unknown Globalisation ''%s'' (''merit'' or ''funnel'').', op.Globalisation)
            end
            reg = lower(string(op.Regularise));
            if ~ismember(reg, ["none", "project", "mirror", "gershgorin-lm", "project-reduced"])
                error('rtimpc:Regularise', 'Unknown Regularise ''%s'' (''none'', ''project'', ''mirror'', ''gershgorin-lm'' or ''project-reduced'').', reg)
            end
            info.RegularisedBlocks = 0;
            info.InertiaDelta = 0;   % largest inertia-correction delta of this step's QPs (Riccati solver)
            ngl = 0;  if pc.on, ngl = pc.ng; end
            lam = struct('pi', zeros(nx, M+1), 'lg', zeros(ngl, M+1), 'ug', zeros(ngl, M+1));   % ug: upper-bound multipliers (ggn)
            if (exact || ggn) && warm && M == N && ~isempty(obj.Multipliers) && isequal(size(obj.Multipliers.lg), size(lam.lg)) ...
                    && isfield(obj.Multipliers, 'ug')
                lam = structfun(sh_idx, obj.Multipliers, 'UniformOutput', false);
            end
            bo = struct('on', false, 'zero_slacks', 0);   % Byrd-Omojokun mode of this step (InfeasibleQP)
            % SQPStopping 'kkt' (when iterating): acados's test at the start of each iteration, on the
            % plan's multipliers carried as acados's (from the previous step's, shifted like the plan)
            if ~ismember(lower(string(op.SQPTolStatScaling)), ["none", "ipopt"])
                error('rtimpc:SQPTolStatScaling', 'Unknown SQPTolStatScaling ''%s'' (''none'' or ''ipopt'').', op.SQPTolStatScaling)
            end
            if ~ismember(lower(string(op.SQPStopping)), ["kkt", "step"])
                error('rtimpc:SQPStopping', 'Unknown SQPStopping ''%s'' (''kkt'' or ''step'').', op.SQPStopping)
            end
            kkt_stop = strcmpi(op.SQPStopping, 'kkt') && max_sqp > 1;
            dual = [];
            if kkt_stop && warm && M == N && ~isempty(obj.NLPDual)
                dual = structfun(sh_idx, obj.NLPDual, 'UniformOutput', false);
            end
            info.KKT = struct('stat', NaN, 'eq', NaN, 'ineq', NaN, 'comp', NaN, 'stat_scale', NaN, 'comp_scale', NaN);
            info.FeasibilityQPs = 0;
            info.QPResolves = 0;   % QPAccuracyTest re-solves
            info.QPColdRetries = 0;   % warm-started QPs that failed and were solved again from a cold start
            info.QPAcceptable = 0;    % QPs ended by the IPM's acceptable termination (TolAcceptable), not its tolerances
            info.QPInertiaCapped = 0; % QPs whose inertia correction reached its cap (the Riccati IPM's delta_max)
            info.FeasibilityResidual = NaN;   % mean violation the last Byrd-Omojokun feasibility QP could not remove (NaN: none ran; Inf: it was unsolved)
            info.DAREConverged = dare_ok;     % TerminalLQR: this step's Riccati iteration met its tolerance (true when none ran)
            obj.IntegratorUnconverged = 0;
            info.InputClip = 0;       % the largest input-bound clip of the plan's inputs (round-off after a solved QP; more: a step from an unsolved QP's iterate, projected)

            sqp_iter = 0;
            min_step_prev = false;   % 'kkt': the last step was below acados's tol_min_step_norm
            while sqp_iter < max_sqp
                sqp_iter = sqp_iter + 1;

                % ===== Preparation: linearise at the current plan (does not need x) =====
                clock = tic;
                J0 = 0;
                inner = ~isempty(G.pts);   % cost points inside coarse intervals ('integrator')
                tf = isfield(ocp, {'SN', 'RN', 'rN'});   % node M's terms in u_{M-1}: rebuilt below when they apply
                if any(tf), names = {'SN', 'RN', 'rN'};  ocp = rmfield(ocp, names(tf)); end
                term_u = false;
                if inner || exact || ggn || stage || reg ~= "none" || obj.BarrierNow > 0   % rebuilt from scratch below (added onto or regularised, not assigned)
                    ocp.Q(:) = 0;  ocp.q(:) = 0;  ocp.S(:) = 0;  ocp.R = Rs .* reshape(d, 1, 1, M);
                end
                if inner, Xin = cell(1, M);  Ain = cell(1, M);  Bin = cell(1, M); end
                if ~isempty(obj.Model.IntervalFcn)   % all intervals in one call
                    if inner, error('rtimpc:IntervalFcn', 'Model.IntervalFcn does not support CostDiscretisation ''integrator''.'), end
                    [Xn, ocp.A, Bn] = obj.interval_fcn(Xb(:,1:M), Ub, d);
                    ocp.B = Bn .* su';
                    ocp.c = Xn - Xb(:,2:M+1);
                else   % rtimpc's own integration, interval by interval
                    for i = 1:M
                        if inner && ~isempty(G.pts{i})
                            [x_next, Ai, Bi, Xin{i}, Ain{i}, Bin{i}] = obj.integrate(Xb(:,i), Ub(:,i), d(i), obj.node_param(i));
                        else
                            [x_next, Ai, Bi] = obj.integrate(Xb(:,i), Ub(:,i), d(i), obj.node_param(i));
                        end
                        ocp.A(:,:,i) = Ai;
                        ocp.B(:,:,i) = Bi .* su';
                        ocp.c(:,i)   = x_next - Xb(:,i+1);
                    end
                end
                % Input cost (held over each interval: x its length)
                du_ref = Ub - uref;
                ocp.r  = (Rw * du_ref) .* su .* d;
                if M == N, J0 = J0 + sum(du_ref .* (Rw * du_ref), 'all');
                else,      J0 = J0 + sum(d .* sum(du_ref .* (Rw * du_ref), 1));
                end
                if ~isempty(G.ulin)   % linear input cost (QP objective is J/2)
                    ocp.r = ocp.r + 0.5 * G.ulin .* su;
                    J0 = J0 + sum(G.ulin .* Ub, 'all');
                end
                % Output tracking (Gauss-Newton). Array column a = prediction node a-1; the
                % output of node i < M may depend on u_i (D-term); node M's is terminal (x_M only).
                for a = 2:M+1
                    Wa = W(:,:,a-1);
                    if ~any(Wa(:))
                        % reset what an earlier SQP iteration of this step put here (a terminal
                        % cost is added below, onto the last node, every iteration)
                        ocp.Q(:,:,a) = 0;  ocp.q(:,a) = 0;
                        if hasD && a <= M, ocp.S(:,:,a) = 0;  ocp.R(:,:,a) = Rs * d(a); end
                        continue
                    end
                    [y, Cy, Dy] = obj.output(Xb(:,a), Ub(:, min(a, M)), hasD, a == M+1);
                    e  = y - ref(:,a-1);
                    CW = Cy' * Wa;
                    ocp.Q(:,:,a) = CW * Cy;
                    ocp.q(:,a)   = CW * e;
                    J0 = J0 + e' * Wa * e;
                    if hasD && a <= M
                        DW = (Dy .* su')' * Wa;
                        ocp.S(:,:,a) = DW * Cy;
                        ocp.R(:,:,a) = Rs * d(a) + DW * (Dy .* su');
                        ocp.r(:,a)   = ocp.r(:,a) + DW * e;
                    end
                end
                if inner
                    % Cost points inside coarse interval i depend on (x_i, u_i): Gauss-Newton
                    % terms through their integration sensitivities, added to stage i
                    for i = 1:M
                        P = G.pts{i};
                        for p = 1:numel(P)
                            s  = P(p).step;
                            Wf = P(p).W;
                            [y, Cy, Dy] = obj.output(Xin{i}(:,s), Ub(:,i), hasD);
                            e  = y - P(p).ref;
                            Cx = Cy * Ain{i}(:,:,s);
                            Cu = (Cy * Bin{i}(:,:,s) + Dy) .* su';
                            ocp.Q(:,:,i) = ocp.Q(:,:,i) + Cx' * Wf * Cx;
                            ocp.S(:,:,i) = ocp.S(:,:,i) + Cu' * Wf * Cx;
                            ocp.R(:,:,i) = ocp.R(:,:,i) + Cu' * Wf * Cu;
                            ocp.q(:,i)   = ocp.q(:,i)   + Cx' * Wf * e;
                            ocp.r(:,i)   = ocp.r(:,i)   + Cu' * Wf * e;
                            J0 = J0 + e' * Wf * e;
                        end
                    end
                end
                if stage   % l(x_i, u_i) at nodes 0..M-1, x the interval length (QP objective is J/2)
                    ix = 1:nx;  iu = nx+1:nx+nu;  nz = nx + nu;
                    vec = obj.Weights.StageCostVectorised;
                    if vec   % all nodes in one call
                        [L, Gs, Hs] = obj.Weights.StageCostFcn(Xb(:,1:M), Ub(:,1:M), ref(:,1:M));
                        if numel(L) ~= M || ~isequal(size(Gs), [nz, M]) || numel(Hs) ~= nz * nz * M
                            error('rtimpc:StageCostFcn', 'With StageCostVectorised, Weights.StageCostFcn must return l 1 x M, the gradients (nx + nu = %d) x M and the Hessians %d x %d x M (M = %d).', nz, nz, nz, M)
                        end
                        Hs = reshape(Hs, nz, nz, M);
                    end
                    for i = 1:M
                        if vec
                            l = L(i);  g = Gs(:,i);  H = Hs(:,:,i);
                        else
                            [l, g, H] = obj.Weights.StageCostFcn(Xb(:,i), Ub(:,i), ref(:,i));
                            g = g(:);
                            if numel(g) ~= nz || ~isequal(size(H), [nz, nz])
                                error('rtimpc:StageCostFcn', 'Weights.StageCostFcn must return a gradient in [x; u] (nx + nu = %d) and its square Hessian.', nz)
                            end
                        end
                        w = 0.5 * d(i);
                        ocp.Q(:,:,i) = ocp.Q(:,:,i) + w * H(ix, ix);
                        ocp.q(:,i)   = ocp.q(:,i)   + w * g(ix);
                        ocp.S(:,:,i) = ocp.S(:,:,i) + w * su .* H(iu, ix);
                        ocp.R(:,:,i) = ocp.R(:,:,i) + w * su .* H(iu, iu) .* su';
                        ocp.r(:,i)   = ocp.r(:,i)   + w * su .* g(iu);
                        J0 = J0 + d(i) * l;
                    end
                end
                if ~isempty(obj.Weights.TerminalCostFcn)
                    [Vf, gf, Hf] = obj.Weights.TerminalCostFcn(Xb(:,M+1), Ub(:,M), ref(:,M));
                    gf = gf(:);
                    if ~ismember(numel(gf), [nx, nx + nu]) || ~isequal(size(Hf), numel(gf) * [1 1])
                        error('rtimpc:TerminalCostFcn', 'Weights.TerminalCostFcn must return a gradient in x_N (nx = %d) or in [x_N; u_last] (nx + nu = %d), and its square Hessian.', nx, nx + nu)
                    end
                    ocp.Q(:,:,M+1) = ocp.Q(:,:,M+1) + 0.5 * Hf(1:nx, 1:nx);
                    ocp.q(:,M+1)   = ocp.q(:,M+1)   + 0.5 * gf(1:nx);
                    if numel(gf) > nx
                        % also in u_last = u_{M-1}: node M's block becomes [Q SN'; SN RN] in
                        % (x_M, du_{M-1}), which the input-rate augmentation carries (its state at
                        % node M is du_{M-1}); switched on here if the step has no rate terms
                        iu = nx+1:nx+nu;
                        if ~term_u, ocp.SN = zeros(nu, nx);  ocp.RN = zeros(nu);  ocp.rN = zeros(nu, 1); end
                        ocp.SN = ocp.SN + 0.5 * su .* Hf(iu, 1:nx);
                        ocp.RN = ocp.RN + 0.5 * su .* Hf(iu, iu) .* su';
                        ocp.rN = ocp.rN + 0.5 * su .* gf(iu);
                        term_u = true;
                    end
                    J0 = J0 + Vf;
                end
                if term_u && ~rate.on, rate.on = true;  obj.RateNow.on = true; end
                if ~isempty(obj.TailNow)   % (x - xbar)'P(x - xbar): QP objective is J/2
                    eN = Xb(:,M+1) - obj.TailNow.xbar;
                    ocp.Q(:,:,M+1) = ocp.Q(:,:,M+1) + obj.TailNow.P;
                    ocp.q(:,M+1)   = ocp.q(:,M+1)   + obj.TailNow.P * eN;
                    J0 = J0 + eN' * obj.TailNow.P * eN;
                end
                if pc.on   % path constraints linearised at the plan: lg <= Cg dx + Dg du <= ug
                    pc.gmin(:,1) = g0min;  pc.gmax(:,1) = g0max;
                    for a = 1:M+1
                        [gv, Cg, Dg] = obj.path_rows(Xb(:,a), Ub(:, min(a, M)), hasD, node_kind(a, M));
                        ocp.Cg(:,:,a) = Cg;
                        ocp.Dg(:,:,a) = Dg .* su';   % node M: zero (terminal output and rows in x_M only)
                        ocp.lg(:,a) = pc.gmin(:,a) - gv;
                        ocp.ug(:,a) = pc.gmax(:,a) - gv;
                        if a == 1   % node 0 (x_0 fixed): a row without u_0 is a constant of the QP, its
                            % multiplier undetermined (on its bound the IPM drives it to infinity, and
                            % the merit weights with it: x_0 is eliminated, so no x_0 bound multiplier
                            % absorbs it, as acados's would); left out of the QP. A hard one also of
                            % this iteration's row bounds (infeasibility, Feasible): the measurement's,
                            % not the step's. A soft one keeps its bounds: its penalty, a constant, is
                            % in the cost, the merit and ConstraintViolation, as acados's node-0 slack.
                            % E.g. a state-only InitialConstraintFcn row: x_0 is the measurement. Without u_0
                            % structurally (at this x_0), not only locally: a zero row of Dg at u_0
                            % must stay zero at probe inputs (node0_input_free), so e.g. u_0^2 >= 1
                            % at u_0 = 0 is kept (an inconsistent linearisation, not a constant)
                            const = all(Dg == 0, 2);
                            if any(const), const = obj.node0_input_free(const, Xb(:,1), Ub(:,1), hasD, su); end
                            ocp.lg(const, 1) = -Inf;  ocp.ug(const, 1) = Inf;
                            hc = const & ~isfinite(pc.L1(:, 1));   % hard ones
                            pc.gmin(hc, 1) = -Inf;  pc.gmax(hc, 1) = Inf;  obj.PathNow = pc;
                        end
                        if obj.BarrierNow > 0   % BarrierHomotopy: the barrier's gradient and (Gauss-Newton) curvature, QP units J/2
                            [~, g1, g2] = obj.barrier_terms(gv, pc.gmin(:,a), pc.gmax(:,a));
                            Ds = ocp.Dg(:,:,a);
                            ocp.Q(:,:,a) = ocp.Q(:,:,a) + 0.5 * Cg' * (g2 .* Cg);
                            ocp.q(:,a)   = ocp.q(:,a)   + 0.5 * Cg' * g1;
                            if a <= M
                                ocp.S(:,:,a) = ocp.S(:,:,a) + 0.5 * Ds' * (g2 .* Cg);
                                ocp.R(:,:,a) = ocp.R(:,:,a) + 0.5 * Ds' * (g2 .* Ds);
                                ocp.r(:,a)   = ocp.r(:,a)   + 0.5 * Ds' * g1;
                            end
                        end
                    end
                end
                if exact
                    ocp = obj.lagrangian_curvature(ocp, Xb, Ub, ref, W, lam, su, hasD, d);
                end
                if ggn && ngl > 0, ocp = obj.ggn_curvature(ocp, Xb, Ub, lam.ug, su, M); end
                if reg ~= "none" && rtimpc.rate_regularised(op, rate)   % on the rate-augmented blocks, which rate_augment then uses
                    [ocp.RateReg, nreg] = rtimpc.regularise_rate_blocks(ocp, rate, su, reg, op.RegulariseEpsilon);
                    info.RegularisedBlocks = info.RegularisedBlocks + nreg;
                elseif reg ~= "none"
                    if ~isfield(ocp, 'SN'), ocp.SN = [];  ocp.RN = []; end
                    if reg == "project-reduced"
                        if ~isempty(ocp.SN), error('rtimpc:Regularise', 'Regularise ''project-reduced'' with a terminal cost on u: not supported.'), end
                        [ocp.Q, ocp.S, ocp.R, nreg] = rtimpc.project_reduced_hessian(ocp.Q, ocp.S, ocp.R, ocp.A, ocp.B, su, op.RegulariseEpsilon);
                    else
                        [ocp.Q, ocp.S, ocp.R, nreg, ocp.SN, ocp.RN] = rtimpc.regularise_blocks(ocp.Q, ocp.S, ocp.R, su, reg, op.RegulariseEpsilon, ocp.SN, ocp.RN);
                    end
                    if isempty(ocp.SN), ocp = rmfield(ocp, {'SN', 'RN'}); end
                    info.RegularisedBlocks = info.RegularisedBlocks + nreg;
                end
                J0 = J0 + obj.soft_penalty(Xb) + obj.path_penalty(Xb, Ub) + obj.barrier_cost(Xb, Ub);   % so J0 is the full NLP cost (merit baseline)
                if rate.on, J0 = J0 + rtimpc.rate_cost(Ub, rate); end
                ocp.lb = (umin - Ub) ./ su;
                ocp.ub = (umax - Ub) ./ su;
                if has_xb
                    ocp.lbx = xmin - Xb;
                    ocp.ubx = xmax - Xb;
                end
                info.TimePreparation = info.TimePreparation + toc(clock);

                % ===== Feedback: embed the measured state, solve the QP =====
                clock = tic;
                ocp.x0 = x - Xb(:,1);
                args = namedargs2cell(op.QPSolverOptions);
                try
                    if ~qp_data_finite(ocp)   % (complex: the model was evaluated outside its domain)
                        error('rtimpc:nonfinite', 'Non-finite or complex linearisation.')
                    end
                    if kkt_stop && ~isempty(dual)
                        [done, info.KKT] = obj.kkt_converged(ocp, Ub, rate, su, dual);
                        if strcmpi(op.Display, 'iter')
                            fprintf('  SQP %4d  KKT stat %.2e (scale %.3g) eq %.2e ineq %.2e comp %.2e%s\n', sqp_iter, info.KKT.stat, ...
                                info.KKT.stat_scale, info.KKT.eq, info.KKT.ineq, info.KKT.comp, ternary_str(done, '  -> converged', ''))
                        end
                        if done   % no QP this iteration
                            sqp_iter = sqp_iter - 1;
                            info.ExitFlag = 1;
                            if isempty(info.QPStatus), info.QPStatus = 'solved'; end   % plan already optimal: no QP needed (not a failure)
                            info.TimeFeedback = info.TimeFeedback + toc(clock);
                            break
                        end
                        if min_step_prev   % the KKT test fails after a step below 1e-12: acados's ACADOS_MINSTEP
                            sqp_iter = sqp_iter - 1;
                            info.ExitFlag = 2;
                            info.TimeFeedback = info.TimeFeedback + toc(clock);
                            break
                        end
                    end
                    [sol, qinfo, bo] = obj.solve_direction(ocp, Ub, rate, su, args, bo);
                    info.FeasibilityQPs = info.FeasibilityQPs + bo.feas_now;
                    if bo.feas_now, info.FeasibilityResidual = bo.l1; end
                    if isfield(qinfo, 'no_step') && qinfo.no_step   % a stall before the first IPM iterate: its 'step' is the
                        error('rtimpc:QPNoStep', 'QP solver broke down before its first iteration (no usable step).')   % starting point (as a QP error: rebuild, retry, OnFailure)
                    end
                catch err
                    if ~qp_failure(err), rethrow(err), end   % anything else is an error to see, not a failed QP
                    if strcmpi(op.Display, 'iter')
                        where = '';  if ~isempty(err.stack), where = sprintf(' (%s, line %d)', err.stack(1).name, err.stack(1).line); end
                        fprintf('  SQP %4d  QP failed: %s%s%s\n', sqp_iter, err.message, where, ternary_str(info.PlanRebuilt, '', ' -> rebuilding the plan'))
                    end
                    info.TimeFeedback = info.TimeFeedback + toc(clock);
                    % QP failure, typically from a plan that has drifted into a region where the
                    % model is invalid. Rebuild the plan by a forward rollout of its inputs from x (no
                    % shooting gaps) and retry once; if that fails too, return the rebuilt plan.
                    Xb = obj.rollout(x, Ub);
                    if ~info.PlanRebuilt
                        info.PlanRebuilt = true;
                        sqp_iter = sqp_iter - 1;
                        continue
                    end
                    info.ExitFlag = -1;
                    info.QPStatus = 'failed';
                    break
                end
                info.QPIterations = info.QPIterations + qinfo.iter;
                info.QPColdRetries = info.QPColdRetries + (isfield(qinfo, 'cold_retry') && qinfo.cold_retry);
                info.QPAcceptable = info.QPAcceptable + (isfield(qinfo, 'acceptable') && qinfo.acceptable);
                info.QPInertiaCapped = info.QPInertiaCapped + (isfield(qinfo, 'inertia_capped') && qinfo.inertia_capped);
                info.QPStatus = qinfo.status;
                if isfield(qinfo, 'inertia_delta'), info.InertiaDelta = max(info.InertiaDelta, qinfo.inertia_delta); end
                info.QPSwitchedToLQ = info.QPSwitchedToLQ || (isfield(qinfo, 'lq') && qinfo.lq);
                if has_xb   % in cost (J) units: the QP objective is J/2
                    info.StateBoundMultiplier = 2 * max([sol.lam_lbx(:); sol.lam_ubx(:)]);
                end
                step_norm = max(max(abs(sol.x), [], 'all'), max(abs(sol.u), [], 'all'));
                qp_solved = strcmp(qinfo.status, 'solved');
                step_tol = op.SQPTol;  if kkt_stop, step_tol = 1e-12; end   % 'kkt': acados's tol_min_step_norm
                converged = step_norm < step_tol && qp_solved;   % a stalled QP's zero step is not convergence
                if bo.feas_now && bo.l1 >= op.SQPTolIneq, converged = false; qp_solved = false; end   % linearisation still infeasible

                alpha = 1;
                ls = struct('nu', NaN, 'viol0', NaN, 'D', NaN, 'n', 0, 'soc', 0);
                if (max_sqp > 1 || op.RTILineSearch) && op.LineSearch && ~converged
                    info.TimeFeedback = info.TimeFeedback + toc(clock);
                    clock = tic;
                    nu_in = nu_merit;
                    [alpha, nu_merit, ls, sol] = obj.merit_line_search(Xb, Ub, sol, ocp, x, ref, uref, W, su, J0, nu_in, ...
                                                                       strcmp(qinfo.status, 'solved'), op.QPAccuracyTest && ~bo.on);
                    % QPAccuracyTest: the step's merit model reduction is below its curvature, so the
                    % QP's residuals dominate it: re-solve with the QP tolerances x1e-2 (at most to
                    % 1e-14, where the IPMs meet round-off), then line-search without the test
                    qopt = op.QPSolverOptions;  tf = {'TolStat', 'TolEq', 'TolComp'};
                    while ls.short
                        tol = cellfun(@(f) qp_tol(qopt, f), tf);
                        if min(tol) * 1e-2 < 1e-14
                            [alpha, nu_merit, ls, sol] = obj.merit_line_search(Xb, Ub, sol, ocp, x, ref, uref, W, su, J0, nu_in, true, false);
                            break
                        end
                        for j = 1:3, qopt.(tf{j}) = tol(j) * 1e-2; end
                        if strcmpi(op.Display, 'iter')
                            fprintf('  SQP %4d  QP accuracy: model reduction %.3e < curvature %.3e -> QP tolerances %.0e\n', sqp_iter, ls.dl, ls.T, min(tol) * 1e-2)
                        end
                        try
                            [sol2, qinfo2, bo] = obj.solve_direction(ocp, Ub, rate, su, namedargs2cell(qopt), bo);
                        catch err
                            if ~qp_failure(err), rethrow(err), end
                            sol2 = [];
                        end
                        info.QPResolves = info.QPResolves + 1;
                        if isempty(sol2) || ~strcmp(qinfo2.status, 'solved')   % keep the solved, less accurate step
                            if ~isempty(sol2), info.QPIterations = info.QPIterations + qinfo2.iter; end
                            [alpha, nu_merit, ls, sol] = obj.merit_line_search(Xb, Ub, sol, ocp, x, ref, uref, W, su, J0, nu_in, true, false);
                            break
                        end
                        info.QPIterations = info.QPIterations + qinfo2.iter;
                        sol = sol2;  qinfo = qinfo2;
                        step_norm = max(max(abs(sol.x), [], 'all'), max(abs(sol.u), [], 'all'));
                        if step_norm < step_tol && ~(bo.feas_now && bo.l1 >= op.SQPTolIneq)   % the accurate step is below SQPTol: converged
                            converged = true;  alpha = 1;  nu_merit = nu_in;    % (the noisy step hid it)
                            ls = struct('nu', NaN, 'viol0', NaN, 'D', NaN, 'n', 0, 'soc', 0, 'dl', NaN, 'T', NaN, 'short', false);
                            break
                        end
                        [alpha, nu_merit, ls, sol] = obj.merit_line_search(Xb, Ub, sol, ocp, x, ref, uref, W, su, J0, nu_in, true, ~bo.on);
                    end
                    info.LineSearchSteps = info.LineSearchSteps + max(ls.n - 1, 0);
                    info.SOCAttempts = info.SOCAttempts + (ls.soc ~= 0);
                    info.SOCAccepted = info.SOCAccepted + (ls.soc == 1);
                    if isfield(ls, 'soc_iter'), info.QPIterations = info.QPIterations + ls.soc_iter; end
                    if ls.soc == 1   % the accepted correction's QP: its status, multipliers and step from here on
                        qinfo = ls.soc_qinfo;  info.QPStatus = qinfo.status;
                        step_norm = max(max(abs(sol.x), [], 'all'), max(abs(sol.u), [], 'all'));
                    end
                    info.TimeLineSearch = info.TimeLineSearch + toc(clock);
                    clock = tic;
                end
                if strcmpi(op.Display, 'iter')
                    if sqp_iter == 1
                        fprintf('  SQP %4s %11s %9s %9s %10s %9s %3s %3s %9s  %s\n', 'it', 'J', 'gaps', 'nu', 'D', 'alpha', 'ls', 'soc', 'step', 'QP')
                    end
                    fprintf('  SQP %4d %11.4g %9.2e %9.2e %10.3e %9.2e %3d %3s %9.2e  %s (%d)\n', sqp_iter, J0, ...
                        ls.viol0, ls.nu, ls.D, alpha, ls.n, ternary_str(ls.soc == 1, 'yes', ternary_str(ls.soc == -1, 'no', '')), ...
                        step_norm, qinfo.status, qinfo.iter)
                end
                % Second stopping test (when line-searching): the merit can decrease by no more
                % than SQPTolCost relative to the cost along this step, and the plan is
                % consistent (gaps at SQPTol per component). It stops the slow Gauss-Newton tail,
                % and turns a line-search failure at such a point (round-off) into convergence.
                stationary = qp_solved && isfinite(ls.D) && abs(ls.D) <= op.SQPTolCost * max(1, abs(J0)) && ...
                    ls.viol0 <= op.SQPTol * nx * M;
                if stationary && (~kkt_stop || alpha == 0)   % 'kkt': only to tell a round-off line-search failure
                    converged = true;
                    kkt_fail = false;
                    if alpha == 0 && kkt_stop && isfield(sol, 'raw_dual')
                        % 'kkt': converged only if the KKT test holds with this QP's multipliers; else
                        % take the smallest step tried and go on, as acados (its line search's MINSTEP
                        % is not a failure: the step is taken), or, with no such step, ExitFlag 2
                        [ok, r] = obj.kkt_converged(ocp, Ub, rate, su, rtimpc.blend_dual(dual, sol.raw_dual, 1));
                        kkt_fail = ~isnan(r.stat) && ~ok;
                        if kkt_fail && isfield(ls, 'alpha_last'), alpha = ls.alpha_last;  converged = false; end
                    end
                    if alpha == 0                 % round-off in the merit: keep the plan, converged
                        info.ExitFlag = 1 + kkt_fail;
                        if kkt_fail, info.KKT = r; end
                        info.TimeFeedback = info.TimeFeedback + toc(clock);
                        break
                    end
                end
                if alpha == 0
                    % Line search failed: the merit cannot be decreased along the QP step
                    % (e.g. iterates reaching a region where the model is invalid). Keep the plan.
                    info.ExitFlag = -2;
                    info.TimeFeedback = info.TimeFeedback + toc(clock);
                    break
                end
                Xb = Xb + alpha * sol.x;
                Un = Ub + alpha * (sol.u .* su);
                Ub = min(max(Un, umin), umax);   % strip IPM round-off
                info.InputClip = max(info.InputClip, max(abs(Un - Ub), [], 'all'));
                % multipliers from a solved QP only: an unsolved one's (iteration cap, stall) are not estimates
                % of the NLP's, and through the exact Hessian (lam' curvature) they spoil the next QPs (ball
                % maze, 2026-10-20: 1e22, after which every relaxed Byrd-Omojokun QP failed); as the merit weights
                solved_now = strcmp(qinfo.status, 'solved');   % (qp_solved is also false while Byrd-Omojokun leaves a violation)
                if kkt_stop && isfield(sol, 'raw_dual') && solved_now, dual = rtimpc.blend_dual(dual, sol.raw_dual, alpha); end
                % and not from a Byrd-Omojokun step's relaxed QP: its rows are relaxed to just what the
                % feasibility QP could not remove, so it is nearly infeasible and its multipliers are large
                % (they price the artificial tightness, not the NLP's rows); through the exact Hessian they
                % made the next QPs unsolvable (ball maze: curvature 3e12; quadprog fails on it too). acados
                % takes them; Gauss-Newton is unaffected (no multipliers). The previous estimates are kept.
                if (exact || ggn) && solved_now && ~bo.feas_now   % the QP's multipliers are the new estimates; move towards them as far as the step
                    lam.pi = lam.pi + alpha * (sol.pi - lam.pi);
                    if ngl > 0
                        lam.lg = lam.lg + alpha * ((sol.lam_ug - sol.lam_lg) - lam.lg);
                        lam.ug = lam.ug + alpha * (sol.lam_ug - lam.ug);
                    end
                end
                info.TimeFeedback = info.TimeFeedback + toc(clock);

                if converged && kkt_stop && sqp_iter < max_sqp
                    % a step below 1e-12 (taken, as acados): the next iteration's KKT test, at the new
                    % linearisation, decides (ExitFlag 1, or 2 if it fails). Testing here, at the old
                    % linearisation with the new multipliers, misses the step's Q d: with a large terminal
                    % weight (an LQR cost-to-go ~1e6) a 1e-12 step moves the gradient by 1e-6.
                    min_step_prev = true;
                    continue
                end
                if converged
                    info.ExitFlag = 1;
                    if kkt_stop   % (last iteration) a step below 1e-12: converged only if the KKT test now holds with this QP's
                        % multipliers (acados: else ACADOS_MINSTEP, "converged to an infeasible point")
                        [ok, r] = obj.kkt_converged(ocp, Ub, rate, su, dual);
                        if ~isnan(r.stat)   % (no residual, e.g. multipliers of another layout: the step test alone, as before)
                            info.KKT = r;  if ~ok, info.ExitFlag = 2; end
                        end
                    end
                    break
                end
            end

            obj.Xbar = Xb;
            obj.Ubar = Ub;
            obj.PlanTimes = G.tau;
            if exact || ggn, obj.Multipliers = lam; end
            if kkt_stop, obj.NLPDual = dual; end
            u = Ub(:,1);
            obj.Uprev = u;
            if info.ExitFlag < 0
                % Do not warm-start the next step from a plan the solver could not improve --
                % unless that plan came from a cold start ('auto'): cold-starting again from the
                % same guess would likely fail the same way, step after step
                switch lower(op.OnFailure)
                    case 'reset', do_reset = true;
                    case 'auto',  do_reset = warm;
                    case 'keep',  do_reset = false;
                    case 'keep-primal'   % the plan kept, its multipliers dropped (as acados after reset())
                        do_reset = false;  obj.QPDual = [];  obj.Multipliers = [];  obj.NLPDual = [];
                    otherwise, error('rtimpc:OnFailure', 'Unknown OnFailure ''%s''.', op.OnFailure)
                end
                if do_reset, obj.reset(); end
            end

            if ~isempty(obj.TerminalP) && ~all(isfinite(obj.TerminalP(:))), obj.TerminalP = []; end   % a failed Riccati solve is not kept ('fixed')
            info.SQPIterations = sqp_iter;
            info.SQPStepNorm = step_norm;
            if has_xb
                info.StateViolation = max(max(xmin - Xb(:,2:end), Xb(:,2:end) - xmax), [], 'all');
                info.StateViolation = max(info.StateViolation, 0);
            end
            % Feasible: the returned plan's hard bounds, rows and rate limits within SQPTolIneq. Soft
            % ones are priced, not required (StateViolation, ConstraintViolation); input bounds hold
            % by the clip; shooting gaps are not included (PlanGap)
            [hxmin, hxmax] = hard_state_bounds(obj, M);
            hv = max([0; reshape(max(hxmin - Xb(:,2:end), Xb(:,2:end) - hxmax), [], 1)]);
            if pc.on
                Gp = obj.path_values(Xb, Ub);
                gv = max(pc.gmin - Gp, Gp - pc.gmax);
                info.ConstraintViolation = max([0; gv(:)]);
                hv = max([hv; reshape(gv(~isfinite(pc.L1)), [], 1)]);
            end
            if rate.on
                dU = [Ub(:,1) - rate.u_prev, diff(Ub, 1, 2)];
                hv = max([hv; reshape(max(rate.min - dU, dU - rate.max), [], 1)]);
            end
            info.Feasible = hv <= op.SQPTolIneq;
            info.Converged = info.ExitFlag == 1;
            info.IntegratorUnconverged = obj.IntegratorUnconverged;
            if op.ComputeCost
                info.Cost = obj.nlp_cost(Xb, Ub, ref, uref, W);
                % the cost is that of a trajectory only if the plan has no gaps
                info.PlanGap = max(abs(obj.interval_ends(Xb(:,1:M), Ub, d) - Xb(:,2:M+1)), [], 'all');
            end
        end
    end

    methods

        function [x1, A, B] = simulate(obj, x, u, p)
        % One sample from x with u held, by the controller's own discretisation (Integrator,
        % IntegratorSubsteps, or Model.IntervalFcn / a discrete StateFcn), and its sensitivities
        % A = dx1/dx, B = dx1/du; p: model parameter (default none). E.g. to simulate a plant with
        % exactly the prediction model.
            if nargin < 4, p = []; end
            if nargout > 1, [x1, A, B] = obj.integrate(x(:), u(:), 1, p); else, x1 = obj.integrate(x(:), u(:), 1, p); end
        end

        function report = validate(obj, x, u, p, ref)
        % Check the derivatives the user supplied against central differences (step: fd_step) at
        % (x, u), with model parameter p and output reference ref (defaults: none, zero). One
        % relative error per derivative, for those that are set:
        %   StateJacobianX/U, OutputJacobianX/U          Jacobian.StateFcn, Jacobian.OutputFcn
        %   TerminalOutputJacobian                       Jacobian.TerminalOutputFcn
        %   IntervalJacobianX/U                          Model.IntervalFcn's A, B (one sample)
        %   ConstraintJacobianX/U, InitialConstraintJacobianX/U, TerminalConstraintJacobian
        %   StageCostGradient/Hessian, TerminalCostGradient/Hessian   the functions' 2nd and 3rd outputs
        %   IntervalHessian, ConstraintLagrangianHessian  the exact-Hessian functions, against
        %                                                differences of the Jacobians
        % Not checked: ConstraintHessianFcn and its initial and terminal forms, which need not be
        % the rows' Hessians (e.g. the convex-over-nonlinear term).
            if nargin < 4, p = []; end
            if nargin < 5 || isempty(ref), ref = zeros(obj.ny, 1); end
            x = x(:);  u = u(:);  nx = obj.nx;  nz = nx + obj.nu;  z = [x; u];
            ix = 1:nx;  iu = nx+1:nz;
            mdl = obj.Model;  jac = obj.Jacobian;  wts = obj.Weights;
            rel = @(M, Mfd) norm(M - Mfd, inf) / max(1, norm(Mfd, inf));
            report = struct();
            if ~isempty(mdl.StateFcn)
                [~, A, B] = obj.eval_model(x, u, p);
                J = fd_jacobian(@(v) obj.eval_model(v(ix), v(iu), p), z);
                report.StateJacobianX = rel(A, J(:, ix));  report.StateJacobianU = rel(B, J(:, iu));
            end
            if ~isempty(mdl.IntervalFcn)
                [~, A, B] = obj.integrate(x, u, 1, p);
                J = fd_jacobian(@(v) obj.integrate(v(ix), v(iu), 1, p), z);
                report.IntervalJacobianX = rel(A, J(:, ix));  report.IntervalJacobianU = rel(B, J(:, iu));
            end
            [~, C, D] = obj.output(x, u);
            J = fd_jacobian(@(v) obj.output(v(ix), v(iu)), z);
            report.OutputJacobianX = rel(C, J(:, ix));  report.OutputJacobianU = rel(D, J(:, iu));
            if ~isempty(mdl.TerminalOutputFcn) && ~isempty(obj.Jacobian.TerminalOutputFcn)
                [~, CN] = obj.output(x, u, false, true);
                report.TerminalOutputJacobian = rel(CN, fd_jacobian(mdl.TerminalOutputFcn, x));
            end
            if ~isempty(mdl.ConstraintFcn) && ~isempty(jac.ConstraintFcn)
                [Cg, Dg] = jac.ConstraintFcn(x, u);
                J = fd_jacobian(@(v) mdl.ConstraintFcn(v(ix), v(iu)), z);
                report.ConstraintJacobianX = rel(Cg, J(:, ix));  report.ConstraintJacobianU = rel(Dg, J(:, iu));
            end
            if ~isempty(mdl.InitialConstraintFcn) && ~isempty(jac.InitialConstraintFcn)
                [Cg, Dg] = jac.InitialConstraintFcn(x, u);
                J = fd_jacobian(@(v) mdl.InitialConstraintFcn(v(ix), v(iu)), z);
                report.InitialConstraintJacobianX = rel(Cg, J(:, ix));  report.InitialConstraintJacobianU = rel(Dg, J(:, iu));
            end
            if ~isempty(mdl.TerminalConstraintFcn) && ~isempty(jac.TerminalConstraintFcn)
                report.TerminalConstraintJacobian = rel(jac.TerminalConstraintFcn(x), fd_jacobian(mdl.TerminalConstraintFcn, x));
            end
            if ~isempty(wts.StageCostFcn)
                f = @(v) wts.StageCostFcn(v(ix), v(iu), ref);
                [~, g, H] = f(z);
                report.StageCostGradient = rel(g(:), fd_jacobian(f, z)');
                report.StageCostHessian  = rel(reshape(H, nz, nz), fd_jacobian(@(v) second_output(f, v), z));
            end
            if ~isempty(wts.TerminalCostFcn)
                [~, g] = wts.TerminalCostFcn(x, u, ref);
                n = numel(g);   % in x_N, or in [x_N; u_last]
                if n > nx, f = @(v) wts.TerminalCostFcn(v(ix), v(iu), ref); else, f = @(v) wts.TerminalCostFcn(v, u, ref); end
                [~, g, H] = f(z(1:n));
                report.TerminalCostGradient = rel(g(:), fd_jacobian(f, z(1:n))');
                report.TerminalCostHessian  = rel(reshape(H, n, n), fd_jacobian(@(v) second_output(f, v), z(1:n)));
            end
            if ~isempty(mdl.IntervalHessianFcn)   % Hessian of w'F(x, u) against differences of [A'w; B'w]
                w = (1:nx)' / nx;
                a = {x, u, 1, w};  if ~isempty(p), a{end+1} = p; end
                Hfd = fd_jacobian(@(v) interval_adjoint(obj, v, p, w), z);
                report.IntervalHessian = rel(reshape(mdl.IntervalHessianFcn(a{:}), nz, nz), (Hfd + Hfd') / 2);
            end
            if ~isempty(mdl.ConstraintLagrangianHessianFcn) && ~isempty(mdl.ConstraintFcn)   % of w'g(x, u)
                ng = numel(mdl.ConstraintFcn(x, u));  w = (1:ng)' / ng;
                Hfd = fd_jacobian(@(v) constraint_adjoint(mdl.ConstraintFcn, jac.ConstraintFcn, v, nx, w), z);
                report.ConstraintLagrangianHessian = rel(reshape(mdl.ConstraintLagrangianHessianFcn(x, u, w), nz, nz), (Hfd + Hfd') / 2);
            end
            if nargout == 0
                fprintf('rtimpc.validate: relative errors of the supplied derivatives (vs central differences)\n')
                disp(report)
            end
        end
    end


    methods (Access = private)

        function check_step_inputs(obj, x, ref, opts, n)
        % Sizes of step's arguments and of the weights it uses (n: the samples planned over)
            nx = obj.nx;  nu = obj.nu;  ny = obj.ny;
            cols = @(v) size(v, 2) == 1 || size(v, 2) >= n;
            bad = @(varargin) error('rtimpc:step:size', varargin{:});
            if numel(x) ~= nx, bad('x must have nx = %d elements (got %d).', nx, numel(x)); end
            if size(ref, 1) ~= ny || ~cols(ref)
                bad('ref must be ny x 1 or ny x N (ny = %d, at least %d columns); got %d x %d.', ny, n, size(ref, 1), size(ref, 2))
            end
            ur = opts.InputReference;
            if ~isempty(ur) && (size(ur, 1) ~= nu || ~cols(ur))
                bad('InputReference must be nu x 1 or nu x N (nu = %d, at least %d columns); got %d x %d.', nu, n, size(ur, 1), size(ur, 2))
            end
            W = opts.OutputWeights;
            if ~isempty(W) && (~isequal(size(W, [1 2]), [ny ny]) || size(W, 3) < n)
                bad('OutputWeights must be ny x ny x N (ny = %d, at least %d pages); got %s.', ny, n, mat2str(size(W)))
            end
            if isempty(W) && ~isscalar(obj.Weights.Qy) && ~isequal(size(obj.Weights.Qy), [ny ny]), bad('Weights.Qy must be ny x ny (ny = %d); got %s.', ny, mat2str(size(obj.Weights.Qy))); end
            if isempty(W) && ~isempty(obj.Weights.Qyf) && ~isscalar(obj.Weights.Qyf) && ~isequal(size(obj.Weights.Qyf), [ny ny]), bad('Weights.Qyf must be ny x ny (ny = %d); got %s.', ny, mat2str(size(obj.Weights.Qyf))); end
            if ~isequal(size(obj.Weights.R), [nu nu]), bad('Weights.R must be nu x nu (nu = %d); got %s.', nu, mat2str(size(obj.Weights.R))); end
        end

        function s = state_snapshot(obj)
        % The controller's state between steps (its private properties: plan, multipliers, QP warm start, ...)
            s = struct();
            for p = obj.state_props(), s.(p) = obj.(p); end
        end

        function state_restore(obj, s)
            for p = obj.state_props(), obj.(p) = s.(p); end
        end

        function set_compute_cost(obj, v)
            obj.Optimisation.ComputeCost = v;
        end

        function set_tail_frozen(obj, v)
            obj.TailFrozen = v;
        end

        function names = state_props(obj)
            P = metaclass(obj).PropertyList;
            P = P(arrayfun(@(q) ischar(q.SetAccess) && strcmp(q.SetAccess, 'private') && ~q.Dependent && ~q.Constant, P));
            names = string({P.Name});
        end

        function [f, A, B] = eval_model(obj, x, u, p)
        % The model (and its Jacobians); with a parameter p (step's Parameters) called as (x, u, p)
            a = {x, u};  if nargin > 3 && ~isempty(p), a = {x, u, p}; end
            if nargout == 1
                f = obj.Model.StateFcn(a{:});
            elseif obj.Model.StateFcnReturnsJacobian
                [f, A, B] = obj.Model.StateFcn(a{:});
            else
                f = obj.Model.StateFcn(a{:});
                [A, B] = obj.Jacobian.StateFcn(a{:});
            end
        end

        function p = node_param(obj, i)
        % Model parameter of node i's interval (the last node's beyond the end); [] without parameters
            if isempty(obj.ParamNow), p = []; return, end
            p = obj.ParamNow(:, min(i, size(obj.ParamNow, 2)));
        end

        function varargout = interval_fcn(obj, X, U, d)
        % Model.IntervalFcn over intervals starting at nodes 1..size(X,2), with their parameters
            if isempty(obj.ParamNow), [varargout{1:nargout}] = obj.Model.IntervalFcn(X, U, d); return, end
            Pn = obj.ParamNow(:, min(1:size(X, 2), size(obj.ParamNow, 2)));
            [varargout{1:nargout}] = obj.Model.IntervalFcn(X, U, d, Pn);
        end

        function [y, C, D] = output(obj, x, u, hasD, terminal)
        % Output and its Jacobians; hasD says whether Jacobian.OutputFcn returns [C, D]. terminal:
        % node N, whose output is Model.TerminalOutputFcn(x) when given, else OutputFcn, which
        % must then not depend on u there (no u_N exists; as acados's separate terminal output)
            if nargin > 4 && terminal && ~isempty(obj.Model.TerminalOutputFcn)
                hN = obj.Model.TerminalOutputFcn;
                y = hN(x);
                if nargout < 2, return, end
                if isempty(obj.Jacobian.TerminalOutputFcn), C = fd_jacobian(hN, x);
                else, C = obj.Jacobian.TerminalOutputFcn(x);
                end
                D = zeros(numel(y), obj.nu);
                return
            end
            if isempty(obj.Model.OutputFcn)
                y = x; C = eye(obj.nx); D = zeros(obj.nx, obj.nu);
                return
            end
            y = obj.Model.OutputFcn(x, u);
            if nargout < 2, return, end
            if nargin < 4, hasD = output_has_D(obj, x, u); end
            if hasD
                [C, D] = obj.Jacobian.OutputFcn(x, u);
            else
                C = obj.Jacobian.OutputFcn(x, u);
                D = zeros(obj.ny, obj.nu);
            end
            if nargin > 4 && terminal && any(D(:))
                error('rtimpc:TerminalOutput', ['The output depends on the input (D ~= 0 at node N), and node N has no ' ...
                      'input: set Model.TerminalOutputFcn = @(x) y_N (and Jacobian.TerminalOutputFcn).'])
            end
        end

        function hasD = output_has_D(obj, x, u)
        % Does Jacobian.OutputFcn return a second output D = dy/du? Checked once per output Jacobian
        % function (cached until Jacobian.OutputFcn is replaced). By the call alone, not by D's
        % values: a D that is zero at this point need not be zero elsewhere.
            hasD = false;
            if isempty(obj.Model.OutputFcn), return, end
            c = obj.HasDCache;
            if ~isempty(c) && isequal(c.fcn, obj.Jacobian.OutputFcn), hasD = c.hasD; return, end
            try
                [~, D] = obj.Jacobian.OutputFcn(x, u);
                hasD = ~isempty(D);
            catch err   % a Jacobian that returns C alone; any other error is the user's function's
                if ~ismember(err.identifier, {'MATLAB:TooManyOutputs', 'MATLAB:needMoreRhsOutputs', 'MATLAB:maxlhs', 'MATLAB:unassignedOutputs'})
                    rethrow(err)
                end
            end
            obj.HasDCache = struct('fcn', obj.Jacobian.OutputFcn, 'hasD', hasD);
        end

        function X = rollout(obj, x0, U)
        % Simulate an input sequence from x0 over the step's shooting grid (a gap-free plan)
            if ~isempty(obj.GridNow) && numel(obj.GridNow.d) == size(U, 2), d = obj.GridNow.d;
            else, d = ones(1, size(U, 2));
            end
            M = size(U, 2);
            X = zeros(obj.nx, M+1);
            X(:,1) = x0;
            for i = 1:M, X(:,i+1) = obj.integrate(X(:,i), U(:,i), d(i), obj.node_param(i)); end
        end

        function d = horizon_grid(obj, N)
        % Shooting-interval lengths in fine samples (summing to N), nearest first: fine remainder
        % samples, then StepsPerTier blocks per near tier, then the far tier, which takes the rest
        % (SD-OPT's build_horizon_grid; a shorter N truncates it progressively). No decimation:
        % all ones. IntervalSamples gives the grid explicitly instead; a shorter N keeps its first
        % N samples (the interval crossing N is clipped).
            iv = obj.Optimisation.IntervalSamples;
            if ~isempty(iv)
                iv = iv(:)';
                if ~isempty(obj.Optimisation.DecimationFactors) || ~isempty(obj.Optimisation.StepsPerTier)
                    error('rtimpc:IntervalSamples', 'Set either IntervalSamples or DecimationFactors/StepsPerTier, not both.')
                end
                if any(iv < 1 | iv ~= round(iv)) || sum(iv) ~= obj.PredictionHorizon
                    error('rtimpc:IntervalSamples', 'IntervalSamples must be positive integers summing to the prediction horizon (%d samples); got sum %g.', ...
                        obj.PredictionHorizon, sum(iv))
                end
                c = cumsum(iv);  k = find(c >= N, 1);
                d = iv(1:k);  d(k) = N - (c(k) - iv(k));
                return
            end
            df = sort(obj.Optimisation.DecimationFactors(:)', 'ascend');
            if isempty(df) || all(df == 1), d = ones(1, N); return, end
            spt = obj.Optimisation.StepsPerTier;  if isempty(spt), spt = df(end); end
            budget = N;  nb = zeros(1, numel(df));
            for i = 1:numel(df)-1
                nb(i) = min(spt, floor(budget / df(i)));
                budget = budget - nb(i) * df(i);
            end
            nb(end) = floor(budget / df(end));
            d = [ones(1, budget - nb(end) * df(end)), repelem(df, nb)];
        end

        function G = node_costs(obj, d, W, ref, uref)
        % Map per-sample costs (W, ref, uref over samples 1..n) onto the shooting grid. Node j
        % (the end of interval j) uses its own sample's reference. 'euler': it carries the summed
        % weight of its interval's samples. 'integrator': a coarse interval's cost is split over its
        % integration steps (each carries the weight of the samples it spans, and the reference
        % interpolated at its time); the node carries the last step's share. Fine intervals: as is.
            M = numel(d);  tau = [0, cumsum(d)];
            G = struct('M', M, 'd', d, 'tau', tau, 'W', W, 'ref', ref, 'uref', uref, 'pts', {{}});
            if all(d == 1), return, end
            ny = size(ref, 1);
            G.W = zeros(ny, ny, M);  G.ref = zeros(ny, M);  G.uref = zeros(size(uref, 1), M);
            integ = strcmpi(obj.Optimisation.CostDiscretisation, 'integrator');
            ns = obj.Optimisation.IntegratorSubsteps;
            if ~obj.Model.IsContinuousTime, ns = NaN; end   % discrete model: one step per sample
            if integ, G.pts = cell(1, M); end
            for j = 1:M
                f = tau(j)+1 : tau(j+1);
                G.uref(:,j) = mean(uref(:, f), 2);
                G.ref(:,j)  = ref(:, f(end));
                if ~integ || d(j) == 1
                    G.W(:,:,j) = sum(W(:,:,f), 3);
                    continue
                end
                n = ns;  if isnan(n), n = d(j); end
                span = d(j) / n;                                   % samples per integration step
                pts = struct('step', {}, 'W', {}, 'ref', {});
                for st = 1:n
                    a = (st-1) * span;  b = st * span;             % step covers (a, b] in the interval
                    Wst = zeros(ny);
                    for k = 1:d(j)                                 % sample k covers (k-1, k]
                        ov = max(0, min(k, b) - max(k-1, a));
                        if ov > 0, Wst = Wst + ov * W(:,:,f(k)); end
                    end
                    if st == n
                        G.W(:,:,j) = Wst;                          % node: the last step's share
                    elseif any(Wst(:))
                        tq = tau(j) + b;                           % step time, in samples
                        lo = max(floor(tq), 1);  hi = min(lo + 1, tau(end));  w = tq - lo;
                        pts(end+1) = struct('step', st, 'W', Wst, 'ref', (1-w)*ref(:,lo) + w*ref(:,hi)); %#ok<AGROW>
                    end
                end
                G.pts{j} = pts;
            end
        end

        function g = node_linear_cost(obj, G)
        % InputLinear on the step's grid: node j's input is held over its interval's samples, so it
        % carries their summed g (nu x M); [] when there is none
            g = obj.Weights.InputLinear;
            if isempty(g) || ~any(g(:)), g = []; return, end
            N = obj.PredictionHorizon;
            if size(g, 2) == 1, g = repmat(g(:), 1, N); end
            if ~isequal(size(g), [obj.nu, N])
                error('rtimpc:InputLinear', 'Weights.InputLinear must be nu x 1 or nu x N (N = %d).', N)
            end
            g = cumsum([zeros(obj.nu, 1), g(:, 1:G.tau(end))], 2);
            g = g(:, G.tau(2:end) + 1) - g(:, G.tau(1:end-1) + 1);
        end

        function [X, U] = shift_plan(obj, Xold, Uold, tau_old, tau)
        % Warm start across grids: re-sample the previous plan (node times tau_old) one sample
        % later onto the new grid (node times tau). Inputs piecewise constant; states interpolated
        % between old nodes, and integrated with the last input past the old plan's end.
            if isempty(tau_old), tau_old = 0:size(Uold, 2); end
            t = tau + 1;                                       % new node times on the old axis
            Mo = size(Uold, 2);  To = tau_old(end);  M = numel(tau) - 1;
            X = zeros(obj.nx, M+1);  U = zeros(obj.nu, M);
            for j = 1:M
                U(:,j) = Uold(:, min(find(tau_old(1:end-1) <= t(j), 1, 'last'), Mo));
            end
            for j = 1:M+1
                if t(j) <= To
                    k = find(tau_old <= t(j), 1, 'last');
                    if tau_old(k) == t(j), X(:,j) = Xold(:,k);
                    else
                        w = (t(j) - tau_old(k)) / (tau_old(k+1) - tau_old(k));
                        X(:,j) = (1 - w) * Xold(:,k) + w * Xold(:,k+1);
                    end
                else
                    X(:,j) = obj.integrate(Xold(:,end), Uold(:,end), t(j) - To, obj.node_param(M));
                end
            end
        end

        function [x1, A, B, Xs, As, Bs] = integrate(obj, x0, u, len, p)
        % One shooting interval of len samples (default 1) with u held, and its sensitivities
        % A = dx1/dx0, B = dx1/du. IntegratorSubsteps steps per interval, so a coarse interval takes
        % larger steps (a discrete model is applied len times). Optionally also the state after
        % each step, Xs(:,s), with its sensitivities As(:,:,s), Bs(:,:,s). p: the interval's model
        % parameter ([] -> none).
            if nargin < 4, len = 1; end
            if nargin < 5, p = []; end
            need_sens = nargout > 1;  keep = nargout > 3;
            nx = obj.nx;
            if ~isempty(obj.Model.IntervalFcn)
                if keep, error('rtimpc:IntervalFcn', 'Model.IntervalFcn does not support CostDiscretisation ''integrator''.'), end
                a = {x0, u, len};  if ~isempty(p), a = {x0, u, len, p}; end
                if need_sens, [x1, A, B] = obj.Model.IntervalFcn(a{:}); else, x1 = obj.Model.IntervalFcn(a{:}); end
                return
            end
            if ~obj.Model.IsContinuousTime
                if len == 1 && ~keep
                    if need_sens, [x1, A, B] = obj.eval_model(x0, u, p); else, x1 = obj.eval_model(x0, u, p); end
                    return
                end
                x1 = x0;
                if need_sens, A = eye(nx);  B = zeros(nx, obj.nu); end
                if keep, Xs = zeros(nx, len);  As = zeros(nx, nx, len);  Bs = zeros(nx, obj.nu, len); end
                for s = 1:len
                    if need_sens
                        [x1, Ai, Bi] = obj.eval_model(x1, u, p);
                        B = Ai*B + Bi;  A = Ai*A;
                    else
                        x1 = obj.eval_model(x1, u, p);
                    end
                    if keep, Xs(:,s) = x1;  As(:,:,s) = A;  Bs(:,:,s) = B; end
                end
                return
            end
            M = obj.Optimisation.IntegratorSubsteps;
            h = len * obj.Ts / M;
            I = eye(nx);
            % The model as one function handle [f, A, B] = fm(x, u), resolved once: a method call
            % per RK4 stage (obj.eval_model) costs more than the arithmetic around it
            if obj.Model.StateFcnReturnsJacobian
                fm = obj.Model.StateFcn;
                if ~isempty(p), fm = @(x, u) obj.Model.StateFcn(x, u, p); end
            else
                Ff = obj.Model.StateFcn;  Fj = obj.Jacobian.StateFcn;
                if isempty(p), fm = @(x, u) f_and_jacobian(Ff, Fj, x, u);
                else, fm = @(x, u) f_and_jacobian(@(z, v) Ff(z, v, p), @(z, v) Fj(z, v, p), x, u);
                end
            end
            x1 = x0;
            if need_sens, A = I; B = zeros(nx, obj.nu); end
            if keep, Xs = zeros(nx, M);  As = zeros(nx, nx, M);  Bs = zeros(nx, obj.nu, M); end

            for s = 1:M
                switch obj.Optimisation.Integrator
                    case 'rk4'
                        if ~need_sens
                            k1 = fm(x1, u);
                            k2 = fm(x1 + h/2*k1, u);
                            k3 = fm(x1 + h/2*k2, u);
                            k4 = fm(x1 + h*k3, u);
                            x1 = x1 + h/6*(k1 + 2*k2 + 2*k3 + k4);
                            if keep, Xs(:,s) = x1; end
                            continue
                        end
                        % Forward sensitivities through the RK4 stages
                        [k1, A1, B1] = fm(x1, u);
                        [k2, A2, B2] = fm(x1 + h/2*k1, u);
                        K2x = A2*(I + h/2*A1);   K2u = A2*(h/2*B1) + B2;
                        [k3, A3, B3] = fm(x1 + h/2*k2, u);
                        K3x = A3*(I + h/2*K2x);  K3u = A3*(h/2*K2u) + B3;
                        [k4, A4, B4] = fm(x1 + h*k3, u);
                        K4x = A4*(I + h*K3x);    K4u = A4*(h*K3u) + B4;
                        x1 = x1 + h/6*(k1 + 2*k2 + 2*k3 + k4);
                        Ax = I + h/6*(A1 + 2*K2x + 2*K3x + K4x);
                        Bu = h/6*(B1 + 2*K2u + 2*K3u + K4u);

                    case 'trapezoidal'
                        % Implicit trapezoidal rule; Newton on the implicit equation
                        [f0, A0, B0] = obj.eval_model(x1, u, p);
                        xn = x1 + h*f0;  newton_ok = false;
                        for it = 1:20
                            [fn, An] = obj.eval_model(xn, u, p);
                            dxn = (I - h/2*An) \ (xn - x1 - h/2*(f0 + fn));
                            xn = xn - dxn;
                            if norm(dxn, inf) < 1e-13 * max(1, norm(xn, inf)), newton_ok = true; break, end
                        end
                        if ~newton_ok, obj.IntegratorUnconverged = obj.IntegratorUnconverged + 1; end
                        if need_sens
                            [~, An, Bn] = obj.eval_model(xn, u, p);
                            Jn = I - h/2*An;
                            Ax = Jn \ (I + h/2*A0);
                            Bu = Jn \ (h/2*(B0 + Bn));
                        end
                        x1 = xn;

                    case 'irk'
                        % Implicit Runge-Kutta (collocation): stage slopes K_i = f(x1 + h sum_j a_ij K_j, u),
                        % solved by Newton from K_i = f(x1, u); sensitivities by the implicit function theorem
                        [Ab, bb] = irk_tableau(obj.Optimisation.IRKStages, obj.Optimisation.IRKScheme);
                        ns = numel(bb);  nits = obj.Optimisation.IRKNewtonIterations;
                        [f0, A0, B0] = fm(x1, u);
                        K = repmat(f0, 1, ns);  J = repmat({A0}, 1, ns);  Bs_ = repmat({B0}, 1, ns);
                        newton_ok = ~isempty(nits);   % a fixed iteration count is not a convergence failure
                        for it = 1:ternary_num(isempty(nits), 30, nits)
                            Z = x1 + h * K * Ab';                  % stage states
                            R = zeros(nx, ns);
                            for i = 1:ns, [fz, J{i}, Bs_{i}] = fm(Z(:,i), u);  R(:,i) = K(:,i) - fz; end
                            Mn = eye(nx * ns);
                            for i = 1:ns, for j = 1:ns
                                Mn((i-1)*nx + (1:nx), (j-1)*nx + (1:nx)) = Mn((i-1)*nx + (1:nx), (j-1)*nx + (1:nx)) - h * Ab(i,j) * J{i};
                            end, end
                            dK = reshape(Mn \ R(:), nx, ns);
                            K = K - dK;
                            if isempty(nits) && norm(dK(:), inf) < 1e-13 * max(1, norm(K(:), inf)), newton_ok = true; break, end
                        end
                        if ~newton_ok, obj.IntegratorUnconverged = obj.IntegratorUnconverged + 1; end
                        if need_sens   % at the final stages
                            Z = x1 + h * K * Ab';
                            for i = 1:ns, [~, J{i}, Bs_{i}] = fm(Z(:,i), u); end
                            Mn = eye(nx * ns);
                            for i = 1:ns, for j = 1:ns
                                Mn((i-1)*nx + (1:nx), (j-1)*nx + (1:nx)) = Mn((i-1)*nx + (1:nx), (j-1)*nx + (1:nx)) - h * Ab(i,j) * J{i};
                            end, end
                            dKdx = Mn \ vertcat(J{:});  dKdu = Mn \ vertcat(Bs_{:});
                            Ax = I;  Bu = zeros(nx, obj.nu);
                            for j = 1:ns
                                Ax = Ax + h * bb(j) * dKdx((j-1)*nx + (1:nx), :);
                                Bu = Bu + h * bb(j) * dKdu((j-1)*nx + (1:nx), :);
                            end
                        end
                        x1 = x1 + h * K * bb;

                    otherwise
                        error('rtimpc:Integrator', 'Unknown Integrator ''%s''.', obj.Optimisation.Integrator)
                end
                if need_sens
                    B = Ax*B + Bu;
                    A = Ax*A;
                end
                if keep, Xs(:,s) = x1;  As(:,:,s) = A;  Bs(:,:,s) = B; end
            end
        end

        function Xn = interval_ends(obj, X, U, d)
        % End states of the intervals starting at X(:,i) with U(:,i) held over d(i) samples
            if ~isempty(obj.Model.IntervalFcn), Xn = obj.interval_fcn(X, U, d); return, end
            Xn = zeros(size(X));
            for i = 1:size(X, 2), Xn(:,i) = obj.integrate(X(:,i), U(:,i), d(i), obj.node_param(i)); end
        end

        function [qp, eqr] = prepare_qp(obj, ocp, Ub, rate, su)
        % The QP as passed to the solver: rate-augmented when rate terms are set; for the Riccati
        % solver, hard first-node rows pinned to one value as its equality rows (first_node_equalities)
            qp = ocp;  eqr = false(0, 1);
            if rate.on, qp = obj.rate_augment(ocp, Ub, rate, su); end
            if strcmp(obj.Optimisation.QPSolver, 'riccati'), [qp, eqr] = rtimpc.first_node_equalities(qp); end
        end

        function [ok, r] = kkt_converged(obj, ocp, Ub, rate, su, dual)
        % SQPStopping 'kkt': acados's NLP residuals (ocp_nlp_res_compute) at the current plan, i.e. the
        % KKT residuals of this iteration's QP at the zero step with the plan's multipliers (QP units:
        % cost J/2 as acados's; inputs back to their own units), each against its tolerance. With
        % SQPTolStatScaling 'ipopt' the stationarity and the complementarity are first divided by
        % r.stat_scale and r.comp_scale (IPOPT's s_d and s_c).
            op = obj.Optimisation;
            qp = obj.prepare_qp(ocp, Ub, rate, su);
            xs = [];  if rate.on, xs = obj.nx + (1:obj.nu); end   % rate augmentation: scaled previous input steps
            if ~dual_fits(qp, dual)   % multipliers of another layout (e.g. rate terms switched on this step): no test now
                r = struct('stat', NaN, 'eq', NaN, 'ineq', NaN, 'comp', NaN, 'stat_scale', NaN, 'comp_scale', NaN);  ok = false;
                return
            end
            r = ocp_nlp_res_compute(qp, dual, su, xs);
            % The plan's x_0 against the measurement: the QP eliminates x_0 = x (ocp.x0 = x - Xb(:,1)
            % enters only its linearisation), acados keeps it as an equality bound on the iterate,
            % whose violation is in its ineq residual. Without it a changed measurement passed when
            % the linearisation hides it (A = 0, Cg = 0 at the old x_0)
            r.ineq = max(r.ineq, max([0; abs(ocp.x0(:))]));
            r.stat_scale = 1;  r.comp_scale = 1;
            if strcmpi(op.SQPTolStatScaling, 'ipopt'), [r.stat_scale, r.comp_scale] = rtimpc.ipopt_stat_scale(qp, dual); end
            ok = r.stat / r.stat_scale < op.SQPTolStat && r.eq < op.SQPTolEq && r.ineq < op.SQPTolIneq && r.comp / r.comp_scale < op.SQPTolComp;
        end

        function [sol, qinfo] = solve_qp(obj, ocp, Ub, rate, su, args)
        % Solve the step's OCP-QP (with the input-rate augmentation when rate terms are set).
        % sol.raw_dual: the solver's multipliers in the QP's own layout (SQPStopping 'kkt')
            [qp, eqr] = obj.prepare_qp(ocp, Ub, rate, su);
            if ~isempty(obj.QPDualInit) && size(obj.QPDualInit.lam_lbx, 1) == size(qp.A, 1)
                qp.lam_init = obj.QPDualInit;
            end
            if ~ismember(obj.Optimisation.QPSolver, {'riccati', 'condensed'})
                error('rtimpc:QPSolver', 'Unknown QPSolver ''%s''.', obj.Optimisation.QPSolver)
            end
            args = qp_solver_args(args, obj.Optimisation.QPSolver);
            switch obj.Optimisation.QPSolver
                case 'riccati'
                    if ~any(strcmp(args(1:2:end), 'InertiaDeltaHint')), args = [args, {'InertiaDeltaHint', obj.InertiaDelta}]; end
                    [sol, qinfo] = ocp_qp_riccati_ipm(qp, args{:});
                    if ~strcmp(qinfo.status, 'solved') && isfield(qp, 'lam_init')
                        % a warm-started QP that fails: once more from a cold start (acados cold-starts
                        % HPIPM by default; a warm start can drive the complementarity to zero before
                        % stationarity, and the factorisation then breaks down)
                        it_warm = qinfo.iter;
                        [sol, qinfo] = ocp_qp_riccati_ipm(rmfield(qp, 'lam_init'), args{:});
                        qinfo.iter = qinfo.iter + it_warm;  qinfo.cold_retry = true;
                    end
                    sol.raw_dual = rtimpc.raw_dual(sol);
                    if any(eqr)   % their multiplier back as the rows' net one (lam_ug - lam_lg)
                        sol.lam_ug(eqr, 1) = max(sol.nu_eq1, 0);  sol.lam_lg(eqr, 1) = max(-sol.nu_eq1, 0);
                    end
                    if qinfo.inertia_delta > 0 && strcmp(qinfo.status, 'solved'), obj.InertiaDelta = qinfo.inertia_delta; end
                    obj.QPDual = struct('lam_lb', sol.lam_lb, 'lam_ub', sol.lam_ub, ...
                                        'lam_lbx', sol.lam_lbx, 'lam_ubx', sol.lam_ubx);
                    if isfield(sol, 'lam_lg'), obj.QPDual.lam_lg = sol.lam_lg;  obj.QPDual.lam_ug = sol.lam_ug; end
                    if ~strcmp(qinfo.status, 'solved')   % unconverged multipliers (possibly diverging): never a warm start
                        obj.QPDual = [];  obj.QPDualInit = [];
                    elseif obj.Optimisation.QPWarmStart
                        obj.QPDualInit = obj.QPDual;     % later SQP iterations
                    end
                case 'condensed'
                    [sol, qinfo] = ocp_qp_condensed_ipm(qp, args{:});
                    sol.raw_dual = rtimpc.raw_dual(sol);
            end
            if rate.on, sol = rtimpc.rate_deaugment(sol, obj.nx); end
        end

        function [sol, qinfo, bo] = solve_direction(obj, ocp, Ub, rate, su, args, bo)
        % The SQP step's QP. InfeasibleQP 'byrd-omojokun' (acados's SQP_WITH_FEASIBLE_QP direction):
        % solve the QP; if it is unsolved (infeasible linearisation: iteration cap or stall), switch
        % to two QPs per iteration for the rest of the step: a feasibility QP (every hard state bound
        % and constraint row softened with unit L1 price, Hessian 1e-4 I, no cost; soft rows free,
        % input bounds and rate limits kept hard), then the QP with each softened row relaxed by
        % 1.00001 x the violation the feasibility QP could not remove. InfeasibleQPSticky false (the
        % default): the plain QP is tried first at every iteration; true (acados's): back to one QP
        % only after the feasibility QPs find (nearly) no violation twice.
            bo.feas_now = 0;
            mode = lower(obj.Optimisation.InfeasibleQP);
            if ~bo.on || ~obj.Optimisation.InfeasibleQPSticky
                [sol, qinfo] = obj.solve_qp(ocp, Ub, rate, su, args);
                if strcmp(qinfo.status, 'solved') || strcmp(mode, 'none'), bo.on = false; return, end
                bo.on = true;   % unsolved: Byrd-Omojokun from here on (this iteration included)
                n0 = qinfo.iter;
            else
                n0 = 0;
            end
            [fq, hard] = feasibility_qp(ocp);
            rz = rate;  rz.R = zeros(size(rate.R));        % no rate cost in the feasibility QP
            keep = {obj.QPDual, obj.QPDualInit};  obj.QPDualInit = [];   % its own (cold) start; not a warm start for others
            [fs, finfo] = obj.solve_qp(fq, Ub, rz, su, args);
            [obj.QPDual, obj.QPDualInit] = keep{:};
            bo.feas_now = 1;  bo.l1 = Inf;
            if ~strcmp(finfo.status, 'solved')
                sol = fs;  qinfo = finfo;  qinfo.iter = qinfo.iter + n0;
                return
            end
            % relax the softened rows by what the feasibility QP could not remove
            q = ocp;  f = 1.00001;  l1 = 0;
            if isfield(q, 'lbx')
                sl = fs.slack_lbx .* hard.xl;  su_ = fs.slack_ubx .* hard.xu;
                q.lbx = q.lbx - f * sl;  q.ubx = q.ubx + f * su_;  l1 = l1 + sum(sl + su_, 'all');
            end
            if isfield(q, 'lg')
                sl = fs.slack_lg .* hard.gl;  su_ = fs.slack_ug .* hard.gu;
                q.lg = q.lg - f * sl;  q.ug = q.ug + f * su_;  l1 = l1 + sum(sl + su_, 'all');
            end
            [sol, qinfo] = obj.solve_qp(q, Ub, rate, su, args);
            qinfo.iter = qinfo.iter + finfo.iter + n0;
            bo.l1 = l1 / max(1, hard.n);
            if bo.l1 < obj.Optimisation.SQPTolIneq          % acados: tol_ineq
                bo.zero_slacks = bo.zero_slacks + 1;
                if bo.zero_slacks == 2, bo.on = false;  bo.zero_slacks = 0; end
            end
        end

        function ocp = lagrangian_curvature(obj, ocp, Xb, Ub, ref, W, lam, su, hasD, d)
        % Hessian 'exact': adds to the QP Hessian the second-order terms Gauss-Newton leaves out, so
        % that it is the Hessian of the Lagrangian (QP units: objective J/2, the QP's multipliers)
        %   L = sum_a 1/2 e_a'W_a e_a + sum_i pi_{i+1}'(F(x_i, u_i) - x_{i+1}) + sum_a lg_a'g(x_a, u_a)
        % (lg = lam_ug - lam_lg: the rows' net multipliers): per stage, the central-difference
        % Jacobian of the stage gradient [A'pi + C'W e + Cg'lg; B'pi + D'W e + Dg'lg] in (x_i, u_i)
        % (node M: in x_M only), minus the Gauss-Newton part [C D]'W[C D] already in the QP. Not
        % convexified: the Riccati QP needs its reduced Hessians positive definite.
        % Difference steps: 1e-5 in the QP's own variables (states as they are, inputs in InputScale
        % units). eps^(1/3) ~ 6e-6 balances truncation and round-off for exact first derivatives;
        % 1e-5 also keeps an implicit integrator's Newton tolerance (1e-13) at <= 1e-8 in the result.
        % An absolute step is exact to round-off while eps |x| << 1e-5, i.e. |x| << 1e10.
        % Model.IntervalHessianFcn and Model.ConstraintLagrangianHessianFcn (e.g. CasADi, all nodes in one
        % call) replace the differences of the dynamics and of the ConstraintFcn rows.
            nx = obj.nx;  nu = obj.nu;  M = numel(d);
            ng = size(lam.lg, 1);
            hx = 1e-5;  hu = 1e-5 * su;
            mdl = obj.Model;
            Hdyn = [];   % the dynamics hook's Hessians, (nx+nu) x (nx+nu) x M
            if ~isempty(mdl.IntervalHessianFcn)
                Pi = lam.pi(:, 2:M+1);
                if isempty(obj.ParamNow), Hdyn = mdl.IntervalHessianFcn(Xb(:,1:M), Ub, d, Pi);
                else, Hdyn = mdl.IntervalHessianFcn(Xb(:,1:M), Ub, d, Pi, obj.ParamNow(:, min(1:M, size(obj.ParamNow, 2))));
                end
            end
            Hrow = [];  rp = [];   % the rows hook's, (nx+nu) x (nx+nu) x M-1 (inner nodes 1..M-1)
            pc = obj.PathNow;
            if ~isempty(mdl.ConstraintLagrangianHessianFcn) && ng > 0 && pc.ngp > 0 && M >= 2
                rp = numel(pc.yrows) + (1:pc.ngp);   % the ConstraintFcn rows (path_rows' order)
                Hrow = mdl.ConstraintLagrangianHessianFcn(Xb(:,2:M), Ub(:,2:M), lam.lg(rp, 2:M));
            end
            for i = 1:M+1
                term = i == M+1;
                x = Xb(:,i);  u = Ub(:, min(i, M));
                if term || ~isempty(Hdyn), p = []; else, p = lam.pi(:, i+1);  if ~any(p), p = []; end, end
                Wi = [];  if i >= 2, Wi = W(:,:,i-1);  if ~any(Wi(:)), Wi = []; end, end
                l = [];   if ng > 0, l = lam.lg(:, i);  if ~isempty(Hrow) && node_kind(i, M) == 1, l(rp) = 0; end, if ~any(l), l = []; end, end
                nz = nx + nu * ~term;   % node M: in x_M only (terminal output and rows)
                H = zeros(nz);   % the hooks' part
                if ~term && ~isempty(Hdyn), H = H + Hdyn(:,:,i); end
                if ~isempty(Hrow) && node_kind(i, M) == 1, H = H + Hrow(:,:,i-1); end
                if ~isempty(p) || ~isempty(Wi) || ~isempty(l)   % the rest by differences
                    Hd = zeros(nz);
                    for j = 1:nx
                        e = zeros(nx, 1);  e(j) = hx;
                        Hd(:,j) = (stage_gradient(x + e, u) - stage_gradient(x - e, u)) / (2 * hx);
                    end
                    for j = 1:nz-nx
                        e = zeros(nu, 1);  e(j) = hu(j);
                        Hd(:,nx+j) = (stage_gradient(x, u + e) - stage_gradient(x, u - e)) / (2 * hu(j));
                    end
                    H = H + Hd;
                elseif ~any(H(:))
                    continue
                end
                H = (H + H') / 2;
                if ~isempty(Wi)   % the Gauss-Newton part, already in the QP
                    [~, Cy, Dy] = obj.output(x, u, hasD, term);
                    Jy = Cy;  if nz > nx, Jy = [Cy, dy(Dy)]; end
                    H = H - Jy' * Wi * Jy;
                end
                ocp.Q(:,:,i) = ocp.Q(:,:,i) + H(1:nx, 1:nx);
                if ~term
                    ocp.S(:,:,i) = ocp.S(:,:,i) + su .* H(nx+1:end, 1:nx);
                    ocp.R(:,:,i) = ocp.R(:,:,i) + su .* H(nx+1:end, nx+1:end) .* su';
                end
            end

            function g = stage_gradient(x, u)
            % gradient of the stage's part of L in (x, u) (x only at node M), unscaled inputs
                g = zeros(nz, 1);
                if ~isempty(p)
                    [~, A, B] = obj.integrate(x, u, d(i), obj.node_param(i));
                    g = g + [A' * p; B' * p];
                end
                if ~isempty(Wi)   % (names distinct from the parent's: nested functions share its variables)
                    [yn, Cn, Dn] = obj.output(x, u, hasD, i == M+1);
                    wr = Wi * (yn - ref(:, i-1));
                    gc = Cn' * wr;  if nz > nx, gc = [gc; dy(Dn)' * wr]; end
                    g = g + gc;
                end
                if ~isempty(l)
                    [~, Cg, Dg] = obj.path_rows(x, u, hasD, node_kind(i, M));
                    gg = Cg' * l;  if nz > nx, gg = [gg; Dg' * l]; end
                    g = g + gg;
                end
            end
            function D = dy(D)   % output Jacobian in u (zero when the output has no D-term)
                if ~hasD || isempty(D), D = zeros(obj.ny, nu); end
            end
        end

        function ocp = ggn_curvature(obj, ocp, Xb, Ub, mu, su, M)
        % Gauss-Newton with row curvature (generalised Gauss-Newton; Messerer, Baumgaertner & Diehl,
        % "Survey of sequential convex programming and generalized Gauss-Newton methods", ESAIM
        % Proc. 2021): adds sum_k mu_k H_k at each node, H_k from the node's ConstraintHessianFcn
        % (inner nodes), InitialConstraintHessianFcn (node 0) or TerminalConstraintHessianFcn
        % (node M, x only), mu the rows' upper-bound multipliers (QP units, as the QP Hessian). For
        % g_k = phi_k(r(x, u)) with phi_k convex and H_k = Jr' phi_k'' Jr this is acados's term
        % for its convex-over-nonlinear constraints (ocp_nlp_constraints_bgp.c: the upper multiplier
        % only, so the term is positive semidefinite). Rows at their lower bound add nothing.
            pc = obj.PathNow;  nx = obj.nx;
            ny_ = numel(pc.yrows);
            rows = {ny_ + pc.ngp + pc.ngt + (1:pc.ng0), ny_ + (1:pc.ngp), ny_ + pc.ngp + (1:pc.ngt)};   % by node_kind 0, 1, 2
            fcn = {obj.Model.InitialConstraintHessianFcn, obj.Model.ConstraintHessianFcn, obj.Model.TerminalConstraintHessianFcn};
            for a = 1:M+1
                k = node_kind(a, M);  f = fcn{k+1};  r = rows{k+1};
                if isempty(f) || isempty(r), continue, end
                m = max(mu(r, a), 0);
                if ~any(m), continue, end
                x = Xb(:, a);
                if k == 2, Hs = f(x); else, Hs = f(x, Ub(:, min(a, M))); end
                nz = nx + obj.nu * (k ~= 2);
                if ~isequal(size(Hs, [1 2 3]), [nz, nz, numel(r)])
                    error('rtimpc:ConstraintHessianFcn', 'Constraint Hessians must be %d x %d x %d at node %d.', nz, nz, numel(r), a - 1)
                end
                H = reshape(reshape(Hs, nz * nz, []) * m, nz, nz);  H = (H + H') / 2;
                ocp.Q(:,:,a) = ocp.Q(:,:,a) + H(1:nx, 1:nx);
                if k ~= 2
                    ocp.S(:,:,a) = ocp.S(:,:,a) + su .* H(nx+1:end, 1:nx);
                    ocp.R(:,:,a) = ocp.R(:,:,a) + su .* H(nx+1:end, nx+1:end) .* su';
                end
            end
        end

        function [P, ok] = terminal_lqr(obj, xN, uN)
        % Infinite-horizon LQR cost-to-go at (xN, uN): DARE on the one-sample discrete linearisation
        % with the stage weights C'Qy C (output Jacobian at xN) and R, in cost (J) units
            [~, A, B] = obj.integrate(xN, uN, 1, obj.node_param(Inf));
            [~, C] = obj.output(xN, uN, false);
            Q = C' * obj.Weights.Qy * C;
            [P, ok] = rtimpc.dare(A, B, (Q + Q')/2, obj.Weights.R);
        end

        function [alpha, nu_merit, ls, sol] = merit_line_search(obj, Xb, Ub, sol, ocp, x, ref, uref, W, su, J0, nu_merit, qp_solved, accuracy_test)
        % Armijo backtracking on the l1 merit  phi = J + sum w .* |gaps| + w0 .* |x_0 - x| + hard state
        % bound, row and input-rate-limit violations weighted likewise; soft-bound violations are
        % already priced into J.
        % MeritWeights 'elementwise' (Leineweber 1999, M5.1; acados's MERIT_BACKTRACKING): each gap
        % and bound has its own weight, from its multiplier: w = 2.2|lam| on the first SQP iteration
        % (2: the QP objective is J/2; 1.1: a margin), then w = max(2.2|lam|, (2.2|lam| + w)/2). Each
        % residual is priced in its own units, so a state measured in large units (a temperature in
        % K) does not dominate the merit (the default). 'component': one weight per state component,
        % 2.2 max|lam| over the nodes (non-decreasing within a step). 'scalar': one weight, 2.2 max|lam|,
        % for every residual.
        % (NMPC suite: scalar stalls on the CSTR's cold start; elementwise and component give the same
        % closed loops, elementwise with as many or fewer SQP iterations.)
        % SecondOrderCorrection: if the full step is rejected, re-solve the QP with the gaps the full
        % step actually opens (minus its linear prediction), which removes their second-order part,
        % and try the corrected full step before backtracking on the original one. Returns the
        % step taken in sol (the corrected one if it was accepted).
        % accuracy_test (QPAccuracyTest): before backtracking, the step's reduction of the merit's
        % linear model, dl = -D - (the weighted residuals of the linearised dynamics and x_0 that the
        % inexact QP leaves), must reach its curvature T (step_curvature): the step's QP objective no
        % worse than the zero step's, Termination Test 1 of Curtis, Johnson, Robinson & Waechter
        % (2014), whose (2.17) it implies. An exact QP solution meets it with a factor 2 (dl >= 2T, as
        % the weights exceed the multipliers). If not, ls.short is set and nothing else is done: the
        % caller re-solves the QP more accurately.
            op = obj.Optimisation;
            if nargin < 14, accuracy_test = op.QPAccuracyTest; end
            M = size(Xb, 2) - 1;
            % merit weights (QP objective is J/2): gaps (nodes 2..M+1), x_0, lower/upper hard bounds
            % x_0 = x: the QP solvers eliminate node 0, so sol.pi(:,1) is not its multiplier; take it
            % from node 0's stationarity, lam_0 = Q_0 x_0 + q_0 + S_0'u_0 + A_0'pi_1 + Cg_0'(lam_ug - lam_lg)
            % (Q_0, q_0: costs at node 0, i.e. StageCostFcn and 'integrator' cost points; else zero)
            lam_x0 = ocp.Q(:,:,1) * sol.x(:, 1) + ocp.q(:, 1) + ocp.A(:,:,1)' * sol.pi(:, 2) + ocp.S(:,:,1)' * sol.u(:, 1);
            pc = obj.PathNow;  ng = 0;  if ~isempty(pc) && pc.on, ng = pc.ng; end
            if ng > 0, lam_x0 = lam_x0 + ocp.Cg(:,:,1)' * (sol.lam_ug(:, 1) - sol.lam_lg(:, 1)); end
            lam = struct('g', abs(sol.pi(:, 2:end)), 'x0', abs(lam_x0), 'l', zeros(obj.nx, M), 'u', zeros(obj.nx, M), ...
                         'gl', zeros(ng, M+1), 'gu', zeros(ng, M+1));
            if isfield(sol, 'lam_lbx'), lam.l = sol.lam_lbx(:, 2:end);  lam.u = sol.lam_ubx(:, 2:end); end
            if ng > 0, lam.gl = sol.lam_lg;  lam.gu = sol.lam_ug; end   % rows at nodes 0..M
            % hard input-rate limits: with the rate augmentation on, the QP's input box is the box on
            % v_i = du_i - du_{i-1} (rate_augment), its multipliers per scaled input: / su per unit change
            rt = obj.RateNow;  rate_on = ~isempty(rt) && rt.on;
            lam.rl = zeros(obj.nu, M);  lam.ru = zeros(obj.nu, M);
            if rate_on && isfield(sol, 'lam_lb'), lam.rl = sol.lam_lb ./ su;  lam.ru = sol.lam_ub ./ su; end
            lam = structfun(@(v) 2 * op.MeritWeightMargin * v, lam, 'UniformOutput', false);
            fn = fieldnames(lam)';
            if ~qp_solved && isstruct(nu_merit)
                % A QP stopped at its iteration cap (or stalled) has unconverged multipliers, which can
                % be huge (1e60 seen on a diverging model): keep this step's earlier weights
            elseif strcmpi(op.MeritWeights, 'component')     % per component (state or row), uniform over the nodes
                cm = @(v) max([v, zeros(size(v, 1), 1)], [], 2) .* ones(1, size(v, 2));
                gx0 = cm([lam.g, lam.x0]);
                lam.x0 = gx0(:, end);  lam.g = gx0(:, 1:end-1);
                for f = setdiff(fn, {'g', 'x0'}), lam.(f{1}) = cm(lam.(f{1})); end
                if isstruct(nu_merit), for f = fn, lam.(f{1}) = max(lam.(f{1}), nu_merit.(f{1})); end, end
                nu_merit = lam;
            elseif strcmpi(op.MeritWeights, 'scalar')
                top = max(structfun(@(v) max([v(:); 0]), lam));
                if isstruct(nu_merit), top = max(top, nu_merit.g(1)); else, top = max(top, nu_merit); end
                nu_merit = structfun(@(v) top * ones(size(v)), lam, 'UniformOutput', false);
            elseif ~isstruct(nu_merit)                   % first SQP iteration of this step
                nu_merit = lam;
            else
                for f = fn, nu_merit.(f{1}) = max(lam.(f{1}), (lam.(f{1}) + nu_merit.(f{1})) / 2); end
            end
            w = nu_merit;
            [xmin, xmax] = hard_state_bounds(obj, M);
            if ng > 0, hard = ~isfinite(pc.L1); else, hard = false(0, M+1); end   % hard path rows (soft ones are priced in J)
            [rlv, ruv] = deal(@(U) 0);   % hard rate-limit violations (lower, upper), nu x M
            if rate_on
                dUf = @(U) [U(:,1) - rt.u_prev, diff(U, 1, 2)];
                rlv = @(U) max(rt.min - dUf(U), 0);  ruv = @(U) max(dUf(U) - rt.max, 0);
            end
            weighted = @(X, gaps, G, U) sum(w.g .* abs(gaps), 'all') + sum(w.x0 .* abs(X(:,1) - x)) + ...
                sum(w.l .* max(xmin - X(:, 2:end), 0), 'all') + sum(w.u .* max(X(:, 2:end) - xmax, 0), 'all') + ...
                sum(w.gl .* max(pc_min(pc, ng) - G, 0) .* hard, 'all') + sum(w.gu .* max(G - pc_max(pc, ng), 0) .* hard, 'all') + ...
                sum(w.rl .* rlv(U), 'all') + sum(w.ru .* ruv(U), 'all');
            rate_viol = @(U) sum(rlv(U), 'all') + sum(ruv(U), 'all');
            G0 = obj.path_values(Xb, Ub);
            viol0 = sum(abs(ocp.c), 'all') + sum(abs(ocp.x0)) + rtimpc.bound_violation(Xb, xmin, xmax) + rate_viol(Ub);
            if ng > 0
                viol0 = viol0 + sum((max(pc.gmin - G0, 0) + max(G0 - pc.gmax, 0)) .* hard, 'all');
            end
            pen0  = weighted(Xb, ocp.c, G0, Ub);
            phi0  = J0 + pen0;
            dJ = 2 * (sum(ocp.q .* sol.x, 'all') + sum(ocp.r .* sol.u, 'all'));   % node 0: sol.x(:,1) = x - x_0 (costs at node 0: StageCostFcn, 'integrator' points)
            if isfield(ocp, 'rN'), dJ = dJ + 2 * (ocp.rN' * sol.u(:,M)); end   % terminal cost in u_{M-1}
            dJ = dJ + obj.soft_penalty_slope(Xb, sol.x);
            if ng > 0   % soft path rows: slope of their penalty along the linearised rows
                dG = zeros(ng, M+1);
                for a = 1:M+1
                    dG(:,a) = ocp.Cg(:,:,a) * sol.x(:,a) + ocp.Dg(:,:,a) * sol.u(:, min(a, M));
                end
                dJ = dJ + rtimpc.soft_slope(G0, dG, pc.gmin, pc.gmax, pc.L1, pc.L2);
                % (BarrierHomotopy: the barrier's slope is in ocp.q, ocp.r and, node M's rows' part in
                % u_{M-1}, ocp.rN: all above)
            end
            if ~isempty(obj.RateNow) && obj.RateNow.on
                dJ = dJ + rtimpc.rate_slope(Ub, sol.u .* su, obj.RateNow);
            end
            D  = dJ - pen0;                      % directional derivative of phi along the step
            [umin, umax] = input_bounds(obj);
            ls = struct('nu', max(structfun(@(v) max([v(:); 0]), w)), 'viol0', viol0, 'D', D, 'n', 0, 'soc', 0, ...
                        'dl', NaN, 'T', NaN, 'short', false, 'soc_iter', 0);
            d = obj.GridNow.d;
            if accuracy_test && qp_solved
                lin = zeros(size(ocp.c));   % the linearised dynamics' residuals at the QP's solution
                for k = 1:M
                    lin(:,k) = ocp.A(:,:,k) * sol.x(:,k) + ocp.B(:,:,k) * sol.u(:,k) + ocp.c(:,k) - sol.x(:,k+1);
                end
                ls.dl = -D - sum(w.g .* abs(lin), 'all') - sum(w.x0 .* abs(sol.x(:,1) - ocp.x0));
                ls.T = obj.step_curvature(ocp, sol, su);
                if ls.dl < ls.T
                    ls.short = true;  alpha = NaN;
                    return
                end
            end

            function [alpha, ls] = funnel_search(ls)
            % Globalisation 'funnel': acados's FUNNEL_L1PEN_LINESEARCH (ocp_nlp_globalization_funnel.c,
            % e35cc32d9; Kiessling, Leyffer & Vanaret 2025), its defaults: funnel
            % width tau = max(1, 15 h_0) at the step's first line search; f-type steps (switching
            % condition alpha pred_f >= 1e-3 pred_h, Armijo on J) leave tau; h-type steps (h <= 0.9 tau)
            % shrink it, tau = 0.1 h + 0.9 tau; else the merit mu J + h, entering penalty mode, which a
            % step with h <= 0.9 tau leaves. h: the l1 infeasibility (viol0). Backtracking with
            % LineSearchFactor; at LineSearchMinStep the step is taken, as with the merit. The Armijo
            % factor is acados's eps_sufficient_descent, 1e-4, which its funnel always applies (its
            % merit line search applies none by default, as LineSearchArmijo's 0).
                fs = obj.FunnelNow;
                if isempty(fs), fs = struct('tau', max(1, 15 * viol0), 'mu', 1, 'pmode', false); end
                pred_f = -dJ;  pred_h = viol0;
                if pred_f < 0 && pred_f > -1e-4, pred_f = 0; end   % acados: predicted objective noise
                if fs.mu * pred_f + pred_h < 1e-6 * pred_h
                    fs.mu = max(0, min(0.5 * fs.mu, (1 - 1e-6) * pred_h / (-pred_f + 1e-9)));
                end
                m0 = fs.mu * J0 + viol0;  pred_m = fs.mu * pred_f + pred_h;
                arm = @(actual, pred, a) actual >= 1e-4 * a * max(0, pred - 1e-9);
                alpha = 1;  ls.type = '';
                for n = 1:op.LineSearchMaxSteps
                    ls.n = n;
                    [~, Ja, ~, ~, h] = merit(sol, alpha);
                    if isfinite(Ja) && h <= fs.tau
                        mt = fs.mu * Ja + h;
                        if ~fs.pmode
                            if alpha * pred_f >= 1e-3 * pred_h
                                if arm(J0 - Ja, pred_f, alpha), ls.type = 'f'; end
                            elseif h <= 0.9 * fs.tau
                                ls.type = 'h';  fs.tau = 0.1 * h + 0.9 * fs.tau;
                            elseif h < viol0 && arm(m0 - mt, pred_m, alpha)
                                ls.type = 'b';  fs.pmode = true;
                            end
                        elseif arm(m0 - mt, pred_m, alpha)
                            ls.type = 'p';
                            if h <= 0.9 * fs.tau, fs.tau = 0.1 * h + 0.9 * fs.tau;  fs.pmode = false; end
                        end
                    end
                    if strcmpi(op.Display, 'iter') && n <= 3
                        fprintf('        funnel alpha %8.2e: J %11.5g  h %10.3e  tau %10.3e  mu %8.2e  %s\n', alpha, Ja, h, fs.tau, fs.mu, ls.type)
                    end
                    if ~isempty(ls.type), break, end
                    if op.LineSearchMinStep > 0 && op.LineSearchFactor * alpha < op.LineSearchMinStep
                        ls.type = 'm';  break   % the minimum step, taken (acados: MINSTEP)
                    end
                    alpha = op.LineSearchFactor * alpha;
                end
                if isempty(ls.type), alpha = 0; end   % budget spent, no step accepted: reject (as the merit search), not the untested shorter one
                obj.FunnelNow = fs;
                ls.nu = fs.mu;
            end

            function [phi, Ja, pen, gaps, h] = merit(step, alpha)
                Xa = Xb + alpha * step.x;
                Ua = min(max(Ub + alpha * (step.u .* su), umin), umax);
                gaps = obj.interval_ends(Xa(:,1:end-1), Ua, d) - Xa(:,2:end);
                Ga = obj.path_values(Xa, Ua);
                pen = weighted(Xa, gaps, Ga, Ua);
                Ja = obj.nlp_cost(Xa, Ua, ref, uref, W);
                phi = Ja + pen;
                if nargout > 4   % the l1 infeasibility, unweighted (as viol0)
                    h = sum(abs(gaps), 'all') + sum(abs(Xa(:,1) - x)) + rtimpc.bound_violation(Xa, xmin, xmax) + rate_viol(Ua);
                    if ng > 0, h = h + sum((max(pc.gmin - Ga, 0) + max(Ga - pc.gmax, 0)) .* hard, 'all'); end
                end
            end
            function ok = armijo(phi, alpha, tag)
                if strcmpi(op.Display, 'iter') && ls.n <= 3
                    fprintf('        ls %s alpha %8.2e: J %11.5g  penalty %11.5g  phi-phi0 %+11.4g  (Armijo needs <= %+.4g)\n', ...
                        tag, alpha, Ja, pen, phi - phi0, op.LineSearchArmijo * alpha * min(D, 0))
                end
                % Armijo, with a round-off allowance (IPOPT's Compare_le): near convergence the decrease is below
                % the resolution of phi itself
                if op.LineSearchRoundoff > 0
                    ok = isfinite(phi) && phi <= phi0 + op.LineSearchArmijo * alpha * min(D, 0) + op.LineSearchRoundoff * eps(phi0);
                else   % acados's test: merit1 < merit0 + eps_sufficient_descent * slope * alpha (strict)
                    ok = isfinite(phi) && phi < phi0 + op.LineSearchArmijo * alpha * min(D, 0);
                end
            end

            if strcmpi(op.Globalisation, 'funnel')
                [alpha, ls] = funnel_search(ls);
                return
            end

            alpha = 1;
            for n_ls = 1:op.LineSearchMaxSteps
                ls.n = n_ls;
                [phi, Ja, pen, gaps] = merit(sol, alpha);
                if armijo(phi, alpha, '   '), return, end
                if n_ls == 1 && op.SecondOrderCorrection
                    % gaps of the full step minus what the linearisation predicted for them
                    lin = zeros(size(ocp.c));
                    for k = 1:numel(d)
                        lin(:,k) = ocp.A(:,:,k) * sol.x(:,k) + ocp.B(:,:,k) * sol.u(:,k) - sol.x(:,k+1);
                    end
                    soc_ocp = ocp;  soc_ocp.c = gaps - lin;
                    ls.soc = -1;                 % attempted
                    try
                        [sol2, q2] = obj.solve_qp(soc_ocp, Ub, obj.RateNow, su, namedargs2cell(op.QPSolverOptions));
                        ls.soc_iter = q2.iter;   % its work, counted by the caller
                        if strcmp(q2.status, 'solved')   % a correction from a solved QP only (an unsolved one's
                            % step and multipliers are not estimates: backtrack on the original instead)
                            [phi, Ja, pen] = merit(sol2, 1);
                            if armijo(phi, 1, 'SOC')
                                sol = sol2;  ls.soc = 1;  ls.soc_qinfo = q2;   % accepted: the caller takes its status
                                return
                            end
                        end
                    catch err   % SOC QP failed: fall back to backtracking on the original step
                        if ~qp_failure(err), rethrow(err), end
                    end
                end
                if op.LineSearchMinStep > 0 && op.LineSearchFactor * alpha < op.LineSearchMinStep
                    % take the smallest step tried (acados's MINSTEP), unless the step is already
                    % stationary by the caller's test: there no alpha passes only by round-off, and
                    % alpha = 0 tells the caller to keep the plan as converged
                    stationary = abs(D) <= op.SQPTolCost * max(1, abs(J0)) && viol0 <= op.SQPTol * obj.nx * M;
                    if ~stationary, return, end
                    ls.alpha_last = alpha;   % 'kkt': the caller takes it if the KKT test fails (acados)
                    break
                end
                alpha = op.LineSearchFactor * alpha;
            end
            alpha = 0;   % no acceptable step: reject it (caller stops iterating)
        end

        function T = step_curvature(obj, ocp, sol, su)
        % J's second-order term along the QP step (d'H d with H the QP's Hessian, the QP objective
        % being J/2), for QPAccuracyTest: the stage blocks [Q S'; S R] (node 0's Q unused: x_0 is
        % fixed), node M's Q and terminal input terms, and the input-rate terms. Left out: the soft
        % slacks' L2 curvature, which can only make T smaller, so the test easier to pass.
            M = size(ocp.R, 3);  X = sol.x;  U = sol.u;
            T = 0;
            for k = 1:M
                if k > 1, T = T + X(:,k)' * ocp.Q(:,:,k) * X(:,k); end
                T = T + 2 * U(:,k)' * ocp.S(:,:,k) * X(:,k) + U(:,k)' * ocp.R(:,:,k) * U(:,k);
            end
            T = T + X(:,M+1)' * ocp.Q(:,:,M+1) * X(:,M+1);
            if isfield(ocp, 'SN') && ~isempty(ocp.SN)
                T = T + 2 * U(:,M)' * ocp.SN * X(:,M+1) + U(:,M)' * ocp.RN * U(:,M);
            end
            rate = obj.RateNow;
            if ~isempty(rate) && rate.on   % sum dE' RateR dE, dE the step's input changes (unscaled)
                dU = U .* su;  dE = [dU(:,1), diff(dU, 1, 2)];
                T = T + sum(dE .* rate_times(rate.R, dE), 'all');
            end
        end

        function dP = soft_penalty_slope(obj, X, dX)
        % One-sided directional derivative of the soft state-bound penalty in J along dX
        % (piecewise linear in the l1 part: a node on a bound counts if the step leaves the box)
            [xmin, xmax, L1, L2] = state_bounds(obj, size(X, 2) - 1);
            soft = isfinite(L1) & (isfinite(xmin) | isfinite(xmax));     % nx x M
            if ~any(soft, 'all'), dP = 0; return, end
            X = X(:, 2:end);  dX = dX(:, 2:end);
            L1(~soft) = 0;  L2(~soft) = 0;
            vl = max(xmin - X, 0);  vu = max(X - xmax, 0);
            tol = 1e-12 * max(1, abs(X));
            gl = (vl > tol | (abs(xmin - X) <= tol & dX < 0)) .* -dX .* soft;    % d(lower violation)
            gu = (vu > tol | (abs(X - xmax) <= tol & dX > 0)) .*  dX .* soft;    % d(upper violation)
            dP = sum(L1 .* (gl + gu) + 2 * L2 .* (vl .* gl + vu .* gu), 'all');
        end

        function J = nlp_cost(obj, X, U, ref, uref, W)
        % NLP objective of a plan on the step's shooting grid (node-level ref, uref, W)
            G  = obj.GridNow;  d = G.d;  M = numel(d);
            dU = U - uref;
            if all(d == 1), J = sum(dU .* (obj.Weights.R * dU), 'all');
            else,           J = sum(d .* sum(dU .* (obj.Weights.R * dU), 1));
            end
            if isfield(G, 'ulin') && ~isempty(G.ulin), J = J + sum(G.ulin .* U, 'all'); end
            for i = 2:M+1
                Wi = W(:,:,i-1);
                if ~any(Wi(:)), continue, end
                e = obj.output(X(:,i), U(:, min(i, M)), false, i == M+1) - ref(:,i-1);
                J = J + e' * Wi * e;
            end
            for i = 1:numel(G.pts)   % 'integrator': cost points inside coarse intervals
                P = G.pts{i};
                if isempty(P), continue, end
                [~, ~, ~, Xs] = obj.integrate(X(:,i), U(:,i), d(i), obj.node_param(i));
                for p = 1:numel(P)
                    e = obj.output(Xs(:,P(p).step), U(:,i)) - P(p).ref;
                    J = J + e' * P(p).W * e;
                end
            end
            if ~isempty(obj.Weights.StageCostFcn)
                if obj.Weights.StageCostVectorised
                    J = J + sum(d .* reshape(obj.Weights.StageCostFcn(X(:,1:M), U(:,1:M), ref(:,1:M)), 1, M));
                else
                    for i = 1:M, J = J + d(i) * obj.Weights.StageCostFcn(X(:,i), U(:,i), ref(:,i)); end
                end
            end
            if ~isempty(obj.Weights.TerminalCostFcn)
                J = J + obj.Weights.TerminalCostFcn(X(:,M+1), U(:,M), ref(:,M));
            end
            if ~isempty(obj.TailNow)
                eN = X(:,M+1) - obj.TailNow.xbar;
                J = J + eN' * obj.TailNow.P * eN;
            end
            J = J + obj.soft_penalty(X) + obj.path_penalty(X, U) + obj.barrier_cost(X, U);
            if ~isempty(obj.RateNow) && obj.RateNow.on, J = J + rtimpc.rate_cost(U, obj.RateNow); end
        end

        function B = barrier_cost(obj, X, U)
        % Optimisation.BarrierHomotopy: BarrierNow x the relaxed log barriers of every finite row bound (0 when off)
            B = 0;  pc = obj.PathNow;
            if obj.BarrierNow == 0 || isempty(pc) || ~pc.on, return, end
            B = sum(obj.barrier_terms(obj.path_values(X, U), pc.gmin, pc.gmax), 'all');
        end

        function [b, g1, g2] = barrier_terms(obj, G, gmin, gmax)
        % BarrierNow x relaxed log barrier of the rows' bound distances, per row (b), and its first and
        % second derivatives in the row value (g1, g2): beta(z) = -log z for z > delta, else
        % ((z - 2 delta)^2 / delta^2 - 1) / 2 - log delta (C2-matched; Feller & Ebenbauer 2017)
            mu = obj.BarrierNow;  dl = obj.Optimisation.BarrierRelax;
            [b, g1, g2] = deal(zeros(size(G)));
            for side = [1, -1]
                if side == 1, lim = gmin; else, lim = gmax; end
                on = isfinite(lim) & true(size(G));
                z = side * (G - lim);  z(~on) = 1;
                hi = z > dl;
                v = -log(max(z, dl));  v(~hi) = ((z(~hi) - 2*dl).^2 / dl^2 - 1) / 2 - log(dl);
                d1 = -1 ./ max(z, dl);  d1(~hi) = (z(~hi) - 2*dl) / dl^2;
                d2 = 1 ./ max(z, dl).^2;  d2(~hi) = 1 / dl^2;
                b = b + mu * v .* on;  g1 = g1 + side * mu * d1 .* on;  g2 = g2 + mu * d2 .* on;
            end
        end

        function pc = path_setup(obj, x, u, hasD, d)
        % Constraint rows of this step, stacked: output bounds (rows of y with a finite bound; nodes
        % 1..M), path constraints (ConstraintFcn; nodes 1..M-1), terminal constraints
        % (TerminalConstraintFcn; node M), initial constraints (InitialConstraintFcn; node 0).
        % Bounds per node (ng x M+1, column a = node a-1; -+Inf where a row does not apply) and
        % node-level soft weights: per-sample weights x interval length for the output and path rows
        % (as the state bounds) and for the initial rows (node 0's input is held over interval 1),
        % as given for the terminal rows. Stored in PathNow.
            ny = obj.ny;  M = numel(d);  Cn = obj.Constraints;
            ymin = Cn.OutputMin(:);  ymax = Cn.OutputMax(:);
            if isempty(ymin), ymin = -inf(ny, 1); end
            if isempty(ymax), ymax =  inf(ny, 1); end
            ymin = ymin .* ones(ny, 1);  ymax = ymax .* ones(ny, 1);
            yrows = reshape(find(isfinite(ymin) | isfinite(ymax)), [], 1);   % a column also when ny = 1
            ngp = 0;  if ~isempty(obj.Model.ConstraintFcn), ngp = numel(obj.Model.ConstraintFcn(x, u)); end
            ngt = 0;  if ~isempty(obj.Model.TerminalConstraintFcn), ngt = numel(obj.Model.TerminalConstraintFcn(x)); end
            ng0 = 0;  if ~isempty(obj.Model.InitialConstraintFcn), ng0 = numel(obj.Model.InitialConstraintFcn(x, u)); end
            bnd = @(v, n, s) ternary_num(isempty(v), s * inf(n, 1), v(:)) .* ones(n, 1);
            w = @(v, def, n) ternary_num(isempty(v), def, v(:)) .* ones(n, 1);
            ny_ = numel(yrows);  ng = ny_ + ngp + ngt + ng0;
            ip = ny_ + (1:ngp);  it = ny_ + ngp + (1:ngt);  i0 = ny_ + ngp + ngt + (1:ng0);
            gmin = -inf(ng, M+1);  gmax = inf(ng, M+1);
            gmin(1:ny_, 2:M+1) = repmat(ymin(yrows), 1, M);  gmax(1:ny_, 2:M+1) = repmat(ymax(yrows), 1, M);
            gmin(ip, 2:M) = repmat(bnd(Cn.ConstraintMin, ngp, -1), 1, M-1);
            gmax(ip, 2:M) = repmat(bnd(Cn.ConstraintMax, ngp,  1), 1, M-1);
            gmin(it, M+1) = bnd(Cn.TerminalConstraintMin, ngt, -1);
            gmax(it, M+1) = bnd(Cn.TerminalConstraintMax, ngt,  1);
            gmin(i0, 1) = bnd(Cn.InitialConstraintMin, ng0, -1);
            gmax(i0, 1) = bnd(Cn.InitialConstraintMax, ng0,  1);
            yL1 = w(Cn.OutputSoftL1, 1e4, ny);  yL2 = w(Cn.OutputSoftL2, 0, ny);
            yL1 = reshape(yL1(yrows), [], 1);  yL2 = reshape(yL2(yrows), [], 1);   % the bounded rows (columns, also when ny = 1)
            dn = [0, d];  e0 = [d(1), zeros(1, M)];  eN = [zeros(1, M), 1];       % per node 0..M
            L1 = [yL1 .* dn; w(Cn.ConstraintSoftL1, 1e4, ngp) .* dn; w(Cn.TerminalConstraintSoftL1, 1e4, ngt) .* eN; ...
                  w(Cn.InitialConstraintSoftL1, 1e4, ng0) .* e0];
            L2 = [yL2 .* dn; w(Cn.ConstraintSoftL2, 0, ngp) .* dn;   w(Cn.TerminalConstraintSoftL2, 0, ngt) .* eN; ...
                  w(Cn.InitialConstraintSoftL2, 0, ng0) .* e0];
            hard = [~isfinite(yL1); ~isfinite(w(Cn.ConstraintSoftL1, 1e4, ngp)); ~isfinite(w(Cn.TerminalConstraintSoftL1, 1e4, ngt)); ...
                    ~isfinite(w(Cn.InitialConstraintSoftL1, 1e4, ng0))];
            L1(hard, :) = Inf;  L1(isnan(L1)) = Inf;   % Inf x 0 (a row at a node where it does not apply) is unused
            L2(isnan(L2)) = 0;
            pc = struct('on', ng > 0, 'ng', ng, 'yrows', yrows, 'ngp', ngp, 'ngt', ngt, 'ng0', ng0, 'hasD', hasD, ...
                        'gmin', gmin, 'gmax', gmax, 'L1', L1, 'L2', L2, 'M', M);
            obj.PathNow = pc;
        end

        function [g, C, D] = path_rows(obj, x, u, hasD, kind)
        % Constraint-row values and Jacobians at (x, u) of a node of the given kind (node_kind: 0
        % initial, 1 inner, 2 terminal): output rows (zero at node 0), then path rows (inner nodes),
        % terminal rows (node M) and initial rows (node 0), each zero elsewhere
            pc = obj.PathNow;  nx = obj.nx;  nu = obj.nu;
            g = zeros(0, 1);  C = zeros(0, nx);  D = zeros(0, nu);
            jac = nargout > 1;
            if ~isempty(pc.yrows)
                nyr = numel(pc.yrows);
                if kind == 0
                    y = zeros(nyr, 1);  Cy = zeros(nyr, nx);  Dy = zeros(nyr, nu);
                elseif jac
                    [y, Cy, Dy] = obj.output(x, u, hasD, kind == 2);
                    y = y(pc.yrows);  Cy = Cy(pc.yrows, :);  Dy = Dy(pc.yrows, :);
                else
                    y = obj.output(x, u, false, kind == 2);  y = y(pc.yrows);
                end
                g = y;  if jac, C = Cy;  D = Dy; end
            end
            if pc.ngp > 0
                if kind == 2
                    gp = zeros(pc.ngp, 1);  Cp = zeros(pc.ngp, nx);  Dp = zeros(pc.ngp, nu);
                else
                    gp = obj.Model.ConstraintFcn(x, u);
                    if jac, [Cp, Dp] = constraint_jacobian(obj.Model.ConstraintFcn, obj.Jacobian.ConstraintFcn, x, u); end
                end
                g = [g; gp(:)];
                if jac, C = [C; Cp];  D = [D; Dp]; end
            end
            if pc.ngt > 0
                if kind == 2
                    gt = obj.Model.TerminalConstraintFcn(x);
                    if jac, Ct = constraint_jacobian(@(z, ~) obj.Model.TerminalConstraintFcn(z), obj.Jacobian.TerminalConstraintFcn, x, zeros(0, 1)); end
                else
                    gt = zeros(pc.ngt, 1);  Ct = zeros(pc.ngt, nx);
                end
                g = [g; gt(:)];
                if jac, C = [C; Ct];  D = [D; zeros(pc.ngt, nu)]; end
            end
            if pc.ng0 > 0
                if kind == 0
                    g0 = obj.Model.InitialConstraintFcn(x, u);
                    if jac, [C0, D0] = constraint_jacobian(obj.Model.InitialConstraintFcn, obj.Jacobian.InitialConstraintFcn, x, u); end
                else
                    g0 = zeros(pc.ng0, 1);  C0 = zeros(pc.ng0, nx);  D0 = zeros(pc.ng0, nu);
                end
                g = [g; g0(:)];
                if jac, C = [C; C0];  D = [D; D0]; end
            end
        end

        function free = node0_input_free(obj, free, x, u, hasD, su)
        % Node-0 rows (logical, ng x 1) whose input Jacobian is zero at u stay marked free only if it is
        % also zero at two probe inputs u + su .* r (x fixed: x_0 is the measurement). A row that
        % depends on u_0 vanishes there only on a measure-zero set, so a generic probe finds it. The
        % directions r are fixed (reproducible, no rng state), on the scale of InputScale (su: the
        % QP's input scaling, the inputs' declared or bound-derived typical size). A non-finite
        % Jacobian at a probe (outside the function's domain) is no evidence either way: ignored.
            nu = obj.nu;  j = (1:nu)';
            for r = [cos(1.1 * j + 0.3), sin(2.3 * j + 0.7)]
                [~, ~, D] = obj.path_rows(x, u + su(:) .* r, hasD, 0);
                dep = any(D ~= 0 & isfinite(D), 2);
                free = free & ~dep(:);
                if ~any(free), return, end
            end
        end

        function G = path_values(obj, X, U)
        % Constraint-row values of a plan at nodes 0..M (ng x M+1; node M: terminal output rows)
            pc = obj.PathNow;
            if isempty(pc) || ~pc.on, G = zeros(0, size(U, 2) + 1); return, end
            M = size(U, 2);  G = zeros(pc.ng, M+1);
            for a = 1:M+1, G(:,a) = obj.path_rows(X(:,a), U(:, min(a, M)), false, node_kind(a, M)); end
        end

        function P = path_penalty(obj, X, U)
        % Soft path-constraint penalty in J (at the minimal slacks: the violations)
            pc = obj.PathNow;  P = 0;
            if isempty(pc) || ~pc.on || ~any(isfinite(pc.L1), 'all'), return, end
            G = obj.path_values(X, U);
            soft = isfinite(pc.L1);
            v = (max(pc.gmin - G, 0) + max(G - pc.gmax, 0)) .* soft;
            L1 = pc.L1;  L1(~soft) = 0;  L2 = pc.L2;  L2(~soft) = 0;
            P = sum(L1 .* v + L2 .* v.^2, 'all');
        end

        function rate = rate_setup(obj, u_prev, umin, umax, tau, rref)
        % Input-rate terms for this step: u_{-1}, weight per node-level change (nu x nu x M: change i
        % of the grid happens at fine sample tau(i), so it takes RateR's page tau(i)+1), and limits. On a
        % decimated grid both are per sample of the fine grid: the change into an interval of d samples
        % stands for d sample changes (the fine grid's input, which the coarse part approximates), so its
        % weight is RateR / d and its limits d x RateMin, d x RateMax (a ramp costs and is allowed the same
        % on any grid). Uniform grids: d = 1, unchanged.
            nu = obj.nu;  M = numel(tau) - 1;
            Rr = obj.Weights.RateR;
            if size(Rr, 3) > 1
                if size(Rr, 3) ~= obj.PredictionHorizon
                    error('rtimpc:RateR', 'Weights.RateR must be nu x nu or nu x nu x N (N = %d).', obj.PredictionHorizon)
                end
                Rr = Rr(:, :, tau(1:M) + 1);
            end
            rmin = obj.Constraints.RateMin(:); if isempty(rmin), rmin = -inf(nu, 1); end
            rmax = obj.Constraints.RateMax(:); if isempty(rmax), rmax =  inf(nu, 1); end
            rmin = rmin .* ones(nu, 1);  rmax = rmax .* ones(nu, 1);
            if isempty(Rr), Rr = zeros(nu); end
            Rr = Rr .* ones(1, 1, M);
            rate.on = any(Rr(:)) || any(isfinite([rmin; rmax]));
            d = diff(tau(1:M+1));                               % samples per shooting interval
            Rr = Rr ./ reshape(d, 1, 1, M);  rmin = rmin .* d;  rmax = rmax .* d;   % per sample of the fine grid (see above)
            if isempty(u_prev), u_prev = obj.Uprev; end
            if isempty(u_prev), u_prev = min(max(zeros(nu, 1), umin), umax); end
            rate.u_prev = u_prev(:);
            rate.R = Rr;  rate.min = rmin;  rate.max = rmax;
            % reference changes per node-level change: change i (into node i at fine sample tau(i))
            % gets the sum of the fine references over its samples tau(i-1)+1..tau(i) (change 0: r_0)
            rate.ref = zeros(nu, M);
            if nargin > 5 && ~isempty(rref)
                if size(rref, 1) ~= nu || size(rref, 2) < tau(M) + 1
                    error('rtimpc:RateReference', 'RateReference must be nu x N (nu = %d, N = %d).', nu, obj.PredictionHorizon)
                end
                Cr = cumsum(rref, 2);  rate.ref = Cr(:, tau(1:M) + 1) - [zeros(nu, 1), Cr(:, tau(1:M-1) + 1)];
            end
        end

        function qp = rate_augment(obj, ocp, Ub, rate, su)
        % QP with input-rate terms, in Riccati (stage-wise) form. With the input step du_i, the
        % previous input step is carried as an extra state xi_i = du_{i-1} (xi_1 = 0: u_{-1} is
        % fixed), and the decision becomes v_i = du_i - du_{i-1}:
        %   [dx; xi]_{i+1} = [A B; 0 I][dx; xi]_i + [B; I] v_i + [c; 0]
        % The input bounds become (hard) bounds on xi at nodes 2..N+1; the rate limits a box on v.
            [nx, nu, N] = size(ocp.B);
            n = nx + nu;
            E = [Ub(:,1) - rate.u_prev, diff(Ub, 1, 2)] - rate.ref;   % current input changes less their reference (unscaled)
            qp = struct();
            qp.A = zeros(n, n, N);  qp.B = zeros(n, nu, N);  qp.c = [ocp.c; zeros(nu, N)];
            qp.q = zeros(n, N+1);  qp.r = zeros(nu, N);
            for i = 1:N
                A = ocp.A(:,:,i);  B = ocp.B(:,:,i);
                qp.A(:,:,i) = [A, B; zeros(nu, nx), eye(nu)];
                qp.B(:,:,i) = [B; eye(nu)];
                qp.q(:,i)   = [ocp.q(:,i); ocp.r(:,i)];
                qp.r(:,i)   = ocp.r(:,i) + su .* (rate.R(:,:,i) * E(:,i));
            end
            qp.q(1:nx,N+1) = ocp.q(:,N+1);
            if isfield(ocp, 'SN'), qp.q(nx+1:end,N+1) = ocp.rN; end   % terminal cost in u_{N-1}: node N's state xi = du_{N-1}
            if isfield(ocp, 'RateReg')   % Optimisation.RegulariseRate: the blocks regularised in the main loop
                qp.Q = ocp.RateReg.Q;  qp.S = ocp.RateReg.S;  qp.R = ocp.RateReg.R;
            else
                [qp.Q, qp.S, qp.R] = rtimpc.rate_hessian(ocp, rate, su);
            end
            qp.x0 = [ocp.x0; zeros(nu, 1)];
            % rate limits on v (scaled), input bounds on xi
            dU = E + rate.ref;   % the limits are on the changes themselves (E is less RateReference)
            qp.lb = (rate.min - dU) ./ su;  qp.ub = (rate.max - dU) ./ su;
            if isfield(ocp, 'lbx'), lbx = ocp.lbx; ubx = ocp.ubx; else, lbx = -inf(nx, N+1); ubx = inf(nx, N+1); end
            qp.lbx = [lbx; -inf(nu, 1), ocp.lb];
            qp.ubx = [ubx;  inf(nu, 1), ocp.ub];
            zl = inf(nx, N+1); zu = zl; Zl = zeros(nx, N+1); Zu = Zl;
            if isfield(ocp, 'zlx'), zl = ocp.zlx .* ones(nx, N+1); zu = ocp.zux .* ones(nx, N+1); end
            if isfield(ocp, 'Zlx'), Zl = ocp.Zlx .* ones(nx, N+1); Zu = ocp.Zux .* ones(nx, N+1); end
            qp.zlx = [zl; inf(nu, N+1)];  qp.zux = [zu; inf(nu, N+1)];   % input bounds stay hard
            qp.Zlx = [Zl; zeros(nu, N+1)];  qp.Zux = [Zu; zeros(nu, N+1)];
            if isfield(ocp, 'Cg')   % general rows: Cg dx + Dg du with du = xi + v (node N+1: du_N = xi)
                ng = size(ocp.Cg, 1);
                qp.Cg = zeros(ng, n, N+1);  qp.Dg = ocp.Dg(:,:,1:N);
                for i = 1:N+1, qp.Cg(:,:,i) = [ocp.Cg(:,:,i), ocp.Dg(:,:,i)]; end
                for f = {'lg', 'ug', 'zlg', 'zug', 'Zlg', 'Zug'}, qp.(f{1}) = ocp.(f{1}); end
            end
        end

        function w = node_lengths(obj, M)
        % Interval length of each node 1..M (held-x weight of node-level penalties)
            if ~isempty(obj.GridNow) && numel(obj.GridNow.d) == M, w = obj.GridNow.d; else, w = ones(1, M); end
        end

        function P = soft_penalty(obj, X)
        % Soft state-bound penalty in J, at the minimal slacks (the violations), nodes 1..N
            [xmin, xmax, xL1, xL2] = state_bounds(obj, size(X, 2) - 1);
            soft = isfinite(xL1) & (isfinite(xmin) | isfinite(xmax));
            P = 0;
            if any(soft, 'all')
                v = (max(xmin - X(:,2:end), 0) + max(X(:,2:end) - xmax, 0)) .* soft;
                xL1(~soft) = 0;  xL2(~soft) = 0;
                P = sum(xL1 .* v + xL2 .* v.^2, 'all');
            end
        end

        function W = stage_weights(obj, override)
            N = obj.PredictionHorizon;
            if ~isempty(override)
                W = override;
                return
            end
            Qyf = obj.Weights.Qyf;
            if isempty(Qyf), Qyf = obj.Weights.Qy; end
            W = repmat(obj.Weights.Qy, 1, 1, N);
            W(:,:,N) = Qyf;
        end

        function [xmin, xmax, L1, L2] = state_bounds(obj, M)
        % State bounds (nx x 1) and the soft-bound weights of nodes 1..M (nx x M). Weights are given
        % per sample (scalar, nx x 1 or nx x N); a node carries the sum over the samples of its
        % interval (as the output weights: held-x), so a coarse interval costs d(j) times a fine
        % one. A node with any hard (Inf) sample is hard.
            nx = obj.nx;  N = obj.PredictionHorizon;
            xmin = obj.Constraints.StateMin(:); if isempty(xmin), xmin = -inf(nx, 1); end
            xmax = obj.Constraints.StateMax(:); if isempty(xmax), xmax =  inf(nx, 1); end
            L1 = obj.Constraints.StateSoftL1;  if isempty(L1), L1 = 1e4; end
            L2 = obj.Constraints.StateSoftL2;  if isempty(L2), L2 = 0;   end
            L1 = per_sample(L1, 'StateSoftL1');  L2 = per_sample(L2, 'StateSoftL2');
            if ~isempty(obj.GridNow) && numel(obj.GridNow.d) == M && any(obj.GridNow.d ~= 1)
                tau = obj.GridNow.tau;
                [S1, S2] = deal(zeros(nx, M));
                for j = 1:M
                    f = tau(j)+1 : tau(j+1);
                    S1(:,j) = sum(L1(:, f), 2);  S2(:,j) = sum(L2(:, f), 2);
                end
                L1 = S1;  L2 = S2;
            else
                L1 = L1(:, 1:M);  L2 = L2(:, 1:M);
            end

            function W = per_sample(W, name)
                if isscalar(W) || isequal(size(W), [nx, 1]) || isequal(size(W), [1, nx])
                    W = repmat(W(:), 1, N) .* ones(nx, N);
                elseif ~isequal(size(W), [nx, N])
                    error('rtimpc:Constraints', '%s must be a scalar, nx x 1 or nx x N (per node).', name)
                end
            end
        end

        function [xmin, xmax] = hard_state_bounds(obj, M)
        % The state bounds of nodes 1..M (nx x M) with StateSoftL1 = Inf (soft ones relaxed to +-Inf)
            [xmin, xmax, L1] = state_bounds(obj, M);
            xmin = xmin .* ones(1, M);  xmax = xmax .* ones(1, M);
            xmin(isfinite(L1)) = -inf;  xmax(isfinite(L1)) = inf;
        end

        function [umin, umax, su] = input_bounds(obj)
            nu = obj.nu;
            umin = obj.Constraints.InputMin(:); if isempty(umin), umin = -inf(nu, 1); end
            umax = obj.Constraints.InputMax(:); if isempty(umax), umax =  inf(nu, 1); end
            su = obj.Optimisation.InputScale(:);
            if isempty(su)
                su = (umax - umin) / 2;
                su(~isfinite(su) | su <= 0) = 1;
            end
        end
    end

    methods (Static)
        function [fred, jred] = dae_reduce(f, g, z0, tol, maxit)
        % A semi-explicit index-1 DAE  x' = f(x, z, u),  0 = g(x, z, u)  as its reduced ODE
        %   x' = fred(x, u) = f(x, z(x, u), u),  [A, B] = jred(x, u)
        % for Model.StateFcn and Jacobian.StateFcn. z(x, u) solves g = 0 by Newton from z0 (to the
        % step tolerance tol, default 1e-14 relative; at most maxit iterations, default 50; an
        % error if it does not converge), and the Jacobian follows from the implicit function
        % theorem, z_x = -g_z \ g_x, z_u = -g_z \ g_u, with the partial derivatives of f and g by
        % complex steps (f, g must be analytic; dg/dz nonsingular: index 1). With the 'irk'
        % integrator this equals collocation of the DAE itself (the stage equations force g = 0 at
        % every stage): checked against acados's IRK on a DAE to 1e-15 (NMPC suite: pendulum_dae,
        % turbo_engine).
            if nargin < 4 || isempty(tol), tol = 1e-14; end
            if nargin < 5 || isempty(maxit), maxit = 50; end
            fred = @(x, u) f(x, solve_z(x, u), u);
            jred = @jacobian;
            function z = solve_z(x, u)
                z = z0(:);
                for it = 1:maxit
                    r = g(x, z, u);
                    dz = -cs_partial(@(zz) g(x, zz, u), z) \ r;
                    z = z + dz;
                    if max(abs(dz)) <= tol * (1 + max(abs(z))), return, end   % ODE solvers' mixed test, atol = rtol = tol
                end
                error('rtimpc:dae_reduce', 'Newton on g(x, z, u) = 0 did not converge in %d iterations.', maxit)
            end
            function [A, B] = jacobian(x, u)
                z = solve_z(x, u);
                fx = cs_partial(@(xx) f(xx, z, u), x);  fz = cs_partial(@(zz) f(x, zz, u), z);  fu = cs_partial(@(uu) f(x, z, uu), u);
                gx = cs_partial(@(xx) g(xx, z, u), x);  gz = cs_partial(@(zz) g(x, zz, u), z);  gu = cs_partial(@(uu) g(x, z, uu), u);
                A = fx - fz * (gz \ gx);  B = fu - fz * (gz \ gu);
            end
        end

        function [P, converged] = dare(A, B, Q, R, tol)
        % Stabilising solution of the discrete algebraic Riccati equation
        %   P = A'PA - A'PB (R + B'PB)^-1 B'PA + Q
        % by the structure-preserving doubling algorithm (quadratic convergence; no toolboxes).
        % converged: the iteration met tol within 100 iterations. Called with one output, a
        % non-finite result is an error and an unconverged finite one a warning.
            if nargin < 5, tol = 1e-12; end
            n = size(A, 1);  I = eye(n);  converged = false;
            G = B * (R \ B');  G = (G + G')/2;  H = Q;
            for it = 1:100
                S  = (I + G * H) \ [A, G];          % one factorisation, two right-hand sides
                A1 = S(:, 1:n);                     % (I + G H)^-1 A
                G1 = G + A * S(:, n+1:end) * A';
                H1 = H + A' * H * A1;
                A  = A * A1;
                G  = (G1 + G1')/2;
                dH = norm(H1 - H, 1);  H = (H1 + H1')/2;
                if isfinite(dH) && dH <= tol * max(1, norm(H, 1)), converged = true; break, end   % (Inf <= Inf: not convergence)
            end
            P = H;
            if nargout < 2 && ~all(isfinite(P(:)))
                error('rtimpc:dare', 'The Riccati iteration gave a non-finite result ((A, B) not stabilisable, or non-finite data).')
            elseif nargout < 2 && ~converged
                warning('rtimpc:dare:notConverged', 'The Riccati iteration did not meet its tolerance in 100 iterations.')
            end
        end

        function [Q, S, R, n] = project_reduced_hessian(Q, S, R, A, B, su, min_eig)
        % Regularise 'project-reduced': acados's PROJECT_REDUC_HESS (ocp_nlp_reg_project_reduc_hess.c,
        % its defaults: thr_eig 1e-12, min_pivot 1e-12, pivoting on; min_eig = RegulariseEpsilon), ported
        % line by line except the pivot swap (acados's indices are wrong for nu >= 3). Backwards over the stages, the stage's reduced Hessian L = H + [B A]' P [B A]
        % (H = [R S'; S Q], P the cost-to-go): its input block's eigenvalues below thr_eig are lifted to
        % min_eig, added to the QP's R; the state block Q is corrected so that the QP's own Riccati
        % recursion passes back the unregularised cost-to-go (P from the original L, by elimination).
        % Stage 0: the input block alone (acados: the whole block with x0's cost; x0 is eliminated here).
        % Inputs in the user's units (InputScale undone,
        % as regularise_blocks). n: the stages changed. acados marks the method experimental.
            thr_eig = 1e-12;  min_pivot = 1e-12;  pivoting = true;
            nx = size(Q, 1);  nu = size(R, 1);  M = size(R, 3);  nt = nu + nx;  n = 0;
            R = R ./ (su(:) * su(:)');  S = S ./ su(:);  B = B ./ su(:)';   % user units
            P = Q(:,:,M+1);                                  % terminal stage: no inputs
            for k = M:-1:2                                   % stage k-1 = node k-1 (MATLAB index k)
                H = [R(:,:,k), S(:,:,k); S(:,:,k)', Q(:,:,k)];
                BA = [B(:,:,k)'; A(:,:,k)'];
                L = H + BA * P * BA';  L = tril(L) + tril(L, -1)';
                [V, d] = eig((L(1:nu,1:nu) + L(1:nu,1:nu)') / 2, 'vector');
                e = (d < thr_eig) .* (min_eig - d);  do_reg = any(d < thr_eig);
                L3 = zeros(nt);  L3(:, 1:nu) = L(:, 1:nu);   % the original input columns
                if do_reg
                    Rg = V * (e .* V');  Rg = (Rg + Rg') / 2;
                    R(:,:,k) = R(:,:,k) + Rg;  L(1:nu,1:nu) = L(1:nu,1:nu) + Rg;  n = n + 1;
                end
                F = rtimpc.potrf_l_mn(L(:, 1:nu));  Ls = F(nu+1:end, :);   % acados: blasfeo_dpotrf_l_mn
                L2xx = -tril(Ls * Ls');
                if do_reg   % the unregularised Schur complement by (pivoted) elimination of L3, as acados
                    for jj = 1:nu
                        if pivoting
                            pivot = L3(jj, jj);  idx = jj;
                            for kk = jj+1:nu
                                t = L3(kk, kk);
                                if t < pivot && t > -pivot, pivot = L3(kk, kk);  idx = kk; end
                            end
                            if idx ~= jj   % symmetric swap of rows/columns jj and idx, lower triangle
                                % acados swaps (kk, jj) with (idx-jj, idx-kk) (0-based), which equals
                                % the right partner, (idx, idx) for kk = jj and (idx, kk) for
                                % jj < kk < idx, only when jj = 0 and idx <= 2: wrong for nu >= 3.
                                t = L3(jj, jj);  L3(jj, jj) = L3(idx, idx);  L3(idx, idx) = t;
                                for kk = jj+1:idx-1
                                    t = L3(kk, jj);  L3(kk, jj) = L3(idx, kk);  L3(idx, kk) = t;
                                end
                                for kk = idx+1:nt
                                    t = L3(kk, jj);  L3(kk, jj) = L3(kk, idx);  L3(kk, idx) = t;
                                end
                            end
                        end
                        pivot = L3(jj, jj);
                        if pivot < min_pivot && pivot > -min_pivot
                            if pivot < 0, pivot = min_pivot; else, pivot = -min_pivot; end
                        end
                        pivot = 1 / pivot;
                        for kk = jj+1:nt
                            t = pivot * L3(kk, jj);
                            for ll = kk:nt, L3(ll, kk) = L3(ll, kk) - L3(ll, jj) * t; end
                        end
                    end
                    L3xx = tril(L3(nu+1:end, nu+1:end));
                    P = tril(L(nu+1:end, nu+1:end)) + L3xx;
                    dQ = -L2xx + L3xx;  Q(:,:,k) = Q(:,:,k) + dQ + tril(dQ, -1)';
                else
                    P = tril(L(nu+1:end, nu+1:end)) + L2xx;
                end
                P = P + tril(P, -1)';
            end
            % stage 0: acados projects its whole block [R S'; S Q]; here x0 is fixed and eliminated (its
            % cost not in Q, as for regularise_blocks), so the input block alone, R + B'PB
            Luu = R(:,:,1) + B(:,:,1)' * P * B(:,:,1);
            [V, d] = eig((Luu + Luu') / 2, 'vector');
            e = (d < thr_eig) .* (min_eig - d);
            if any(e)
                G = V * (e .* V');  R(:,:,1) = R(:,:,1) + (G + G') / 2;  n = n + 1;
            end
            R = R .* (su(:) * su(:)');  S = S .* su(:);       % back to the QP's scaled inputs
        end

        function F = potrf_l_mn(C)
        % Cholesky of the top square block of the m x n panel C (m >= n) and the rows below it divided by
        % its transpose, as blasfeo_dpotrf_l_mn: a non-positive pivot gives a zero column (its inverse
        % set to 0, blasfeo's convention) instead of an error, so a nearly singular block that the
        % regularisation leaves alone (eigenvalues just above thr_eig) factorises as in acados.
            [m, n] = size(C);  F = zeros(m, n);
            for j = 1:n
                c = C(j, j) - F(j, 1:j-1) * F(j, 1:j-1)';
                if c > 0, inv_d = 1 / sqrt(c); else, inv_d = 0; end
                F(j, j) = c * inv_d;
                F(j+1:m, j) = (C(j+1:m, j) - F(j+1:m, 1:j-1) * F(j, 1:j-1)') * inv_d;
            end
        end

        function [Q, S, R, n, SN, RN] = regularise_blocks(Q, S, R, su, method, epsilon, SN, RN)
        % Optimisation.Regularise: make each node's QP Hessian block positive definite, as acados's
        % regularize_method (ocp_nlp_reg_project.c, ocp_nlp_reg_glm.c) does for its stage blocks:
        %   'project'       : eigenvalues floored at epsilon (acados PROJECT)
        %   'mirror'        : eigenvalues replaced by their magnitudes, those within +-epsilon by
        %                     epsilon (acados MIRROR, acados_mirror in ocp_nlp_reg_common.c)
        %   'gershgorin-lm' : + alpha I, from the Gershgorin lower bound b on the smallest eigenvalue:
        %                     alpha = |b| + epsilon if b < 0, epsilon if 0 <= b < epsilon (acados
        %                     GERSHGORIN_LEVENBERG_MARQUARDT)
        % Blocks: [Q S'; S R] at nodes 1..M-1, R at node 0 (x_0 is fixed and eliminated, so u_0 is
        % its only variable; acados keeps x_0 in the block, where its Q only mixes into R_0), Q at
        % node M, or [Q SN'; SN RN] when the terminal cost also depends on u_{M-1} (SN, RN given,
        % nu x nx and nu x nu). Regularised with the inputs in the user's units (undoing InputScale su), so the
        % result does not depend on the scaling; the QP objective is J/2, as acados's with the
        % suite's weights (cost 1/2 e'We). n: the number of blocks changed.
        % Q: nx x nx x (M+1), S: nu x nx x M, R: nu x nu x M (scaled inputs, as in the QP).
            nx = size(Q, 1);  nu = size(R, 1);  M = size(R, 3);  n = 0;
            if nargin < 7, SN = [];  RN = []; end
            for i = 1:M+1
                if i == 1,          H = R(:,:,1);  t = su(:);
                elseif i <= M,      H = [Q(:,:,i), S(:,:,i)'; S(:,:,i), R(:,:,i)];  t = [ones(nx, 1); su(:)];
                elseif isempty(SN), H = Q(:,:,i);  t = ones(nx, 1);
                else,               H = [Q(:,:,i), SN'; SN, RN];  t = [ones(nx, 1); su(:)];
                end
                if isempty(H), continue, end
                H = (H + H') / 2 ./ (t * t');
                [H, changed] = rtimpc.regularise_block(H, method, epsilon);
                if ~changed, continue, end
                n = n + 1;
                H = H .* (t * t');
                if i == 1,          R(:,:,1) = H;
                elseif i <= M,      Q(:,:,i) = H(1:nx, 1:nx);  S(:,:,i) = H(nx+1:nx+nu, 1:nx);  R(:,:,i) = H(nx+1:nx+nu, nx+1:nx+nu);
                elseif isempty(SN), Q(:,:,i) = H;
                else,               Q(:,:,i) = H(1:nx, 1:nx);  SN = H(nx+1:nx+nu, 1:nx);  RN = H(nx+1:nx+nu, nx+1:nx+nu);
                end
            end
        end

        function [B, n] = regularise_rate_blocks(ocp, rate, su, method, epsilon)
        % Optimisation.RegulariseRate: the rate-augmented QP's Hessian blocks (rate_hessian) regularised
        % in the coordinates (dx, du_{i-1}, du_i) of acados's augmented state (the previous input as a
        % state, the rate term in the stage cost): stage blocks over (x_i, u_{i-1}, u_i) at nodes 1..N
        % (x_0 kept in node 1's block, as acados), the terminal block over (x_N, u_{N-1}). Inputs in
        % the user's units, as regularise_blocks. B: the blocks (Q, S, R) for rate_augment; n: the
        % number changed.
            [Q, S, R] = rtimpc.rate_hessian(ocp, rate, su);
            nu = size(R, 1);  nb = size(Q, 1);  nx = nb - nu;  N = size(R, 3);  n = 0;
            L = blkdiag(eye(nx), [eye(nu), zeros(nu); eye(nu), eye(nu)]);   % (dx, xi, v) -> (dx, du_{i-1}, du_i)
            t = [ones(nx, 1); su(:); su(:)];  T = t * t';
            for i = 1:N
                H = [Q(:,:,i), S(:,:,i)'; S(:,:,i), R(:,:,i)];
                H = (H + H') / 2 ./ T;
                [Ha, changed] = rtimpc.regularise_block(L' \ H / L, method, epsilon);
                if ~changed, continue, end
                n = n + 1;
                H = L' * Ha * L;  H = (H + H') / 2 .* T;
                Q(:,:,i) = H(1:nb, 1:nb);  S(:,:,i) = H(nb+1:end, 1:nb);  R(:,:,i) = H(nb+1:end, nb+1:end);
            end
            tN = t(1:nb) * t(1:nb)';
            [H, changed] = rtimpc.regularise_block((Q(:,:,N+1) + Q(:,:,N+1)') / 2 ./ tN, method, epsilon);
            if changed, Q(:,:,N+1) = H .* tN;  n = n + 1; end
            B = struct('Q', Q, 'S', S, 'R', R);
        end
    end

    methods (Static, Access = private)
        function v = check_fields(name, v)
        % An option struct on assignment: exactly the fields of its default (a misspelt name is an error)
            persistent known
            if isempty(known)
                P = ?rtimpc;  P = P.PropertyList;  known = struct();
                for n = {'Model', 'Jacobian', 'Weights', 'Constraints', 'Optimisation'}
                    known.(n{1}) = fieldnames(P(strcmp({P.Name}, n{1})).DefaultValue);
                end
            end
            if ~isstruct(v) || ~isscalar(v), error('rtimpc:Options', '%s must be a scalar struct.', name), end
            fn = fieldnames(v);
            extra = fn(~ismember(fn, known.(name)));
            if ~isempty(extra), error('rtimpc:Options', 'Unknown field ''%s'' in %s (see the class file for its fields).', extra{1}, name), end
            if numel(fn) < numel(known.(name))
                gone = known.(name)(~ismember(known.(name), fn));
                error('rtimpc:Options', '%s is missing its field ''%s''.', name, gone{1})
            end
        end

        function [H, changed] = regularise_block(H, method, epsilon)
        % One symmetric block made positive definite by Optimisation.Regularise's rule (regularise_blocks)
            changed = true;
            switch method
                case "project"
                    [V, e] = eig(H, 'vector');
                    if min(e) >= epsilon, changed = false;  return, end
                    H = V * (max(e, epsilon) .* V');
                case "mirror"
                    [V, e] = eig(H, 'vector');
                    if min(e) > epsilon, changed = false;  return, end
                    e = abs(e);  e(e <= epsilon) = epsilon;
                    H = V * (e .* V');
                case "gershgorin-lm"
                    a = abs(diag(H));
                    b = min(diag(H) - (sum(abs(H), 2) - a));
                    if b >= epsilon, changed = false;  return, end
                    H = H + (abs(min(b, 0)) + epsilon) * eye(size(H, 1));
            end
            H = (H + H') / 2;
        end

        function r = rate_regularised(op, rate)
        % Optimisation.RegulariseRate applies: block regularisation on the rate-augmented QP
            reg = lower(string(op.Regularise));
            r = op.RegulariseRate && rate.on && ~ismember(reg, ["none", "project-reduced"]);
        end

        function [Q, S, R] = rate_hessian(ocp, rate, su)
        % The rate-augmented QP's Hessian blocks (rate_augment: state (dx, xi), xi = du_{i-1}, decision
        % v = du_i - du_{i-1}; the stage cost with du = xi + v, the rate term on v)
            [nx, nu, N] = size(ocp.B);  n = nx + nu;
            Rrs = su .* rate.R .* su';                          % rate weight on scaled inputs (per stage)
            Q = zeros(n, n, N+1);  S = zeros(nu, n, N);  R = zeros(nu, nu, N);
            for i = 1:N
                Qi = ocp.Q(:,:,i);  Si = ocp.S(:,:,i);  Ri = ocp.R(:,:,i);
                Q(:,:,i) = [Qi, Si'; Si, Ri];
                S(:,:,i) = [Si, Ri];
                R(:,:,i) = Ri + Rrs(:,:,i);
            end
            Q(1:nx,1:nx,N+1) = ocp.Q(:,:,N+1);
            if isfield(ocp, 'SN'), Q(:,:,N+1) = [ocp.Q(:,:,N+1), ocp.SN'; ocp.SN, ocp.RN]; end
        end

        function [qp, eqr] = first_node_equalities(qp)
        % Hard general rows of the first node pinned to one value (lg = ug: e.g. InitialConstraintFcn
        % rows tying inputs together) as equality rows on u_1 (the QP solver's Eq1, solved exactly):
        % as two-sided rows they leave its interior point no interior. x_1 is fixed, so each such row
        % Cg x_1 + Dg u_1 = lg constrains u_1 alone. eqr marks them; their bounds are dropped. Only
        % when their Dg rows are linearly independent (the QP solver's requirement); else none.
            eqr = false(0, 1);
            if ~isfield(qp, 'Cg') || isempty(qp.Cg) || size(qp.Cg, 1) == 0, return, end
            ng = size(qp.Cg, 1);  hard = true(ng, 1);
            if isfield(qp, 'zlg')
                zl = qp.zlg .* ones(ng, 1);  zu = qp.zug .* ones(ng, 1);   % (ng x N+1 or ng x 1: column 1)
                hard = ~isfinite(zl(:, 1)) & ~isfinite(zu(:, 1));
            end
            eqr = isfinite(qp.lg(:, 1)) & qp.lg(:, 1) == qp.ug(:, 1) & hard;
            if ~any(eqr), return, end
            E = qp.Dg(eqr, :, 1);   % the exact step needs independent rows: else (e.g. u1 u2 = 0 linearised
            if rank(E) < size(E, 1), eqr(:) = false; return, end   % at u = 0, a zero row) leave them as rows
            qp.Eq1 = qp.Dg(eqr, :, 1);  qp.eq1 = qp.lg(eqr, 1) - qp.Cg(eqr, :, 1) * qp.x0;
            qp.lg(eqr, 1) = -Inf;  qp.ug(eqr, 1) = Inf;
        end

        function d = blend_dual(d, q, alpha)
        % The plan's multipliers after a step alpha, as acados's (without full_step_dual):
        % (1 - alpha) old + alpha QP; none yet (cold) -> alpha QP, as acados from zero
            for f = fieldnames(q)'
                if isstruct(d) && isfield(d, f{1}) && isequal(size(d.(f{1})), size(q.(f{1})))
                    q.(f{1}) = (1 - alpha) * d.(f{1}) + alpha * q.(f{1});
                else
                    q.(f{1}) = alpha * q.(f{1});
                end
            end
            d = q;
        end

        function [sd, sc] = ipopt_stat_scale(qp, dual)
        % IPOPT's scaled optimality error (Waechter & Biegler 2006, eqs. 5-6; IPOPT's s_max = 100):
        % s_d = max(s_max, sum|all multipliers| / (n + m)) / s_max divides the stationarity,
        % s_c = max(s_max, sum|inequality multipliers| / n_i) / s_max the complementarity, with n
        % the QP's variables (states of nodes 2..N+1, inputs), m its equality and general rows
        % (dynamics, rows) and n_i its inequality multipliers (IPOPT's bound multipliers z: its
        % inequalities are bounds on slacks). The paper's reason: with very large multipliers the
        % unscaled test can be impossible to meet (its terms carry their round-off). Both are 1
        % while the mean multipliers are <= 100. (The option: decision D1, 2026-10-17; no suite
        % problem needs it.)
            [nx, nu, N] = size(qp.B);  ng = 0;
            if isfield(qp, 'Cg') && ~isempty(qp.Cg), ng = size(qp.Cg, 1); end
            % inequality multipliers of the bounds and rows that exist (finite sides; IPOPT has no
            % multipliers for absent bounds), with the QP fields holding each side's limits
            side = struct('lam_lb', 'lb', 'lam_ub', 'ub', 'lam_lbx', 'lbx', 'lam_ubx', 'ubx', 'lam_lg', 'lg', 'lam_ug', 'ug');
            f = fieldnames(dual);  tot = 0;  totz = 0;  nz = 0;
            for k = 1:numel(f)
                v = dual.(f{k});
                if ~isnumeric(v), continue, end
                if isfield(side, f{k})
                    lim = [];  if isfield(qp, side.(f{k})), lim = qp.(side.(f{k})); end
                    if isequal(size(lim), size(v)), v = v(isfinite(lim)); end
                    totz = totz + sum(abs(v(:)));  nz = nz + numel(v);
                end
                tot = tot + sum(abs(v(:)));
            end
            smax = 100;
            sd = max(smax, tot / (N * (nx + nu) + N * nx + ng * (N + 1))) / smax;
            sc = max(smax, totz / max(nz, 1)) / smax;
        end

        function d = raw_dual(sol)
        % A QP solution's multipliers (the fields ocp_nlp_res_compute reads)
            d = struct();
            for f = {'pi', 'lam_lb', 'lam_ub', 'lam_lbx', 'lam_ubx', 'lam_lg', 'lam_ug', 'nu_eq1'}
                if isfield(sol, f{1}), d.(f{1}) = sol.(f{1}); end
            end
        end

        function sol = rate_deaugment(sol, nx)
        % Back to (dx, du): du_i = xi_{i+1}
            sol.u = sol.x(nx+1:end, 2:end);
            sol.x = sol.x(1:nx, :);
            sol.pi = sol.pi(1:nx, :);
            for f = {'lam_lbx', 'lam_ubx', 'slack_lbx', 'slack_ubx'}
                if isfield(sol, f{1}), sol.(f{1}) = sol.(f{1})(1:nx, :); end
            end
        end

        function J = rate_cost(U, rate)
        % sum_i e_i' RateR e_i, e_i = (u_i - u_{i-1}) - r_i, with u_{-1} the previous input
            E = [U(:,1) - rate.u_prev, diff(U, 1, 2)] - rate.ref;
            J = sum(E .* rate_times(rate.R, E), 'all');
        end

        function d = rate_slope(U, dU, rate)
        % Directional derivative of rate_cost at U along dU (unscaled)
            E  = [U(:,1) - rate.u_prev, diff(U, 1, 2)] - rate.ref;
            dE = [dU(:,1), diff(dU, 1, 2)];
            d = 2 * sum(E .* rate_times(rate.R, dE), 'all');
        end

        function dP = soft_slope(G, dG, gmin, gmax, L1, L2)
        % One-sided directional derivative of the soft penalty sum L1 v + L2 v^2 (v: the violation of
        % gmin <= G <= gmax, soft rows: finite L1) along dG; as soft_penalty_slope
            soft = isfinite(L1) & (isfinite(gmin) | isfinite(gmax));
            if ~any(soft, 'all'), dP = 0; return, end
            L1(~soft) = 0;  L2(~soft) = 0;
            vl = max(gmin - G, 0);  vu = max(G - gmax, 0);
            tol = 1e-12 * max(1, abs(G));
            gl = (vl > tol | (abs(gmin - G) <= tol & dG < 0)) .* -dG .* soft;
            gu = (vu > tol | (abs(G - gmax) <= tol & dG > 0)) .*  dG .* soft;
            dP = sum(L1 .* (gl + gu) + 2 * L2 .* (vl .* gl + vu .* gu), 'all');
        end

        function v = bound_violation(X, xmin, xmax)
        % l1 violation of the state bounds over nodes 1..N (node 0 is the measurement)
            X = X(:, 2:end);
            v = sum(max(xmin - X, 0), 'all') + sum(max(X - xmax, 0), 'all');
        end
    end
end

function [A, b, c] = irk_tableau(s, scheme)
% Butcher tableau of an s-stage collocation method on [0, 1]: Gauss-Legendre (nodes: roots of the
% shifted Legendre polynomial P_s; order 2s) or Radau IIA (roots of P_s - P_{s-1}, c_s = 1; order
% 2s-1). a_ij = integral_0^c_i l_j, b_j = integral_0^1 l_j, l_j the Lagrange basis on the nodes.
% Cached per (s, scheme).
    persistent cache
    if isempty(cache), cache = containers.Map(); end
    key = sprintf('%s_%d', lower(scheme), s);
    if isKey(cache, key), t = cache(key);  A = t.A;  b = t.b;  c = t.c;  return, end
    Pm = {1, [1 0]};   % Legendre polynomials on [-1, 1] (coefficients, highest power first)
    for k = 1:s-1, Pm{k+2} = ((2*k+1) * [Pm{k+1} 0] - k * [0 0 Pm{k}]) / (k+1); end
    switch lower(scheme)
        case 'gauss-legendre', pc = Pm{s+1};
        case 'radau', pc = Pm{s+1} - [0 Pm{s}];
        otherwise, error('rtimpc:IRKScheme', 'Unknown IRKScheme ''%s''.', scheme)
    end
    c = sort((real(roots(pc)) + 1) / 2);
    if strcmpi(scheme, 'radau'), c(end) = 1; end
    A = zeros(s);  b = zeros(s, 1);
    for j = 1:s
        e = zeros(s, 1);  e(j) = 1;
        lj = polyint(polyfit(c, e, s - 1));
        b(j) = polyval(lj, 1) - polyval(lj, 0);
        for i = 1:s, A(i, j) = polyval(lj, c(i)) - polyval(lj, 0); end
    end
    cache(key) = struct('A', A, 'b', b, 'c', c);
end

function P = node_parameters(P, tau, M, n)
% Model parameters per node: np x N per sample (or np x 1, held) -> np x M, each interval taking
% its first sample's; [] -> none
    if isempty(P), P = []; return, end
    if size(P, 2) == 1, P = repmat(P, 1, M); return, end
    if size(P, 2) < n
        error('rtimpc:Parameters', 'Parameters must be np x 1 or np x N (one column per sample 0..N-1; %d needed).', n)
    end
    P = P(:, tau(1:M) + 1);
end

function [f, A, B] = f_and_jacobian(Ff, Fj, x, u)
% Model and (separately supplied) Jacobians as one call; the Jacobians only when asked for
    f = Ff(x, u);
    if nargout > 1, [A, B] = Fj(x, u); end
end

function RE = rate_times(R, E)
% Columns R(:,:,i) * E(:,i) of the per-stage rate weights
    RE = reshape(pagemtimes(R, reshape(E, size(E, 1), 1, [])), size(E));
end

function s = ternary_str(c, a, b)
    if c, s = a; else, s = b; end
end

function [fq, hard] = feasibility_qp(ocp)
% Byrd-Omojokun feasibility QP (as acados's): the QP's constraints with every hard state bound and
% general row softened (unit L1 price), the soft ones free (L1 0, L2 1e-4), the cost replaced by
% 1e-4 I on all steps (no gradient). Input bounds, hard rows at node 0 (on the first input alone)
% and the dynamics stay hard. hard: masks of the softened bounds/rows (with a finite bound) and
% their count.
    fq = ocp;  h = 1e-4;
    [nx, nu, N] = size(ocp.B);
    fq.Q = repmat(h * eye(nx), 1, 1, N + 1);  fq.q = zeros(nx, N + 1);
    fq.R = repmat(h * eye(nu), 1, 1, N);      fq.r = zeros(nu, N);
    fq.S = zeros(nu, nx, N);
    if isfield(fq, 'SN'), fq = rmfield(fq, {'SN', 'RN', 'rN'}); end   % terminal cost in u_{N-1}
    if isfield(fq, 'RateReg'), fq = rmfield(fq, 'RateReg'); end      % (RegulariseRate: the step QP's blocks)
    hard =struct('xl', false(nx, N + 1), 'xu', false(nx, N + 1), 'gl', [], 'gu', [], 'n', 0);
    if isfield(ocp, 'lbx')
        zl = inf(nx, N + 1);  zu = zl;
        if isfield(ocp, 'zlx'), zl = ocp.zlx .* ones(nx, N + 1);  zu = ocp.zux .* ones(nx, N + 1); end
        hard.xl = ~isfinite(zl) & isfinite(ocp.lbx);  hard.xu = ~isfinite(zu) & isfinite(ocp.ubx);
        hard.xl(:, 1) = false;  hard.xu(:, 1) = false;   % node 0 is fixed
        fq.zlx = double(~isfinite(zl));  fq.zux = double(~isfinite(zu));   % hard -> 1, soft -> 0
        fq.Zlx = h * isfinite(zl);       fq.Zux = h * isfinite(zu);
    end
    if isfield(ocp, 'lg')
        ng = size(ocp.lg, 1);
        zl = inf(ng, N + 1);  zu = zl;
        if isfield(ocp, 'zlg'), zl = ocp.zlg .* ones(ng, N + 1);  zu = ocp.zug .* ones(ng, N + 1); end
        hard.gl = ~isfinite(zl) & isfinite(ocp.lg);  hard.gu = ~isfinite(zu) & isfinite(ocp.ug);
        fq.zlg = double(~isfinite(zl));  fq.zug = double(~isfinite(zu));
        fq.Zlg = h * isfinite(zl);       fq.Zug = h * isfinite(zu);
        % node 0 (x fixed): its hard rows constrain the first input alone, so they stay hard, as the
        % input bounds (e.g. equality rows tying scenario inputs: the QP solver then keeps them exact)
        fq.zlg(~isfinite(zl(:, 1)), 1) = Inf;  fq.zug(~isfinite(zu(:, 1)), 1) = Inf;
        hard.gl(:, 1) = false;  hard.gu(:, 1) = false;
    end
    hard.n = nnz(hard.xl) + nnz(hard.xu) + nnz(hard.gl) + nnz(hard.gu);
end

function [C, D] = constraint_jacobian(f, J, x, u)
% [C, D] of a constraint function: the user's Jacobian J (D omitted for a terminal one), else
% central differences (step: fd_step)
    if ~isempty(J)
        if isempty(u), C = J(x); D = zeros(size(C, 1), 0); else, [C, D] = J(x, u); end
        return
    end
    g0 = f(x, u);  C = zeros(numel(g0), numel(x));  D = zeros(numel(g0), numel(u));
    h = fd_step();
    for j = 1:numel(x)
        e = zeros(numel(x), 1);  e(j) = h;
        C(:,j) = (f(x + e, u) - f(x - e, u)) / (2*h);
    end
    for j = 1:numel(u)
        e = zeros(numel(u), 1);  e(j) = h;
        D(:,j) = (f(x, u + e) - f(x, u - e)) / (2*h);
    end
end

function J = fd_jacobian(f, z)
% Jacobian of f at z by central differences (validate)
    h = fd_step();  f0 = f(z);  J = zeros(numel(f0), numel(z));
    for j = 1:numel(z)
        e = zeros(numel(z), 1);  e(j) = h;
        J(:, j) = (reshape(f(z + e), [], 1) - reshape(f(z - e), [], 1)) / (2 * h);
    end
end

function g = second_output(f, v)
    [~, g] = f(v);  g = g(:);
end

function v = interval_adjoint(obj, z, p, w)
% [A'w; B'w] of one sample's sensitivities at z = [x; u] (validate: IntervalHessianFcn)
    [~, A, B] = obj.integrate(z(1:obj.nx), z(obj.nx+1:end), 1, p);
    v = [A' * w; B' * w];
end

function v = constraint_adjoint(f, J, z, nx, w)
% [Cg'w; Dg'w] of the constraint rows at z = [x; u] (validate: ConstraintLagrangianHessianFcn)
    [C, D] = constraint_jacobian(f, J, z(1:nx), z(nx+1:end));
    v = [C' * w; D' * w];
end

function h = fd_step()
% Step of the central differences of user functions (constraint Jacobians, validate): 1e-5,
% absolute. A central difference's error is h^2 |f'''| / 6 (truncation) + eps |f| / h (round-off),
% least near h = eps^(1/3) ~ 6e-6 for a function of unit scale: about 2e-11 at 1e-5. The step is
% not scaled by |x|: a variable's magnitude says nothing about the scale its function varies on,
% and an absolute step stays exact to round-off while eps |x| << 1e-5, i.e. |x| << 1e10 (as the
% steps of lagrangian_curvature).
    h = 1e-5;
end

function t = qp_tol(qopt, f)
% A QP solver tolerance as set in QPSolverOptions, else the solvers' default (1e-8)
    t = 1e-8;  if isfield(qopt, f), t = qopt.(f); end
end

function k = node_kind(a, M)
% Node a-1 of a grid with M intervals: 0 initial, 1 inner, 2 terminal
    k = 1;  if a == 1, k = 0; elseif a == M+1, k = 2; end
end

function v = ternary_num(c, a, b)
    if c, v = a; else, v = b; end
end

function g = pc_min(pc, ng)
% Path-constraint lower bounds (ng x 1; empty without path constraints)
    if ng > 0, g = pc.gmin; else, g = zeros(0, 1); end
end

function g = pc_max(pc, ng)
    if ng > 0, g = pc.gmax; else, g = zeros(0, 1); end
end

function J = cs_partial(F, v)
% Jacobian of F at v by complex steps (exact to round-off for analytic F; rtimpc.dae_reduce)
    h = 1e-30;  v = v(:);  F0 = F(v);  J = zeros(numel(F0), numel(v));
    for j = 1:numel(v), e = zeros(numel(v), 1);  e(j) = 1i * h;  J(:, j) = imag(F(v + e)) / h; end
end

function ok = qp_data_finite(ocp)
% The QP's data are finite and real (bounds may be +-Inf, not NaN)
    ok = true;
    for f = {'A', 'B', 'c', 'Q', 'q', 'R', 'S', 'r', 'x0', 'SN', 'RN', 'rN', 'Cg', 'Dg'}
        if isfield(ocp, f{1}) && (~isreal(ocp.(f{1})) || ~all(isfinite(ocp.(f{1})), 'all')), ok = false; return, end
    end
    for f = {'lb', 'ub', 'lbx', 'ubx', 'lg', 'ug'}
        if isfield(ocp, f{1}) && (~isreal(ocp.(f{1})) || any(isnan(ocp.(f{1})), 'all')), ok = false; return, end
    end
end

function tf = qp_failure(err)
% Errors that mean "this QP could not be solved" (handled: rebuild, retry, OnFailure): a non-finite
% linearisation and the QP solvers' numerical breakdowns. Any other error is rethrown by the callers.
    tf = ismember(err.identifier, {'rtimpc:nonfinite', 'rtimpc:QPNoStep', 'rtimpc:chol', 'rtimpc:inertia'});
end

function ok = dual_fits(qp, dual)
% Do carried multipliers have this QP's layout (the fields ocp_nlp_res_compute reads)?
    ok = true;
    if ~isstruct(dual), return, end
    [nx, nu, N] = size(qp.B);
    want = struct('pi', [nx, N+1], 'lam_lb', [nu, N], 'lam_ub', [nu, N], 'lam_lbx', [nx, N+1], 'lam_ubx', [nx, N+1]);
    if isfield(qp, 'Cg') && ~isempty(qp.Cg), want.lam_lg = [size(qp.Cg, 1), N+1];  want.lam_ug = want.lam_lg; end
    if isfield(qp, 'Eq1') && ~isempty(qp.Eq1), want.nu_eq1 = [size(qp.Eq1, 1), 1]; end
    for f = fieldnames(want)'
        if isfield(dual, f{1}) && ~isempty(dual.(f{1})) && ~isequal(size(dual.(f{1})), want.(f{1})), ok = false; return, end
    end
end

function args = qp_solver_args(args, solver)
% Optimisation.QPSolverOptions for one QP solver: the names it takes; a name neither solver takes is an error
    if isempty(args), return, end
    takes.riccati   = {'MaxIter', 'TolStat', 'TolEq', 'TolComp', 'TolAcceptable', 'AcceptableIter', 'StepFrac', 'Mu0', ...
                       'WarmFloor', 'Refine', 'RefineAbove', 'InertiaCorrection', 'InertiaDeltaHint'};
    takes.condensed = {'MaxIter', 'TolStat', 'TolEq', 'TolComp', 'StepFrac', 'Mu0', 'Factor', 'Refine', 'Proximal', 'Scale'};
    names = args(1:2:end);
    bad = setdiff(names, [takes.riccati, takes.condensed]);
    if ~isempty(bad)
        error('rtimpc:QPSolverOptions', 'Unknown QPSolverOptions field ''%s''. Riccati: %s. Condensed: %s.', bad{1}, ...
            strjoin(takes.riccati, ', '), strjoin(takes.condensed, ', '))
    end
    keep = find(ismember(names, takes.(solver)));
    args = args(sort([2 * keep - 1, 2 * keep]));
end

function barrier_done(obj, ws, cap)
% BarrierHomotopy: the controller's settings after (or on an error in) barrier_step
    obj.BarrierNow = 0;  obj.Optimisation.WarmStart = ws;  obj.Optimisation.SQPMaxIter = cap;
end
