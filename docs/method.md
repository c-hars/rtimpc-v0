# Method

## 1. The problem

At every sample, `rtimpc` solves, from the measured state $\hat{x}$,

$$
\min_{x,\,u,\,s}\; J \quad \text{such that} \quad
\begin{cases}
x_0 = \hat{x} \\
x_{i+1} = F(x_i, u_i, p_i) \\
c_{\min} - s^- \le c(x_i, u_i) \le c_{\max} + s^+, & s^-, s^+ \ge 0
\end{cases}
$$

$$
\begin{aligned}
J = &\sum_{i=1}^{N-1} \lVert h(x_i, u_i) - r_i \rVert_{W_i}^2 + \lVert h_N(x_N) - r_N \rVert_{W_N}^2 \\
&+ \sum_{i=0}^{N-1} \Big( \lVert u_i - u^{\mathrm{ref}}_i \rVert_R^2 + g_i^\top u_i + \lVert u_i - u_{i-1} - \rho_i \rVert_{R_\Delta}^2 + \ell(x_i, u_i) \Big) + V_f(x_N) + \sum \big( \lambda_1 s + \lambda_2 s^2 \big)
\end{aligned}
$$

and applies $u_0$. Here $F$ is one sample of a discrete model, or of an integrator applied to $\dot{x} = f(x, u, p)$ with the input held (section 2); $p_i$ are known parameters over interval $i$. They can also enter the outputs, constraints and costs as one more argument: $h(x_i, u_i, p_i)$, $c(x_i, u_i, p_i)$, $\ell(x_i, u_i, p_i)$.

> $u_{-1}$ is the input applied over the previous sample.

**Cost.** $h$ is the output function (the state by default), and $h_N$ the terminal output, a function of the state alone ($h$ by default, if $h$ does not depend on the input). By default $W_i = Q_y$ for $i < N$ and $W_N = Q_{yf}$; per-node weights can be given instead. $\ell$ and $V_f$ are user functions, for costs that are not weighted squares.

> Note: the quadratic terms carry no factor $\tfrac{1}{2}$, which matters when comparing against tools such as acados [4].

**Constraints.** $c$ is any constraint in the table below. Bounds may be infinite, and may vary along the horizon.

| Constraint | Form | Nodes | Default |
|---|---|---|---|
| Input bounds | $u_{\min} \le u_i \le u_{\max}$ | $0$ to $N-1$ | hard |
| Input-rate limits | $\Delta_{\min} \le u_i - u_{i-1} \le \Delta_{\max}$ | $0$ to $N-1$ | hard |
| State bounds | $x_{\min} \le x_i \le x_{\max}$ | $1$ to $N$ | soft |
| Output bounds | $y_{\min} \le h(x_i, u_i) \le y_{\max}$ ($h_N$ at $N$) | $1$ to $N$ | soft |
| Path constraints | $g_{\min} \le g(x_i, u_i) \le g_{\max}$ | $1$ to $N-1$ | soft |
| Initial constraints | $g_{0,\min} \le g_0(x_0, u_0) \le g_{0,\max}$ | $0$ | soft |
| Terminal constraints | $g_{N,\min} \le g_N(x_N) \le g_{N,\max}$ | $N$ | soft |

- **Node 0** is the measurement: only initial constraints apply there, and an initial constraint without $u_0$ is left out.
- **Node $N$** has no path constraints: use a terminal constraint or a state bound.

**Slacks.** Defaults: $\lambda_1 = 10^4$, $\lambda_2 = 0$, except for input bounds and rate limits, which are hard constraints with no slacks ($\lambda_1 = \infty$, so that $s^- = s^+ = 0$).

> Note: a soft constraint yields the same solution as a hard constraint, whenever the hard problem has a solution, and provided $\lambda_1$ exceeds the hard problem's multiplier, i.e. identical solution quality in the typical case, without losing feasibility outside of that.

## 2. Transcription

The problem is transcribed by multiple shooting, with the states at every node as decision variables. At a plan $(\bar{x}, \bar{u})$ that is not yet a solution, the model leaves gaps

$$
d_i = F(\bar{x}_i, \bar{u}_i, p_i) - \bar{x}_{i+1}
$$

which, at each SQP step, the solver closes to first order.

**Shooting grid.** Each shooting interval is one sample long by default. Optionally, a "decimated" (non-uniform) grid can be used with move blocking, reducing node count over long horizons.

**Integration.** $F$ uses a fixed-step integrator, with the input held over each interval: RK4 (the default), the implicit trapezoidal rule, or collocation (Gauss-Legendre or Radau IIA) for stiff models. An interval may be split into several substeps.

**Sensitivities.** The derivatives of $F$ are the exact derivatives of the integration scheme, computed from the user-supplied Jacobians of $f$.

> The Jacobians are user functions. Analytic or symbolic expressions are best, though numerical differentiation also works.

## 3. SQP

The problem is solved by sequential quadratic programming (SQP). Each iteration solves a quadratic program (QP) for the step $w_i = (\delta x_i, \delta u_i)$ from the current plan, with $w_N = \delta x_N$:

$$
\min_{\delta x,\,\delta u}\; \sum_{i=0}^{N} \Big( \tfrac{1}{2} w_i^\top H_i\, w_i + \nabla_i^\top w_i \Big)
\quad \text{such that} \quad
\begin{cases}
\delta x_0 = \hat{x} - \bar{x}_0 \\
\delta x_{i+1} = A_i\, \delta x_i + B_i\, \delta u_i + d_i \\
u_{\min} - \bar{u}_i \le \delta u_i \le u_{\max} - \bar{u}_i \\
c_{\min} - \bar{c}_i \le C_i\, \delta x_i + D_i\, \delta u_i \le c_{\max} - \bar{c}_i
\end{cases}
$$

Here $\nabla_i$ is the gradient of $J/2$, $A_i = \partial F / \partial x$ and $B_i = \partial F / \partial u$, and $\bar{c}_i$, $C_i$ and $D_i$ are the values and Jacobians of the constraints, all at the plan. The slacks of soft constraints stay in the QP as variables, with their penalties.

**Hessian.** By default, $H_i$ is the Gauss-Newton Hessian of the least-squares terms,

$$
H_i = G_i^\top W_i\, G_i + \begin{bmatrix} 0 & 0 \\ 0 & R \end{bmatrix}, \qquad G_i = \begin{bmatrix} \partial h / \partial x & \partial h / \partial u \end{bmatrix}
$$

to which $\ell$ and $V_f$ add their own Hessians. It needs no second derivatives of the model, and is positive semidefinite whenever the weights are. For costs with too little curvature, such as linear ones, the exact Hessian of the Lagrangian can be used instead, regularised to keep the QP convex.

**Input changes.** The rate cost and rate limits are made stage-wise by carrying $u_{i-1}$ as an extra state.


**Plan update.** The plan is updated with $\bar{x} \leftarrow \bar{x} + \alpha\, \delta x$ and $\bar{u} \leftarrow \bar{u} + \alpha\, \delta u$. RTI takes the full step, $\alpha = 1$. Otherwise, $\alpha$ is chosen by backtracking on the merit function

$$
\phi = J + \sum_k \nu_k \lvert v_k \rvert
$$

where the $v_k$ are the gaps and hard-constraint violations, and each $\nu_k$ is the magnitude of the corresponding QP multiplier [5]. Starting from $\alpha = 1$, the step is shortened by a factor of $0.7$ until $\phi$ decreases. Below $\alpha = 0.05$, the shortened step is taken as it is.

**Stopping condition.** When iterating to convergence, SQP iterations stop once the stationarity residual, the gaps, the hard-constraint violation and the complementarity are all below $10^{-6}$.

## 4. Real-time iteration

Real-time iteration (RTI) [1] uses a fixed number of SQP iterations per call, one by default.

**Warm start.** The initial guess is the previous plan, shifted by one sample:

$$
\bar{u} \leftarrow (\bar{u}_1, \dots, \bar{u}_{N-1}, \bar{u}_{N-1}), \qquad \bar{x} \leftarrow \big(\bar{x}_1, \dots, \bar{x}_N, F(\bar{x}_N, \bar{u}_{N-1})\big)
$$

The QP's multipliers (duals) are also shifted in the same way.

**Cold start.** With no previous plan, the first call iterates to convergence (at most 50 iterations by default) using a forward rollout:

$$
\bar{u}_i = \min\big(\max(0, u_{\min}), u_{\max}\big), \qquad \bar{x}_0 = \hat{x}, \qquad \bar{x}_{i+1} = F(\bar{x}_i, \bar{u}_i, p_i)
$$

i.e. a rollout from the measured state $\hat{x}$, under zero input $\bar{u}_i$ (clipped within bounds). Alternatively, the user can supply their own initial guess.


**Delay compensation.** If the input is applied one sample after the measurement, the problem is posed from the predicted state $F(\hat{x}, u_{-1})$.

## 5. QP solver

QPs are solved using a primal-dual interior-point method (IPM), Mehrotra's predictor-corrector [2, 3]. 

With a slack $t \ge 0$ and multiplier $\lambda \ge 0$ for each inequality, each IPM iteration reduces to an LQ problem whose weights are adjusted by the inequalities, for example $\tilde{R}_i = R_i + \operatorname{diag}(\lambda / t)$ for input bounds. A Riccati recursion solves it in $O\big(N (n_x + n_u)^3\big)$ operations.

**Stopping condition.** When the stationarity residual, the equality residual and the largest complementarity product $t \lambda$ are all below $10^{-8}$ (absolute), the stopping condition is satisfied and the IPM declares convergence.

Iterates that stay within $10^{-6}$ for 15 iterations are also accepted, as in IPOPT [6]. A QP that meets neither test within 50 iterations is treated as failed (Section 6).

**Regularisation.** If a Riccati factorisation fails because the QP's Hessian is not positive semidefinite, $\delta I$ is added to every Hessian block and the factorisation is repeated, with $\delta$ increased until it succeeds - *inertia correction*, as in IPOPT [6]. 

This is possible when using the exact Hessian, or an indefinite $\ell$ – not when using the standard positive (semi)-definite weights $Q$ and $R$.

**Condensed QP.** A condensed solver (states eliminated) is also included, mainly as a cross-check: it is $O(N^3)$ rather than $O(N)$, has no warm start, and can be ill-conditioned, notably for open-loop unstable models over long horizons.

## 6. Failures and status

**Infeasible linearisation.** Hard constraints that hold for the nonlinear problem can contradict each other once linearised, leaving the QP without a solution. Two QPs then replace it: the first minimises the linearised hard-constraint violation, and the second solves the original QP with the constraints relaxed by the remaining violation.

**Numerical failure.** If the QP's data are not finite or the solver breaks down, the plan's states are re-initialised by forward simulation of its inputs from $\hat{x}$, and the QP is solved once more. If that also fails, the call returns status $-1$.

**Retry.** By default, a call that fails from a warm start is repeated from a cold start.

**Status.** Every call returns the first input of its plan, a status, and a flag for whether the plan satisfies its hard constraints:

| Status | Meaning |
|---|---|
| 1 | converged |
| 2 | negligible step, not converged |
| 0 | iteration limit reached (the normal outcome in real time) |
| -1 | QP not solved, also after rebuilding the plan |
| -2 | the line search found no acceptable step |

## Appendix A. Verification

Each part of the method is checked against a reference implementation: the QP solvers against `quadprog`, the SQP solver against acados and IPOPT on standard test problems (including 72 Hock-Schittkowski problems), and the integrators against their nominal order and `ode45`. The examples reproduce `nlmpc`'s results on the same problems. The full verification suite will be published later.

## References

1. M. Diehl, H. G. Bock, J. P. Schlöder et al., *Real-time optimization and nonlinear model predictive control of processes governed by differential-algebraic equations*, J. Process Control, 2002.
2. G. Frison, M. Diehl, *HPIPM: a high-performance quadratic programming framework for model predictive control*, IFAC 2020.
3. C. V. Rao, S. J. Wright, J. B. Rawlings, *Application of interior-point methods to model predictive control*, JOTA, 1998.
4. R. Verschueren et al., *acados — a modular open-source framework for fast embedded optimal control*, Math. Prog. Comp., 2022.
5. J. Nocedal, S. J. Wright, *Numerical Optimization*, 2nd ed., Springer, 2006.
6. A. Wächter, L. T. Biegler, *On the implementation of an interior-point filter line-search algorithm for large-scale nonlinear programming*, Math. Prog., 2006.
