# GNSS Least Squares — Four Unknowns: x, y, z and Clock Bias

This guide develops the mathematics for a basic GNSS Single Point Positioning (SPP) solver, starting from a distance measurement and ending with an iterative least-squares algorithm. It builds on the [linear least-squares explanation](README.md).

**Status:** this is the mathematical plan. The GNSS Python implementation will follow separately. The existing `lstsq()` function can solve each linear correction step.

## 1. What is known, and what is unknown?

At one measurement epoch, satellite i supplies one pseudorange observation. For this introductory model, use a single constellation, known satellite positions, and corrected pseudoranges. Receiver position and clock bias are shared by all observations at that epoch.

| Quantity | Meaning | Unit | Known or estimated? |
| --- | --- | --- | --- |
| $(X_i,Y_i,Z_i)$ | Position of satellite i | metres | Known input |
| $P_i$ | Corrected pseudorange to satellite i | metres | Measured input |
| $(x,y,z)$ | Receiver position | metres | Three unknowns |
| $\delta t$ | Receiver clock offset relative to system time | seconds | One unknown |
| $b=c\delta t$ | The same receiver clock offset expressed as distance | metres | Alternative representation of the fourth unknown |
| $c$ | Speed of light: 299,792,458 | metres per second | Known constant |

Satellite and receiver positions must use a consistent Earth-centred, Earth-fixed (ECEF) frame. The coordinates $x,y,z$ are Cartesian coordinates, not latitude, longitude, and height.

There are **four unknowns**, using either of these equivalent state vectors:

```math
\boldsymbol{\theta}_b=\begin{bmatrix}x & y & z & b\end{bmatrix}^T,
\qquad
\boldsymbol{\theta}_t=\begin{bmatrix}x & y & z & \delta t\end{bmatrix}^T.
```

We do not estimate b and delta t independently. They describe the same clock offset. The derivation below uses b so every state component has units of metres.

## 2. Why is there a clock unknown?

Distance follows from signal travel time:

```math
\text{distance}=c\times\text{travel time}.
```

A receiver measures travel time using its own clock. If that clock is ahead of system time by a positive delta t, the measured pseudorange contains a positive distance offset b:

```math
b=c\delta t,
\qquad
\delta t=\frac{b}{c}.
```

For example, one microsecond of receiver clock offset corresponds to

```math
\delta t=10^{-6}\ \mathrm{s},
\qquad
b=299\,792\,458\times10^{-6}=299.792458\ \mathrm{m}.
```

The same receiver clock contributes to every satellite observation at this epoch. That shared term is what allows it to be estimated together with position.

## 3. The pseudorange observation equation

By the three-dimensional Pythagorean theorem, the geometric distance to satellite i is

```math
\rho_i(x,y,z)=\sqrt{(x-X_i)^2+(y-Y_i)^2+(z-Z_i)^2}.
```

After correcting satellite clock and other modelled effects, write

```math
\boxed{P_i=\rho_i(x,y,z)+b+\varepsilon_i}.
```

The error term epsilon includes remaining measurement noise and modelling errors. Raw observations require satellite-clock, atmospheric, and other corrections; real satellite coordinates must account for transmission time and Earth rotation during signal travel. A first synthetic example can supply consistent positions and generate observations directly from this simplified model. The standard observation model and linearisation are described in [ESA Navipedia: Code Based Positioning](https://gssc.esa.int/navipedia/index.php/Code_Based_Positioning_%28SPS%29).

For four satellites, the equations have this structure:

```math
\begin{aligned}
P_1&=\sqrt{(x-X_1)^2+(y-Y_1)^2+(z-Z_1)^2}+b+\varepsilon_1,\\ P_2&=\sqrt{(x-X_2)^2+(y-Y_2)^2+(z-Z_2)^2}+b+\varepsilon_2,\\ P_3&=\sqrt{(x-X_3)^2+(y-Y_3)^2+(z-Z_3)^2}+b+\varepsilon_3,\\ P_4&=\sqrt{(x-X_4)^2+(y-Y_4)^2+(z-Z_4)^2}+b+\varepsilon_4.
\end{aligned}
```

Each extra satellite adds one equation, not another receiver unknown. Four satellites are the minimum for this unconstrained four-unknown model, but satellite geometry must also provide independent information.

## 4. What does least squares minimise?

Define the predicted observation and residual:

```math
f_i(\boldsymbol{\theta}_b)=\rho_i(x,y,z)+b,
\qquad
r_i(\boldsymbol{\theta}_b)=P_i-f_i(\boldsymbol{\theta}_b).
```

For M satellites, the nonlinear least-squares objective is

```math
J(\boldsymbol{\theta}_b)=\sum_{i=1}^{M}\left[P_i-\rho_i(x,y,z)-b\right]^2.
```

We want the position and clock bias giving the smallest total squared residual. Equal weighting is assumed here.

Unlike fitting a straight line, the unknown coordinates appear inside a square root. We cannot pass these nonlinear equations directly to `lstsq()`. We first approximate the model by a linear system near a current estimate, solve for a correction, and repeat.

## 5. Start from a current estimate

At iteration k, let the current estimate be

```math
\boldsymbol{\theta}^{(k)}=\begin{bmatrix}x^{(k)} & y^{(k)} & z^{(k)} & b^{(k)}\end{bmatrix}^T.
```

For each satellite, calculate the current geometric range and predicted pseudorange:

```math
\rho_i^{(k)}=\sqrt{(x^{(k)}-X_i)^2+(y^{(k)}-Y_i)^2+(z^{(k)}-Z_i)^2},
\qquad
\hat P_i^{(k)}=\rho_i^{(k)}+b^{(k)}.
```

Subtract prediction from observation:

```math
\ell_i^{(k)}=P_i-\hat P_i^{(k)}.
```

This is the prefit residual: the mismatch before applying this iteration's correction. A positive value means the measured pseudorange exceeds the current prediction.

The correction vector is

```math
\Delta\boldsymbol{\theta}=\begin{bmatrix}\Delta x & \Delta y & \Delta z & \Delta b\end{bmatrix}^T.
```

These are changes to the current estimate, not absolute coordinates or the absolute clock bias.

## 6. Linearise using a first-order Taylor expansion

For a small correction, Taylor expansion gives

```math
f_i(\boldsymbol{\theta}^{(k)}+\Delta\boldsymbol{\theta})
\approx f_i(\boldsymbol{\theta}^{(k)})
+\left.\frac{\partial f_i}{\partial x}\right|_k\Delta x
+\left.\frac{\partial f_i}{\partial y}\right|_k\Delta y
+\left.\frac{\partial f_i}{\partial z}\right|_k\Delta z
+\left.\frac{\partial f_i}{\partial b}\right|_k\Delta b.
```

The vertical bar means evaluate the derivative at the current estimate. Terms involving products or higher powers of the corrections are omitted. Repeating the expansion after updating the estimate reduces the error from this approximation near a solution.

To derive the x derivative, set

```math
q_i=(x-X_i)^2+(y-Y_i)^2+(z-Z_i)^2,
\qquad \rho_i=q_i^{1/2}.
```

Apply the chain rule:

```math
\frac{\partial\rho_i}{\partial x}
=\frac{1}{2}q_i^{-1/2}\,2(x-X_i)
=\frac{x-X_i}{\rho_i}.
```

The remaining derivatives follow the same rule. Since b is added directly, its derivative is one:

```math
\frac{\partial f_i}{\partial x}=\frac{x-X_i}{\rho_i},
\quad
\frac{\partial f_i}{\partial y}=\frac{y-Y_i}{\rho_i},
\quad
\frac{\partial f_i}{\partial z}=\frac{z-Z_i}{\rho_i},
\quad
\frac{\partial f_i}{\partial b}=1.
```

Substitute the expansion into the observation equation and subtract the current prediction:

```math
\ell_i^{(k)}\approx
\frac{x^{(k)}-X_i}{\rho_i^{(k)}}\Delta x
+\frac{y^{(k)}-Y_i}{\rho_i^{(k)}}\Delta y
+\frac{z^{(k)}-Z_i}{\rho_i^{(k)}}\Delta z
+\Delta b+\varepsilon_i.
```

This is now linear in the four corrections.

## 7. Build the geometry matrix

Define one row for each satellite:

```math
H_i=\begin{bmatrix}
\dfrac{x^{(k)}-X_i}{\rho_i^{(k)}} &
\dfrac{y^{(k)}-Y_i}{\rho_i^{(k)}} &
\dfrac{z^{(k)}-Z_i}{\rho_i^{(k)}} & 1
\end{bmatrix}.
```

Stack these rows and the prefit residuals:

```math
H=\begin{bmatrix}H_1\\ H_2\\ \vdots\\ H_M\end{bmatrix},
\qquad
\boldsymbol{\ell}=\begin{bmatrix}\ell_1^{(k)}\\ \ell_2^{(k)}\\ \vdots\\ \ell_M^{(k)}\end{bmatrix},
\qquad
H\Delta\boldsymbol{\theta}\approx\boldsymbol{\ell}.
```

The dimensions are

```math
H\in\mathbb{R}^{M\times4},
\qquad
\Delta\boldsymbol{\theta}\in\mathbb{R}^{4},
\qquad
\boldsymbol{\ell}\in\mathbb{R}^{M}.
```

All four columns of H are dimensionless in the distance-bias formulation. Multiplying by corrections in metres gives a predicted pseudorange change in metres.

**Sign check:** the position coefficients use receiver minus satellite, and the residual uses observed minus predicted. For a satellite directly in the positive x direction, the x coefficient is negative: moving towards it reduces the range. Increasing b increases every predicted pseudorange, so the fourth coefficient is positive one. The update below therefore adds the computed correction.

## 8. Solve the linear least-squares problem

At a fixed iteration, minimise the squared mismatch of the linearised model:

```math
Q(\Delta\boldsymbol{\theta})=
(\boldsymbol{\ell}-H\Delta\boldsymbol{\theta})^T
(\boldsymbol{\ell}-H\Delta\boldsymbol{\theta}).
```

Differentiate and set the gradient to zero:

```math
\nabla Q=-2H^T\boldsymbol{\ell}+2H^TH\Delta\boldsymbol{\theta}=\mathbf{0}.
```

The normal equations are

```math
\boxed{H^TH\Delta\boldsymbol{\theta}=H^T\boldsymbol{\ell}}.
```

There are four correction unknowns regardless of how many satellite rows are available. A unique linear correction requires

```math
M\geq4,\qquad \mathrm{rank}(H)=4.
```

With four independent rows, the linear system can be solved exactly, even when the observations contain noise. That does not mean the estimated position is exact. More than four satellites provide redundant measurements, which generally cannot all be fitted exactly.

Full rank guarantees a unique correction for the current linearised system. It does not guarantee a unique global solution to the original nonlinear problem or convergence from every starting point.

The existing solver performs this step as `lstsq(H, ell)`. It forms the normal equations and uses Gaussian elimination, rather than explicitly inverting a matrix. Its second argument is the residual vector, not the scalar clock bias b.

## 9. Update and repeat

Apply all four corrections:

```math
\begin{aligned}
x^{(k+1)}&=x^{(k)}+\Delta x,\\ y^{(k+1)}&=y^{(k)}+\Delta y,\\ z^{(k+1)}&=z^{(k)}+\Delta z,\\ b^{(k+1)}&=b^{(k)}+\Delta b.
\end{aligned}
```

Recompute the ranges, predictions, residuals, and H at the updated state. This procedure is a Gauss–Newton iteration for nonlinear least squares.

An illustrative stopping condition checks both position and clock corrections:

```math
\sqrt{(\Delta x)^2+(\Delta y)^2+(\Delta z)^2}\lt\tau_p,
\qquad
|\Delta b|\lt\tau_b.
```

For a synthetic demonstration, both thresholds could be 0.001 metres, with at most 20 iterations. These are proposed algorithm settings, not a claim of millimetre positioning accuracy. Noise and model errors can remain after the updates become small.

Start from an approximate receiver state, with zero clock bias if no better estimate is available. A poor initial position or weak satellite geometry can prevent convergence. The implementation must detect invalid ranges, a singular solve, non-finite values, and reaching the iteration limit without convergence.

After stopping, evaluate residuals using the final **nonlinear** model:

```math
r_i^{\mathrm{final}}=P_i-\rho_i(\hat x,\hat y,\hat z)-\hat b,
\qquad
\mathrm{RMS}=\sqrt{\frac{1}{M}\sum_{i=1}^{M}(r_i^{\mathrm{final}})^2}.
```

This RMS describes measurement fit; it is not the receiver's position error. With simulated data, the known true position makes a separate position-error check possible.

## 10. What changes if the fourth unknown is delta t?

With clock offset expressed in seconds, the model becomes

```math
f_i(x,y,z,\delta t)=\rho_i(x,y,z)+c\delta t,
\qquad
\frac{\partial f_i}{\partial\delta t}=c.
```

Use the same first three columns but change the fourth column to c:

```math
H_{t,i}=\begin{bmatrix}
\dfrac{x^{(k)}-X_i}{\rho_i^{(k)}} &
\dfrac{y^{(k)}-Y_i}{\rho_i^{(k)}} &
\dfrac{z^{(k)}-Z_i}{\rho_i^{(k)}} & c
\end{bmatrix},
\qquad
\Delta\boldsymbol{\theta}_t=\begin{bmatrix}\Delta x & \Delta y & \Delta z & \Delta\delta t\end{bmatrix}^T.
```

The two formulations are equivalent because

```math
\Delta b=c\,\Delta\delta t,
\qquad
H_b\Delta\boldsymbol{\theta}_b=H_t\Delta\boldsymbol{\theta}_t.
```

| Clock representation | Fourth matrix entry | Clock correction unit | Final clock output |
| --- | --- | --- | --- |
| Distance bias b | 1 | metres | Divide b by c to obtain seconds |
| Time offset delta t | c | seconds | Already in seconds |

Using b avoids a column of approximately 300 million alongside position coefficients whose absolute values are at most one. This improves column scaling for the educational normal-equation solver; it does not fix poor satellite geometry. We will solve for b and report both b and delta t.

Do not combine a fourth matrix entry of one with a clock correction in seconds. Do not estimate b and delta t as separate states: their columns would be dependent.

## 11. Planned implementation sequence

1. Accept M satellite coordinate triples and M corrected pseudoranges, all in metres.
2. Check matching lengths, finite inputs, and at least four satellites.
3. Initialise the receiver state `(x, y, z, b)`.
4. Calculate each range, predicted pseudorange, prefit residual, and geometry row.
5. Call `lstsq(H, ell)` to obtain `(dx, dy, dz, db)`.
6. Add the corrections to the current state.
7. Stop when both correction thresholds pass, or repeat from step 4 up to the iteration limit.
8. Recompute final nonlinear residuals and return the state, clock offset in seconds, iteration count, and convergence status.

For verification, generate synthetic measurements from a known state:

```math
P_i^{\mathrm{sim}}=
\sqrt{(x_{\mathrm{true}}-X_i)^2+(y_{\mathrm{true}}-Y_i)^2+(z_{\mathrm{true}}-Z_i)^2}
+b_{\mathrm{true}}+\varepsilon_i.
```

First use no noise and a nearby initial estimate to check recovery of the known state. Then add controlled noise and compare estimated position and clock bias against truth. Additional checks should cover both clock representations, derivative signs, inadequate geometry, and non-convergence.

This first version will use equal weights and simulated corrected observations. Processing raw GNSS data and estimating additional inter-system clock offsets are later extensions.
