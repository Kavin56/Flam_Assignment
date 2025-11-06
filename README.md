# 📘 Research & Development / AI — Parametric Curve Fitting Assignment

## 1) Problem Overview

Find the unknown parameters (θ, M, X) in the parametric curve equations:

$$
x = t \cdot \cos(\theta) - e^{M|t|} \cdot \sin(0.3t) \cdot \sin(\theta) + X
$$

$$
y = 42 + t \cdot \sin(\theta) + e^{M|t|} \cdot \sin(0.3t) \cdot \cos(\theta)
$$

**Parameter Ranges:**
- $0° < \theta < 50°$
- $-0.05 < M < 0.05$
- $0 < X < 100$
- $6 < t < 60$ (t-domain)

**Given:** A dataset of (x, y) points that lie on this curve for t in [6, 60].

---

## 2) Conceptual Explanation

This curve is a rotated and translated oscillating trajectory:

- The linear terms \(t\cos\theta\) and \(t\sin\theta\) form a baseline straight line.
- The term \(e^{M|t|}\sin(0.3t)\) injects a sinusoidal oscillation with slowly varying amplitude controlled by \(M\) (growth/decay).
- Rotation by \(\theta\) mixes these components; \(X\) and constant 42 shift the curve horizontally and vertically.

Goal: recover numeric \((\theta, M, X)\) that best match the observed data.

---

## 3) Approach (What we did and Why)

### 3.1 Coordinate Transform Insight

The key insight is to transform the problem to a simpler coordinate system:

1. **Translation:** Subtract X from x and 42 from y
2. **Rotation:** Rotate by -θ to align with the parameter space

After transformation:
- $u = (x - X) \cos(\theta) + (y - 42) \sin(\theta) \approx t$
- $v = -(x - X) \sin(\theta) + (y - 42) \cos(\theta) \approx e^{Mu} \sin(0.3u)$

This reduces the problem to fitting the simpler function $v = e^{Mu} \sin(0.3u)$ after finding the correct transformation parameters.

### 3.2 Optimization Strategy

We use multiple optimization approaches:

#### A) Residuals

- **UV-space residual:**
  \[ r_{uv}(\theta, M, X) = v_{obs} - e^{Mu}\sin(0.3u) \]
  where \(u=(x-X)\cos\theta+(y-42)\sin\theta\) and \(v=-(x-X)\sin\theta+(y-42)\cos\theta\).

- **Direct-XY residual:**
  \[ r_{xy}(\theta, M, X) = (x,y)_{obs} - (x,y)_{model}(t{=}u) \]

- **Hybrid residual:** weighted concatenation of the two to leverage both spaces.

#### B) Least Squares Optimization
- **Method:** Scipy's `least_squares` with different algorithms:
  - Levenberg-Marquardt (`lm`)
  - Trust Region Reflective (`trf`)
  - Dogleg (`dogbox`)
- **Loss Functions:** `linear`, `huber`, `soft_l1`
- **Residual Functions:**
  1. **UV Space Residual:** Fit in transformed (u,v) coordinates
  2. **Direct XY Residual:** Fit directly in original (x,y) space
  3. **Hybrid Residual:** Combine both approaches

#### C) Global & Multi-start Optimization
- **Differential Evolution:** Global search to avoid local minima
- **Basin Hopping:** Multi-start optimization with local minimization

#### D) Refinement
- Use best solution as starting point for additional fine-tuning
- Extended search with perturbations around best solution

### 3.3 Objective / Metric

The primary metric is **L1 distance** between uniformly sampled curve points and the data:

1. Generate 1000 uniformly spaced t values in [6, 60]
2. Compute corresponding (x, y) points on the fitted curve
3. For each curve point, find nearest data point using L1 (Manhattan) distance
4. Return mean L1 distance

This directly measures how well the curve matches the data distribution.

### 3.4 Comprehensive Testing

The script tests:
- Multiple initial guesses (systematic grid + random sampling)
- Different residual formulations
- Various optimization algorithms
- Different loss functions
- Global and local optimization methods
- Refinement iterations

---

## 4) Repository Structure

- `solve_params.ipynb`: Structured, end-to-end notebook (prints results & plots)
- `parameter_fitting_comprehensive.py`: Comprehensive CLI script
- `xy_data.csv`: Provided data
- `best_solution.txt`: Optional text output when running the script
- `optimization_log.txt`: Optional log when running the script

---

## 5) How to Run

### Option A: Notebook (recommended)
1. Open `solve_params.ipynb`
2. Run all cells
3. The notebook will search with multiple strategies, print the best result and LaTeX string, and plot diagnostics.

### Option B: Python script
```bash
python parameter_fitting_comprehensive.py
```
Notes:
- Let it run for thoroughness; the best-so-far is printed (and optionally written to `best_solution.txt`).
- You can stop anytime (Ctrl+C) and still retain the latest best.

---

## 6) Results

| Parameter | Estimated Value              | Interpretation                               |
| :-------- | :--------------------------- | :-------------------------------------------- |
| \(\theta\) | 29.9999735° (≈ 0.523598 rad) | Curve rotated ~30° from x-axis                |
| \(M\)     | 0.0299999938                | Slight amplitude growth of oscillation        |
| \(X\)     | 54.9999977449               | Horizontal shift of the entire curve          |
| \(t\)     | 6 < t < 60                  | Parametric domain                             |

| Metric                    | Value        |
| :------------------------ | ----------: |
| Mean L1 distance (XY)     | 0.02581908  |
| Best method (from search) | residuals_uv_space_differential_evolution |

### 6.1 Submission (LaTeX)

```
\left(t*\cos(0.523598)-e^{0.030000\left|t\right|}\cdot\sin(0.3t)\sin(0.523598)+54.999998,\
42+t*\sin(0.523598)+e^{0.030000\left|t\right|}\cdot\sin(0.3t)\cos(0.523598)\right)
```

### 6.2 Split Equations

```
x(t) = t\cos(0.523598) - e^{0.030000|t|}\sin(0.3t)\sin(0.523598) + 54.999998
y(t) = 42 + t\sin(0.523598) + e^{0.030000|t|}\sin(0.3t)\cos(0.523598)
```

### 6.3 Top-3 Best Trials (from run logs)

| Rank | L1 Distance   | Method                                        | θ (deg)       | M            | X             |
| :--- | :------------ | :-------------------------------------------- | :------------ | :----------- | :------------ |
| 1    | 0.0258190805  | residuals_uv_space_differential_evolution     | 29.9999735077 | 0.0299999938 | 54.9999977449 |
| 2    | 0.0258190855  | residuals_direct_xy_least_squares_dogbox_soft_l1_init1 | 29.9999729323 | 0.0299999969 | 54.9999982132 |
| 3    | 0.0258190805  | residuals_uv_space_differential_evolution     | 29.9999735077 | 0.0299999938 | 54.9999977449 |

### Where to submit
Paste the LaTeX string into the Desmos calculator:
`https://www.desmos.com/calculator/rfj91yrxob`

---

## 7) Why This Approach Works

1. **Transformation reduces complexity:** Instead of fitting 3 parameters in a complex 2D curve, we transform to a simpler 1D function relationship.

2. **Multiple methods ensure robustness:** Different optimizers can find different local minima, and we keep the globally best solution.

3. **Proper evaluation metric:** L1 distance directly measures assignment criteria, ensuring we optimize for what matters.

4. **Comprehensive search:** Multiple initial guesses and methods reduce the chance of missing the global optimum.

---

## 8) Expected Runtime

- **Quick test:** 1-5 minutes (100 iterations)
- **Comprehensive:** 1-3 hours (full search)
- **Extended:** Can run overnight for maximum thoroughness

The script automatically saves/prints the best solution; you can stop at any time and still have the best-so-far.

---

## 9) Screenshots (leave space to insert)

### 8.1 Result screenshot (printed values & LaTeX)

<paste or drag your result screenshot here>

### 8.2 Plot screenshot – transformed (u,v) vs model and XY vs fitted curve

<paste or drag your plot screenshot(s) here>

---

## 10) Reproducibility Notes

- Random seeds are fixed where applicable
- Parameter bounds strictly enforced
- Notebook prints a LaTeX string directly usable in Desmos

