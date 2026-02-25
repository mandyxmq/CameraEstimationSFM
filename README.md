# Camera Estimation Practice

A hands-on coding exercise for implementing classical camera estimation algorithms from scratch in Python and NumPy.

The tests in this repo are **AI-curated** — designed to progressively build up a full structure-from-motion pipeline, one algorithm at a time. Each round isolates a concept, validates your implementation on synthetic data, and gates the next round on the previous one passing.

## What You'll Implement

Fill in the 10 core functions in [`estimate_camera.py`](estimate_camera.py):

| # | Function | Concept |
|---|----------|---------|
| 1 | `normalize_points` | Isotropic scaling for DLT conditioning |
| 2 | `eight_point_F` | 8-point algorithm for the fundamental matrix |
| 3 | `enforce_rank2` | SVD-based rank-2 enforcement |
| 4 | `sampson_error` | Sampson distance (first-order geometric error) |
| 5 | `ransac_F` | RANSAC with adaptive iteration count |
| 6 | `decompose_E` | Essential matrix decomposition into R, t |
| 7 | `triangulate_DLT` | Linear triangulation via DLT |
| 8 | `pnp_dlt` | Perspective-n-Point via DLT |
| 9 | `ransac_pnp` | RANSAC + inlier refit for PnP |
| 10 | `bundle_adjustment` | Levenberg-Marquardt bundle adjustment (3 cameras) |

Helper functions (`so3_exp`, `project_point`, `project_points`, `reprojection_errors`, `refine_pose_gauss_newton`) are provided.

## Test Rounds

Run the tests at any stage — they skip rounds whose dependencies aren't implemented yet.

```
python test.py
```

| Round | Tests | Depends on |
|-------|-------|------------|
| 1 | 8-point F on clean data | `eight_point_F` |
| 2 | RANSAC F with 35% outliers | + `sampson_error`, `ransac_F` |
| 3 | Pose recovery + cheirality | + `decompose_E`, `triangulate_DLT` |
| 4 | PnP + RANSAC | + `pnp_dlt`, `ransac_pnp` |
| 5 | 3-view incremental SfM | all of the above |
| 6 | Bundle adjustment | + `bundle_adjustment` |
