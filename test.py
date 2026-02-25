import numpy as np
from estimate_camera import eight_point_F, sampson_error, ransac_F
from estimate_camera import decompose_E, triangulate_DLT
from estimate_camera import ransac_pnp, reprojection_errors, project_points, bundle_adjustment

# ----------------------------
# Helpers: geometry + synthesis
# ----------------------------

def make_K(fx=800.0, fy=800.0, cx=640.0, cy=360.0, skew=0.0):
    return np.array([[fx, skew, cx],
                     [0.0, fy,   cy],
                     [0.0, 0.0,  1.0]], dtype=float)

def rodrigues(axis, theta):
    axis = np.asarray(axis, dtype=float)
    axis /= (np.linalg.norm(axis) + 1e-12)
    x, y, z = axis
    K = np.array([[0, -z, y],
                  [z, 0, -x],
                  [-y, x, 0]], dtype=float)
    I = np.eye(3)
    return I + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)

def project(P, X):
    # P: (3,4), X: (N,3) in world coords
    N = X.shape[0]
    Xh = np.hstack([X, np.ones((N, 1))])  # (N,4)
    x = (P @ Xh.T).T  # (N,3)
    x = x[:, :2] / x[:, 2:3]
    return x

def skew(t):
    tx, ty, tz = t
    return np.array([[0, -tz, ty],
                     [tz, 0, -tx],
                     [-ty, tx, 0]], dtype=float)

def F_from_RtK(R, t, K1, K2):
    # Essential: E = [t]_x R, Fundamental: F = K2^{-T} E K1^{-1}
    E = skew(t) @ R
    F = np.linalg.inv(K2).T @ E @ np.linalg.inv(K1)
    # normalize scale for comparisons
    return F / (np.linalg.norm(F) + 1e-12)

def to_h(pts):
    return np.hstack([pts, np.ones((pts.shape[0], 1))])

def epipolar_residuals(F, pts1, pts2):
    x1 = to_h(pts1)
    x2 = to_h(pts2)
    r = np.sum(x2 * (x1 @ F.T), axis=1)  # x2^T F x1
    return r

def add_outliers(pts1, pts2, outlier_ratio, rng):
    N = pts1.shape[0]
    M = int(round(outlier_ratio * N))
    idx = rng.choice(N, size=M, replace=False)
    perm = rng.permutation(N)
    pts2_ol = pts2.copy()
    pts2_ol[idx] = pts2[perm[idx]]  # shuffle matches for outliers
    mask_inlier_gt = np.ones(N, dtype=bool)
    mask_inlier_gt[idx] = False
    return pts1, pts2_ol, mask_inlier_gt

def sample_scene_and_views(
    rng,
    N=200,
    image_size=(1280, 720),
    noise_px=1.0,
    outlier_ratio=0.0,
    baseline=0.5,
    f=900.0
):
    W, H = image_size
    K1 = make_K(fx=f, fy=f, cx=W/2, cy=H/2)
    K2 = make_K(fx=f, fy=f, cx=W/2, cy=H/2)

    # Random rotation (small-ish) and translation (baseline)
    axis = rng.normal(size=3)
    theta = rng.uniform(0.0, np.deg2rad(15.0))
    R = rodrigues(axis, theta)
    t_dir = rng.normal(size=3)
    t_dir /= (np.linalg.norm(t_dir) + 1e-12)
    t = t_dir * baseline

    # Camera matrices in world coords: P = K [R|t] with cam1 at world origin
    P1 = K1 @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K2 @ np.hstack([R, t.reshape(3, 1)])

    # Random 3D points in front of cam1 (Z positive)
    # Keep points in a frustum-like region
    X = np.empty((N, 3), dtype=float)
    X[:, 0] = rng.uniform(-1.0, 1.0, size=N) * 2.0
    X[:, 1] = rng.uniform(-1.0, 1.0, size=N) * 1.5
    X[:, 2] = rng.uniform(3.0, 8.0, size=N)

    pts1 = project(P1, X)
    pts2 = project(P2, X)

    # Add pixel noise
    pts1 += rng.normal(scale=noise_px, size=pts1.shape)
    pts2 += rng.normal(scale=noise_px, size=pts2.shape)

    # Add outliers (shuffle some correspondences)
    pts1, pts2, inlier_gt = add_outliers(pts1, pts2, outlier_ratio, rng)

    # Ground-truth F (scaled)
    F_gt = F_from_RtK(R, t, K1, K2)

    return {
        "pts1": pts1, "pts2": pts2,
        "K1": K1, "K2": K2,
        "R": R, "t": t,
        "F_gt": F_gt,
        "inlier_gt": inlier_gt,
        "image_size": image_size
    }

def normalized_fro_error(F, F_gt):
    # Compare up to scale: normalize both then L2 difference
    Fn = F / (np.linalg.norm(F) + 1e-12)
    Gn = F_gt / (np.linalg.norm(F_gt) + 1e-12)
    # also account for sign ambiguity
    e1 = np.linalg.norm(Fn - Gn)
    e2 = np.linalg.norm(Fn + Gn)
    return min(e1, e2)

def print_stats(name, vals):
    vals = np.asarray(vals)
    print(f"{name}: mean={vals.mean():.4g}, median={np.median(vals):.4g}, "
          f"p90={np.percentile(vals, 90):.4g}, max={vals.max():.4g}")
    
def sample_three_views(rng, N=500, image_size=(1280, 720), noise_px=1.0, outlier_ratio=0.25, f=900.0):
    W, H = image_size
    K = make_K(fx=f, fy=f, cx=W/2, cy=H/2)

    # World points
    X = np.empty((N, 3), dtype=float)
    X[:, 0] = rng.uniform(-2.0, 2.0, size=N)
    X[:, 1] = rng.uniform(-1.5, 1.5, size=N)
    X[:, 2] = rng.uniform(3.0, 8.0, size=N)

    # Camera 1: identity
    R1 = np.eye(3)
    t1 = np.zeros(3)

    # Camera 2 and 3: random poses
    def rand_pose(baseline):
        axis = rng.normal(size=3)
        theta = rng.uniform(0.0, np.deg2rad(15.0))
        R = rodrigues(axis, theta)
        t_dir = rng.normal(size=3)
        t_dir /= (np.linalg.norm(t_dir) + 1e-12)
        t = t_dir * baseline
        return R, t

    R2, t2 = rand_pose(baseline=0.8)
    R3, t3 = rand_pose(baseline=1.1)

    P1 = K @ np.hstack([R1, t1.reshape(3,1)])
    P2 = K @ np.hstack([R2, t2.reshape(3,1)])
    P3 = K @ np.hstack([R3, t3.reshape(3,1)])

    pts1 = project(P1, X) + rng.normal(scale=noise_px, size=(N,2))
    pts2 = project(P2, X) + rng.normal(scale=noise_px, size=(N,2))
    pts3 = project(P3, X) + rng.normal(scale=noise_px, size=(N,2))

    # Add outliers independently for pairwise matches (1<->2) and (1<->3)
    pts1_12, pts2_12, inlier12 = add_outliers(pts1, pts2, outlier_ratio, rng)
    pts1_13, pts3_13, inlier13 = add_outliers(pts1, pts3, outlier_ratio, rng)

    return dict(
        K=K, X_gt=X,
        R1=R1, t1=t1, R2=R2, t2=t2, R3=R3, t3=t3,
        pts1=pts1, pts2=pts2, pts3=pts3,
        pts1_12=pts1_12, pts2_12=pts2_12, inlier12=inlier12,
        pts1_13=pts1_13, pts3_13=pts3_13, inlier13=inlier13,
    )

# ----------------------------
# Round 1 test: 8-point F
# ----------------------------

def test_round1_eight_point(eight_point_F, seed=0):
    rng = np.random.default_rng(seed)
    data = sample_scene_and_views(rng, N=200, noise_px=0.5, outlier_ratio=0.0, baseline=0.8, f=900.0)
    pts1, pts2 = data["pts1"], data["pts2"]
    F_gt = data["F_gt"]

    F = eight_point_F(pts1, pts2)
    errF = normalized_fro_error(F, F_gt)

    r = epipolar_residuals(F, pts1, pts2)
    print("\n[Round 1] Eight-point on clean-ish data")
    print(f"F error (scale/sign-invariant Fro): {errF:.4g}")
    print_stats("epipolar residual x2^T F x1", np.abs(r))

    # Basic sanity checks (tune thresholds if needed)
    assert np.isfinite(F).all()
    assert errF < 0.5, "F too far from ground truth (check normalization/denormalization/rank2)."
    assert np.median(np.abs(r)) < 5.0, "Residual too large for low-noise synthetic data."

# ----------------------------
# Round 2 test: RANSAC + Sampson
# ----------------------------

def test_round2_ransac(ransac_F, sampson_error, seed=1):
    rng = np.random.default_rng(seed)
    data = sample_scene_and_views(rng, N=500, noise_px=1.0, outlier_ratio=0.35, baseline=0.8, f=900.0)
    pts1, pts2 = data["pts1"], data["pts2"]
    inlier_gt = data["inlier_gt"]
    F_gt = data["F_gt"]

    # Threshold: Sampson is roughly "squared pixels" when using pixel coords.
    # Start with 2px -> tau = 4.0, adjust if your sampson uses slightly different scale.
    tau = 4.0
    F, inliers = ransac_F(pts1, pts2, threshold=tau, max_iters=5000, p=0.99)

    print("\n[Round 2] RANSAC with outliers")
    print(f"Estimated inliers: {inliers.sum()} / {len(inliers)} = {inliers.mean():.3f}")

    # Measure quality on GT inliers and estimated inliers
    errF = normalized_fro_error(F, F_gt)
    print(f"F error (scale/sign-invariant Fro): {errF:.4g}")

    errs_all = sampson_error(F, pts1, pts2)
    print_stats("Sampson error (all)", errs_all)
    print_stats("Sampson error (estimated inliers)", errs_all[inliers])

    # Compare estimated inliers to GT (not perfect, but should be decent)
    tp = np.logical_and(inliers, inlier_gt).sum()
    fp = np.logical_and(inliers, ~inlier_gt).sum()
    fn = np.logical_and(~inliers, inlier_gt).sum()
    prec = tp / (tp + fp + 1e-12)
    rec  = tp / (tp + fn + 1e-12)
    print(f"Precision={prec:.3f}, Recall={rec:.3f} (vs synthetic GT outliers)")

    assert np.isfinite(F).all()
    assert inliers.sum() > 0.4 * len(inliers), "Too few inliers; check threshold, sampson, or ransac."
    assert np.median(errs_all[inliers]) < 10.0, "Inlier Sampson error too big; likely a bug or bad threshold."
    assert errF < 1.0, "F far from GT; check denormalization, rank2, or ransac scoring."

# ----------------------------
# Round 3 (optional): Pose + cheirality
# ----------------------------
# This requires your functions:
#   E = K2.T @ F @ K1
#   decompose_E(E) -> list of (R,t) candidates
#   triangulate_DLT(P1, P2, x1, x2) -> X (3,)
#   cheirality_select(candidates, pts1, pts2, K1, K2) -> (R,t)
#
# If you have them, this test checks cheirality and pose angle errors.

def rotation_angle_deg(R_est, R_gt):
    R = R_est @ R_gt.T
    cos = (np.trace(R) - 1) / 2
    cos = np.clip(cos, -1.0, 1.0)
    return np.degrees(np.arccos(cos))

def unit(v):
    v = np.asarray(v, dtype=float).reshape(-1)
    return v / (np.linalg.norm(v) + 1e-12)

def test_round3_pose_from_F(
    eight_point_F,
    sampson_error,
    ransac_F,
    decompose_E,
    triangulate_DLT,
    seed=2
):
    rng = np.random.default_rng(seed)
    data = sample_scene_and_views(rng, N=400, noise_px=1.0, outlier_ratio=0.25, baseline=1.0, f=900.0)
    pts1, pts2 = data["pts1"], data["pts2"]
    K1, K2 = data["K1"], data["K2"]
    R_gt, t_gt = data["R"], data["t"]

    # Estimate F robustly
    tau = 4.0
    F, inliers = ransac_F(pts1, pts2, threshold=tau, max_iters=5000, p=0.99)
    pts1_in = pts1[inliers]
    pts2_in = pts2[inliers]

    # Build E and enforce essential structure (you can do inside your code)
    E = K2.T @ F @ K1

    candidates = decompose_E(E)  # list of (R,t_dir) with t as direction or arbitrary scale

    # Cheirality selection by triangulating a subset
    P1 = np.hstack([np.eye(3), np.zeros((3, 1))])

    best = None
    best_count = -1
    best_stats = None

    # Use normalized camera coordinates for triangulation (why?)
    x1n = (np.linalg.inv(K1) @ to_h(pts1_in).T).T
    x2n = (np.linalg.inv(K2) @ to_h(pts2_in).T).T

    # Take a subset for speed
    M = min(50, x1n.shape[0])
    idx = np.arange(M)

    for (R, t) in candidates:
        t = t.reshape(3, 1)
        P2 = np.hstack([R, t])

        # cheirality_select
        count = 0
        for i in idx:
            X = triangulate_DLT(P1, P2, x1n[i], x2n[i])  # should return 3D in cam1 frame
            z1 = X[2]
            X2 = (R @ X.reshape(3, 1) + t).reshape(3)
            z2 = X2[2]
            if z1 > 0 and z2 > 0:
                count += 1

        if count > best_count:
            best_count = count
            best = (R, t.reshape(3))
            best_stats = count

    R_est, t_est = best
    ang_R = rotation_angle_deg(R_est, R_gt)
    ang_t = np.degrees(np.arccos(np.clip(unit(t_est) @ unit(t_gt), -1.0, 1.0)))
    ang_t = min(ang_t, 180 - ang_t)  # account for sign ambiguity

    print("\n[Round 3] Pose + cheirality")
    print(f"Cheirality positives (subset): {best_stats} / {M}")
    print(f"Rotation error (deg): {ang_R:.3f}")
    print(f"Translation direction error (deg, sign-free): {ang_t:.3f}")

    assert best_count > 0.7 * M, "Cheirality weak; decomposition/triangulation likely wrong."
    assert ang_R < 10.0, "Rotation error too large for this synthetic setup."
    assert ang_t < 15.0, "Translation direction error too large (check cheirality + t extraction)."


def test_round4_pnp_ransac(ransac_pnp, seed=3):
    rng = np.random.default_rng(seed)

    # Create camera + points
    W, H = 1280, 720
    K = make_K(fx=900.0, fy=900.0, cx=W/2, cy=H/2)

    # Generate random 3D points
    N = 400
    X = np.empty((N, 3), dtype=float)
    X[:, 0] = rng.uniform(-2.0, 2.0, size=N)
    X[:, 1] = rng.uniform(-1.5, 1.5, size=N)
    X[:, 2] = rng.uniform(3.0, 8.0, size=N)

    # Generate camera poses 
    axis = rng.normal(size=3)
    theta = rng.uniform(0.0, np.deg2rad(20.0))
    R_gt = rodrigues(axis, theta)
    t_dir = rng.normal(size=3); t_dir /= (np.linalg.norm(t_dir) + 1e-12)
    t_gt = t_dir * 1.0

    uv = project_points(K, R_gt, t_gt, X)
    uv += rng.normal(scale=1.0, size=uv.shape)

    # Add outliers in 2D by shuffling a subset
    outlier_ratio = 0.3
    idx = rng.choice(N, size=int(outlier_ratio*N), replace=False)
    perm = rng.permutation(N)
    uv[idx] = uv[perm[idx]]

    R_est, t_est, inliers = ransac_pnp(X, uv, K, threshold_px=3.0, max_iters=5000, p=0.99)

    print("\n[Round 4] PnP + RANSAC")
    print(f"Estimated inliers: {inliers.sum()} / {N} = {inliers.mean():.3f}")

    ang_R = rotation_angle_deg(R_est, R_gt)
    ang_t = np.degrees(np.arccos(np.clip(unit(t_est) @ unit(t_gt), -1.0, 1.0)))
    ang_t = min(ang_t, 180 - ang_t)

    errs = reprojection_errors(K, R_est, t_est, X, uv)
    print(f"Rotation error (deg): {ang_R:.3f}")
    print(f"Translation dir error (deg): {ang_t:.3f}")
    print_stats("Reproj error (all)", errs)
    print_stats("Reproj error (inliers)", errs[inliers])

    assert inliers.sum() > 0.5 * N, "Too few inliers; check PnP or threshold."
    assert ang_R < 10.0, "Rotation error too large."


def test_round5_three_view_incremental(ransac_F, sampson_error, decompose_E, triangulate_DLT, ransac_pnp, seed=4):
    rng = np.random.default_rng(seed)
    data = sample_three_views(rng, N=600, noise_px=1.0, outlier_ratio=0.25, f=900.0)
    K = data["K"]
    X_gt = data["X_gt"]

    # --- Step 1: initialize using views 1 & 2 (like your Round 3) ---
    pts1, pts2 = data["pts1_12"], data["pts2_12"]

    F12, in12 = ransac_F(pts1, pts2, threshold=4.0, max_iters=5000, p=0.99)
    orig_idx = np.where(in12)[0]          # indices into the original arrays
    pts1_in  = pts1[orig_idx]
    pts2_in  = pts2[orig_idx]

    E12 = K.T @ F12 @ K
    candidates = decompose_E(E12)

    # Choose pose by cheirality
    x1n = (np.linalg.inv(K) @ to_h(pts1_in).T).T
    x2n = (np.linalg.inv(K) @ to_h(pts2_in).T).T
    P1 = np.hstack([np.eye(3), np.zeros((3,1))])

    best = None
    best_count = -1
    M = min(80, x1n.shape[0])

    for (R, t) in candidates:
        t = t.reshape(3,1)
        P2 = np.hstack([R, t])
        count = 0
        for i in range(M):
            X = triangulate_DLT(P1, P2, x1n[i], x2n[i])
            z1 = X[2]
            z2 = (R @ X.reshape(3,1) + t).reshape(3)[2]
            if z1 > 0 and z2 > 0:
                count += 1
        if count > best_count:
            best_count = count
            best = (R, t.reshape(3))

    R2_est, t2_est = best

    # --- Step 2: triangulate a set of 3D points from inlier matches (1&2) ---
    P2_est = np.hstack([R2_est, t2_est.reshape(3,1)])

    X_est = []
    idx_est = []

    for j in range(len(pts1_in)):
        X = triangulate_DLT(P1, P2_est, x1n[j], x2n[j])
        X_est.append(X)
        idx_est.append(orig_idx[j])

    X_est = np.array(X_est)
    idx_est = np.array(idx_est)

    # Take the corresponding uv in view3 from GT data (same index in original array)
    # We need to map back to original indices: simplest is to just reuse a subset aligned by index.
    # Here we approximate by using first len(X_est) points from GT (works for synthetic sanity).
    if len(X_est) < 50:
        raise AssertionError("Too few triangulated points; check pose selection/triangulation.")

    # For a clean incremental test, use GT correspondences for view3 (no outliers) on the same 3D points:
    # We'll just take the first M3 points of X_est and project GT view3 measurements.
    M3 = min(200, len(X_est))
    X_for_pnp = X_est[:M3]
    uv_for_pnp = data["pts3"][idx_est[:M3]]  # synthetic alignment convenience

    # --- Step 3: estimate camera 3 pose using PnP + RANSAC ---
    R3_est, t3_est, in3 = ransac_pnp(X_for_pnp, uv_for_pnp, K, threshold_px=3.0, max_iters=5000, p=0.99)

    print("\n[Round 5] 3-view incremental (init 1-2, PnP for 3)")
    ang_R3 = rotation_angle_deg(R3_est, data["R3"])
    ang_t3 = np.degrees(np.arccos(np.clip(unit(t3_est) @ unit(data["t3"]), -1.0, 1.0)))
    ang_t3 = min(ang_t3, 180 - ang_t3)
    print(f"Cam3 rotation error (deg): {ang_R3:.3f}")
    print(f"Cam3 translation dir error (deg): {ang_t3:.3f}")
    print(f"PnP inliers: {in3.sum()} / {len(in3)}")

    assert ang_R3 < 15.0, "Cam3 rotation error too large (check PnP)."


def test_round6_bundle_adjustment(bundle_adjustment, seed=6):
    rng = np.random.default_rng(seed)
    data = sample_three_views(rng, N=200, noise_px=1.0, outlier_ratio=0.0, f=900.0)
    K = data["K"]
    X_gt = data["X_gt"]

    # Build observations: each point j seen in all 3 cameras
    obs = []
    for j in range(len(X_gt)):
        obs.append((0, j, data["pts1"][j]))
        obs.append((1, j, data["pts2"][j]))
        obs.append((2, j, data["pts3"][j]))

    # Initial guess: perturb GT (or use your estimated poses + triangulated points)
    R1, t1 = np.eye(3), np.zeros(3)
    R2, t2 = data["R2"], data["t2"]
    R3, t3 = data["R3"], data["t3"]

    # Slightly noisy initialization
    X0 = X_gt + rng.normal(scale=0.05, size=X_gt.shape)
    R2_0, t2_0 = R2, t2 + rng.normal(scale=0.05, size=3)
    R3_0, t3_0 = R3, t3 + rng.normal(scale=0.05, size=3)

    # Run BA (camera0 fixed inside BA)
    (R2_opt, t2_opt, R3_opt, t3_opt, X_opt,
     cost_before, cost_after) = bundle_adjustment(K, R2_0, t2_0, R3_0, t3_0, X0, obs)

    print("\n[Round 6] Bundle adjustment (3 cams)")
    print("cost before:", cost_before)
    print("cost after:", cost_after)

    assert cost_after < 0.5 * cost_before, "BA didn't reduce reprojection error enough."

# ----------------------------
# Entry point
# ----------------------------

def run_all_tests(
    eight_point_F,
    sampson_error=None,
    ransac_F=None,
    decompose_E=None,
    triangulate_DLT=None,
    ransac_pnp=None,
    bundle_adjustment=None
):
    test_round1_eight_point(eight_point_F)

    if sampson_error is not None and ransac_F is not None:
        test_round2_ransac(ransac_F, sampson_error)

    if (sampson_error is not None and ransac_F is not None
        and decompose_E is not None and triangulate_DLT is not None):
        test_round3_pose_from_F(eight_point_F, sampson_error, ransac_F, decompose_E, triangulate_DLT)

    if ransac_pnp is not None:
        test_round4_pnp_ransac(ransac_pnp)

    if ransac_pnp is not None:
        test_round5_three_view_incremental(ransac_F, sampson_error, decompose_E, triangulate_DLT, ransac_pnp)

    if bundle_adjustment is not None:
        test_round6_bundle_adjustment(bundle_adjustment)


    print("\n✅ Done.")



if __name__ == "__main__":
    run_all_tests(
        eight_point_F=eight_point_F,
        sampson_error=sampson_error,
        ransac_F=ransac_F,
        decompose_E=decompose_E,
        triangulate_DLT=triangulate_DLT,
        ransac_pnp=ransac_pnp,
        bundle_adjustment=bundle_adjustment
    )
