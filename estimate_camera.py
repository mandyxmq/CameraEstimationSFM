import numpy as np
import random

def so3_exp(w):
    """w: (3,) axis-angle vector. Returns R (3,3)."""
    theta = np.linalg.norm(w)
    if theta < 1e-12:
        return np.eye(3)
    axis = w / theta
    x, y, z = axis
    K = np.array([[0, -z, y],
                  [z, 0, -x],
                  [-y, x, 0]], dtype=float)
    I = np.eye(3)
    return I + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)

def normalize_points(pts):
    # pts is Nx3 (homogeneous 2D points)
    # Compute mean and center
    # Compute average distance from origin and scale so avg dist = sqrt(2)
    # Build 3x3 similarity transform T_normalize
    # Return T_normalize, normalized pts
    
    N = len(pts)
    mux = np.mean(pts[:,0])
    muy = np.mean(pts[:,1])

    dist = np.mean(np.sqrt((pts[:,0] - mux)*(pts[:,0] - mux)+(pts[:,1] - muy)*(pts[:,1] - muy)))
    s =  np.sqrt(2) / dist

    T_normalize = np.array([[s, 0, -s*mux],[0,s,-s*muy],[0,0,1]])

    pts_normalized = (T_normalize @ pts.T).T
   
    return T_normalize, pts_normalized

def eight_point_F(pts1, pts2):
    # x2^T F x1 = 0
    # Homogenize pts1 and pts2 to Nx3
    # Normalize both sets of points (call normalize_points)
    # Build Nx9 matrix A from outer products of normalized correspondences
    # Solve via SVD: F = last row of Vt reshaped to 3x3
    # Enforce rank-2 (call enforce_rank2)
    # Denormalize: F = T2.T @ F_normalized @ T1

    N = len(pts1)
    pts1_h = np.hstack([pts1, np.ones((N, 1))])
    pts2_h = np.hstack([pts2, np.ones((N, 1))])

    T1, pts1_normalized = normalize_points(pts1_h)
    T2, pts2_normalized = normalize_points(pts2_h)

    A = np.zeros((N, 9))
    for i in range(N):
        u1,v1 = pts1_normalized[i,:2]
        u2,v2 = pts2_normalized[i,:2]
        A[i, :] = np.array([u1*u2, v1*u2, u2, u1*v2, v1*v2, v2, u1, v1, 1])


    U, S, Vt = np.linalg.svd(A)
    v = Vt[-1]
    F = T2.T @ np.reshape(v, (3, 3)) @ T1
    
    F = enforce_rank2(F)

    return F

def enforce_rank2(F):
    # SVD, zero out smallest singular value, reconstruct
    U, S, Vt = np.linalg.svd(F)
    S[-1] = 0
    Smat = np.diag(S)

    return U @ Smat @ Vt

def sampson_error(F, pts1, pts2):
    # pts1, pts2 are Nx2
    # Homogenize to Nx3
    # For each point pair compute: (x2^T F x1)^2 / (||Fx1||_xy^2 + ||F^T x2||_xy^2)
    # Return array of N errors
    
    N = len(pts1)

    pts1_h = np.hstack([pts1, np.ones((N, 1))])
    pts2_h = np.hstack([pts2, np.ones((N, 1))])

    error = np.zeros(N,)
    for i in range(N):
        x2 = pts2_h[i]
        x1 = pts1_h[i]
        error[i] = np.dot(x2, F@x1)**2 / ((F@x1)[0]**2 + (F@x1)[1]**2 + (F.T@x2)[0]**2 + (F.T@x2)[1]**2)

    return error

def ransac_F(pts1, pts2, threshold, max_iters, p):
    # Sample 8 points, estimate F, compute Sampson errors on all points
    # Track best F and inlier mask
    # Adaptively update iteration count using RANSAC formula
    # Return best_F, best_inliers
    N = len(pts1)
    best_F = np.zeros((3, 3))
    samplenum = 8
    best_inliers = None
    best_count = 0

    # pts1_h = np.hstack([pts1, np.ones((N, 1))])
    # pts2_h = np.hstack([pts2, np.ones((N, 1))])

    #for i in range(max_iters):

    best_k = max_iters
    it = 0
    while it < best_k:
        idx = random.sample(range(N), samplenum)
        curpts1 = pts1[idx]
        curpts2 = pts2[idx]
        curF = eight_point_F(curpts1, curpts2)
        cur_error = sampson_error(curF, pts1, pts2)
        
        # # a simpler error
        # temp = curF @ pts1_h.T
        # cur_error = pts2_h[:,0]*temp[0,:] + pts2_h[:,1]*temp[1,:] + pts2_h[:,2]*temp[2,:]

        cur_inliers = cur_error < threshold
        cur_count = int(np.sum(cur_inliers))
        if cur_count > best_count:
            best_F = curF
            best_inliers = cur_inliers
            best_count = cur_count

            # update iterations
            w = best_count / float(N)
            denom = 1 - w**samplenum
            denom = min(max(denom, 1e-12), 1 - 1e-12)
            best_k = min(best_k, int(np.log(1 - p) / np.log(denom)))

        it+=1

    return best_F, best_inliers


def decompose_E(E):
    # 1) SVD of E
    # 2) Enforce essential structure: singular values -> (1,1,0), re-SVD
    # 3) Define W = [[0,-1,0],[1,0,0],[0,0,1]]
    # 4) Two candidate rotations: R1 = U W Vt, R2 = U W^T Vt
    # 5) Fix improper rotations (det = -1 -> negate)
    # 6) Translation: t = U[:, 2] (last column of U)
    # Return list of 4 candidates: [(R1,t),(R1,-t),(R2,t),(R2,-t)]
    
    U, S, Vt = np.linalg.svd(E)
    E_enf = U @ np.diag([1.0, 1.0, 0.0]) @ Vt
    U, S, Vt = np.linalg.svd(E_enf)
    W = np.array([[0, -1, 0],[1, 0, 0],[0, 0, 1]])

    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt
    
    if np.linalg.det(R1) < 0:
        R1 = -R1
    if np.linalg.det(R2) < 0:
        R2 = -R2

    t = U[:, -1]

    return (R1, t), (R1, -t), (R2, t), (R2, -t)

def triangulate_DLT(P1, P2, x1, x2):
    """
    P1, P2: (3,4) camera matrices
    x1, x2: (3,) homogeneous image points (u,v,1)
    Returns:
      X: (3,) 3D point in Euclidean coords (in the coordinate frame of P1/P2)
    """
    # Build 4x4 matrix A:
    #   row 0: x1[0]*P1[2] - P1[0]
    #   row 1: x1[1]*P1[2] - P1[1]
    #   row 2: x2[0]*P2[2] - P2[0]
    #   row 3: x2[1]*P2[2] - P2[1]
    # Solve via SVD: X = last row of Vt, dehomogenize

    A = np.zeros((4, 4))
    A[0] = x1[0]*P1[2] - P1[0]
    A[1] = x1[1]*P1[2] - P1[1]
    A[2] = x2[0]*P2[2] - P2[0]
    A[3] = x2[1]*P2[2] - P2[1]

    U, S, Vt = np.linalg.svd(A)
    X = Vt[-1]
    X = X / (X[3] + 1e-12)

    return X[:3]

def project_point(K, R, t, X):
    """Project one 3D point X (3,) to pixel uv (2,)."""
    X_c = R @ X + t
    X_nor = K @ X_c
    X_nor = X_nor / (X_nor[-1] + 1e-12)

    return X_nor[:2]
    

def project_points(K, R, t, X):
    X_c = (R @ X.T).T + t
    X_nor = (K @ X_c.T).T
    X_nor = X_nor / (X_nor[:,-1] + 1e-12)[:,None]
    
    return X_nor[:, :2]

def reprojection_errors(K, R, t, X, uv_obs):
    uv_proj = project_points(K, R, t, X)
    d = uv_proj - uv_obs
    return np.sum(d * d, axis=1)

def pnp_dlt(X_world, uv, K):
    # Needs at least 6 correspondences
    # Homogenize uv, unproject to normalized camera coords via K^{-1}
    # Build 2Nx12 DLT matrix A from 3D-2D correspondences
    # Solve via SVD: P = last row of Vt reshaped to 3x4
    # Extract R,t: SVD of P[:,:3], enforce SO(3), recover scale, t = p4/scale

    N = len(X_world)
    A = np.zeros((2*N, 12))

    uv_h = np.hstack([uv, np.ones((N, 1))]) 
    # convert to normalized camera coordinates
    x = (np.linalg.inv(K) @ uv_h.T).T
    x = x / (x[:, 2:3] + 1e-12)
    u = x[:, 0]
    v = x[:, 1]

    Xh = np.hstack([X_world, np.ones((N, 1))])  # (N,4)

    for i in range(N):
        Xi = Xh[i]  # (4,)
        # u*(p3^T Xi) - (p1^T Xi) = 0
        # v*(p3^T Xi) - (p2^T Xi) = 0
        A[2*i,   0:4] = -Xi
        A[2*i,   8:12] = u[i] * Xi
        A[2*i+1, 4:8] = -Xi
        A[2*i+1, 8:12] = v[i] * Xi

    U, S, Vt = np.linalg.svd(A)
    P = Vt[-1].reshape((3, 4))

    M = P[:, :3]
    p4 = P[:, 3]
    
    U, S, Vt = np.linalg.svd(M)
    R = U @ Vt

    # Fix improper rotation
    if np.linalg.det(R) < 0:
        R = -R
        p4 = -p4

    s = (S[0] + S[1] + S[2])/3
    t = p4 / s
    
    return R, t

def refine_pose_gauss_newton(K, R_init, t_init, X_world, uv_obs,
                             iters=10, lam=1e-3, eps=1e-6):
    """
    Pose-only LM/GN refinement.
    X_world: (N,3), uv_obs: (N,2)
    Returns refined R,t.
    """
    R = R_init.copy()
    t = t_init.copy().reshape(3)

    X_world = np.asarray(X_world, float)
    uv_obs = np.asarray(uv_obs, float)
    N = X_world.shape[0]
    if N < 4:
        return R, t

    def residual(R, t):
        r = np.zeros((2 * N,), dtype=float)
        for i in range(N):
            uv_hat = project_point(K, R, t, X_world[i])
            r[2*i:2*i+2] = (uv_hat - uv_obs[i])
        return r

    for _ in range(iters):
        r0 = residual(R, t)
        cost0 = float(r0 @ r0)

        # Numeric Jacobian wrt 6 params: [dw(3), dt(3)]
        J = np.zeros((2 * N, 6), dtype=float)

        # d/dw
        for j in range(3):
            dw = np.zeros(3); dw[j] = eps
            Rp = so3_exp(dw) @ R
            rp = residual(Rp, t)
            J[:, j] = (rp - r0) / eps

        # d/dt
        for j in range(3):
            dt = np.zeros(3); dt[j] = eps
            rp = residual(R, t + dt)
            J[:, 3 + j] = (rp - r0) / eps

        A = J.T @ J + lam * np.eye(6)
        b = -J.T @ r0

        try:
            delta = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            lam *= 10.0
            continue

        dw = delta[:3]
        dt = delta[3:]

        R_new = so3_exp(dw) @ R
        t_new = t + dt
        r_new = residual(R_new, t_new)
        cost_new = float(r_new @ r_new)

        # LM accept/reject
        if cost_new < cost0:
            R, t = R_new, t_new
            lam = max(lam / 3.0, 1e-12)
        else:
            lam *= 10.0

    return R, t

def ransac_pnp(X_world, uv, K, threshold_px=3.0, max_iters=5000, p=0.99):
    # Sample 6 points, estimate pose via pnp_dlt
    # Evaluate reprojection errors on all points
    # Track best pose and inlier mask
    # Adaptively update iteration count using RANSAC formula
    # Refit on all inliers, then refine with refine_pose_gauss_newton
    # Return best_R, best_t, best_inliers

    N = len(X_world)
    best_R = np.zeros((3, 3))
    best_t = np.zeros(3,)
    best_inliers = None
    best_k = max_iters
    best_count = 0

    samplenum = 6
    it = 0
    while it < best_k:

        idx = random.sample(range(N), samplenum)
        X_cur = X_world[idx]
        uv_cur = uv[idx]

        try:
            R, t = pnp_dlt(X_cur, uv_cur, K)
        except Exception:
            it += 1
            continue

        cur_error = reprojection_errors(K, R, t, X_world, uv)
        cur_inliers = cur_error < threshold_px ** 2
        cur_count = int(np.sum(cur_inliers))
        if cur_count > best_count:
            best_R = R
            best_t = t
            best_count = cur_count
            best_inliers = cur_inliers

            w = cur_count / float(N)
            # 1 - (1 - w**samplenum)**k > p
            k = np.log(1-p) / np.log(1-w**samplenum)
            if k < best_k:
                best_k = k

        it += 1
    
    # Refit on all inliers for a more accurate final pose
    if best_inliers is not None and best_count >= 6:
        best_R, best_t = pnp_dlt(X_world[best_inliers], uv[best_inliers], K)
        errs = reprojection_errors(K, best_R, best_t, X_world, uv)
        best_inliers = errs < threshold_px ** 2

        R_ref, t_ref = refine_pose_gauss_newton(K, best_R, best_t,
                                            X_world[best_inliers], uv[best_inliers],
                                            iters=10, lam=1e-3)
        best_R, best_t = R_ref, t_ref


    return best_R, best_t, best_inliers


# 3 camera bundle adjustment
def bundle_adjustment(K,
                    R2_init, t2_init,
                    R3_init, t3_init,
                    X_init, obs,
                    iters=8, lam=1e-2, eps=1e-6):
    """
    Full BA for 3 cameras: cam0 fixed as [I|0].
    Optimizes cam2, cam3 poses and all 3D points.

    obs: list of (cam_idx, point_idx, uv) tuples
    Returns: R2, t2, R3, t3, X_opt, cost_before, cost_after
    """
    # Initialize state: R2,t2,R3,t3,X from inits
    # Define residual function: for each observation project point, compute (uv_hat - uv_obs)
    # Record cost_before
    # LM loop (iters):
    #   Build Jacobian numerically: (2M) x (12 + 3N)
    #     - cols 0-5:  cam2 rotation (so3_exp perturbation) and translation
    #     - cols 6-11: cam3 rotation and translation
    #     - cols 12+:  3D point perturbations (3 cols per point)
    #   Solve normal equations: (J^T J + lam I) delta = -J^T r
    #   Apply update multiplicatively for rotations, additively for t and X
    #   Accept if cost decreases (lam /= 3), else reject (lam *= 10)
    # Record cost_after
    # Return R2, t2, R3, t3, X, cost_before, cost_after

    X0 = np.asarray(X_init, float).copy()
    Np = len(X0)
   
    # Parameter vector p = [w2(3), t2(3), w3(3), t3(3), X(3N)]
    # We need initial w2,w3; easiest: start at 0 if your init R is already close,
    # but better: keep R as matrix and update multiplicatively. We'll do that:
    # store state as (R2,t2,R3,t3,X), and build numeric Jacobian wrt increments.

    R2 = R2_init.copy(); t2 = np.asarray(t2_init, float).reshape(3)
    R3 = R3_init.copy(); t3 = np.asarray(t3_init, float).reshape(3)
    X  = X0

    def residual(R2,t2,R3,t3,X):
        r = np.zeros((2 * len(obs),), dtype=float)
        for k,(ci,pi,uv) in enumerate(obs):
            if ci == 0:
                R, t = np.eye(3), np.zeros(3)
            elif ci == 1:
                R, t = R2, t2
            elif ci == 2:
                R, t = R3, t3
            else:
                raise ValueError("cam_idx must be 0,1,2")
            uv_hat = project_point(K, R, t, X[pi])
            r[2*k:2*k+2] = (uv_hat - uv)
        return r

    r0 = residual(R2,t2,R3,t3,X)
    cost_before = float(r0 @ r0)

    for _ in range(iters):
        r0 = residual(R2,t2,R3,t3,X)
        cost0 = float(r0 @ r0)

        # Jacobian size: (2M) x (12 + 3N)
        M = len(obs)
        D = 12 + 3 * Np
        J = np.zeros((2 * M, D), dtype=float)

        # --- cam2 increments ---
        for j in range(3):
            dw = np.zeros(3); dw[j] = eps
            rp = residual(so3_exp(dw) @ R2, t2, R3, t3, X)
            J[:, j] = (rp - r0) / eps
        for j in range(3):
            dt = np.zeros(3); dt[j] = eps
            rp = residual(R2, t2 + dt, R3, t3, X)
            J[:, 3 + j] = (rp - r0) / eps

        # --- cam3 increments ---
        base = 6
        for j in range(3):
            dw = np.zeros(3); dw[j] = eps
            rp = residual(R2, t2, so3_exp(dw) @ R3, t3, X)
            J[:, base + j] = (rp - r0) / eps
        for j in range(3):
            dt = np.zeros(3); dt[j] = eps
            rp = residual(R2, t2, R3, t3 + dt, X)
            J[:, base + 3 + j] = (rp - r0) / eps

        # --- point increments ---
        for pi in range(Np):
            for j in range(3):
                dX = np.zeros((Np,3), dtype=float)
                dX[pi, j] = eps
                rp = residual(R2, t2, R3, t3, X + dX)
                col = 12 + 3*pi + j
                J[:, col] = (rp - r0) / eps

        A = J.T @ J + lam * np.eye(D)
        b = -J.T @ r0

        try:
            delta = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            lam *= 10.0
            continue

        # Apply update
        dw2 = delta[0:3]; dt2 = delta[3:6]
        dw3 = delta[6:9]; dt3 = delta[9:12]
        dX  = delta[12:].reshape(Np,3)

        R2_new = so3_exp(dw2) @ R2
        t2_new = t2 + dt2
        R3_new = so3_exp(dw3) @ R3
        t3_new = t3 + dt3
        X_new  = X + dX

        r_new = residual(R2_new,t2_new,R3_new,t3_new,X_new)
        cost_new = float(r_new @ r_new)

        if cost_new < cost0:
            R2,t2,R3,t3,X = R2_new,t2_new,R3_new,t3_new,X_new
            lam = max(lam / 3.0, 1e-12)
        else:
            lam *= 10.0

    r_final = residual(R2,t2,R3,t3,X)
    cost_after = float(r_final @ r_final)
    
    return R2, t2, R3, t3, X, cost_before, cost_after
