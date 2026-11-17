"""Camera-motion compensation: align every frame to the reference frame the desk zones were drawn on.

Why: zones are fixed pixel polygons. If the camera pans, tilts, zooms or gets bumped, the zones would
slide off the desks. We match static background features (ORB) between the current frame and the
reference frame, estimate a homography with RANSAC (moving people are rejected as outliers), and warp
the frame into the reference view before detection.

Speed: features are computed on a downscaled image, and the homography is only re-estimated every
`every` calls (camera motion is slow compared with the frame rate); in between, the last one is reused.
"""
from __future__ import annotations

import cv2
import numpy as np


class Stabilizer:
    def __init__(self, reference: np.ndarray, n_features: int = 2000, min_inliers: int = 30,
                 scale: float = 0.5, every: int = 1):
        self.size = (reference.shape[1], reference.shape[0])
        self.scale, self.every = scale, max(1, every)
        self.orb = cv2.ORB_create(n_features)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.min_inliers = min_inliers
        self.reference = reference
        self.ref_kp, self.ref_des = self.orb.detectAndCompute(self._small_gray(reference), None)
        self.H = np.eye(3)          # last good homography (current -> reference), full-resolution coords
        self.failures = 0
        self._calls = 0
        # Converts a homography estimated on downscaled images to full resolution.
        self._S = np.diag([scale, scale, 1.0])
        self._S_inv = np.diag([1 / scale, 1 / scale, 1.0])

    def _small_gray(self, img: np.ndarray) -> np.ndarray:
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.resize(g, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA) if self.scale != 1 else g

    def align(self, frame: np.ndarray) -> np.ndarray:
        """Return `frame` warped into the reference view (falls back to the last good alignment)."""
        if self._calls % self.every == 0:
            self._estimate(frame)
        self._calls += 1
        return self._warp(frame)

    def _estimate(self, frame: np.ndarray) -> None:
        kp, des = self.orb.detectAndCompute(self._small_gray(frame), None)
        if des is None or self.ref_des is None or len(kp) < self.min_inliers:
            self.failures += 1
            return
        pairs = self.matcher.knnMatch(des, self.ref_des, k=2)
        good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < 0.75 * n.distance]
        if len(good) < self.min_inliers:
            self.failures += 1
            return
        src = np.float32([kp[m.queryIdx].pt for m in good])
        dst = np.float32([self.ref_kp[m.trainIdx].pt for m in good])
        H, inliers = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
        if H is None or int(inliers.sum()) < self.min_inliers:
            self.failures += 1
            return
        self.H = self._S_inv @ H @ self._S
        self.failures = 0

    def _warp(self, frame: np.ndarray) -> np.ndarray:
        """Warp into the reference view; fill edges the camera no longer covers from the reference image."""
        warped = cv2.warpPerspective(frame, self.H, self.size)
        coverage = cv2.warpPerspective(np.full(frame.shape[:2], 255, np.uint8), self.H, self.size)
        coverage = cv2.erode(coverage, np.ones((3, 3), np.uint8))
        warped[coverage == 0] = self.reference[coverage == 0]
        return warped
