# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 10:58:15 2026

@author: gauthambekal93
"""

import numpy as np

# ---------------- task generator ----------------
def sample_task_params():
    w1 = np.random.randn(2).astype(np.float32)
    w1 /= np.linalg.norm(w1) + 1e-9

    w2 = np.random.randn(2).astype(np.float32)
    w2 /= np.linalg.norm(w2) + 1e-9

    b1 = np.random.uniform(0, 2 * np.pi)
    b2 = np.random.uniform(0, 2 * np.pi)

    return w1, w2, b1, b2


def sample_from_task(params, n=512, noise=0.02):
    w1, w2, b1, b2 = params

    X = np.random.randn(n, 2).astype(np.float32)

    score = (
        np.sin(3.0 * (X @ w1) + b1)
        + 0.5 * np.cos(5.0 * (X @ w2) + b2)
    )

    y = (score > 0).astype(np.int64)

    if noise > 0:
        flip = np.random.rand(n) < noise
        y[flip] = 1 - y[flip]

    return X, y




