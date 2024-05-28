from dataclasses import dataclass

import numpy as np


@dataclass
class ProblemInst:
    s: int
    t: int
    c: np.ndarray
    A_ub: np.ndarray
    b_ub: np.ndarray
    A_eq: np.ndarray
    b_eq: np.ndarray
    sol: np.ndarray
