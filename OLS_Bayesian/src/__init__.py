import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import math

import scipy.sparse as sp
from scipy.sparse.linalg import inv as sparse_inv, spsolve
from scipy.linalg import inv as dense_inv
from scipy.linalg import orth

import os

np.random.seed(42)
