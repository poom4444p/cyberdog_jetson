"""The Go2's legs, as the motor bus numbers them.

Order is Unitree's (unitree_sdk2_python example/go2/low_level/unitree_legged_const.py)
and the sim's actuator order (unitree_mujoco unitree_robots/go2/go2.xml): FR, FL,
RR, RL, each hip (abduction), thigh, calf. A policy trained elsewhere (Isaac
Lab orders joints by type, not by leg) is remapped to this order before it
reaches the board, never after.
"""
import numpy as np

N = 12
NAMES = [f"{leg}_{part}" for leg in ("FR", "FL", "RR", "RL")
         for part in ("hip", "thigh", "calf")]

# Joint ranges (rad), from go2.xml: abduction, front_hip / back_hip, knee.
_HIP = (-1.0472, 1.0472)
_FRONT_THIGH = (-1.5708, 3.4907)
_BACK_THIGH = (-0.5236, 4.5379)
_CALF = (-2.7227, -0.83776)
Q_MIN = np.array([lo for lo, _ in (_HIP, _FRONT_THIGH, _CALF) * 2
                  + (_HIP, _BACK_THIGH, _CALF) * 2])
Q_MAX = np.array([hi for _, hi in (_HIP, _FRONT_THIGH, _CALF) * 2
                  + (_HIP, _BACK_THIGH, _CALF) * 2])
Q_MARGIN = 0.05           # rad kept clear of each end stop

# Peak torque (N*m), go2.xml ctrlrange: hip and thigh 23.7, calf 45.43.
TAU_MAX = np.array([23.7, 23.7, 45.43] * 4)

# Poses from Unitree's go2_stand_example.py.
LIE = np.array([0.0, 1.36, -2.65, 0.0, 1.36, -2.65,
                -0.2, 1.36, -2.65, 0.2, 1.36, -2.65])
STAND = np.array([0.0, 0.67, -1.3] * 4)

KP_STAND, KD_STAND = 60.0, 5.0     # the example's gains for posing
KD_DAMP = 3.0                      # damping only: legs go limp, but slowly
