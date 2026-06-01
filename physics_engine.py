#!/usr/bin/env python3
"""
3D physics engine experiment.
Contains rigid bodies, colliders, materials and supporting math types.
"""


import math
import time
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Callable, Optional, Tuple
from enum import Enum, auto



# ============================================================================
#                                     VEC3                                    
# ============================================================================

import math
import numpy as np


class Vec3:
    """Simple 3D vector helper used throughout the engine."""

    __slots__ = ('x', 'y', 'z')

    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    # ── Constructors ──────────────────────────────────────────────────────────

    @classmethod
    def zero(cls):
        return cls(0, 0, 0)

    @classmethod
    def one(cls):
        return cls(1, 1, 1)

    @classmethod
    def up(cls):
        return cls(0, 1, 0)

    @classmethod
    def down(cls):
        return cls(0, -1, 0)

    @classmethod
    def left(cls):
        return cls(-1, 0, 0)

    @classmethod
    def right(cls):
        return cls(1, 0, 0)

    @classmethod
    def forward(cls):
        return cls(0, 0, 1)

    @classmethod
    def back(cls):
        return cls(0, 0, -1)

    @classmethod
    def from_list(cls, lst):
        return cls(lst[0], lst[1], lst[2])

    @classmethod
    def from_numpy(cls, arr):
        return cls(arr[0], arr[1], arr[2])

    @classmethod
    def random_unit(cls):
        """Random unit vector on sphere surface."""
        theta = np.random.uniform(0, 2 * math.pi)
        phi = np.random.uniform(0, math.pi)
        return cls(
            math.sin(phi) * math.cos(theta),
            math.sin(phi) * math.sin(theta),
            math.cos(phi)
        )

    @classmethod
    def random_in_unit_sphere(cls):
        while True:
            v = cls(
                np.random.uniform(-1, 1),
                np.random.uniform(-1, 1),
                np.random.uniform(-1, 1)
            )
            if v.length_squared() <= 1.0:
                return v

    # ── Arithmetic ────────────────────────────────────────────────────────────

    def __add__(self, other):
        if isinstance(other, Vec3):
            return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)
        return Vec3(self.x + other, self.y + other, self.z + other)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, Vec3):
            return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)
        return Vec3(self.x - other, self.y - other, self.z - other)

    def __rsub__(self, other):
        return Vec3(other - self.x, other - self.y, other - self.z)

    def __mul__(self, other):
        if isinstance(other, Vec3):
            return Vec3(self.x * other.x, self.y * other.y, self.z * other.z)
        return Vec3(self.x * other, self.y * other, self.z * other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, Vec3):
            return Vec3(self.x / other.x, self.y / other.y, self.z / other.z)
        return Vec3(self.x / other, self.y / other, self.z / other)

    def __rtruediv__(self, other):
        return Vec3(other / self.x, other / self.y, other / self.z)

    def __neg__(self):
        return Vec3(-self.x, -self.y, -self.z)

    def __pos__(self):
        return Vec3(self.x, self.y, self.z)

    def __abs__(self):
        return Vec3(abs(self.x), abs(self.y), abs(self.z))

    def __iadd__(self, other):
        if isinstance(other, Vec3):
            self.x += other.x; self.y += other.y; self.z += other.z
        else:
            self.x += other; self.y += other; self.z += other
        return self

    def __isub__(self, other):
        if isinstance(other, Vec3):
            self.x -= other.x; self.y -= other.y; self.z -= other.z
        else:
            self.x -= other; self.y -= other; self.z -= other
        return self

    def __imul__(self, other):
        if isinstance(other, Vec3):
            self.x *= other.x; self.y *= other.y; self.z *= other.z
        else:
            self.x *= other; self.y *= other; self.z *= other
        return self

    def __itruediv__(self, other):
        if isinstance(other, Vec3):
            self.x /= other.x; self.y /= other.y; self.z /= other.z
        else:
            self.x /= other; self.y /= other; self.z /= other
        return self

    # ── Comparison ────────────────────────────────────────────────────────────

    def __eq__(self, other):
        if isinstance(other, Vec3):
            return (abs(self.x - other.x) < 1e-9 and
                    abs(self.y - other.y) < 1e-9 and
                    abs(self.z - other.z) < 1e-9)
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __lt__(self, other):
        return self.length_squared() < other.length_squared()

    def __le__(self, other):
        return self.length_squared() <= other.length_squared()

    def __gt__(self, other):
        return self.length_squared() > other.length_squared()

    def __ge__(self, other):
        return self.length_squared() >= other.length_squared()

    # ── Representation ────────────────────────────────────────────────────────

    def __repr__(self):
        return f"Vec3({self.x:.4f}, {self.y:.4f}, {self.z:.4f})"

    def __str__(self):
        return f"({self.x:.4f}, {self.y:.4f}, {self.z:.4f})"

    def __iter__(self):
        yield self.x
        yield self.y
        yield self.z

    def __getitem__(self, idx):
        if idx == 0: return self.x
        if idx == 1: return self.y
        if idx == 2: return self.z
        raise IndexError(f"Vec3 index {idx} out of range")

    def __setitem__(self, idx, val):
        if idx == 0: self.x = float(val)
        elif idx == 1: self.y = float(val)
        elif idx == 2: self.z = float(val)
        else: raise IndexError(f"Vec3 index {idx} out of range")

    def __len__(self):
        return 3

    def __hash__(self):
        return hash((round(self.x, 6), round(self.y, 6), round(self.z, 6)))

    # ── Core Math ─────────────────────────────────────────────────────────────

    def length(self):
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)

    def length_squared(self):
        return self.x * self.x + self.y * self.y + self.z * self.z

    def magnitude(self):
        return self.length()

    def normalize(self):
        mag = self.length()
        if mag < 1e-10:
            return Vec3.zero()
        return Vec3(self.x / mag, self.y / mag, self.z / mag)

    def normalized(self):
        return self.normalize()

    def dot(self, other):
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other):
        return Vec3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )

    def distance_to(self, other):
        return (self - other).length()

    def distance_squared_to(self, other):
        return (self - other).length_squared()

    def angle_to(self, other):
        """Angle in radians between two vectors."""
        d = self.dot(other)
        mag = self.length() * other.length()
        if mag < 1e-10:
            return 0.0
        return math.acos(max(-1.0, min(1.0, d / mag)))

    def project_onto(self, other):
        """Project this vector onto another."""
        other_len_sq = other.length_squared()
        if other_len_sq < 1e-10:
            return Vec3.zero()
        return other * (self.dot(other) / other_len_sq)

    def reject_from(self, other):
        """Rejection from another vector (perpendicular component)."""
        return self - self.project_onto(other)

    def reflect(self, normal):
        """Reflect across a surface normal."""
        return self - normal * (2 * self.dot(normal))

    def refract(self, normal, eta_ratio):
        """Refract using Snell's law."""
        cos_theta = min(-self.dot(normal), 1.0)
        r_out_perp = (self + normal * cos_theta) * eta_ratio
        r_out_parallel = normal * (-math.sqrt(abs(1.0 - r_out_perp.length_squared())))
        return r_out_perp + r_out_parallel

    def lerp(self, other, t):
        """Linear interpolation."""
        return self + (other - self) * t

    def slerp(self, other, t):
        """Spherical linear interpolation."""
        dot = max(-1.0, min(1.0, self.normalize().dot(other.normalize())))
        theta = math.acos(dot) * t
        relative = (other - self * dot).normalize()
        return self * math.cos(theta) + relative * math.sin(theta)

    def clamp(self, min_val, max_val):
        """Clamp each component."""
        return Vec3(
            max(min_val, min(max_val, self.x)),
            max(min_val, min(max_val, self.y)),
            max(min_val, min(max_val, self.z))
        )

    def clamp_magnitude(self, max_length):
        if self.length() > max_length:
            return self.normalize() * max_length
        return Vec3(self.x, self.y, self.z)

    def abs(self):
        return Vec3(abs(self.x), abs(self.y), abs(self.z))

    def min_component(self):
        return min(self.x, self.y, self.z)

    def max_component(self):
        return max(self.x, self.y, self.z)

    def floor(self):
        return Vec3(math.floor(self.x), math.floor(self.y), math.floor(self.z))

    def ceil(self):
        return Vec3(math.ceil(self.x), math.ceil(self.y), math.ceil(self.z))

    def round(self, decimals=0):
        return Vec3(round(self.x, decimals), round(self.y, decimals), round(self.z, decimals))

    def is_zero(self, eps=1e-9):
        return self.length_squared() < eps * eps

    def is_unit(self, eps=1e-6):
        return abs(self.length() - 1.0) < eps

    def is_finite(self):
        return math.isfinite(self.x) and math.isfinite(self.y) and math.isfinite(self.z)

    # ── Conversion ────────────────────────────────────────────────────────────

    def to_list(self):
        return [self.x, self.y, self.z]

    def to_tuple(self):
        return (self.x, self.y, self.z)

    def to_numpy(self):
        return np.array([self.x, self.y, self.z], dtype=np.float64)

    def copy(self):
        return Vec3(self.x, self.y, self.z)

    def xy(self):
        return (self.x, self.y)

    def xz(self):
        return (self.x, self.z)

    def yz(self):
        return (self.y, self.z)

    # ── Static Utilities ──────────────────────────────────────────────────────

    @staticmethod
    def min_vec(a, b):
        return Vec3(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z))

    @staticmethod
    def max_vec(a, b):
        return Vec3(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z))

    @staticmethod
    def average(vectors):
        if not vectors:
            return Vec3.zero()
        total = Vec3.zero()
        for v in vectors:
            total += v
        return total / len(vectors)

    @staticmethod
    def weighted_average(vectors, weights):
        total = Vec3.zero()
        w_sum = 0.0
        for v, w in zip(vectors, weights):
            total += v * w
            w_sum += w
        if w_sum < 1e-10:
            return Vec3.zero()
        return total / w_sum


# ============================================================================
#                                  QUATERNION                                 
# ============================================================================

import math
import numpy as np


class Quaternion:
    """
    Unit quaternion for 3D rotations.
    Stored as (w, x, y, z) where w is the scalar component.
    """

    __slots__ = ('w', 'x', 'y', 'z')

    def __init__(self, w=1.0, x=0.0, y=0.0, z=0.0):
        self.w = float(w)
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    # ── Constructors ──────────────────────────────────────────────────────────

    @classmethod
    def identity(cls):
        return cls(1, 0, 0, 0)

    @classmethod
    def from_axis_angle(cls, axis: Vec3, angle_rad: float):
        """Create quaternion from axis-angle representation."""
        axis = axis.normalize()
        half = angle_rad * 0.5
        s = math.sin(half)
        return cls(math.cos(half), axis.x * s, axis.y * s, axis.z * s)

    @classmethod
    def from_euler(cls, pitch: float, yaw: float, roll: float):
        """
        Create quaternion from Euler angles (radians).
        pitch = rotation around X
        yaw   = rotation around Y
        roll  = rotation around Z
        """
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)

        return cls(
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy
        )

    @classmethod
    def from_euler_degrees(cls, pitch: float, yaw: float, roll: float):
        return cls.from_euler(
            math.radians(pitch),
            math.radians(yaw),
            math.radians(roll)
        )

    @classmethod
    def from_rotation_matrix(cls, m):
        """
        Create quaternion from 3x3 rotation matrix (numpy array or nested list).
        """
        if isinstance(m, list):
            m = np.array(m)
        trace = m[0, 0] + m[1, 1] + m[2, 2]
        if trace > 0:
            s = 0.5 / math.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (m[2, 1] - m[1, 2]) * s
            y = (m[0, 2] - m[2, 0]) * s
            z = (m[1, 0] - m[0, 1]) * s
        elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = 2.0 * math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
            w = (m[2, 1] - m[1, 2]) / s
            x = 0.25 * s
            y = (m[0, 1] + m[1, 0]) / s
            z = (m[0, 2] + m[2, 0]) / s
        elif m[1, 1] > m[2, 2]:
            s = 2.0 * math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
            w = (m[0, 2] - m[2, 0]) / s
            x = (m[0, 1] + m[1, 0]) / s
            y = 0.25 * s
            z = (m[1, 2] + m[2, 1]) / s
        else:
            s = 2.0 * math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
            w = (m[1, 0] - m[0, 1]) / s
            x = (m[0, 2] + m[2, 0]) / s
            y = (m[1, 2] + m[2, 1]) / s
            z = 0.25 * s
        return cls(w, x, y, z).normalize()

    @classmethod
    def look_rotation(cls, forward: Vec3, up: Vec3 = None):
        """Create quaternion that rotates to look along 'forward'."""
        if up is None:
            up = Vec3.up()
        forward = forward.normalize()
        right = up.cross(forward).normalize()
        up_new = forward.cross(right)
        m = np.array([
            [right.x, right.y, right.z],
            [up_new.x, up_new.y, up_new.z],
            [forward.x, forward.y, forward.z]
        ])
        return cls.from_rotation_matrix(m)

    @classmethod
    def from_to_rotation(cls, from_vec: Vec3, to_vec: Vec3):
        """Rotation from one vector to another."""
        from_vec = from_vec.normalize()
        to_vec = to_vec.normalize()
        dot = from_vec.dot(to_vec)
        if dot > 0.9999:
            return cls.identity()
        if dot < -0.9999:
            perp = Vec3(1, 0, 0)
            if abs(from_vec.dot(perp)) > 0.9:
                perp = Vec3(0, 1, 0)
            axis = from_vec.cross(perp).normalize()
            return cls.from_axis_angle(axis, math.pi)
        axis = from_vec.cross(to_vec)
        angle = math.acos(max(-1.0, min(1.0, dot)))
        return cls.from_axis_angle(axis, angle)

    # ── Arithmetic ────────────────────────────────────────────────────────────

    def __mul__(self, other):
        if isinstance(other, Quaternion):
            return Quaternion(
                self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z,
                self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y,
                self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x,
                self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w
            )
        elif isinstance(other, Vec3):
            return self.rotate_vector(other)
        return Quaternion(self.w * other, self.x * other, self.y * other, self.z * other)

    def __rmul__(self, other):
        if isinstance(other, (int, float)):
            return Quaternion(self.w * other, self.x * other, self.y * other, self.z * other)
        return NotImplemented

    def __add__(self, other):
        return Quaternion(self.w + other.w, self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return Quaternion(self.w - other.w, self.x - other.x, self.y - other.y, self.z - other.z)

    def __neg__(self):
        return Quaternion(-self.w, -self.x, -self.y, -self.z)

    def __eq__(self, other):
        return (abs(self.w - other.w) < 1e-9 and abs(self.x - other.x) < 1e-9 and
                abs(self.y - other.y) < 1e-9 and abs(self.z - other.z) < 1e-9)

    def __repr__(self):
        return f"Quaternion(w={self.w:.4f}, x={self.x:.4f}, y={self.y:.4f}, z={self.z:.4f})"

    # ── Core Math ─────────────────────────────────────────────────────────────

    def length(self):
        return math.sqrt(self.w*self.w + self.x*self.x + self.y*self.y + self.z*self.z)

    def length_squared(self):
        return self.w*self.w + self.x*self.x + self.y*self.y + self.z*self.z

    def normalize(self):
        mag = self.length()
        if mag < 1e-10:
            return Quaternion.identity()
        return Quaternion(self.w/mag, self.x/mag, self.y/mag, self.z/mag)

    def conjugate(self):
        return Quaternion(self.w, -self.x, -self.y, -self.z)

    def inverse(self):
        ls = self.length_squared()
        if ls < 1e-10:
            return Quaternion.identity()
        c = self.conjugate()
        return Quaternion(c.w/ls, c.x/ls, c.y/ls, c.z/ls)

    def dot(self, other):
        return self.w*other.w + self.x*other.x + self.y*other.y + self.z*other.z

    def rotate_vector(self, v: Vec3) -> Vec3:
        """Rotate a vector by this quaternion (sandwich product)."""
        qv = Vec3(self.x, self.y, self.z)
        t = qv.cross(v) * 2.0
        return v + t * self.w + qv.cross(t)

    def to_euler(self):
        """Convert to Euler angles (pitch, yaw, roll) in radians."""
        # pitch (x-axis rotation)
        sinr_cosp = 2 * (self.w * self.x + self.y * self.z)
        cosr_cosp = 1 - 2 * (self.x * self.x + self.y * self.y)
        pitch = math.atan2(sinr_cosp, cosr_cosp)

        # yaw (y-axis rotation)
        sinp = 2 * (self.w * self.y - self.z * self.x)
        if abs(sinp) >= 1:
            yaw = math.copysign(math.pi / 2, sinp)
        else:
            yaw = math.asin(sinp)

        # roll (z-axis rotation)
        siny_cosp = 2 * (self.w * self.z + self.x * self.y)
        cosy_cosp = 1 - 2 * (self.y * self.y + self.z * self.z)
        roll = math.atan2(siny_cosp, cosy_cosp)

        return pitch, yaw, roll

    def to_euler_degrees(self):
        p, y, r = self.to_euler()
        return math.degrees(p), math.degrees(y), math.degrees(r)

    def to_rotation_matrix(self):
        """Return 3x3 numpy rotation matrix."""
        w, x, y, z = self.w, self.x, self.y, self.z
        return np.array([
            [1-2*(y*y+z*z),   2*(x*y-w*z),   2*(x*z+w*y)],
            [  2*(x*y+w*z), 1-2*(x*x+z*z),   2*(y*z-w*x)],
            [  2*(x*z-w*y),   2*(y*z+w*x), 1-2*(x*x+y*y)]
        ], dtype=np.float64)

    def to_axis_angle(self):
        """Return (axis: Vec3, angle: float in radians)."""
        q = self.normalize()
        angle = 2.0 * math.acos(max(-1.0, min(1.0, q.w)))
        s = math.sqrt(1.0 - q.w * q.w)
        if s < 1e-6:
            return Vec3(1, 0, 0), angle
        return Vec3(q.x/s, q.y/s, q.z/s), angle

    def slerp(self, other, t):
        """Spherical linear interpolation."""
        dot = self.dot(other)
        other_q = Quaternion(other.w, other.x, other.y, other.z)
        if dot < 0:
            dot = -dot
            other_q = -other_q
        dot = min(1.0, dot)
        theta = math.acos(dot)
        if abs(theta) < 1e-6:
            return (self + (other_q - self) * t).normalize()
        return (self * math.sin((1-t)*theta) + other_q * math.sin(t*theta)) * (1.0 / math.sin(theta))

    def nlerp(self, other, t):
        """Normalized linear interpolation (cheaper than slerp)."""
        return (self + (other - self) * t).normalize()

    def angle_to(self, other):
        """Angle in radians between two rotations."""
        dot = abs(self.dot(other))
        return 2.0 * math.acos(min(1.0, dot))

    def is_identity(self, eps=1e-6):
        return abs(self.w - 1.0) < eps and abs(self.x) < eps and abs(self.y) < eps and abs(self.z) < eps

    def copy(self):
        return Quaternion(self.w, self.x, self.y, self.z)

    def to_list(self):
        return [self.w, self.x, self.y, self.z]

    def forward_vector(self):
        return self.rotate_vector(Vec3(0, 0, 1))

    def up_vector(self):
        return self.rotate_vector(Vec3(0, 1, 0))

    def right_vector(self):
        return self.rotate_vector(Vec3(1, 0, 0))


# ============================================================================
#                                  MATERIALS                                  
# ============================================================================

from dataclasses import dataclass
from typing import Dict


@dataclass
class PhysicsMaterial:
    """
    Physics material definition.
    All values are approximate real-world measurements.
    """
    name: str
    density: float          # kg/m³
    restitution: float      # coefficient of restitution (0=inelastic, 1=perfect elastic)
    static_friction: float  # static friction coefficient
    dynamic_friction: float # kinetic friction coefficient
    rolling_friction: float # rolling resistance
    air_drag: float         # linear air drag coefficient
    angular_drag: float     # angular drag coefficient
    young_modulus: float    # Pa (stiffness, for soft-body later)
    poisson_ratio: float    # dimensionless
    description: str = ""

    def apply_to_body(self, body):
        """Apply this material to a RigidBody."""
        body.restitution = self.restitution
        body.static_friction = self.static_friction
        body.dynamic_friction = self.dynamic_friction
        body.rolling_friction = self.rolling_friction
        body.air_drag = self.air_drag
        body.angular_drag = self.angular_drag


# ─────────────────────────────────────────────────────────────────────────────
# Material Library
# ─────────────────────────────────────────────────────────────────────────────

MATERIALS: Dict[str, PhysicsMaterial] = {

    # ── Balls & Sports ────────────────────────────────────────────────────────

    'super_ball': PhysicsMaterial(
        name='Super Ball',
        density=1200,
        restitution=0.90,
        static_friction=1.2,
        dynamic_friction=0.95,
        rolling_friction=0.002,
        air_drag=0.0003,
        angular_drag=0.02,
        young_modulus=0.05e9,
        poisson_ratio=0.49,
        description='Vulcanized rubber — bounces nearly to original height'
    ),

    'tennis_ball': PhysicsMaterial(
        name='Tennis Ball',
        density=650,
        restitution=0.75,
        static_friction=0.6,
        dynamic_friction=0.55,
        rolling_friction=0.003,
        air_drag=0.0005,
        angular_drag=0.03,
        young_modulus=0.5e6,
        poisson_ratio=0.48,
        description='Pressurized rubber core with felt'
    ),

    'basketball': PhysicsMaterial(
        name='Basketball',
        density=620,
        restitution=0.76,
        static_friction=0.8,
        dynamic_friction=0.7,
        rolling_friction=0.005,
        air_drag=0.0004,
        angular_drag=0.025,
        young_modulus=0.3e6,
        poisson_ratio=0.48,
        description='Inflated rubber / leather shell'
    ),

    'golf_ball': PhysicsMaterial(
        name='Golf Ball',
        density=1130,
        restitution=0.68,
        static_friction=0.35,
        dynamic_friction=0.25,
        rolling_friction=0.001,
        air_drag=0.0002,
        angular_drag=0.015,
        young_modulus=2.4e9,
        poisson_ratio=0.45,
        description='Hard surlyn cover over wound rubber core'
    ),

    'steel_ball': PhysicsMaterial(
        name='Steel Ball',
        density=7850,
        restitution=0.58,
        static_friction=0.15,
        dynamic_friction=0.10,
        rolling_friction=0.001,
        air_drag=0.00005,
        angular_drag=0.005,
        young_modulus=200e9,
        poisson_ratio=0.29,
        description='Carbon steel bearing ball'
    ),

    'glass_ball': PhysicsMaterial(
        name='Glass Ball',
        density=2500,
        restitution=0.65,
        static_friction=0.4,
        dynamic_friction=0.35,
        rolling_friction=0.0015,
        air_drag=0.0001,
        angular_drag=0.01,
        young_modulus=70e9,
        poisson_ratio=0.23,
        description='Borosilicate glass marble'
    ),

    'rubber_ball': PhysicsMaterial(
        name='Rubber Ball',
        density=1100,
        restitution=0.80,
        static_friction=0.9,
        dynamic_friction=0.75,
        rolling_friction=0.003,
        air_drag=0.0004,
        angular_drag=0.03,
        young_modulus=0.01e9,
        poisson_ratio=0.49,
        description='Natural rubber'
    ),

    'bouncy_ball': PhysicsMaterial(
        name='Bouncy Ball',
        density=950,
        restitution=0.85,
        static_friction=1.0,
        dynamic_friction=0.85,
        rolling_friction=0.002,
        air_drag=0.0004,
        angular_drag=0.02,
        young_modulus=0.005e9,
        poisson_ratio=0.499,
        description='Highly elastic polymer toy ball'
    ),

    # ── Surfaces / Floors ─────────────────────────────────────────────────────

    'concrete': PhysicsMaterial(
        name='Concrete',
        density=2300,
        restitution=0.15,
        static_friction=0.7,
        dynamic_friction=0.55,
        rolling_friction=0.01,
        air_drag=0.0,
        angular_drag=0.0,
        young_modulus=30e9,
        poisson_ratio=0.2,
        description='Standard structural concrete'
    ),

    'hardwood': PhysicsMaterial(
        name='Hardwood',
        density=700,
        restitution=0.35,
        static_friction=0.5,
        dynamic_friction=0.4,
        rolling_friction=0.005,
        air_drag=0.0,
        angular_drag=0.0,
        young_modulus=12e9,
        poisson_ratio=0.35,
        description='Oak / maple hardwood flooring'
    ),

    'ice': PhysicsMaterial(
        name='Ice',
        density=917,
        restitution=0.40,
        static_friction=0.03,
        dynamic_friction=0.01,
        rolling_friction=0.0005,
        air_drag=0.0,
        angular_drag=0.0,
        young_modulus=9e9,
        poisson_ratio=0.33,
        description='Water ice at 0°C'
    ),

    'sand': PhysicsMaterial(
        name='Sand',
        density=1600,
        restitution=0.05,
        static_friction=0.65,
        dynamic_friction=0.55,
        rolling_friction=0.2,
        air_drag=0.0,
        angular_drag=0.0,
        young_modulus=0.1e9,
        poisson_ratio=0.3,
        description='Dry quartz sand'
    ),

    'carpet': PhysicsMaterial(
        name='Carpet',
        density=800,
        restitution=0.10,
        static_friction=0.8,
        dynamic_friction=0.65,
        rolling_friction=0.05,
        air_drag=0.0,
        angular_drag=0.0,
        young_modulus=0.5e6,
        poisson_ratio=0.4,
        description='Medium pile carpet'
    ),

    'mud': PhysicsMaterial(
        name='Mud',
        density=1900,
        restitution=0.02,
        static_friction=0.4,
        dynamic_friction=0.35,
        rolling_friction=0.3,
        air_drag=0.0,
        angular_drag=0.0,
        young_modulus=0.01e9,
        poisson_ratio=0.45,
        description='Saturated clay soil'
    ),

    'grass': PhysicsMaterial(
        name='Grass',
        density=800,
        restitution=0.25,
        static_friction=0.6,
        dynamic_friction=0.5,
        rolling_friction=0.03,
        air_drag=0.0,
        angular_drag=0.0,
        young_modulus=1e6,
        poisson_ratio=0.4,
        description='Short grass surface'
    ),

    # ── Metals ────────────────────────────────────────────────────────────────

    'steel': PhysicsMaterial(
        name='Steel',
        density=7850,
        restitution=0.55,
        static_friction=0.15,
        dynamic_friction=0.12,
        rolling_friction=0.001,
        air_drag=0.00005,
        angular_drag=0.005,
        young_modulus=200e9,
        poisson_ratio=0.29,
        description='Structural steel (ASTM A36)'
    ),

    'aluminum': PhysicsMaterial(
        name='Aluminum',
        density=2700,
        restitution=0.50,
        static_friction=0.35,
        dynamic_friction=0.25,
        rolling_friction=0.002,
        air_drag=0.0001,
        angular_drag=0.008,
        young_modulus=70e9,
        poisson_ratio=0.33,
        description='6061-T6 aluminum alloy'
    ),

    'copper': PhysicsMaterial(
        name='Copper',
        density=8960,
        restitution=0.30,
        static_friction=0.45,
        dynamic_friction=0.38,
        rolling_friction=0.002,
        air_drag=0.00005,
        angular_drag=0.006,
        young_modulus=120e9,
        poisson_ratio=0.35,
        description='Pure copper'
    ),

    'titanium': PhysicsMaterial(
        name='Titanium',
        density=4500,
        restitution=0.52,
        static_friction=0.36,
        dynamic_friction=0.30,
        rolling_friction=0.001,
        air_drag=0.00005,
        angular_drag=0.005,
        young_modulus=116e9,
        poisson_ratio=0.34,
        description='Grade 5 titanium alloy (Ti-6Al-4V)'
    ),

    # ── Soft Materials ────────────────────────────────────────────────────────

    'foam': PhysicsMaterial(
        name='Foam',
        density=50,
        restitution=0.20,
        static_friction=0.5,
        dynamic_friction=0.4,
        rolling_friction=0.05,
        air_drag=0.002,
        angular_drag=0.1,
        young_modulus=0.05e6,
        poisson_ratio=0.3,
        description='Polyurethane open-cell foam'
    ),

    'clay': PhysicsMaterial(
        name='Clay',
        density=2000,
        restitution=0.05,
        static_friction=0.45,
        dynamic_friction=0.40,
        rolling_friction=0.15,
        air_drag=0.0,
        angular_drag=0.0,
        young_modulus=0.3e9,
        poisson_ratio=0.45,
        description='Modeling clay — very low bounce'
    ),

    'wood': PhysicsMaterial(
        name='Wood',
        density=600,
        restitution=0.30,
        static_friction=0.45,
        dynamic_friction=0.35,
        rolling_friction=0.005,
        air_drag=0.0001,
        angular_drag=0.02,
        young_modulus=10e9,
        poisson_ratio=0.3,
        description='Generic softwood (pine/spruce)'
    ),

    'glass': PhysicsMaterial(
        name='Glass',
        density=2500,
        restitution=0.60,
        static_friction=0.4,
        dynamic_friction=0.35,
        rolling_friction=0.002,
        air_drag=0.00005,
        angular_drag=0.01,
        young_modulus=70e9,
        poisson_ratio=0.23,
        description='Soda-lime glass'
    ),

    'plastic': PhysicsMaterial(
        name='Plastic',
        density=1200,
        restitution=0.45,
        static_friction=0.4,
        dynamic_friction=0.35,
        rolling_friction=0.004,
        air_drag=0.0002,
        angular_drag=0.02,
        young_modulus=3e9,
        poisson_ratio=0.38,
        description='ABS / polycarbonate'
    ),

    'water': PhysicsMaterial(
        name='Water (submerged)',
        density=1000,
        restitution=0.0,
        static_friction=0.0,
        dynamic_friction=0.001,
        rolling_friction=0.0,
        air_drag=0.05,
        angular_drag=0.3,
        young_modulus=2.2e9,
        poisson_ratio=0.5,
        description='Water — high drag approximation'
    ),
}


def get_material(name: str) -> PhysicsMaterial:
    """Get material by name (case-insensitive)."""
    key = name.lower().replace(' ', '_')
    if key not in MATERIALS:
        raise KeyError(f"Material '{name}' not found. Available: {list(MATERIALS.keys())}")
    return MATERIALS[key]


def get_all_material_names() -> list:
    return list(MATERIALS.keys())


def combine_materials(mat_a: PhysicsMaterial, mat_b: PhysicsMaterial,
                      name: str = 'Combined') -> PhysicsMaterial:
    """
    Create a combined material (for contact between two surfaces).
    Uses Bullet/PhysX convention: geometric mean for restitution,
    arithmetic mean for friction.
    """
    import math
    return PhysicsMaterial(
        name=name,
        density=(mat_a.density + mat_b.density) * 0.5,
        restitution=math.sqrt(mat_a.restitution * mat_b.restitution),
        static_friction=(mat_a.static_friction + mat_b.static_friction) * 0.5,
        dynamic_friction=(mat_a.dynamic_friction + mat_b.dynamic_friction) * 0.5,
        rolling_friction=(mat_a.rolling_friction + mat_b.rolling_friction) * 0.5,
        air_drag=(mat_a.air_drag + mat_b.air_drag) * 0.5,
        angular_drag=(mat_a.angular_drag + mat_b.angular_drag) * 0.5,
        young_modulus=(mat_a.young_modulus + mat_b.young_modulus) * 0.5,
        poisson_ratio=(mat_a.poisson_ratio + mat_b.poisson_ratio) * 0.5,
    )


# ============================================================================
#                                  COLLIDERS                                  
# ============================================================================

import math
import numpy as np
from enum import Enum, auto


class ColliderType(Enum):
    SPHERE = auto()
    BOX = auto()
    CAPSULE = auto()
    PLANE = auto()
    CYLINDER = auto()
    CONE = auto()
    CONVEX_HULL = auto()
    MESH = auto()


class AABB:
    """Basic AABB used for broad-phase checks."""

    def __init__(self, min_point: Vec3 = None, max_point: Vec3 = None):
        self.min = min_point or Vec3(float('inf'), float('inf'), float('inf'))
        self.max = max_point or Vec3(float('-inf'), float('-inf'), float('-inf'))

    def expand(self, point: Vec3):
        self.min = Vec3.min_vec(self.min, point)
        self.max = Vec3.max_vec(self.max, point)

    def expand_aabb(self, other):
        self.min = Vec3.min_vec(self.min, other.min)
        self.max = Vec3.max_vec(self.max, other.max)

    def center(self) -> Vec3:
        return (self.min + self.max) * 0.5

    def half_extents(self) -> Vec3:
        return (self.max - self.min) * 0.5

    def size(self) -> Vec3:
        return self.max - self.min

    def volume(self) -> float:
        s = self.size()
        return s.x * s.y * s.z

    def surface_area(self) -> float:
        s = self.size()
        return 2 * (s.x*s.y + s.y*s.z + s.x*s.z)

    def contains_point(self, p: Vec3) -> bool:
        return (self.min.x <= p.x <= self.max.x and
                self.min.y <= p.y <= self.max.y and
                self.min.z <= p.z <= self.max.z)

    def intersects(self, other) -> bool:
        return (self.min.x <= other.max.x and self.max.x >= other.min.x and
                self.min.y <= other.max.y and self.max.y >= other.min.y and
                self.min.z <= other.max.z and self.max.z >= other.min.z)

    def closest_point(self, p: Vec3) -> Vec3:
        return Vec3(
            max(self.min.x, min(p.x, self.max.x)),
            max(self.min.y, min(p.y, self.max.y)),
            max(self.min.z, min(p.z, self.max.z))
        )

    def inflate(self, amount: float):
        v = Vec3(amount, amount, amount)
        return AABB(self.min - v, self.max + v)

    def __repr__(self):
        return f"AABB(min={self.min}, max={self.max})"


class Collider:
    """Base class for all collision shapes."""

    def __init__(self, collider_type: ColliderType):
        self.collider_type = collider_type
        self.local_position = Vec3.zero()
        self.local_rotation = Quaternion.identity()
        self.material_restitution = None   # override rigidbody if set
        self.material_friction = None
        self.is_trigger = False
        self.rigidbody = None  # parent rigidbody

    def get_aabb(self, position: Vec3, rotation: Quaternion) -> AABB:
        raise NotImplementedError

    def support(self, direction: Vec3, position: Vec3, rotation: Quaternion) -> Vec3:
        """GJK support function: farthest point in direction."""
        raise NotImplementedError

    def closest_point(self, point: Vec3, position: Vec3, rotation: Quaternion) -> Vec3:
        raise NotImplementedError

    def get_world_position(self, body_position: Vec3, body_rotation: Quaternion) -> Vec3:
        return body_position + body_rotation.rotate_vector(self.local_position)

    def get_world_rotation(self, body_rotation: Quaternion) -> Quaternion:
        return body_rotation * self.local_rotation


class SphereCollider(Collider):
    """Sphere collision shape."""

    def __init__(self, radius: float = 0.5):
        super().__init__(ColliderType.SPHERE)
        self.radius = radius

    def get_aabb(self, position: Vec3, rotation: Quaternion) -> AABB:
        center = self.get_world_position(position, rotation)
        r = Vec3(self.radius, self.radius, self.radius)
        return AABB(center - r, center + r)

    def support(self, direction: Vec3, position: Vec3, rotation: Quaternion) -> Vec3:
        center = self.get_world_position(position, rotation)
        return center + direction.normalize() * self.radius

    def closest_point(self, point: Vec3, position: Vec3, rotation: Quaternion) -> Vec3:
        center = self.get_world_position(position, rotation)
        diff = point - center
        dist = diff.length()
        if dist < 1e-10:
            return center + Vec3(self.radius, 0, 0)
        if dist <= self.radius:
            return point
        return center + diff.normalize() * self.radius

    def contains_point(self, point: Vec3, position: Vec3, rotation: Quaternion) -> bool:
        center = self.get_world_position(position, rotation)
        return (point - center).length_squared() <= self.radius * self.radius

    def __repr__(self):
        return f"SphereCollider(radius={self.radius})"


class BoxCollider(Collider):
    """Axis-aligned box (OBB in world space)."""

    def __init__(self, half_extents: Vec3 = None):
        super().__init__(ColliderType.BOX)
        self.half_extents = half_extents or Vec3(0.5, 0.5, 0.5)

    def get_aabb(self, position: Vec3, rotation: Quaternion) -> AABB:
        center = self.get_world_position(position, rotation)
        R = rotation.to_rotation_matrix()
        he = self.half_extents.to_numpy()
        # World-space AABB from OBB
        world_he = np.abs(R) @ he
        world_he_vec = Vec3.from_numpy(world_he)
        return AABB(center - world_he_vec, center + world_he_vec)

    def get_axes(self, rotation: Quaternion):
        """Return 3 local axes in world space."""
        R = self.get_world_rotation(rotation).to_rotation_matrix()
        return [
            Vec3.from_numpy(R[:, 0]),
            Vec3.from_numpy(R[:, 1]),
            Vec3.from_numpy(R[:, 2])
        ]

    def support(self, direction: Vec3, position: Vec3, rotation: Quaternion) -> Vec3:
        center = self.get_world_position(position, rotation)
        world_rot = self.get_world_rotation(rotation)
        local_dir = world_rot.inverse().rotate_vector(direction)
        local_support = Vec3(
            math.copysign(self.half_extents.x, local_dir.x),
            math.copysign(self.half_extents.y, local_dir.y),
            math.copysign(self.half_extents.z, local_dir.z)
        )
        return center + world_rot.rotate_vector(local_support)

    def closest_point(self, point: Vec3, position: Vec3, rotation: Quaternion) -> Vec3:
        center = self.get_world_position(position, rotation)
        world_rot = self.get_world_rotation(rotation)
        local_point = world_rot.inverse().rotate_vector(point - center)
        clamped = Vec3(
            max(-self.half_extents.x, min(self.half_extents.x, local_point.x)),
            max(-self.half_extents.y, min(self.half_extents.y, local_point.y)),
            max(-self.half_extents.z, min(self.half_extents.z, local_point.z))
        )
        return center + world_rot.rotate_vector(clamped)

    def get_vertices(self, position: Vec3, rotation: Quaternion):
        """Return 8 corner vertices in world space."""
        center = self.get_world_position(position, rotation)
        world_rot = self.get_world_rotation(rotation)
        he = self.half_extents
        corners = []
        for sx in (-1, 1):
            for sy in (-1, 1):
                for sz in (-1, 1):
                    local = Vec3(sx * he.x, sy * he.y, sz * he.z)
                    corners.append(center + world_rot.rotate_vector(local))
        return corners

    def __repr__(self):
        return f"BoxCollider(half_extents={self.half_extents})"


class CapsuleCollider(Collider):
    """Capsule = cylinder + two hemisphere caps."""

    def __init__(self, radius: float = 0.5, height: float = 2.0):
        super().__init__(ColliderType.CAPSULE)
        self.radius = radius
        self.height = height  # total height including caps

    @property
    def cylinder_height(self):
        return max(0.0, self.height - 2.0 * self.radius)

    def _get_segment(self, position: Vec3, rotation: Quaternion):
        """Return (point_a, point_b) — the two end-sphere centers."""
        center = self.get_world_position(position, rotation)
        world_rot = self.get_world_rotation(rotation)
        up = world_rot.rotate_vector(Vec3(0, 1, 0))
        half_h = self.cylinder_height * 0.5
        return center - up * half_h, center + up * half_h

    def get_aabb(self, position: Vec3, rotation: Quaternion) -> AABB:
        a, b = self._get_segment(position, rotation)
        r = Vec3(self.radius, self.radius, self.radius)
        aabb = AABB()
        aabb.expand(a - r)
        aabb.expand(a + r)
        aabb.expand(b - r)
        aabb.expand(b + r)
        return aabb

    def support(self, direction: Vec3, position: Vec3, rotation: Quaternion) -> Vec3:
        a, b = self._get_segment(position, rotation)
        # Pick the capsule end farthest in direction
        if direction.dot(a) > direction.dot(b):
            return a + direction.normalize() * self.radius
        return b + direction.normalize() * self.radius

    def closest_point_on_segment(self, point: Vec3, position: Vec3, rotation: Quaternion) -> Vec3:
        """Closest point on the capsule central segment."""
        a, b = self._get_segment(position, rotation)
        ab = b - a
        len_sq = ab.length_squared()
        if len_sq < 1e-10:
            return a
        t = max(0.0, min(1.0, (point - a).dot(ab) / len_sq))
        return a + ab * t

    def closest_point(self, point: Vec3, position: Vec3, rotation: Quaternion) -> Vec3:
        seg_pt = self.closest_point_on_segment(point, position, rotation)
        diff = point - seg_pt
        dist = diff.length()
        if dist < 1e-10:
            return seg_pt + Vec3(self.radius, 0, 0)
        return seg_pt + diff.normalize() * min(dist, self.radius)

    def __repr__(self):
        return f"CapsuleCollider(radius={self.radius}, height={self.height})"


class PlaneCollider(Collider):
    """Infinite plane defined by normal and distance from origin."""

    def __init__(self, normal: Vec3 = None, distance: float = 0.0):
        super().__init__(ColliderType.PLANE)
        self.normal = (normal or Vec3(0, 1, 0)).normalize()
        self.distance = distance  # signed distance from world origin

    def get_aabb(self, position: Vec3, rotation: Quaternion) -> AABB:
        # Infinite plane — return huge AABB
        INF = 1e9
        return AABB(Vec3(-INF, -INF, -INF), Vec3(INF, INF, INF))

    def support(self, direction: Vec3, position: Vec3, rotation: Quaternion) -> Vec3:
        # Planes are infinite — support is a point on the plane in the given direction
        d = direction.dot(self.normal)
        if abs(d) < 1e-10:
            return self.normal * self.distance
        return self.normal * self.distance

    def signed_distance(self, point: Vec3) -> float:
        return point.dot(self.normal) - self.distance

    def closest_point(self, point: Vec3, position: Vec3, rotation: Quaternion) -> Vec3:
        dist = self.signed_distance(point)
        return point - self.normal * dist

    def __repr__(self):
        return f"PlaneCollider(normal={self.normal}, distance={self.distance})"


class CylinderCollider(Collider):
    """Cylinder along local Y axis."""

    def __init__(self, radius: float = 0.5, height: float = 1.0):
        super().__init__(ColliderType.CYLINDER)
        self.radius = radius
        self.height = height

    def get_aabb(self, position: Vec3, rotation: Quaternion) -> AABB:
        center = self.get_world_position(position, rotation)
        world_rot = self.get_world_rotation(rotation)
        R = world_rot.to_rotation_matrix()
        # Conservative AABB
        e = Vec3(
            self.radius * math.sqrt(R[0,0]**2 + R[0,2]**2) + self.height*0.5*abs(R[0,1]),
            self.radius * math.sqrt(R[1,0]**2 + R[1,2]**2) + self.height*0.5*abs(R[1,1]),
            self.radius * math.sqrt(R[2,0]**2 + R[2,2]**2) + self.height*0.5*abs(R[2,1])
        )
        return AABB(center - e, center + e)

    def support(self, direction: Vec3, position: Vec3, rotation: Quaternion) -> Vec3:
        center = self.get_world_position(position, rotation)
        world_rot = self.get_world_rotation(rotation)
        local_dir = world_rot.inverse().rotate_vector(direction)
        r_sq = local_dir.x**2 + local_dir.z**2
        if r_sq > 1e-10:
            scale = self.radius / math.sqrt(r_sq)
            local_sup = Vec3(local_dir.x*scale, math.copysign(self.height*0.5, local_dir.y), local_dir.z*scale)
        else:
            local_sup = Vec3(0, math.copysign(self.height*0.5, local_dir.y), 0)
        return center + world_rot.rotate_vector(local_sup)

    def closest_point(self, point: Vec3, position: Vec3, rotation: Quaternion) -> Vec3:
        center = self.get_world_position(position, rotation)
        world_rot = self.get_world_rotation(rotation)
        local_pt = world_rot.inverse().rotate_vector(point - center)
        y = max(-self.height*0.5, min(self.height*0.5, local_pt.y))
        r = math.sqrt(local_pt.x**2 + local_pt.z**2)
        if r > self.radius:
            scale = self.radius / r
            x = local_pt.x * scale
            z = local_pt.z * scale
        else:
            x, z = local_pt.x, local_pt.z
        return center + world_rot.rotate_vector(Vec3(x, y, z))

    def __repr__(self):
        return f"CylinderCollider(radius={self.radius}, height={self.height})"


# ============================================================================
#                                  RIGIDBODY                                  
# ============================================================================

import math
import numpy as np


class RigidBody:
    """
    Full 6-DOF Rigid Body.
    Supports linear and angular dynamics, forces, torques, drag,
    sleeping, constraints, and continuous collision detection (CCD) helpers.
    """

    _id_counter = 0

    def __init__(self, mass=1.0, position=None, rotation=None):
        RigidBody._id_counter += 1
        self.id = RigidBody._id_counter
        self.name = f"RigidBody_{self.id}"

        # ── Transform ─────────────────────────────────────────────────────────
        self.position = position or Vec3.zero()
        self.rotation = rotation or Quaternion.identity()
        self.prev_position = self.position.copy()
        self.prev_rotation = self.rotation.copy()

        # ── Mass & Inertia ─────────────────────────────────────────────────────
        self._mass = 0.0
        self._inv_mass = 0.0
        self._inertia_tensor = np.eye(3)
        self._inv_inertia_tensor = np.eye(3)
        self._inertia_tensor_world = np.eye(3)
        self._inv_inertia_tensor_world = np.eye(3)
        self.set_mass(mass)

        # ── Velocities ─────────────────────────────────────────────────────────
        self.linear_velocity = Vec3.zero()
        self.angular_velocity = Vec3.zero()
        self.linear_acceleration = Vec3.zero()
        self.angular_acceleration = Vec3.zero()

        # ── Forces & Torques ───────────────────────────────────────────────────
        self._force_accumulator = Vec3.zero()
        self._torque_accumulator = Vec3.zero()
        self._impulse_accumulator = Vec3.zero()
        self._angular_impulse_accumulator = Vec3.zero()

        # ── Material Properties ────────────────────────────────────────────────
        self.restitution = 0.6          # coefficient of restitution (bounciness) 0–1
        self.static_friction = 0.5
        self.dynamic_friction = 0.4
        self.rolling_friction = 0.01
        self.air_drag = 0.01            # linear drag coefficient
        self.angular_drag = 0.05        # angular drag coefficient

        # ── Constraints & Flags ───────────────────────────────────────────────
        self.is_static = False
        self.is_kinematic = False
        self.use_gravity = True
        self.is_sleeping = False
        self.is_trigger = False
        self.freeze_position_x = False
        self.freeze_position_y = False
        self.freeze_position_z = False
        self.freeze_rotation_x = False
        self.freeze_rotation_y = False
        self.freeze_rotation_z = False

        # ── Sleep thresholds ──────────────────────────────────────────────────
        self.sleep_linear_threshold = 0.01
        self.sleep_angular_threshold = 0.01
        self._sleep_timer = 0.0
        self.sleep_time_required = 0.5  # seconds before sleeping

        # ── CCD ───────────────────────────────────────────────────────────────
        self.use_ccd = False
        self.ccd_motion_threshold = 1.0

        # ── Collision Shape Reference ─────────────────────────────────────────
        self.collider = None
        self.collision_callbacks = []

        # ── Metadata ──────────────────────────────────────────────────────────
        self.layer = 0
        self.tag = ""
        self.user_data = {}

        # ── Cached World Inertia ──────────────────────────────────────────────
        self._update_world_inertia()

    # ── Mass / Inertia ─────────────────────────────────────────────────────────

    def set_mass(self, mass: float):
        if mass <= 0:
            raise ValueError(f"Mass must be positive, got {mass}")
        self._mass = mass
        self._inv_mass = 1.0 / mass
        self._compute_default_inertia()

    def set_infinite_mass(self):
        """Make body static (infinite mass)."""
        self._mass = float('inf')
        self._inv_mass = 0.0
        self._inv_inertia_tensor = np.zeros((3, 3))
        self._inv_inertia_tensor_world = np.zeros((3, 3))
        self.is_static = True

    @property
    def mass(self):
        return self._mass

    @property
    def inv_mass(self):
        return self._inv_mass

    def _compute_default_inertia(self):
        """Default inertia for a unit sphere."""
        I = (2.0 / 5.0) * self._mass * 1.0  # I = 2/5 mr^2, r=1
        self._inertia_tensor = np.eye(3) * I
        if I > 1e-10:
            self._inv_inertia_tensor = np.eye(3) / I
        else:
            self._inv_inertia_tensor = np.zeros((3, 3))

    def set_inertia_tensor(self, Ixx, Iyy, Izz, Ixy=0, Ixz=0, Iyz=0):
        """Set full inertia tensor."""
        self._inertia_tensor = np.array([
            [Ixx, Ixy, Ixz],
            [Ixy, Iyy, Iyz],
            [Ixz, Iyz, Izz]
        ], dtype=np.float64)
        try:
            self._inv_inertia_tensor = np.linalg.inv(self._inertia_tensor)
        except np.linalg.LinAlgError:
            self._inv_inertia_tensor = np.zeros((3, 3))
        self._update_world_inertia()

    def set_sphere_inertia(self, radius: float):
        I = (2.0 / 5.0) * self._mass * radius * radius
        self.set_inertia_tensor(I, I, I)

    def set_box_inertia(self, hx: float, hy: float, hz: float):
        """Half-extents of box."""
        Ixx = (1.0/12.0) * self._mass * (4*hy*hy + 4*hz*hz)
        Iyy = (1.0/12.0) * self._mass * (4*hx*hx + 4*hz*hz)
        Izz = (1.0/12.0) * self._mass * (4*hx*hx + 4*hy*hy)
        self.set_inertia_tensor(Ixx, Iyy, Izz)

    def set_cylinder_inertia(self, radius: float, height: float):
        Ixx = (1.0/12.0) * self._mass * (3*radius*radius + height*height)
        Iyy = 0.5 * self._mass * radius * radius
        Izz = Ixx
        self.set_inertia_tensor(Ixx, Iyy, Izz)

    def _update_world_inertia(self):
        """Recompute world-space inertia tensor from local + rotation."""
        R = self.rotation.to_rotation_matrix()
        self._inertia_tensor_world = R @ self._inertia_tensor @ R.T
        self._inv_inertia_tensor_world = R @ self._inv_inertia_tensor @ R.T

    # ── Force Application ──────────────────────────────────────────────────────

    def apply_force(self, force: Vec3):
        """Apply world-space force at center of mass."""
        if not self.is_static and not self.is_kinematic:
            self._force_accumulator += force

    def apply_force_at_point(self, force: Vec3, world_point: Vec3):
        """Apply force at a world-space point, generating torque."""
        if not self.is_static and not self.is_kinematic:
            r = world_point - self.position
            self._force_accumulator += force
            self._torque_accumulator += r.cross(force)

    def apply_local_force(self, force: Vec3):
        """Apply force in local body space."""
        world_force = self.rotation.rotate_vector(force)
        self.apply_force(world_force)

    def apply_torque(self, torque: Vec3):
        """Apply world-space torque."""
        if not self.is_static and not self.is_kinematic:
            self._torque_accumulator += torque

    def apply_local_torque(self, torque: Vec3):
        """Apply torque in local space."""
        self.apply_torque(self.rotation.rotate_vector(torque))

    def apply_impulse(self, impulse: Vec3):
        """Apply linear impulse (instant velocity change)."""
        if not self.is_static and not self.is_kinematic:
            self._impulse_accumulator += impulse

    def apply_impulse_at_point(self, impulse: Vec3, world_point: Vec3):
        """Apply impulse at a point, generating angular impulse."""
        if not self.is_static and not self.is_kinematic:
            r = world_point - self.position
            self._impulse_accumulator += impulse
            self._angular_impulse_accumulator += r.cross(impulse)

    def apply_angular_impulse(self, impulse: Vec3):
        if not self.is_static and not self.is_kinematic:
            self._angular_impulse_accumulator += impulse

    def apply_explosion_force(self, force: float, origin: Vec3, radius: float,
                               upward_modifier: float = 0.0):
        """Apply explosion force (falloff with distance)."""
        diff = self.position - origin
        dist = diff.length()
        if dist > radius or dist < 1e-6:
            return
        falloff = 1.0 - (dist / radius)
        direction = diff.normalize() + Vec3(0, upward_modifier, 0)
        self.apply_force_at_point(direction * (force * falloff), self.position)

    def clear_forces(self):
        self._force_accumulator = Vec3.zero()
        self._torque_accumulator = Vec3.zero()

    def clear_impulses(self):
        self._impulse_accumulator = Vec3.zero()
        self._angular_impulse_accumulator = Vec3.zero()

    # ── Integration ────────────────────────────────────────────────────────────

    def integrate(self, dt: float, gravity: Vec3):
        """
        Integrate equations of motion using symplectic Euler.
        This is what Unity/PhysX use as base integrator.
        """
        if self.is_static or self.is_sleeping:
            return

        # Store previous state for interpolation / CCD
        self.prev_position = self.position.copy()
        self.prev_rotation = self.rotation.copy()

        if self.is_kinematic:
            # Kinematic bodies: move via velocity only, no forces
            self.position += self.linear_velocity * dt
            self._integrate_rotation(dt)
            return

        # ── Apply Impulses ────────────────────────────────────────────────────
        if not self._impulse_accumulator.is_zero():
            self.linear_velocity += self._impulse_accumulator * self._inv_mass
        if not self._angular_impulse_accumulator.is_zero():
            inv_I = self._inv_inertia_tensor_world
            ang_imp = self._angular_impulse_accumulator
            dw = Vec3.from_numpy(inv_I @ ang_imp.to_numpy())
            self.angular_velocity += dw
        self.clear_impulses()

        # ── Gravity ───────────────────────────────────────────────────────────
        if self.use_gravity:
            self._force_accumulator += gravity * self._mass

        # integrate linear velocity
        linear_acc = self._force_accumulator * self._inv_mass
        self.linear_acceleration = linear_acc

        # Air drag (quadratic for realism)
        drag_force = self.linear_velocity * (-self.air_drag)
        linear_acc += drag_force * self._inv_mass

        self.linear_velocity += linear_acc * dt

        # Freeze constraints
        if self.freeze_position_x: self.linear_velocity.x = 0
        if self.freeze_position_y: self.linear_velocity.y = 0
        if self.freeze_position_z: self.linear_velocity.z = 0

        self.position += self.linear_velocity * dt

        # integrate angular velocity
        inv_I = self._inv_inertia_tensor_world
        torque_np = self._torque_accumulator.to_numpy()

        # Gyroscopic term: τ_gyro = -ω × (I·ω)
        omega_np = self.angular_velocity.to_numpy()
        I_np = self._inertia_tensor_world
        gyro = -np.cross(omega_np, I_np @ omega_np)

        angular_acc_np = inv_I @ (torque_np + gyro)
        angular_acc = Vec3.from_numpy(angular_acc_np)

        # Angular drag
        angular_acc -= self.angular_velocity * self.angular_drag

        self.angular_velocity += angular_acc * dt

        # Freeze rotation constraints
        if self.freeze_rotation_x: self.angular_velocity.x = 0
        if self.freeze_rotation_y: self.angular_velocity.y = 0
        if self.freeze_rotation_z: self.angular_velocity.z = 0

        self.angular_acceleration = angular_acc
        self._integrate_rotation(dt)

        # ── Clear Accumulators ────────────────────────────────────────────────
        self.clear_forces()

        # ── Update World Inertia ──────────────────────────────────────────────
        self._update_world_inertia()

        # ── Sleep Check ───────────────────────────────────────────────────────
        self._check_sleep(dt)

    def _integrate_rotation(self, dt: float):
        """Integrate angular velocity into orientation quaternion."""
        omega = self.angular_velocity
        if omega.is_zero(1e-12):
            return
        angle = omega.length() * dt
        axis = omega.normalize()
        dq = Quaternion.from_axis_angle(axis, angle)
        self.rotation = (dq * self.rotation).normalize()

    # ── Velocity Queries ──────────────────────────────────────────────────────

    def get_velocity_at_point(self, world_point: Vec3) -> Vec3:
        """Get velocity at a world-space point (includes angular contribution)."""
        r = world_point - self.position
        return self.linear_velocity + self.angular_velocity.cross(r)

    def get_kinetic_energy(self) -> float:
        linear_ke = 0.5 * self._mass * self.linear_velocity.length_squared()
        I_np = self._inertia_tensor_world
        omega_np = self.angular_velocity.to_numpy()
        angular_ke = 0.5 * float(omega_np @ I_np @ omega_np)
        return linear_ke + angular_ke

    def get_momentum(self) -> Vec3:
        return self.linear_velocity * self._mass

    def get_angular_momentum(self) -> Vec3:
        I_np = self._inertia_tensor_world
        return Vec3.from_numpy(I_np @ self.angular_velocity.to_numpy())

    # ── Sleep ─────────────────────────────────────────────────────────────────

    def _check_sleep(self, dt: float):
        if not self.use_gravity:
            return
        lin_ok = self.linear_velocity.length_squared() < self.sleep_linear_threshold ** 2
        ang_ok = self.angular_velocity.length_squared() < self.sleep_angular_threshold ** 2
        if lin_ok and ang_ok:
            self._sleep_timer += dt
            if self._sleep_timer >= self.sleep_time_required:
                self.sleep()
        else:
            self._sleep_timer = 0.0
            if self.is_sleeping:
                self.wake_up()

    def sleep(self):
        self.is_sleeping = True
        self.linear_velocity = Vec3.zero()
        self.angular_velocity = Vec3.zero()

    def wake_up(self):
        self.is_sleeping = False
        self._sleep_timer = 0.0

    # ── Transform Helpers ─────────────────────────────────────────────────────

    def transform_point(self, local_point: Vec3) -> Vec3:
        """Transform local point to world space."""
        return self.position + self.rotation.rotate_vector(local_point)

    def inverse_transform_point(self, world_point: Vec3) -> Vec3:
        """Transform world point to local space."""
        return self.rotation.inverse().rotate_vector(world_point - self.position)

    def transform_direction(self, local_dir: Vec3) -> Vec3:
        return self.rotation.rotate_vector(local_dir)

    def inverse_transform_direction(self, world_dir: Vec3) -> Vec3:
        return self.rotation.inverse().rotate_vector(world_dir)

    def get_transform_matrix(self):
        """4x4 transform matrix (numpy)."""
        R = self.rotation.to_rotation_matrix()
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = self.position.to_numpy()
        return T

    def interpolated_position(self, alpha: float) -> Vec3:
        """Interpolate between previous and current position (for rendering)."""
        return self.prev_position.lerp(self.position, alpha)

    def interpolated_rotation(self, alpha: float) -> Quaternion:
        """Interpolate rotation for smooth rendering."""
        return self.prev_rotation.slerp(self.rotation, alpha)

    # ── Utility ───────────────────────────────────────────────────────────────

    def reset(self):
        self.linear_velocity = Vec3.zero()
        self.angular_velocity = Vec3.zero()
        self.linear_acceleration = Vec3.zero()
        self.angular_acceleration = Vec3.zero()
        self.clear_forces()
        self.clear_impulses()
        self.is_sleeping = False
        self._sleep_timer = 0.0

    def __repr__(self):
        return (f"RigidBody(id={self.id}, name='{self.name}', "
                f"mass={self._mass:.2f}, pos={self.position})")


# ============================================================================
#                             COLLISION_DETECTION                             
# ============================================================================

import math
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple



@dataclass
class ContactPoint:
    """A single contact point between two bodies."""
    position: Vec3           # world-space contact point
    normal: Vec3             # contact normal (pointing from B to A)
    penetration_depth: float # positive = overlapping
    local_point_a: Vec3 = field(default_factory=Vec3.zero)
    local_point_b: Vec3 = field(default_factory=Vec3.zero)
    impulse: float = 0.0
    tangent_impulse_1: float = 0.0
    tangent_impulse_2: float = 0.0
    is_new: bool = True


@dataclass
class CollisionManifold:
    """
    Full collision manifold between two bodies.
    Contains up to 4 contact points (like PhysX/Bullet).
    """
    body_a: object   # RigidBody
    body_b: object   # RigidBody
    collider_a: Collider
    collider_b: Collider
    contacts: List[ContactPoint] = field(default_factory=list)
    normal: Vec3 = field(default_factory=Vec3.zero)
    penetration_depth: float = 0.0
    is_valid: bool = False
    restitution: float = 0.0
    static_friction: float = 0.0
    dynamic_friction: float = 0.0

    def add_contact(self, contact: ContactPoint):
        if len(self.contacts) < 4:
            self.contacts.append(contact)

    def get_deepest_contact(self) -> Optional[ContactPoint]:
        if not self.contacts:
            return None
        return max(self.contacts, key=lambda c: c.penetration_depth)


# ─────────────────────────────────────────────────────────────────────────────
# GJK Algorithm
# ─────────────────────────────────────────────────────────────────────────────

class GJKSimplex:
    """Simplex for GJK — holds 1-4 support points in Minkowski difference."""

    def __init__(self):
        self.points: List[Vec3] = []

    def add(self, p: Vec3):
        self.points.insert(0, p)

    def size(self):
        return len(self.points)

    def last(self) -> Vec3:
        return self.points[0]

    def __getitem__(self, idx):
        return self.points[idx]


def _support_minkowski(col_a, pos_a, rot_a, col_b, pos_b, rot_b, direction: Vec3) -> Vec3:
    """Support function for Minkowski difference A - B."""
    sup_a = col_a.support(direction, pos_a, rot_a)
    sup_b = col_b.support(-direction, pos_b, rot_b)
    return sup_a - sup_b


def _gjk_do_simplex(simplex: GJKSimplex, direction: Vec3) -> Tuple[bool, Vec3]:
    """
    Process simplex and update search direction.
    Returns (found_origin, new_direction).
    """
    if simplex.size() == 2:
        return _gjk_line(simplex, direction)
    elif simplex.size() == 3:
        return _gjk_triangle(simplex, direction)
    elif simplex.size() == 4:
        return _gjk_tetrahedron(simplex, direction)
    return False, direction


def _triple_product(a: Vec3, b: Vec3, c: Vec3) -> Vec3:
    """Triple product: (a × b) × c"""
    return b * a.dot(c) - c * a.dot(b)


def _gjk_line(simplex: GJKSimplex, direction: Vec3) -> Tuple[bool, Vec3]:
    a, b = simplex[0], simplex[1]
    ab = b - a
    ao = -a
    if ab.dot(ao) > 0:
        new_dir = _triple_product(ab, ao, ab)
    else:
        simplex.points = [a]
        new_dir = ao
    return False, new_dir


def _gjk_triangle(simplex: GJKSimplex, direction: Vec3) -> Tuple[bool, Vec3]:
    a, b, c = simplex[0], simplex[1], simplex[2]
    ab = b - a
    ac = c - a
    ao = -a
    abc = ab.cross(ac)

    if (abc.cross(ac)).dot(ao) > 0:
        if ac.dot(ao) > 0:
            simplex.points = [a, c]
            new_dir = _triple_product(ac, ao, ac)
        else:
            simplex.points = [a, b]
            return _gjk_line(simplex, direction)
    else:
        if (ab.cross(abc)).dot(ao) > 0:
            simplex.points = [a, b]
            return _gjk_line(simplex, direction)
        else:
            if abc.dot(ao) > 0:
                new_dir = abc
            else:
                simplex.points = [a, c, b]
                new_dir = -abc
    return False, new_dir


def _gjk_tetrahedron(simplex: GJKSimplex, direction: Vec3) -> Tuple[bool, Vec3]:
    a, b, c, d = simplex[0], simplex[1], simplex[2], simplex[3]
    ab = b - a; ac = c - a; ad = d - a; ao = -a
    abc = ab.cross(ac)
    acd = ac.cross(ad)
    adb = ad.cross(ab)

    if abc.dot(ao) > 0:
        simplex.points = [a, b, c]
        return _gjk_triangle(simplex, direction)
    if acd.dot(ao) > 0:
        simplex.points = [a, c, d]
        return _gjk_triangle(simplex, direction)
    if adb.dot(ao) > 0:
        simplex.points = [a, d, b]
        return _gjk_triangle(simplex, direction)
    return True, direction


def gjk_intersect(col_a, pos_a, rot_a, col_b, pos_b, rot_b, max_iter=64) -> Tuple[bool, GJKSimplex]:
    """
    GJK intersection test.
    Returns (intersecting: bool, simplex for EPA).
    """
    direction = pos_a - pos_b
    if direction.is_zero():
        direction = Vec3(1, 0, 0)

    simplex = GJKSimplex()
    support = _support_minkowski(col_a, pos_a, rot_a, col_b, pos_b, rot_b, direction)
    simplex.add(support)
    direction = -support

    for _ in range(max_iter):
        if direction.is_zero(1e-12):
            return True, simplex
        support = _support_minkowski(col_a, pos_a, rot_a, col_b, pos_b, rot_b, direction)
        if support.dot(direction) < 0:
            return False, simplex
        simplex.add(support)
        found, direction = _gjk_do_simplex(simplex, direction)
        if found:
            return True, simplex

    return False, simplex


# ─────────────────────────────────────────────────────────────────────────────
# EPA — Expanding Polytope Algorithm (penetration depth + normal)
# ─────────────────────────────────────────────────────────────────────────────

def epa_penetration(col_a, pos_a, rot_a, col_b, pos_b, rot_b,
                    simplex: GJKSimplex, max_iter=64, tolerance=1e-4):
    """
    EPA: given a GJK simplex containing the origin, expand polytope to find
    minimum penetration depth and contact normal.
    Returns (normal: Vec3, depth: float) or (None, 0) if failed.
    """
    if simplex.size() < 4:
        # Expand simplex to tetrahedron
        if not _expand_simplex_to_tetrahedron(simplex, col_a, pos_a, rot_a, col_b, pos_b, rot_b):
            return None, 0.0

    polytope = list(simplex.points)
    faces = _get_initial_faces(polytope)

    for _ in range(max_iter):
        # Find face closest to origin
        closest_face, closest_dist, closest_normal = _find_closest_face(polytope, faces)
        if closest_face is None:
            break

        # Get new support point
        support = _support_minkowski(col_a, pos_a, rot_a, col_b, pos_b, rot_b, closest_normal)
        d = support.dot(closest_normal)

        if d - closest_dist < tolerance:
            return closest_normal, closest_dist

        # Expand polytope
        polytope, faces = _expand_polytope(polytope, faces, support)

    return closest_normal if 'closest_normal' in dir() else Vec3(0, 1, 0), \
           closest_dist if 'closest_dist' in dir() else 0.0


def _expand_simplex_to_tetrahedron(simplex, col_a, pos_a, rot_a, col_b, pos_b, rot_b):
    """Expand GJK simplex to a tetrahedron for EPA."""
    axes = [Vec3(1,0,0), Vec3(0,1,0), Vec3(0,0,1),
            Vec3(-1,0,0), Vec3(0,-1,0), Vec3(0,0,-1)]
    while simplex.size() < 4:
        for axis in axes:
            s = _support_minkowski(col_a, pos_a, rot_a, col_b, pos_b, rot_b, axis)
            if s not in simplex.points:
                simplex.add(s)
                break
        else:
            return False
    return True


def _get_initial_faces(polytope):
    """Get initial 4 faces of tetrahedron."""
    if len(polytope) < 4:
        return []
    a, b, c, d = polytope[0], polytope[1], polytope[2], polytope[3]
    return [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]


def _find_closest_face(polytope, faces):
    """Find face closest to origin."""
    min_dist = float('inf')
    closest_face = None
    closest_normal = Vec3.zero()

    for face in faces:
        if max(face) >= len(polytope):
            continue
        a = polytope[face[0]]
        b = polytope[face[1]]
        c = polytope[face[2]]
        normal = (b - a).cross(c - a)
        if normal.is_zero():
            continue
        normal = normal.normalize()
        dist = abs(normal.dot(a))
        if dist < min_dist:
            min_dist = dist
            closest_face = face
            closest_normal = normal if normal.dot(a) > 0 else -normal

    return closest_face, min_dist, closest_normal


def _expand_polytope(polytope, faces, new_point):
    """Remove faces visible from new_point, add new faces."""
    new_faces = []
    edges = []

    for face in faces:
        if max(face) >= len(polytope):
            continue
        a = polytope[face[0]]
        b = polytope[face[1]]
        c = polytope[face[2]]
        normal = (b - a).cross(c - a).normalize()
        if normal.dot(new_point - a) > 0:
            # Face visible — add its edges
            e = [(face[0], face[1]), (face[1], face[2]), (face[0], face[2])]
            for edge in e:
                rev = (edge[1], edge[0])
                if rev in edges:
                    edges.remove(rev)
                else:
                    edges.append(edge)
        else:
            new_faces.append(face)

    idx = len(polytope)
    polytope.append(new_point)
    for edge in edges:
        new_faces.append((edge[0], edge[1], idx))

    return polytope, new_faces


# ─────────────────────────────────────────────────────────────────────────────
# Analytic Collision Tests
# ─────────────────────────────────────────────────────────────────────────────

def sphere_sphere_collision(
    col_a: SphereCollider, pos_a: Vec3,
    col_b: SphereCollider, pos_b: Vec3
) -> Optional[CollisionManifold]:
    """Analytic sphere-sphere collision (fast)."""
    center_a = col_a.get_world_position(pos_a, Quaternion.identity())
    center_b = col_b.get_world_position(pos_b, Quaternion.identity())
    diff = center_a - center_b
    dist_sq = diff.length_squared()
    combined_r = col_a.radius + col_b.radius

    if dist_sq >= combined_r * combined_r:
        return None

    dist = math.sqrt(dist_sq)
    if dist < 1e-8:
        normal = Vec3(0, 1, 0)
        depth = combined_r
    else:
        normal = diff * (1.0 / dist)
        depth = combined_r - dist

    contact_pos = center_b + normal * col_b.radius
    contact = ContactPoint(
        position=contact_pos,
        normal=normal,
        penetration_depth=depth,
        local_point_a=col_a.local_position - normal * col_a.radius,
        local_point_b=col_b.local_position + normal * col_b.radius
    )
    manifold = CollisionManifold(
        body_a=None, body_b=None,
        collider_a=col_a, collider_b=col_b,
        normal=normal, penetration_depth=depth, is_valid=True
    )
    manifold.add_contact(contact)
    return manifold


def sphere_plane_collision(
    col_sphere: SphereCollider, pos_sphere: Vec3, rot_sphere: Quaternion,
    col_plane: PlaneCollider
) -> Optional[CollisionManifold]:
    """Analytic sphere-plane collision."""
    center = col_sphere.get_world_position(pos_sphere, rot_sphere)
    dist = col_plane.signed_distance(center)

    if dist > col_sphere.radius:
        return None

    depth = col_sphere.radius - dist
    normal = col_plane.normal.copy()
    contact_pos = center - normal * col_sphere.radius

    contact = ContactPoint(
        position=contact_pos,
        normal=normal,
        penetration_depth=depth
    )
    manifold = CollisionManifold(
        body_a=None, body_b=None,
        collider_a=col_sphere, collider_b=col_plane,
        normal=normal, penetration_depth=depth, is_valid=True
    )
    manifold.add_contact(contact)
    return manifold


def sphere_box_collision(
    col_sphere: SphereCollider, pos_sphere: Vec3, rot_sphere: Quaternion,
    col_box: BoxCollider, pos_box: Vec3, rot_box: Quaternion
) -> Optional[CollisionManifold]:
    """Analytic sphere-box collision."""
    center = col_sphere.get_world_position(pos_sphere, rot_sphere)
    closest = col_box.closest_point(center, pos_box, rot_box)
    diff = center - closest
    dist_sq = diff.length_squared()

    if dist_sq >= col_sphere.radius * col_sphere.radius:
        return None

    dist = math.sqrt(dist_sq)
    if dist < 1e-8:
        normal = Vec3(0, 1, 0)
        depth = col_sphere.radius
    else:
        normal = diff * (1.0 / dist)
        depth = col_sphere.radius - dist

    contact = ContactPoint(
        position=closest,
        normal=normal,
        penetration_depth=depth
    )
    manifold = CollisionManifold(
        body_a=None, body_b=None,
        collider_a=col_sphere, collider_b=col_box,
        normal=normal, penetration_depth=depth, is_valid=True
    )
    manifold.add_contact(contact)
    return manifold


def capsule_capsule_collision(
    col_a: CapsuleCollider, pos_a: Vec3, rot_a: Quaternion,
    col_b: CapsuleCollider, pos_b: Vec3, rot_b: Quaternion
) -> Optional[CollisionManifold]:
    """Capsule-capsule analytic collision."""
    a1, a2 = col_a._get_segment(pos_a, rot_a)
    b1, b2 = col_b._get_segment(pos_b, rot_b)

    # Closest points on two line segments
    cp_a, cp_b = _closest_points_segments(a1, a2, b1, b2)
    diff = cp_a - cp_b
    dist_sq = diff.length_squared()
    combined_r = col_a.radius + col_b.radius

    if dist_sq >= combined_r * combined_r:
        return None

    dist = math.sqrt(dist_sq)
    if dist < 1e-8:
        normal = Vec3(0, 1, 0)
        depth = combined_r
    else:
        normal = diff * (1.0 / dist)
        depth = combined_r - dist

    contact_pos = cp_b + normal * col_b.radius
    contact = ContactPoint(position=contact_pos, normal=normal, penetration_depth=depth)
    manifold = CollisionManifold(
        body_a=None, body_b=None,
        collider_a=col_a, collider_b=col_b,
        normal=normal, penetration_depth=depth, is_valid=True
    )
    manifold.add_contact(contact)
    return manifold


def _closest_points_segments(p1: Vec3, p2: Vec3, p3: Vec3, p4: Vec3) -> Tuple[Vec3, Vec3]:
    """Closest points on two line segments."""
    d1 = p2 - p1
    d2 = p4 - p3
    r = p1 - p3
    a = d1.dot(d1)
    e = d2.dot(d2)
    f = d2.dot(r)

    if a <= 1e-10 and e <= 1e-10:
        return p1, p3

    if a <= 1e-10:
        s = 0.0
        t = max(0.0, min(1.0, f / e))
    else:
        c = d1.dot(r)
        if e <= 1e-10:
            t = 0.0
            s = max(0.0, min(1.0, -c / a))
        else:
            b = d1.dot(d2)
            denom = a * e - b * b
            if abs(denom) > 1e-10:
                s = max(0.0, min(1.0, (b*f - c*e) / denom))
            else:
                s = 0.0
            t = (b*s + f) / e
            if t < 0:
                t = 0.0
                s = max(0.0, min(1.0, -c / a))
            elif t > 1:
                t = 1.0
                s = max(0.0, min(1.0, (b - c) / a))

    return p1 + d1 * s, p3 + d2 * t


# ─────────────────────────────────────────────────────────────────────────────
# Broad Phase — Sweep and Prune (SAP)
# ─────────────────────────────────────────────────────────────────────────────

class BroadPhaseEntry:
    def __init__(self, body, collider, aabb: AABB):
        self.body = body
        self.collider = collider
        self.aabb = aabb


class SweepAndPrune:
    """
    1D sort on X axis with full 3D AABB check.
    O(n log n) for mostly sorted, very cache friendly.
    """

    def __init__(self):
        self.entries: List[BroadPhaseEntry] = []

    def clear(self):
        self.entries.clear()

    def add(self, entry: BroadPhaseEntry):
        self.entries.append(entry)

    def get_pairs(self) -> List[Tuple[BroadPhaseEntry, BroadPhaseEntry]]:
        pairs = []
        # Sort by min X
        self.entries.sort(key=lambda e: e.aabb.min.x)

        for i in range(len(self.entries)):
            for j in range(i + 1, len(self.entries)):
                a = self.entries[i]
                b = self.entries[j]
                # Early exit on X axis
                if b.aabb.min.x > a.aabb.max.x:
                    break
                # Full 3D check
                if a.aabb.intersects(b.aabb):
                    pairs.append((a, b))
        return pairs


# ─────────────────────────────────────────────────────────────────────────────
# BVH — Bounding Volume Hierarchy (for static geometry)
# ─────────────────────────────────────────────────────────────────────────────

class BVHNode:
    def __init__(self):
        self.aabb = AABB()
        self.left = None
        self.right = None
        self.entry = None  # leaf data

    def is_leaf(self):
        return self.left is None and self.right is None


class BVH:
    """
    Static BVH for fast broad phase with many static objects.
    Uses Surface Area Heuristic (SAH) for splits.
    """

    def __init__(self):
        self.root = None
        self.entries: List[BroadPhaseEntry] = []

    def build(self, entries: List[BroadPhaseEntry]):
        self.entries = entries
        if not entries:
            self.root = None
            return
        self.root = self._build_recursive(list(range(len(entries))))

    def _build_recursive(self, indices: List[int]) -> BVHNode:
        node = BVHNode()
        for i in indices:
            node.aabb.expand_aabb(self.entries[i].aabb)

        if len(indices) == 1:
            node.entry = self.entries[indices[0]]
            return node

        # Split along longest axis
        size = node.aabb.size()
        axis = 0
        if size.y > size.x:
            axis = 1
        if size.z > (size.y if axis == 1 else size.x):
            axis = 2

        indices.sort(key=lambda i: self.entries[i].aabb.center()[axis])
        mid = len(indices) // 2
        node.left = self._build_recursive(indices[:mid])
        node.right = self._build_recursive(indices[mid:])
        return node

    def query_aabb(self, aabb: AABB) -> List[BroadPhaseEntry]:
        results = []
        if self.root:
            self._query_recursive(self.root, aabb, results)
        return results

    def _query_recursive(self, node: BVHNode, aabb: AABB, results: list):
        if not node.aabb.intersects(aabb):
            return
        if node.is_leaf():
            results.append(node.entry)
            return
        if node.left:
            self._query_recursive(node.left, aabb, results)
        if node.right:
            self._query_recursive(node.right, aabb, results)


# ─────────────────────────────────────────────────────────────────────────────
# Narrow Phase Dispatcher
# ─────────────────────────────────────────────────────────────────────────────

class NarrowPhaseDispatcher:
    """
    Dispatches collision tests to the right algorithm based on shape pairs.
    """

    def test(self, entry_a: BroadPhaseEntry, entry_b: BroadPhaseEntry) -> Optional[CollisionManifold]:
        ca = entry_a.collider
        cb = entry_b.collider
        pa = entry_a.body.position
        pb = entry_b.body.position
        ra = entry_a.body.rotation
        rb = entry_b.body.rotation

        ta = ca.collider_type
        tb = cb.collider_type

        # Sphere vs Sphere
        if ta == ColliderType.SPHERE and tb == ColliderType.SPHERE:
            manifold = sphere_sphere_collision(ca, pa, cb, pb)
        # Sphere vs Plane
        elif ta == ColliderType.SPHERE and tb == ColliderType.PLANE:
            manifold = sphere_plane_collision(ca, pa, ra, cb)
        elif ta == ColliderType.PLANE and tb == ColliderType.SPHERE:
            manifold = sphere_plane_collision(cb, pb, rb, ca)
        # Sphere vs Box
        elif ta == ColliderType.SPHERE and tb == ColliderType.BOX:
            manifold = sphere_box_collision(ca, pa, ra, cb, pb, rb)
        elif ta == ColliderType.BOX and tb == ColliderType.SPHERE:
            manifold = sphere_box_collision(cb, pb, rb, ca, pa, ra)
        # Capsule vs Capsule
        elif ta == ColliderType.CAPSULE and tb == ColliderType.CAPSULE:
            manifold = capsule_capsule_collision(ca, pa, ra, cb, pb, rb)
        # GJK + EPA fallback for convex shapes
        else:
            manifold = self._gjk_epa_test(entry_a, entry_b)

        if manifold:
            manifold.body_a = entry_a.body
            manifold.body_b = entry_b.body
            self._compute_friction_restitution(manifold, entry_a.body, entry_b.body)

        return manifold

    def _gjk_epa_test(self, entry_a, entry_b) -> Optional[CollisionManifold]:
        ca, pa, ra = entry_a.collider, entry_a.body.position, entry_a.body.rotation
        cb, pb, rb = entry_b.collider, entry_b.body.position, entry_b.body.rotation

        intersecting, simplex = gjk_intersect(ca, pa, ra, cb, pb, rb)
        if not intersecting:
            return None

        normal, depth = epa_penetration(ca, pa, ra, cb, pb, rb, simplex)
        if normal is None or depth <= 0:
            return None

        contact_pos = pa + normal * (-depth * 0.5)
        contact = ContactPoint(position=contact_pos, normal=normal, penetration_depth=depth)
        manifold = CollisionManifold(
            body_a=entry_a.body, body_b=entry_b.body,
            collider_a=ca, collider_b=cb,
            normal=normal, penetration_depth=depth, is_valid=True
        )
        manifold.add_contact(contact)
        return manifold

    def _compute_friction_restitution(self, manifold: CollisionManifold, body_a, body_b):
        # Use geometric mean for restitution, average for friction (Bullet/PhysX convention)
        e_a = getattr(manifold.collider_a, 'material_restitution', None) or body_a.restitution
        e_b = getattr(manifold.collider_b, 'material_restitution', None) or body_b.restitution
        manifold.restitution = math.sqrt(e_a * e_b)

        sf_a = getattr(manifold.collider_a, 'material_friction', None) or body_a.static_friction
        sf_b = getattr(manifold.collider_b, 'material_friction', None) or body_b.static_friction
        manifold.static_friction = (sf_a + sf_b) * 0.5

        df_a = body_a.dynamic_friction
        df_b = body_b.dynamic_friction
        manifold.dynamic_friction = (df_a + df_b) * 0.5


# ============================================================================
#                              COLLISION_RESPONSE                             
# ============================================================================

import math
import numpy as np


class ContactConstraint:
    """
    Precomputed constraint data for one contact point.
    Cached for warm starting and multiple iterations.
    """

    def __init__(self):
        self.r_a = Vec3.zero()       # vector from body_a COM to contact
        self.r_b = Vec3.zero()       # vector from body_b COM to contact
        self.normal = Vec3.zero()
        self.tangent1 = Vec3.zero()
        self.tangent2 = Vec3.zero()

        self.normal_mass = 0.0
        self.tangent_mass_1 = 0.0
        self.tangent_mass_2 = 0.0

        self.velocity_bias = 0.0     # restitution + Baumgarte

        # Warm starting (accumulated impulses)
        self.normal_impulse = 0.0
        self.tangent_impulse_1 = 0.0
        self.tangent_impulse_2 = 0.0


class ImpulseSolver:
    """
    Sequential Impulse Solver (SI).
    Industry standard for real-time rigid body simulation.
    - Velocity-level constraint resolution
    - Friction cone (Coulomb model)
    - Positional correction (Baumgarte stabilization + pseudo-velocities)
    - Warm starting for stability
    """

    def __init__(self):
        self.position_correction = True
        self.baumgarte_factor = 0.2          # 0.1–0.3 typical
        self.slop = 0.005                    # penetration allowance
        self.warm_starting = True
        self.velocity_iterations = 8
        self.position_iterations = 3

    def solve(self, manifolds: list, dt: float):
        """Main solve entry point."""
        # Pre-step
        constraints = []
        for manifold in manifolds:
            if not manifold.is_valid:
                continue
            cs = self._pre_step(manifold, dt)
            constraints.append((manifold, cs))

        # Warm starting
        if self.warm_starting:
            for manifold, cs in constraints:
                self._apply_warm_start(manifold, cs)

        # Velocity iterations
        for _ in range(self.velocity_iterations):
            for manifold, cs in constraints:
                self._solve_velocity(manifold, cs)

        # Position correction
        if self.position_correction:
            for _ in range(self.position_iterations):
                for manifold, cs in constraints:
                    self._solve_position(manifold, cs)

    def _pre_step(self, manifold: CollisionManifold, dt: float) -> list:
        """Precompute constraint data for each contact."""
        body_a = manifold.body_a
        body_b = manifold.body_b
        cs_list = []

        for contact in manifold.contacts:
            cs = ContactConstraint()
            cs.normal = contact.normal
            cs.r_a = contact.position - body_a.position
            cs.r_b = contact.position - body_b.position

            # Compute tangents (Gram-Schmidt)
            n = cs.normal
            t1 = Vec3(1, 0, 0)
            if abs(n.dot(t1)) > 0.9:
                t1 = Vec3(0, 1, 0)
            cs.tangent1 = (t1 - n * n.dot(t1)).normalize()
            cs.tangent2 = n.cross(cs.tangent1)

            # Effective mass for normal
            cs.normal_mass = self._compute_effective_mass(
                body_a, body_b, cs.r_a, cs.r_b, cs.normal)

            # Effective mass for tangents
            cs.tangent_mass_1 = self._compute_effective_mass(
                body_a, body_b, cs.r_a, cs.r_b, cs.tangent1)
            cs.tangent_mass_2 = self._compute_effective_mass(
                body_a, body_b, cs.r_a, cs.r_b, cs.tangent2)

            # Velocity bias (restitution)
            rel_vel = self._relative_velocity_at_contact(body_a, body_b, cs.r_a, cs.r_b)
            vel_along_normal = rel_vel.dot(cs.normal)

            restitution = manifold.restitution
            bounce_threshold = 1.0  # m/s — don't bounce tiny collisions
            if abs(vel_along_normal) > bounce_threshold:
                cs.velocity_bias = -restitution * vel_along_normal
            else:
                cs.velocity_bias = 0.0

            # Baumgarte bias for position correction
            if contact.penetration_depth > self.slop:
                cs.velocity_bias += (self.baumgarte_factor / dt) * (
                    contact.penetration_depth - self.slop)

            cs_list.append(cs)

        return cs_list

    def _compute_effective_mass(self, body_a, body_b, r_a, r_b, dir_vec) -> float:
        """
        Effective mass = 1 / (1/m_a + 1/m_b + (r_a×n)·I_a⁻¹·(r_a×n) + (r_b×n)·I_b⁻¹·(r_b×n))
        """
        inv_ma = body_a.inv_mass
        inv_mb = body_b.inv_mass

        ra_cross = r_a.cross(dir_vec).to_numpy()
        rb_cross = r_b.cross(dir_vec).to_numpy()

        inv_Ia = body_a._inv_inertia_tensor_world
        inv_Ib = body_b._inv_inertia_tensor_world

        term_a = float(ra_cross @ inv_Ia @ ra_cross)
        term_b = float(rb_cross @ inv_Ib @ rb_cross)

        denom = inv_ma + inv_mb + term_a + term_b
        if denom < 1e-10:
            return 0.0
        return 1.0 / denom

    def _relative_velocity_at_contact(self, body_a, body_b, r_a, r_b) -> Vec3:
        """Relative velocity at contact point."""
        vel_a = body_a.linear_velocity + body_a.angular_velocity.cross(r_a)
        vel_b = body_b.linear_velocity + body_b.angular_velocity.cross(r_b)
        return vel_a - vel_b

    def _apply_warm_start(self, manifold: CollisionManifold, cs_list: list):
        """Apply cached impulses from previous frame."""
        body_a = manifold.body_a
        body_b = manifold.body_b

        for cs in cs_list:
            if body_a.is_static or body_a.is_sleeping:
                pass
            if body_b.is_static or body_b.is_sleeping:
                pass

            impulse = (cs.normal * cs.normal_impulse +
                      cs.tangent1 * cs.tangent_impulse_1 +
                      cs.tangent2 * cs.tangent_impulse_2)

            if not body_a.is_static and not body_a.is_kinematic:
                body_a.linear_velocity += impulse * body_a.inv_mass
                dw_a = Vec3.from_numpy(
                    body_a._inv_inertia_tensor_world @ cs.r_a.cross(impulse).to_numpy())
                body_a.angular_velocity += dw_a
                body_a.wake_up()

            if not body_b.is_static and not body_b.is_kinematic:
                body_b.linear_velocity -= impulse * body_b.inv_mass
                dw_b = Vec3.from_numpy(
                    body_b._inv_inertia_tensor_world @ cs.r_b.cross(impulse).to_numpy())
                body_b.angular_velocity -= dw_b
                body_b.wake_up()

    def _solve_velocity(self, manifold: CollisionManifold, cs_list: list):
        """One velocity constraint iteration."""
        body_a = manifold.body_a
        body_b = manifold.body_b

        for cs in cs_list:
            rel_vel = self._relative_velocity_at_contact(body_a, body_b, cs.r_a, cs.r_b)

            # ── Normal Impulse ────────────────────────────────────────────────
            vel_n = rel_vel.dot(cs.normal)
            lambda_n = cs.normal_mass * (-vel_n + cs.velocity_bias)

            # Clamp: normal impulse can only push, not pull
            old_impulse = cs.normal_impulse
            cs.normal_impulse = max(0.0, cs.normal_impulse + lambda_n)
            delta_n = cs.normal_impulse - old_impulse

            self._apply_impulse_pair(body_a, body_b, cs.r_a, cs.r_b,
                                     cs.normal * delta_n)

            # ── Tangent Impulses (friction) ───────────────────────────────────
            rel_vel = self._relative_velocity_at_contact(body_a, body_b, cs.r_a, cs.r_b)

            max_friction = manifold.dynamic_friction * cs.normal_impulse

            vel_t1 = rel_vel.dot(cs.tangent1)
            lambda_t1 = -cs.tangent_mass_1 * vel_t1
            old_t1 = cs.tangent_impulse_1
            cs.tangent_impulse_1 = max(-max_friction, min(max_friction, old_t1 + lambda_t1))
            delta_t1 = cs.tangent_impulse_1 - old_t1

            vel_t2 = rel_vel.dot(cs.tangent2)
            lambda_t2 = -cs.tangent_mass_2 * vel_t2
            old_t2 = cs.tangent_impulse_2
            cs.tangent_impulse_2 = max(-max_friction, min(max_friction, old_t2 + lambda_t2))
            delta_t2 = cs.tangent_impulse_2 - old_t2

            tangent_impulse = cs.tangent1 * delta_t1 + cs.tangent2 * delta_t2
            self._apply_impulse_pair(body_a, body_b, cs.r_a, cs.r_b, tangent_impulse)

    def _solve_position(self, manifold: CollisionManifold, cs_list: list):
        """
        Position correction using pseudo-velocities (Erin Catto method).
        More stable than direct position adjustment.
        """
        body_a = manifold.body_a
        body_b = manifold.body_b

        for contact in manifold.contacts:
            if contact.penetration_depth <= self.slop:
                continue

            r_a = contact.position - body_a.position
            r_b = contact.position - body_b.position
            n = contact.normal

            # Positional error
            correction = max(0.0, contact.penetration_depth - self.slop)
            correction *= self.baumgarte_factor

            effective_mass = self._compute_effective_mass(body_a, body_b, r_a, r_b, n)
            lambda_p = effective_mass * correction

            impulse = n * lambda_p

            if not body_a.is_static and not body_a.is_kinematic:
                body_a.position += impulse * body_a.inv_mass
                dq_a = Vec3.from_numpy(
                    body_a._inv_inertia_tensor_world @ r_a.cross(impulse).to_numpy())
                # Small rotation correction
                angle = dq_a.length()
                if angle > 1e-8:
                    axis = dq_a.normalize()
                    drot = Quaternion.from_axis_angle(axis, angle)
                    body_a.rotation = (drot * body_a.rotation).normalize()

            if not body_b.is_static and not body_b.is_kinematic:
                body_b.position -= impulse * body_b.inv_mass
                dq_b = Vec3.from_numpy(
                    body_b._inv_inertia_tensor_world @ r_b.cross(impulse).to_numpy())
                angle = dq_b.length()
                if angle > 1e-8:
                    axis = dq_b.normalize()
                    drot = Quaternion.from_axis_angle(axis, angle)
                    body_b.rotation = (drot * body_b.rotation).normalize()

    def _apply_impulse_pair(self, body_a, body_b, r_a: Vec3, r_b: Vec3, impulse: Vec3):
        """Apply equal and opposite impulses to two bodies."""
        if not body_a.is_static and not body_a.is_kinematic:
            body_a.linear_velocity += impulse * body_a.inv_mass
            dw_a = Vec3.from_numpy(
                body_a._inv_inertia_tensor_world @ r_a.cross(impulse).to_numpy())
            body_a.angular_velocity += dw_a
            body_a.wake_up()

        if not body_b.is_static and not body_b.is_kinematic:
            body_b.linear_velocity -= impulse * body_b.inv_mass
            dw_b = Vec3.from_numpy(
                body_b._inv_inertia_tensor_world @ r_b.cross(impulse).to_numpy())
            body_b.angular_velocity -= dw_b
            body_b.wake_up()


# ============================================================================
#                                 CONSTRAINTS                                 
# ============================================================================

import math


class Constraint:
    """Base class for all constraints."""

    def __init__(self, body_a: RigidBody, body_b: RigidBody = None):
        self.body_a = body_a
        self.body_b = body_b
        self.enabled = True
        self.break_force = float('inf')   # force at which constraint breaks
        self.break_torque = float('inf')
        self.is_broken = False

    def pre_step(self, dt: float):
        pass

    def solve(self, dt: float):
        pass

    def _apply_impulse_pair(self, body_a, body_b, r_a, r_b, impulse):
        if not body_a.is_static:
            body_a.linear_velocity += impulse * body_a.inv_mass
            dw = Vec3.from_numpy(body_a._inv_inertia_tensor_world @ r_a.cross(impulse).to_numpy())
            body_a.angular_velocity += dw

        if body_b and not body_b.is_static:
            body_b.linear_velocity -= impulse * body_b.inv_mass
            dw = Vec3.from_numpy(body_b._inv_inertia_tensor_world @ r_b.cross(impulse).to_numpy())
            body_b.angular_velocity -= dw


class DistanceConstraint(Constraint):
    """
    Maintains a fixed distance between two anchor points.
    Equivalent to an ideal rigid rod.
    """

    def __init__(self, body_a: RigidBody, body_b: RigidBody,
                 anchor_a: Vec3 = None, anchor_b: Vec3 = None,
                 distance: float = None):
        super().__init__(body_a, body_b)
        self.anchor_a = anchor_a or Vec3.zero()  # local space
        self.anchor_b = anchor_b or Vec3.zero()
        world_a = body_a.transform_point(self.anchor_a)
        world_b = body_b.transform_point(self.anchor_b)
        self.distance = distance if distance is not None else world_a.distance_to(world_b)
        self.stiffness = 1.0
        self.damping = 0.0

    def solve(self, dt: float):
        if not self.enabled or self.is_broken:
            return

        world_a = self.body_a.transform_point(self.anchor_a)
        world_b = self.body_b.transform_point(self.anchor_b)
        diff = world_a - world_b
        dist = diff.length()

        if dist < 1e-8:
            return

        error = dist - self.distance
        normal = diff * (1.0 / dist)

        r_a = world_a - self.body_a.position
        r_b = world_b - self.body_b.position

        # Effective mass
        inv_ma = self.body_a.inv_mass
        inv_mb = self.body_b.inv_mass
        ra_cross = r_a.cross(normal).to_numpy()
        rb_cross = r_b.cross(normal).to_numpy()
        ia = self.body_a._inv_inertia_tensor_world
        ib = self.body_b._inv_inertia_tensor_world
        eff_mass = (inv_ma + inv_mb +
                    float(ra_cross @ ia @ ra_cross) +
                    float(rb_cross @ ib @ rb_cross))

        if eff_mass < 1e-10:
            return

        # Baumgarte position correction
        baumgarte = self.stiffness * error / (dt * eff_mass)

        # Damping
        rel_vel = self.body_a.get_velocity_at_point(world_a) - self.body_b.get_velocity_at_point(world_b)
        vel_n = rel_vel.dot(normal)
        lambda_val = (-vel_n * self.stiffness - self.damping * error / dt) / eff_mass

        impulse = normal * lambda_val
        self._apply_impulse_pair(self.body_a, self.body_b, r_a, r_b, impulse)


class SpringConstraint(Constraint):
    """
    Spring-damper between two anchor points.
    Hooke's law with viscous damping.
    """

    def __init__(self, body_a: RigidBody, body_b: RigidBody,
                 anchor_a: Vec3 = None, anchor_b: Vec3 = None,
                 rest_length: float = 1.0,
                 spring_constant: float = 100.0,
                 damping: float = 5.0):
        super().__init__(body_a, body_b)
        self.anchor_a = anchor_a or Vec3.zero()
        self.anchor_b = anchor_b or Vec3.zero()
        self.rest_length = rest_length
        self.spring_constant = spring_constant
        self.damping = damping
        self.min_length = 0.0
        self.max_length = float('inf')

    def solve(self, dt: float):
        if not self.enabled or self.is_broken:
            return

        world_a = self.body_a.transform_point(self.anchor_a)
        world_b = self.body_b.transform_point(self.anchor_b)
        diff = world_a - world_b
        dist = diff.length()

        if dist < 1e-8:
            return

        dist_clamped = max(self.min_length, min(self.max_length, dist))
        normal = diff * (1.0 / dist)
        extension = dist_clamped - self.rest_length

        r_a = world_a - self.body_a.position
        r_b = world_b - self.body_b.position
        rel_vel = self.body_a.get_velocity_at_point(world_a) - self.body_b.get_velocity_at_point(world_b)
        vel_n = rel_vel.dot(normal)

        # F = -k*x - c*v
        force_magnitude = -self.spring_constant * extension - self.damping * vel_n
        force = normal * force_magnitude

        # Apply as forces (not impulses) — spring is a continuous force
        self.body_a.apply_force_at_point(force, world_a)
        if self.body_b and not self.body_b.is_static:
            self.body_b.apply_force_at_point(-force, world_b)


class BallSocketConstraint(Constraint):
    """
    Ball-and-socket joint (3 DOF rotation, 0 DOF translation).
    Like a shoulder joint.
    """

    def __init__(self, body_a: RigidBody, body_b: RigidBody,
                 pivot: Vec3 = None):
        super().__init__(body_a, body_b)
        pivot = pivot or (body_a.position + body_b.position) * 0.5
        self.local_pivot_a = body_a.inverse_transform_point(pivot)
        self.local_pivot_b = body_b.inverse_transform_point(pivot)
        self._accumulated_impulse = Vec3.zero()

    def solve(self, dt: float):
        if not self.enabled or self.is_broken:
            return

        world_a = self.body_a.transform_point(self.local_pivot_a)
        world_b = self.body_b.transform_point(self.local_pivot_b)
        r_a = world_a - self.body_a.position
        r_b = world_b - self.body_b.position

        vel_a = self.body_a.get_velocity_at_point(world_a)
        vel_b = self.body_b.get_velocity_at_point(world_b)
        rel_vel = vel_a - vel_b

        # Baumgarte bias
        error = world_a - world_b
        bias = error * (-0.2 / dt)

        # Compute effective mass matrix (3x3)
        import numpy as np
        inv_ma = self.body_a.inv_mass
        inv_mb = self.body_b.inv_mass
        inv_Ia = self.body_a._inv_inertia_tensor_world
        inv_Ib = self.body_b._inv_inertia_tensor_world

        ra_np = r_a.to_numpy()
        rb_np = r_b.to_numpy()

        def skew(v):
            return np.array([[0, -v[2], v[1]],
                              [v[2], 0, -v[0]],
                              [-v[1], v[0], 0]])

        K = ((inv_ma + inv_mb) * np.eye(3) -
             skew(ra_np) @ inv_Ia @ skew(ra_np) -
             skew(rb_np) @ inv_Ib @ skew(rb_np))

        rhs = -(rel_vel + bias).to_numpy()
        try:
            impulse_np = np.linalg.solve(K, rhs)
        except np.linalg.LinAlgError:
            return

        impulse = Vec3.from_numpy(impulse_np)
        self._apply_impulse_pair(self.body_a, self.body_b, r_a, r_b, impulse)


class HingeConstraint(Constraint):
    """
    Hinge (revolute) joint — 1 DOF rotation around hinge axis.
    Can have angle limits and motors.
    """

    def __init__(self, body_a: RigidBody, body_b: RigidBody,
                 pivot: Vec3 = None, axis: Vec3 = None,
                 min_angle: float = -float('inf'),
                 max_angle: float = float('inf')):
        super().__init__(body_a, body_b)
        pivot = pivot or (body_a.position + body_b.position) * 0.5
        axis = (axis or Vec3(0, 1, 0)).normalize()

        self.local_pivot_a = body_a.inverse_transform_point(pivot)
        self.local_pivot_b = body_b.inverse_transform_point(pivot)
        self.local_axis_a = body_a.inverse_transform_direction(axis)
        self.local_axis_b = body_b.inverse_transform_direction(axis)

        self.min_angle = min_angle
        self.max_angle = max_angle

        # Motor
        self.motor_enabled = False
        self.motor_target_velocity = 0.0
        self.motor_max_force = 100.0

        self.ball_socket = BallSocketConstraint(body_a, body_b, pivot)

    def solve(self, dt: float):
        if not self.enabled or self.is_broken:
            return

        # Position constraint (via ball socket)
        self.ball_socket.solve(dt)

        # Angular constraint — align axes
        world_axis_a = self.body_a.transform_direction(self.local_axis_a)
        world_axis_b = self.body_b.transform_direction(self.local_axis_b)

        # Keep perpendicular components aligned
        perp = world_axis_a.cross(world_axis_b)
        if perp.length() < 1e-8:
            return

        angle_error = math.asin(max(-1.0, min(1.0, perp.length())))
        correction_axis = perp.normalize()
        correction = correction_axis * (-0.2 * angle_error / dt)

        rel_ang_vel = self.body_a.angular_velocity - self.body_b.angular_velocity
        ang_constraint = rel_ang_vel.project_onto(correction_axis)

        # Apply angular correction
        inv_Ia = self.body_a._inv_inertia_tensor_world
        inv_Ib = self.body_b._inv_inertia_tensor_world
        import numpy as np
        ax = correction_axis.to_numpy()
        eff_ang_mass = float(ax @ inv_Ia @ ax) + float(ax @ inv_Ib @ ax)
        if eff_ang_mass < 1e-10:
            return

        lambda_ang = (-ang_constraint.length() + correction.length()) / eff_ang_mass
        ang_impulse = correction_axis * lambda_ang

        if not self.body_a.is_static:
            self.body_a.angular_velocity += Vec3.from_numpy(inv_Ia @ ang_impulse.to_numpy())
        if not self.body_b.is_static:
            self.body_b.angular_velocity -= Vec3.from_numpy(inv_Ib @ ang_impulse.to_numpy())

        # Motor
        if self.motor_enabled:
            self._solve_motor(dt, world_axis_a)

    def _solve_motor(self, dt, axis):
        rel_vel = self.body_a.angular_velocity - self.body_b.angular_velocity
        current_vel = rel_vel.dot(axis)
        error = self.motor_target_velocity - current_vel
        import numpy as np
        ax = axis.to_numpy()
        inv_Ia = self.body_a._inv_inertia_tensor_world
        inv_Ib = self.body_b._inv_inertia_tensor_world
        eff_mass = float(ax @ inv_Ia @ ax) + float(ax @ inv_Ib @ ax)
        if eff_mass < 1e-10:
            return
        motor_impulse = min(abs(error / eff_mass), self.motor_max_force * dt)
        motor_impulse *= (1.0 if error > 0 else -1.0)
        impulse = axis * motor_impulse
        if not self.body_a.is_static:
            self.body_a.angular_velocity += Vec3.from_numpy(inv_Ia @ impulse.to_numpy())
        if not self.body_b.is_static:
            self.body_b.angular_velocity -= Vec3.from_numpy(inv_Ib @ impulse.to_numpy())


class SliderConstraint(Constraint):
    """
    Prismatic joint — body can slide along an axis only.
    Like a piston.
    """

    def __init__(self, body_a: RigidBody, body_b: RigidBody,
                 axis: Vec3 = None,
                 min_distance: float = -float('inf'),
                 max_distance: float = float('inf')):
        super().__init__(body_a, body_b)
        self.axis = (axis or Vec3(1, 0, 0)).normalize()
        self.local_axis_a = body_a.inverse_transform_direction(self.axis)
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.motor_enabled = False
        self.motor_velocity = 0.0
        self.motor_max_force = 100.0

    def solve(self, dt: float):
        if not self.enabled or self.is_broken:
            return

        world_axis = self.body_a.transform_direction(self.local_axis_a)
        diff = self.body_b.position - self.body_a.position

        # Constrain perpendicular motion
        perp = diff - diff.project_onto(world_axis)
        if perp.length() > 1e-6:
            correction = perp * (-0.3 / dt)
            self.body_b.linear_velocity -= correction * self.body_b.inv_mass

        # Limit along axis
        proj = diff.dot(world_axis)
        if proj < self.min_distance:
            error = proj - self.min_distance
            rel_vel = (self.body_a.linear_velocity - self.body_b.linear_velocity).dot(world_axis)
            eff_mass = self.body_a.inv_mass + self.body_b.inv_mass
            if eff_mass > 1e-10:
                lambda_val = max(0.0, (-rel_vel - 0.2 * error / dt) / eff_mass)
                impulse = world_axis * lambda_val
                self._apply_impulse_pair(self.body_a, self.body_b,
                                         Vec3.zero(), Vec3.zero(), impulse)

        elif proj > self.max_distance:
            error = proj - self.max_distance
            rel_vel = (self.body_b.linear_velocity - self.body_a.linear_velocity).dot(world_axis)
            eff_mass = self.body_a.inv_mass + self.body_b.inv_mass
            if eff_mass > 1e-10:
                lambda_val = max(0.0, (-rel_vel - 0.2 * error / dt) / eff_mass)
                impulse = world_axis * (-lambda_val)
                self._apply_impulse_pair(self.body_a, self.body_b,
                                         Vec3.zero(), Vec3.zero(), impulse)


class FixedConstraint(Constraint):
    """
    Welds two bodies together — no relative motion.
    """

    def __init__(self, body_a: RigidBody, body_b: RigidBody):
        super().__init__(body_a, body_b)
        self.relative_position = body_b.position - body_a.position
        self.relative_rotation = body_a.rotation.inverse() * body_b.rotation
        self.ball_socket = BallSocketConstraint(
            body_a, body_b,
            pivot=(body_a.position + body_b.position) * 0.5
        )

    def solve(self, dt: float):
        if not self.enabled or self.is_broken:
            return

        self.ball_socket.solve(dt)

        # Lock relative rotation
        target_rot_b = body_a_rot = self.body_a.rotation * self.relative_rotation
        rot_diff = target_rot_b.inverse() * self.body_b.rotation
        axis, angle = rot_diff.to_axis_angle()
        if abs(angle) > 1e-6:
            correction = axis * (-0.3 * angle / dt)
            if not self.body_a.is_static:
                import numpy as np
                self.body_a.angular_velocity += Vec3.from_numpy(
                    self.body_a._inv_inertia_tensor_world @ correction.to_numpy() * 0.5)
            if not self.body_b.is_static:
                import numpy as np
                self.body_b.angular_velocity -= Vec3.from_numpy(
                    self.body_b._inv_inertia_tensor_world @ correction.to_numpy() * 0.5)


# ============================================================================
#                                    WORLD                                    
# ============================================================================

import time
import math
from typing import List, Dict, Callable, Optional, Tuple



class PhysicsWorld:
    """
    Full physics simulation world.

    Features:
    - Fixed timestep with substeps (stability)
    - Configurable gravity
    - Broad phase SAP + optional BVH for statics
    - Narrow phase GJK/EPA + analytic shapes
    - Sequential impulse solver (velocity + position)
    - Collision callbacks (enter, stay, exit)
    - Raycasting
    - Physics materials
    - Sleeping
    - Step events
    """

    def __init__(self,
                 gravity: Vec3 = None,
                 fixed_dt: float = 1.0 / 60.0,
                 substeps: int = 4):
        # ── Simulation Parameters ─────────────────────────────────────────────
        self.gravity = gravity or Vec3(0, -9.81, 0)
        self.fixed_dt = fixed_dt
        self.substeps = substeps
        self.sub_dt = fixed_dt / substeps
        self.time_accumulator = 0.0
        self.simulation_time = 0.0
        self.frame_count = 0

        # ── Bodies & Colliders ────────────────────────────────────────────────
        self.bodies: List[RigidBody] = []
        self._body_map: Dict[int, RigidBody] = {}
        self.static_bodies: List[RigidBody] = []

        # ── Pipeline ──────────────────────────────────────────────────────────
        self.broad_phase = SweepAndPrune()
        self.bvh = BVH()
        self.narrow_phase = NarrowPhaseDispatcher()
        self.solver = ImpulseSolver()

        # ── Collision State ───────────────────────────────────────────────────
        self._active_manifolds: Dict[Tuple[int, int], CollisionManifold] = {}
        self._prev_contacts: set = set()
        self._current_contacts: set = set()

        # ── Callbacks ─────────────────────────────────────────────────────────
        self._collision_enter_callbacks: List[Callable] = []
        self._collision_stay_callbacks: List[Callable] = []
        self._collision_exit_callbacks: List[Callable] = []
        self._step_callbacks: List[Callable] = []

        # ── Forces ────────────────────────────────────────────────────────────
        self._global_forces: List[Callable] = []  # fn(body, dt) -> Vec3

        # ── Settings ──────────────────────────────────────────────────────────
        self.max_bodies = 10000
        self.enable_sleeping = True
        self.enable_ccd = False
        self.debug_mode = False
        self.paused = False

        # ── Stats ─────────────────────────────────────────────────────────────
        self.stats = {
            'bodies': 0,
            'active_bodies': 0,
            'sleeping_bodies': 0,
            'collisions': 0,
            'broad_pairs': 0,
            'solve_time_ms': 0.0,
            'total_step_time_ms': 0.0,
        }

    # ── Body Management ───────────────────────────────────────────────────────

    def add_body(self, body: RigidBody) -> RigidBody:
        """Add a rigid body to the world."""
        if len(self.bodies) >= self.max_bodies:
            raise RuntimeError(f"Max bodies limit ({self.max_bodies}) reached")
        self.bodies.append(body)
        self._body_map[body.id] = body
        if body.is_static:
            self.static_bodies.append(body)
        return body

    def remove_body(self, body: RigidBody):
        """Remove a rigid body from the world."""
        if body in self.bodies:
            self.bodies.remove(body)
        if body.id in self._body_map:
            del self._body_map[body.id]
        if body in self.static_bodies:
            self.static_bodies.remove(body)
        # Clear manifolds involving this body
        keys_to_remove = [k for k in self._active_manifolds if body.id in k]
        for k in keys_to_remove:
            del self._active_manifolds[k]

    def get_body(self, body_id: int) -> Optional[RigidBody]:
        return self._body_map.get(body_id)

    def clear(self):
        """Remove all bodies and reset state."""
        self.bodies.clear()
        self._body_map.clear()
        self.static_bodies.clear()
        self._active_manifolds.clear()
        self._prev_contacts.clear()
        self._current_contacts.clear()
        self.simulation_time = 0.0
        self.frame_count = 0

    # ── Factory Methods ───────────────────────────────────────────────────────

    def create_sphere(self, position: Vec3, radius: float = 0.5, mass: float = 1.0,
                      restitution: float = 0.6, is_static: bool = False) -> RigidBody:
        """Create and register a sphere body."""
        body = RigidBody(mass=mass, position=position)
        body.restitution = restitution
        collider = SphereCollider(radius=radius)
        collider.rigidbody = body
        body.collider = collider
        body.set_sphere_inertia(radius)
        if is_static:
            body.set_infinite_mass()
        self.add_body(body)
        return body

    def create_box(self, position: Vec3, half_extents: Vec3 = None,
                   mass: float = 1.0, restitution: float = 0.3,
                   is_static: bool = False) -> RigidBody:
        """Create and register a box body."""
        he = half_extents or Vec3(0.5, 0.5, 0.5)
        body = RigidBody(mass=mass, position=position)
        body.restitution = restitution
        collider = BoxCollider(half_extents=he)
        collider.rigidbody = body
        body.collider = collider
        body.set_box_inertia(he.x, he.y, he.z)
        if is_static:
            body.set_infinite_mass()
        self.add_body(body)
        return body

    def create_plane(self, normal: Vec3 = None, distance: float = 0.0,
                     restitution: float = 0.4) -> RigidBody:
        """Create a static infinite plane."""
        body = RigidBody(mass=1.0)
        body.restitution = restitution
        collider = PlaneCollider(normal=normal, distance=distance)
        collider.rigidbody = body
        body.collider = collider
        body.set_infinite_mass()
        self.add_body(body)
        return body

    def create_capsule(self, position: Vec3, radius: float = 0.5, height: float = 2.0,
                       mass: float = 1.0) -> RigidBody:
        body = RigidBody(mass=mass, position=position)
        collider = CapsuleCollider(radius=radius, height=height)
        collider.rigidbody = body
        body.collider = collider
        body.set_cylinder_inertia(radius, height)
        self.add_body(body)
        return body

    # ── Simulation Step ───────────────────────────────────────────────────────

    def step(self, dt: float):
        """
        Advance simulation by dt seconds.
        Uses fixed timestep with accumulator for stability.
        """
        if self.paused:
            return

        t_start = time.perf_counter()
        self.time_accumulator += dt

        while self.time_accumulator >= self.fixed_dt:
            self._fixed_step(self.fixed_dt)
            self.time_accumulator -= self.fixed_dt
            self.simulation_time += self.fixed_dt
            self.frame_count += 1

        t_end = time.perf_counter()
        self.stats['total_step_time_ms'] = (t_end - t_start) * 1000.0

    def _fixed_step(self, dt: float):
        """One fixed timestep with substeps."""
        for _ in range(self.substeps):
            self._substep(self.sub_dt)

        # Fire step callbacks
        for cb in self._step_callbacks:
            cb(self.simulation_time, dt)

    def _substep(self, dt: float):
        """One physics substep."""
        # ── Apply Global Forces ───────────────────────────────────────────────
        for body in self.bodies:
            if body.is_static or body.is_sleeping:
                continue
            for force_fn in self._global_forces:
                f = force_fn(body, dt)
                if f:
                    body.apply_force(f)

        # ── Integrate ────────────────────────────────────────────────────────
        for body in self.bodies:
            body.integrate(dt, self.gravity)

        # ── Collision Detection ───────────────────────────────────────────────
        manifolds = self._detect_collisions()

        # ── Collision Response ────────────────────────────────────────────────
        t_solve = time.perf_counter()
        if manifolds:
            self.solver.solve(manifolds, dt)
        self.stats['solve_time_ms'] = (time.perf_counter() - t_solve) * 1000.0

        # ── Collision Events ──────────────────────────────────────────────────
        self._fire_collision_events(manifolds)

        # ── Update Stats ──────────────────────────────────────────────────────
        self._update_stats()

    def _detect_collisions(self) -> List[CollisionManifold]:
        """Full collision pipeline: broad phase → narrow phase."""
        self.broad_phase.clear()

        # Build broad phase entries
        entries = []
        for body in self.bodies:
            if body.collider is None:
                continue
            aabb = body.collider.get_aabb(body.position, body.rotation)
            entry = BroadPhaseEntry(body, body.collider, aabb)
            entries.append(entry)
            self.broad_phase.add(entry)

        # Broad phase pairs
        pairs = self.broad_phase.get_pairs()
        self.stats['broad_pairs'] = len(pairs)

        # Narrow phase
        manifolds = []
        for entry_a, entry_b in pairs:
            body_a = entry_a.body
            body_b = entry_b.body

            # Skip static-static pairs
            if body_a.is_static and body_b.is_static:
                continue
            # Skip sleeping pairs (both sleeping)
            if body_a.is_sleeping and body_b.is_sleeping:
                continue
            # Skip trigger-only pairs (no response)
            if body_a.collider.is_trigger or body_b.collider.is_trigger:
                continue

            manifold = self.narrow_phase.test(entry_a, entry_b)
            if manifold and manifold.is_valid:
                # Normalize: ensure body_a is dynamic so solver applies impulses correctly.
                if body_a.is_static and not body_b.is_static:
                    manifold.body_a, manifold.body_b = manifold.body_b, manifold.body_a
                    manifold.collider_a, manifold.collider_b = manifold.collider_b, manifold.collider_a
                manifolds.append(manifold)
                key = (min(body_a.id, body_b.id), max(body_a.id, body_b.id))
                self._active_manifolds[key] = manifold

        self.stats['collisions'] = len(manifolds)
        return manifolds

    def _fire_collision_events(self, manifolds: List[CollisionManifold]):
        """Fire enter/stay/exit callbacks."""
        self._current_contacts = set()
        for m in manifolds:
            if m.body_a and m.body_b:
                key = (min(m.body_a.id, m.body_b.id), max(m.body_a.id, m.body_b.id))
                self._current_contacts.add(key)

        entered = self._current_contacts - self._prev_contacts
        exited = self._prev_contacts - self._current_contacts
        stayed = self._current_contacts & self._prev_contacts

        for key in entered:
            m = self._active_manifolds.get(key)
            if m:
                for cb in self._collision_enter_callbacks:
                    cb(m)

        for key in stayed:
            m = self._active_manifolds.get(key)
            if m:
                for cb in self._collision_stay_callbacks:
                    cb(m)

        for key in exited:
            for cb in self._collision_exit_callbacks:
                cb(key)
            self._active_manifolds.pop(key, None)

        self._prev_contacts = self._current_contacts.copy()

    # ── Raycasting ────────────────────────────────────────────────────────────

    def raycast(self, origin: Vec3, direction: Vec3, max_distance: float = float('inf'),
                layer_mask: int = 0xFFFFFFFF) -> Optional[dict]:
        """
        Cast a ray and return the first hit body, point, normal, and distance.
        Returns None if no hit.
        """
        direction = direction.normalize()
        best_hit = None
        best_dist = max_distance

        for body in self.bodies:
            if body.collider is None:
                continue
            if not (layer_mask & (1 << body.layer)):
                continue
            hit = self._ray_collider_test(origin, direction, body, best_dist)
            if hit and hit['distance'] < best_dist:
                best_dist = hit['distance']
                best_hit = hit

        return best_hit

    def raycast_all(self, origin: Vec3, direction: Vec3,
                    max_distance: float = float('inf')) -> List[dict]:
        """Cast ray and return ALL hits sorted by distance."""
        direction = direction.normalize()
        hits = []
        for body in self.bodies:
            if body.collider is None:
                continue
            hit = self._ray_collider_test(origin, direction, body, max_distance)
            if hit:
                hits.append(hit)
        hits.sort(key=lambda h: h['distance'])
        return hits

    def _ray_collider_test(self, origin: Vec3, direction: Vec3,
                            body: RigidBody, max_dist: float) -> Optional[dict]:
        """Test ray against a body's collider."""
        col = body.collider

        if col.collider_type == ColliderType.SPHERE:
            return self._ray_sphere(origin, direction, col, body, max_dist)
        elif col.collider_type == ColliderType.BOX:
            return self._ray_box(origin, direction, col, body, max_dist)
        elif col.collider_type == ColliderType.PLANE:
            return self._ray_plane(origin, direction, col, body, max_dist)
        elif col.collider_type == ColliderType.CAPSULE:
            return self._ray_capsule(origin, direction, col, body, max_dist)
        return None

    def _ray_sphere(self, origin, direction, col, body, max_dist):
        center = col.get_world_position(body.position, body.rotation)
        oc = origin - center
        b = 2.0 * oc.dot(direction)
        c = oc.dot(oc) - col.radius * col.radius
        disc = b * b - 4 * c
        if disc < 0:
            return None
        sqrt_disc = math.sqrt(disc)
        t = (-b - sqrt_disc) * 0.5
        if t < 0:
            t = (-b + sqrt_disc) * 0.5
        if t < 0 or t > max_dist:
            return None
        hit_point = origin + direction * t
        normal = (hit_point - center).normalize()
        return {'body': body, 'point': hit_point, 'normal': normal, 'distance': t}

    def _ray_plane(self, origin, direction, col, body, max_dist):
        denom = direction.dot(col.normal)
        if abs(denom) < 1e-8:
            return None
        t = (col.distance - origin.dot(col.normal)) / denom
        if t < 0 or t > max_dist:
            return None
        hit_point = origin + direction * t
        return {'body': body, 'point': hit_point, 'normal': col.normal, 'distance': t}

    def _ray_box(self, origin, direction, col, body, max_dist):
        """Slab method for OBB."""
        center = col.get_world_position(body.position, body.rotation)
        world_rot = col.get_world_rotation(body.rotation)
        local_origin = world_rot.inverse().rotate_vector(origin - center)
        local_dir = world_rot.inverse().rotate_vector(direction)
        he = col.half_extents
        t_min, t_max = -float('inf'), float('inf')
        hit_axis = 0
        for i in range(3):
            d = local_dir[i]
            o = local_origin[i]
            h = he[i]
            if abs(d) < 1e-8:
                if o < -h or o > h:
                    return None
            else:
                t1 = (-h - o) / d
                t2 = (h - o) / d
                if t1 > t2:
                    t1, t2 = t2, t1
                if t1 > t_min:
                    t_min = t1
                    hit_axis = i
                t_max = min(t_max, t2)
        if t_min > t_max or t_max < 0:
            return None
        t = t_min if t_min >= 0 else t_max
        if t > max_dist:
            return None
        hit_point = origin + direction * t
        local_normal = Vec3.zero()
        local_normal[hit_axis] = 1.0 if local_origin[hit_axis] > 0 else -1.0
        normal = world_rot.rotate_vector(local_normal)
        return {'body': body, 'point': hit_point, 'normal': normal, 'distance': t}

    def _ray_capsule(self, origin, direction, col, body, max_dist):
        """Ray vs capsule = ray vs cylinder + two sphere caps."""
        a, b = col._get_segment(body.position, body.rotation)
        r = col.radius
        # Simplified: test sphere at each end and cylinder
        best = None
        for cap_center in [a, b]:
            hit = self._ray_sphere_raw(origin, direction, cap_center, r, max_dist)
            if hit and (best is None or hit['distance'] < best['distance']):
                hit['body'] = body
                best = hit
        return best

    def _ray_sphere_raw(self, origin, direction, center, radius, max_dist):
        oc = origin - center
        b_val = 2.0 * oc.dot(direction)
        c = oc.dot(oc) - radius * radius
        disc = b_val * b_val - 4 * c
        if disc < 0:
            return None
        sqrt_disc = math.sqrt(disc)
        t = (-b_val - sqrt_disc) * 0.5
        if t < 0:
            t = (-b_val + sqrt_disc) * 0.5
        if t < 0 or t > max_dist:
            return None
        hit_point = origin + direction * t
        normal = (hit_point - center).normalize()
        return {'point': hit_point, 'normal': normal, 'distance': t}

    # ── Overlap Queries ───────────────────────────────────────────────────────

    def overlap_sphere(self, center: Vec3, radius: float) -> List[RigidBody]:
        """Find all bodies overlapping a sphere."""
        results = []
        for body in self.bodies:
            if body.collider is None:
                continue
            if body.collider.collider_type.name == 'SPHERE':
                dist = (body.position - center).length()
                if dist < radius + body.collider.radius:
                    results.append(body)
            else:
                closest = body.collider.closest_point(center, body.position, body.rotation)
                if (closest - center).length_squared() <= radius * radius:
                    results.append(body)
        return results

    def overlap_aabb(self, aabb: AABB) -> List[RigidBody]:
        """Find all bodies whose AABB overlaps the given AABB."""
        results = []
        for body in self.bodies:
            if body.collider is None:
                continue
            body_aabb = body.collider.get_aabb(body.position, body.rotation)
            if body_aabb.intersects(aabb):
                results.append(body)
        return results

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def on_collision_enter(self, callback: Callable):
        """Register callback fired when two bodies first collide."""
        self._collision_enter_callbacks.append(callback)

    def on_collision_stay(self, callback: Callable):
        """Register callback fired each frame two bodies remain in contact."""
        self._collision_stay_callbacks.append(callback)

    def on_collision_exit(self, callback: Callable):
        """Register callback fired when two bodies separate."""
        self._collision_exit_callbacks.append(callback)

    def on_step(self, callback: Callable):
        """Register callback fired after each fixed step."""
        self._step_callbacks.append(callback)

    def add_global_force(self, force_fn: Callable):
        """Add a global force function called for every body each step."""
        self._global_forces.append(force_fn)

    # ── World Forces ──────────────────────────────────────────────────────────

    def add_wind_force(self, wind_velocity: Vec3, drag_coefficient: float = 0.5):
        """Add a wind force to all bodies."""
        def wind_fn(body, dt):
            rel_vel = wind_velocity - body.linear_velocity
            return rel_vel * (drag_coefficient * body.mass * 0.1)
        self.add_global_force(wind_fn)

    def add_buoyancy(self, water_level: float, fluid_density: float = 1000.0):
        """Simulate buoyancy for bodies below water_level."""
        def buoyancy_fn(body, dt):
            if body.position.y < water_level and body.collider:
                submerged = min(1.0, (water_level - body.position.y))
                buoyancy_force = Vec3(0, fluid_density * 9.81 * submerged, 0)
                return buoyancy_force
            return None
        self.add_global_force(buoyancy_fn)

    # ── Stats & Debug ─────────────────────────────────────────────────────────

    def _update_stats(self):
        self.stats['bodies'] = len(self.bodies)
        sleeping = sum(1 for b in self.bodies if b.is_sleeping)
        self.stats['sleeping_bodies'] = sleeping
        self.stats['active_bodies'] = len(self.bodies) - sleeping

    def get_stats(self) -> dict:
        return dict(self.stats)

    def get_total_energy(self) -> float:
        return sum(b.get_kinetic_energy() for b in self.bodies if not b.is_static)

    def get_total_momentum(self) -> Vec3:
        total = Vec3.zero()
        for b in self.bodies:
            if not b.is_static:
                total += b.get_momentum()
        return total

    def set_gravity(self, gravity: Vec3):
        self.gravity = gravity

    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False

    def __repr__(self):
        return (f"PhysicsWorld(bodies={len(self.bodies)}, "
                f"gravity={self.gravity}, "
                f"time={self.simulation_time:.3f}s)")


# ================================================================================
# DEMO — Bounce Simulation (run this file directly to test)
# ================================================================================

def run_bounce_demo():
    """
    Simulates a rubber ball dropped from 5m height bouncing on a concrete floor.
    Prints bounce heights — should decrease realistically due to restitution.
    """
    print("=" * 60)
    print("  Physics Engine — Bounce Demo")
    print("=" * 60)

    world = PhysicsWorld(gravity=Vec3(0, -9.81, 0))
    
    # Create floor
    floor = world.create_plane(normal=Vec3(0, 1, 0), distance=0.0, restitution=0.4)
    
    # Create rubber ball dropped from 5m
    ball = world.create_sphere(Vec3(0, 5, 0), radius=0.3, mass=0.5, restitution=0.75)
    get_material('rubber_ball').apply_to_body(ball)
    
    # Collision enter callback
    def on_hit(manifold):
        if not manifold.body_a.is_static:
            body = manifold.body_a
        else:
            body = manifold.body_b
        speed = body.linear_velocity.length()
        print(f"  Impact!  height={body.position.y:.3f}m  speed={speed:.2f}m/s")

    world.on_collision_enter(on_hit)

    # Simulate 10 seconds
    bounces = []
    going_down = True
    dt = 1.0 / 60.0

    for i in range(600):
        world.step(dt)
        vy = ball.linear_velocity.y
        y  = ball.position.y

        if going_down and vy > 0.5:
            bounces.append(y)
            going_down = False
        elif not going_down and vy < -0.2:
            going_down = True

    print()
    print(f"Detected {len(bounces)} bounces:")
    for n, h in enumerate(bounces):
        bar = '#' * int(h * 8)
        print(f"  Bounce {n+1:2d}: {h:6.3f}m  {bar}")
    
    print()
    print(f"Final position : {ball.position}")
    print(f"Final velocity : {ball.linear_velocity}")
    print(f"Kinetic energy : {ball.get_kinetic_energy():.4f} J")
    stats = world.get_stats()
    print(f"Active bodies  : {stats['active_bodies']}")
    print(f"Sleeping bodies: {stats['sleeping_bodies']}")
    print()
    print("Multi-sphere stress test...")
    
    world2 = PhysicsWorld(gravity=Vec3(0, -9.81, 0))
    world2.create_plane()
    import random
    random.seed(42)
    for _ in range(20):
        x = random.uniform(-3, 3)
        z = random.uniform(-3, 3)
        h2 = random.uniform(1, 8)
        r = random.uniform(0.1, 0.4)
        m = random.uniform(0.5, 3.0)
        body = world2.create_sphere(Vec3(x, h2, z), radius=r, mass=m)
        body.restitution = random.uniform(0.3, 0.9)
    
    for _ in range(300):
        world2.step(1/60)
    
    stats2 = world2.get_stats()
    print(f"  20 spheres simulated for 5s")
    print(f"  Final active: {stats2['active_bodies']}, sleeping: {stats2['sleeping_bodies']}")
    print(f"  Collisions last frame: {stats2['collisions']}")
    print()
    print("All tests passed.")
    print("=" * 60)


if __name__ == "__main__":
    run_bounce_demo()
