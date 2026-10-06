"""ctypes binding for the native rasterizer (native/raster.c), built on first use."""
from __future__ import annotations

import ctypes
import os
import platform
import subprocess

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, 'native', 'raster.c')
_LIB = os.path.join(_HERE, 'native', 'libnrraster' + ('.dylib' if platform.system() == 'Darwin' else '.so'))


class NRCamera(ctypes.Structure):
    _fields_ = [('origin', ctypes.c_double * 3), ('right', ctypes.c_double * 3), ('up', ctypes.c_double * 3),
                ('fwd', ctypes.c_double * 3), ('fx', ctypes.c_double), ('fy', ctypes.c_double), ('cx', ctypes.c_double),
                ('cy', ctypes.c_double), ('near_clip', ctypes.c_double), ('width', ctypes.c_int32), ('height', ctypes.c_int32)]


def _load():
    if not os.path.exists(_LIB) or os.path.getmtime(_LIB) < os.path.getmtime(_SRC):
        tune = '-mcpu=native' if platform.machine() in {'arm64', 'aarch64'} else '-march=native'
        subprocess.run(['cc', '-O3', tune, '-shared', '-fPIC', '-o', _LIB, _SRC, '-lpthread'], check=True)
    lib = ctypes.CDLL(_LIB)
    p = ctypes.c_void_p
    lib.nr_rasterize.argtypes = [p, ctypes.c_int, ctypes.POINTER(NRCamera), ctypes.c_int, p, p, ctypes.c_int]
    lib.nr_rasterize.restype = ctypes.c_int
    lib.nr_resolve.argtypes = [p, ctypes.POINTER(NRCamera), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                               p, p, ctypes.c_int, p, p, p, p, p, p, p]
    lib.nr_resolve.restype = ctypes.c_int
    lib.nr_build_bvh.argtypes = [p, ctypes.c_int, ctypes.c_int, p, p]
    lib.nr_build_bvh.restype = ctypes.c_int
    return lib


def _default_threads():
    return max(1, os.cpu_count() or 1)


def _ptr(a):
    return a.ctypes.data_as(ctypes.c_void_p)


def camera_struct(cam):
    origin, right, up, fwd = cam.basis()
    fx, fy, cx, cy = cam.intrinsics()
    c = NRCamera()
    c.origin[:], c.right[:], c.up[:], c.fwd[:] = origin, right, up, fwd
    c.fx, c.fy, c.cx, c.cy, c.near_clip = fx, fy, cx, cy, cam.clip_start
    c.width, c.height = cam.width, cam.height
    return c


def build_bvh(tri_verts, leaf=4):
    """Median-split BVH for the GPU ray tracer: (nodes [N, 8] float32, order [T] int32). See raster.c."""
    tri = np.ascontiguousarray(tri_verts, np.float32).reshape(-1, 9)
    nodes = np.empty((2 * len(tri) + 1, 8), np.float32)
    order = np.empty(len(tri), np.int32)
    n = _load().nr_build_bvh(_ptr(tri), len(tri), leaf, _ptr(nodes), _ptr(order))
    return nodes[:n].copy(), order


class Rasterizer:
    def __init__(self, tri_verts, threads=None):
        self.lib = _load()
        self.tri = np.ascontiguousarray(tri_verts, np.float32).reshape(-1, 9)
        self.threads = threads or _default_threads()
        self._buf, self._out = {}, {}

    def _scratch(self, n):
        if n not in self._buf:
            self._buf = {n: (np.empty(n, np.int32), np.empty(n, np.float32))}
        return self._buf[n]

    def _outputs(self, n, P):
        # Worst-case sized; np.empty only reserves address space, pages are touched as they are written.
        if (n, P) not in self._out:
            self._out = {(n, P): dict(g_tri=np.empty(n, np.int32), g_dir=np.empty((n, 3), np.float32), px_ref=np.empty(P, np.int32),
                                      e_group=np.empty(n, np.int32), e_w=np.empty(n, np.float32), px_miss=np.empty(P, np.float32))}
        return self._out[(n, P)]

    def ids(self, cam, S=1, tri=None):
        """Triangle id per sample on the (H*S, W*S) grid, -1 where nothing is hit. Valid until the next call.
        tri: this frame's triangle corners, for scenes that move (default: the ones given at construction)."""
        n = cam.width * S * cam.height * S
        ids, z = self._scratch(n)
        tri = self.tri if tri is None else np.ascontiguousarray(tri, np.float32).reshape(-1, 9)
        self.lib.nr_rasterize(_ptr(tri), len(tri), ctypes.byref(camera_struct(cam)), S, _ptr(ids), _ptr(z), self.threads)
        return ids.reshape(cam.height * S, cam.width * S)

    def resolve(self, cam, subw, S=1, R=1, full=False, sky_id=-1, cap=None, tri=None):
        """Rasterize and collapse into shading groups. Returns views of the arrays the GPU needs for one
        frame (valid until the next call): n_groups rows of g_tri/g_dir, n_entries of e_group/e_w.
        cap: uint8 per triangle (plus the sky id); 1 keeps that triangle at one group per pixel when R > 1."""
        ids = self.ids(cam, S, tri)
        P = cam.width * cam.height
        o = self._outputs(ids.size, P)
        subw = np.ascontiguousarray(subw, np.float32)
        totals = np.zeros(2, np.int64)
        rc = self.lib.nr_resolve(_ptr(ids), ctypes.byref(camera_struct(cam)), S, R, int(full), sky_id, _ptr(subw),
                                 _ptr(cap) if cap is not None else None, self.threads,
                                 _ptr(o['g_tri']), _ptr(o['g_dir']), _ptr(o['px_ref']), _ptr(o['e_group']), _ptr(o['e_w']),
                                 _ptr(o['px_miss']), _ptr(totals))
        if rc != 0:
            raise ValueError('unsupported sampling configuration S=%d R=%d' % (S, R))
        K, E = int(totals[0]), int(totals[1])
        return dict(g_tri=o['g_tri'][:K], g_dir=o['g_dir'][:K], px_ref=o['px_ref'], e_group=o['e_group'][:E], e_w=o['e_w'][:E],
                    px_miss=o['px_miss'], n_groups=K, n_entries=E)
