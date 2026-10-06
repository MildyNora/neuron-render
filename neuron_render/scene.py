"""The compiled scene: geometry, camera model, lights and the baked colour pipeline."""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, replace

import numpy as np

TRI_STRIDE = 24    # v0 v1 v2 | n0 n1 n2 | plane normal, plane d | material | flat
MAT_STRIDE = 12    # roughness metallic transmission coat emission ior-1 specular kind (1 sky, 0.5 textured) | base r g b | -
LIGHT_STRIDE = 20  # type | centre | u axis | v axis | emit dir | hx hy | radius | shape | spot cos, blend | pad
LIGHT_TYPES = {'AREA': 0, 'POINT': 1, 'SPOT': 2, 'SUN': 3}
MAX_LIGHTS = 12


def _unit(v):
    return v / max(float(np.linalg.norm(v)), 1e-20)


def look_at(position, target):
    """Blender camera rotation (columns right, up, -forward) looking from position to target, +Z up."""
    fwd = _unit(np.asarray(target, np.float64) - np.asarray(position, np.float64))
    ref = np.array([0.0, 0.0, 1.0]) if abs(fwd[2]) < 0.9999 else np.array([0.0, 1.0, 0.0])
    right = _unit(np.cross(fwd, ref))
    up = np.cross(right, fwd)
    return np.stack([right, up, -fwd], axis=1)


def build_bvh(tri_verts, leaf=4):
    """Median-split BVH over the triangles for the GPU ray tracer.

    Returns (nodes [N, 8] float32, order [T] int32). A node is (box min, a, box max, b): a leaf has
    b = triangle count > 0 and a = offset into order; an inner node has a = left child, b = -right child.
    """
    v = np.asarray(tri_verts, np.float64).reshape(-1, 3, 3)
    lo, hi = v.min(1), v.max(1)
    cen = 0.5 * (lo + hi)
    order = np.arange(len(v))
    nodes = []

    def build(a, b):
        me = len(nodes)
        nodes.append(None)
        ids = order[a:b]
        bmin, bmax = lo[ids].min(0) - 1e-5, hi[ids].max(0) + 1e-5
        if b - a <= leaf:
            nodes[me] = (*bmin, a, *bmax, b - a)
            return me
        c = cen[ids]
        m = (a + b) // 2
        order[a:b] = ids[np.argpartition(c[:, int(np.argmax(c.max(0) - c.min(0)))], m - a)]
        left = build(a, m)
        right = build(m, b)
        nodes[me] = (*bmin, left, *bmax, -right)
        return me

    build(0, len(v))
    return np.asarray(nodes, np.float32), order.astype(np.int32)


@dataclass(frozen=True)
class Camera:
    matrix: np.ndarray  # 4x4 camera-to-world, Blender convention (-Z forward, +Y up)
    lens: float
    sensor_width: float
    sensor_height: float
    sensor_fit: str
    shift_x: float
    shift_y: float
    clip_start: float
    width: int
    height: int
    pixel_aspect: float = 1.0

    def resized(self, width, height):
        return replace(self, width=int(width), height=int(height))

    def scaled(self, factor):
        return self.resized(max(1, round(self.width * factor)), max(1, round(self.height * factor)))

    def with_matrix(self, matrix):
        return replace(self, matrix=np.asarray(matrix, np.float64))

    def intrinsics(self):
        """(fx, fy, cx, cy) in pixels, matching Blender's sensor-fit rules. +y is down."""
        W, H, ycor = self.width, self.height, self.pixel_aspect
        if self.sensor_fit == 'VERTICAL':
            sensor, horizontal = self.sensor_height, False
        elif self.sensor_fit == 'HORIZONTAL':
            sensor, horizontal = self.sensor_width, True
        else:
            sensor, horizontal = self.sensor_width, W >= H * ycor
        viewfac = W if horizontal else H * ycor
        f = self.lens / sensor * viewfac
        return f, f / ycor, 0.5 * W - self.shift_x * viewfac, 0.5 * H + self.shift_y * viewfac / ycor

    def basis(self):
        R = np.asarray(self.matrix, np.float64)[:3, :3]
        right, up, back = _unit(R[:, 0]), _unit(R[:, 1]), _unit(R[:, 2])
        return np.asarray(self.matrix, np.float64)[:3, 3].copy(), right, up, -back

    def pixel_dirs(self):
        """Unit world-space ray directions through every pixel centre, [H, W, 3] float32."""
        fx, fy, cx, cy = self.intrinsics()
        _, right, up, fwd = self.basis()
        xc = (np.arange(self.width) + 0.5 - cx) / fx
        yc = -(np.arange(self.height) + 0.5 - cy) / fy
        d = right[None, None] * xc[None, :, None] + up[None, None] * yc[:, None, None] + fwd[None, None]
        return (d / np.linalg.norm(d, axis=-1, keepdims=True)).astype(np.float32)


@dataclass
class OrbitDomain:
    """The set of camera poses the neural field is fitted for: a band around the subject."""
    pivot: np.ndarray
    azimuth: tuple
    elevation: tuple
    radius: tuple
    offset: np.ndarray  # camera rotation relative to a pure look-at, so the scene's framing is kept
    hero: tuple         # (azimuth, elevation, radius) of the .blend's own camera

    def position(self, az, el, r):
        a, e = math.radians(az), math.radians(el)
        return self.pivot + r * np.array([math.sin(a) * math.cos(e), -math.cos(a) * math.cos(e), math.sin(e)])

    def pose(self, az, el, r):
        p = self.position(az, el, r)
        M = np.eye(4)
        M[:3, :3] = look_at(p, self.pivot) @ self.offset
        M[:3, 3] = p
        return M

    def separation(self, a, b):
        """Angle in degrees between two (az, el, r) poses as seen from the pivot."""
        u, v = _unit(self.position(*a) - self.pivot), _unit(self.position(*b) - self.pivot)
        return math.degrees(math.acos(float(np.clip(u @ v, -1.0, 1.0))))

    @property
    def full_circle(self):
        return self.azimuth[1] - self.azimuth[0] >= 359.0

    def contains(self, az, el, r, tol=1e-6):
        in_az = self.full_circle or (az - self.azimuth[0]) % 360.0 <= self.azimuth[1] - self.azimuth[0] + tol
        return (in_az and self.elevation[0] - tol <= el <= self.elevation[1] + tol
                and self.radius[0] - tol <= r <= self.radius[1] + tol)

    def sweep(self, t):
        """Azimuth for a turntable at phase t in [0, 1): all the way round, or to and fro inside a partial orbit."""
        if self.full_circle:
            return self.hero[0] + 360.0 * t
        mid, half = 0.5 * (self.azimuth[0] + self.azimuth[1]), 0.5 * (self.azimuth[1] - self.azimuth[0])
        return mid + half * math.sin(2.0 * math.pi * t)

    def sample(self, n, seed=0, exclude=(), min_sep=3.0):
        """n poses: jittered-stratified azimuth, low-discrepancy elevation/radius, kept min_sep degrees
        away from every pose in exclude (the held-out test views)."""
        rng = np.random.default_rng(seed)
        out = []
        for i in range(n):
            for _ in range(64):
                az = self.azimuth[0] + (i + rng.random()) / n * (self.azimuth[1] - self.azimuth[0])
                el = self.elevation[0] + ((i * 0.6180339887 + rng.random() * 0.25) % 1.0) * (self.elevation[1] - self.elevation[0])
                r = self.radius[0] + ((i * 0.7548776662 + rng.random() * 0.25) % 1.0) * (self.radius[1] - self.radius[0])
                if all(self.separation((az, el, r), e) >= min_sep for e in exclude):
                    break
            else:
                continue
            out.append((az, el, r))
        return out

    def to_json(self):
        return dict(pivot=self.pivot.tolist(), azimuth=list(self.azimuth), elevation=list(self.elevation),
                    radius=list(self.radius), offset=self.offset.tolist(), hero=list(self.hero))

    @staticmethod
    def from_json(d):
        return OrbitDomain(np.asarray(d['pivot'], np.float64), tuple(d['azimuth']), tuple(d['elevation']),
                           tuple(d['radius']), np.asarray(d['offset'], np.float64), tuple(d['hero']))


class Scene:
    def __init__(self, bundle):
        self.bundle = bundle
        with open(os.path.join(bundle, 'scene.json')) as f:
            self.meta = json.load(f)
        z = np.load(os.path.join(bundle, 'scene.npz'))
        self.tri_verts = np.ascontiguousarray(z['tri_verts'], np.float32)
        self.tri_normals = z['tri_normals'].astype(np.float32)
        self.tri_color = z['tri_color'].astype(np.float32)
        self.tri_material = z['tri_material'].astype(np.int32)
        self.tri_object = z['tri_object'].astype(np.int32)
        self.lut = z['lut'].astype(np.float32)
        c, r = self.meta['camera'], self.meta['render']
        self.camera = Camera(np.asarray(c['matrix'], np.float64), c['lens'], c['sensor_width'], c['sensor_height'],
                             c['sensor_fit'], c['shift_x'], c['shift_y'], c['clip_start'], r['width'], r['height'],
                             r.get('pixel_aspect', 1.0))
        self.materials = self.meta['materials']
        self.lights = self.meta['lights']
        self.n_tris = len(self.tri_verts)
        self.sky_id = self.n_tris
        epath = os.path.join(bundle, 'env.npy')   # the world on its own, equirectangular (environment.py)
        self.env = np.load(epath).astype(np.float32) if os.path.exists(epath) else None
        apath = os.path.join(bundle, 'anim.npz')
        self.anim = dict(np.load(apath)) if os.path.exists(apath) else None
        self.frames = [int(f) for f in self.anim['frames']] if self.anim is not None else [int(self.meta.get('frame', 1))]
        self.ref_index = self.frames.index(int(self.meta.get('frame', self.frames[0]))) if int(self.meta.get('frame', self.frames[0])) in self.frames else 0
        # Lamps get analytic hints, strongest first, at most MAX_LIGHTS of them (see pack_lights).
        self.light_order = sorted(range(len(self.lights)), key=lambda i: -self.lights[i]['energy'])[:MAX_LIGHTS]
        self.groups = self.meta.get('groups') or ([dict(name=L['name'], kind='light', energy=L['energy'], color=L['color']) for L in self.lights]
                                                  + [dict(name='World', kind='world', strength=1.0)])

    # -- derived geometry -----------------------------------------------------------------------
    @property
    def n_frames(self):
        return len(self.frames)

    def frame_index(self, frame):
        """Index of an animation frame number (the reference pose if the scene was compiled without animation)."""
        if frame is None:
            return self.ref_index
        if int(frame) not in self.frames:
            raise ValueError('frame %s was not compiled (have %d..%d)' % (frame, self.frames[0], self.frames[-1]))
        return self.frames.index(int(frame))

    def geometry(self, fi=None):
        """(triangle corners, corner normals), both [T, 3, 3], at frame index fi (None: the reference pose)."""
        if self.anim is None or fi is None:
            return self.tri_verts, self.tri_normals
        v, n = self.tri_verts.copy(), self.tri_normals.copy()
        for k, ob in enumerate(self.meta['objects']):
            kind = ob.get('motion', 'static')
            if kind == 'static':
                continue
            sel = self.tri_object == k
            if kind == 'rigid':   # x_f = x_ref @ A[:3] + A[3]
                A = self.anim['o%d_affine' % k][fi]
                v[sel] = (self.tri_verts[sel].astype(np.float64) @ A[:3] + A[3]).astype(np.float32)
                nn = self.tri_normals[sel].astype(np.float64) @ np.linalg.inv(A[:3]).T
                n[sel] = (nn / np.maximum(np.linalg.norm(nn, axis=-1, keepdims=True), 1e-20)).astype(np.float32)
            else:
                v[sel], n[sel] = self.anim['o%d_verts' % k][fi], self.anim['o%d_normals' % k][fi]
        return v, n

    def camera_at(self, fi=None):
        if self.anim is None or fi is None:
            return self.camera
        return replace(self.camera, matrix=np.asarray(self.anim['cam_matrix'][fi], np.float64), lens=float(self.anim['cam_lens'][fi]))

    def lights_at(self, fi=None):
        if self.anim is None or fi is None:
            return self.lights
        return [dict(L, matrix=self.anim['light_matrix'][fi][i].tolist(), energy=float(self.anim['light_energy'][fi][i]),
                     color=self.anim['light_color'][fi][i].tolist()) for i, L in enumerate(self.lights)]

    # -- derived geometry -----------------------------------------------------------------------
    @staticmethod
    def planes_of(verts, corner_normals):
        v = verts.astype(np.float64)
        n = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
        ln = np.linalg.norm(n, axis=1, keepdims=True)
        avg = corner_normals.astype(np.float64).mean(1)
        n = np.where(ln > 1e-20, n / np.maximum(ln, 1e-20), avg)
        n = np.where((n * avg).sum(1, keepdims=True) < 0, -n, n)
        return n, (n * v[:, 0]).sum(1)

    def planes(self):
        return self.planes_of(self.tri_verts, self.tri_normals)

    def pack_triangles(self, fi=None):
        """[T + 1, TRI_STRIDE] table for the kernels at frame fi. Row T is the virtual sky triangle."""
        T = self.n_tris
        v, cn = self.geometry(fi)
        n, d = self.planes_of(v, cn)
        ref = self.tri_normals
        flat = (np.abs(ref - ref[:, :1]).max(axis=(1, 2)) < 1e-4)
        tri = np.zeros((T + 1, TRI_STRIDE), np.float32)
        tri[:T, 0:9] = v.reshape(T, 9)
        tri[:T, 9:18] = cn.reshape(T, 9)
        tri[:T, 18:21] = n
        tri[:T, 21] = d
        tri[:T, 22] = self.tri_material
        tri[:T, 23] = flat
        tri[T, 22] = len(self.materials)
        tri[T, 23] = 1.0
        return tri, v

    def pack_lights(self, fi=None):
        """[max(lights, 1), LIGHT_STRIDE]: the lamps the network is given analytic hints for."""
        src = self.lights_at(fi)
        hinted = [src[i] for i in self.light_order]
        lights = np.zeros((max(len(hinted), 1), LIGHT_STRIDE), np.float32)
        for i, L in enumerate(hinted):
            M = np.asarray(L['matrix'], np.float64)
            ax, ay, az = M[:3, 0], M[:3, 1], M[:3, 2]
            row = lights[i]
            row[0] = LIGHT_TYPES.get(L['type'], 1)
            row[1:4] = M[:3, 3]
            row[4:7], row[7:10], row[10:13] = _unit(ax), _unit(ay), -_unit(az)
            if L['type'] == 'AREA':
                row[13] = 0.5 * L['size'] * np.linalg.norm(ax)
                row[14] = 0.5 * L['size_y'] * np.linalg.norm(ay)
                row[16] = 1.0 if L['shape'] in {'DISK', 'ELLIPSE'} else 0.0
            elif L['type'] == 'SUN':
                row[15] = 0.5 * L.get('angle', 0.0)
            else:
                row[15] = L.get('radius', 0.0)
                if L['type'] == 'SPOT':
                    row[17] = math.cos(0.5 * L.get('spot_size', math.pi))
                    row[18] = L.get('spot_blend', 0.0)
        return lights

    def material_table(self, overrides=None, tint=False):
        """[materials + 1, MAT_STRIDE]; the last row is the sky. overrides: {material name: {param: value}} for
        roughness, metallic, transmission, coat, ior, specular and base_color. With tint=True the base colour
        columns are live (the parametric engine multiplies them in); otherwise they are informational."""
        mat = np.zeros((len(self.materials) + 1, MAT_STRIDE), np.float32)
        for i, m in enumerate(self.materials):
            m = dict(m, **(overrides or {}).get(m['name'], {}))
            mat[i, :8] = [m['roughness'], m['metallic'], m['transmission'], m['coat'], math.log1p(max(m['emission'], 0.0)),
                          m['ior'] - 1.0, m['specular'], 0.0 if m['resolved'] else 0.5]   # 0.5: base colour comes from a texture
            constant = m['resolved'] and not m['color_attribute']
            mat[i, 8:11] = m['base_color'][:3] if (constant or 'base_color' in (overrides or {}).get(m['name'], {})) else 1.0
        mat[-1, 7] = 1.0
        mat[-1, 8:11] = 1.0
        return mat

    def color_table(self, tint=False):
        """[T + 1, 9] per-corner base colours. With tint=True, materials with one constant colour are stored
        white here and coloured by their (editable) row of the material table instead."""
        T = self.n_tris
        col = np.zeros((T + 1, 9), np.float32)
        col[:T] = self.tri_color.reshape(T, 9)
        if tint:
            constant = np.array([m['resolved'] and not m['color_attribute'] for m in self.materials])[self.tri_material]
            col[:T][constant] = 1.0
        return col

    def reference_table(self):
        """[T + 1, 9] triangle corners in the reference pose: the canonical coordinates learned features live in."""
        ref = np.zeros((self.n_tris + 1, 9), np.float32)
        ref[:self.n_tris] = self.tri_verts.reshape(self.n_tris, 9)
        return ref

    def pack_frame(self, fi=None):
        """Everything about one frame the kernels need: triangle table, lamp table, BVH."""
        from .native import build_bvh as native_bvh
        tri, v = self.pack_triangles(fi)
        bvh, order = native_bvh(v)
        return dict(tri=tri, verts=v, lights=self.pack_lights(fi), bvh=bvh, bvh_order=order)

    def packed(self):
        """Flat float32 tables for a static scene (the reference pose)."""
        f = self.pack_frame(None)
        return dict(tri=f['tri'], col=self.color_table(), mat=self.material_table(), lights=f['lights'], n_lights=len(self.light_order),
                    bvh=f['bvh'], bvh_order=f['bvh_order'])

    def textured(self):
        """uint8 per triangle (plus the sky): 1 where the base colour comes from a texture the compiler could
        not reduce to per-corner colours, i.e. where appearance varies *inside* a triangle."""
        flag = np.array([0 if m['resolved'] else 1 for m in self.materials] + [0], np.uint8)
        return np.ascontiguousarray(np.concatenate([flag[self.tri_material], [0]]).astype(np.uint8))

    def per_pixel(self, overrides=None, flat_mirrors=False):
        """uint8 per triangle (plus the sky): 1 where one shading sample per pixel block would show, so the
        renderer always shades per pixel there -- textures, and smooth-shaded glossy surfaces (curved glass
        and polished metal, whose reflections change from one pixel to the next).

        overrides: material edits in force ({name: {roughness: ...}}). flat_mirrors: flat faces of polished
        metal and glass count as well; a flat mirror is only as smooth as what it reflects."""
        cn = self.tri_normals
        smooth = np.abs(cn - cn[:, :1]).max(axis=(1, 2)) >= 1e-4
        mats = [dict(m, **(overrides or {}).get(m['name'], {})) for m in self.materials]
        glossy = np.array([m['roughness'] <= 0.3 for m in mats])[self.tri_material]
        flag = (self.textured()[:-1] > 0) | (smooth & glossy)
        if flat_mirrors:
            flag |= np.array([m['roughness'] <= 0.15 and max(m['metallic'], m['transmission']) >= 0.5 for m in mats])[self.tri_material]
        return np.ascontiguousarray(np.concatenate([flag, [False]]).astype(np.uint8))

    # -- view domain ----------------------------------------------------------------------------
    def default_domain(self, elevation_span=6.0, radius_span=0.08, azimuth_span=360.0):
        cam = self.camera
        origin, _, _, fwd = cam.basis()
        v = self.tri_verts.astype(np.float64)
        lo, hi, any_subject = None, None, False
        for o in range(int(self.tri_object.max()) + 1):
            pts = v[self.tri_object == o].reshape(-1, 3)
            if not len(pts):
                continue
            a, b = pts.min(0), pts.max(0)
            if np.linalg.norm(b - a) < 4.0 * np.linalg.norm(0.5 * (a + b) - origin):  # skip floors, sky domes
                lo = a if lo is None else np.minimum(lo, a)
                hi = b if hi is None else np.maximum(hi, b)
                any_subject = True
        if any_subject:
            depth = float((0.5 * (lo + hi) - origin) @ fwd)
        else:
            cz = (v.mean(1) - origin) @ fwd
            depth = float(np.median(cz[cz > cam.clip_start])) if (cz > cam.clip_start).any() else 10.0
        depth = max(depth, 10.0 * cam.clip_start)
        pivot = origin + fwd * depth
        rel = origin - pivot
        r = float(np.linalg.norm(rel))
        el = math.degrees(math.asin(float(np.clip(rel[2] / r, -1, 1))))
        az = math.degrees(math.atan2(rel[0], -rel[1]))
        R = np.stack(cam.basis()[1:3] + (-fwd,), axis=1)
        offset = look_at(origin, pivot).T @ R
        e0 = el - elevation_span
        if el > 1.0:
            e0 = max(e0, 1.0)
        azimuth = (az, az + 360.0) if azimuth_span >= 360.0 else (az - 0.5 * azimuth_span, az + 0.5 * azimuth_span)
        return OrbitDomain(pivot, azimuth, (e0, min(el + elevation_span, 89.0)),
                           (r * (1 - radius_span), r * (1 + radius_span)), offset, (az, el, r))

    def path_domain(self, elevation_span=6.0, radius_span=0.08, azimuth_margin=30.0):
        """The view band around an *animated* camera: everything its path covers, plus margins."""
        base = self.default_domain(elevation_span, radius_span)
        if self.anim is None:
            return base
        az, el, rr = [], [], []
        for fi in range(self.n_frames):
            rel = self.anim['cam_matrix'][fi][:3, 3] - base.pivot
            r = float(np.linalg.norm(rel))
            rr.append(r)
            el.append(math.degrees(math.asin(float(np.clip(rel[2] / r, -1, 1)))))
            az.append(math.degrees(math.atan2(rel[0], -rel[1])))
        az = np.degrees(np.unwrap(np.radians(az)))
        lo, hi = float(az.min()) - azimuth_margin, float(az.max()) + azimuth_margin
        azimuth = (base.hero[0], base.hero[0] + 360.0) if hi - lo >= 359.0 else (lo, hi)
        e0 = min(el) - elevation_span
        if min(el) > 1.0:
            e0 = max(e0, 1.0)
        return OrbitDomain(base.pivot, azimuth, (e0, min(max(el) + elevation_span, 89.0)),
                           (min(rr) * (1 - radius_span), max(rr) * (1 + radius_span)), base.offset, base.hero)

    def camera_pose(self, domain, fi=None):
        """(azimuth, elevation, radius) of the scene camera at frame fi, in a domain's coordinates."""
        rel = np.asarray(self.camera_at(fi).matrix)[:3, 3] - domain.pivot
        r = float(np.linalg.norm(rel))
        return (math.degrees(math.atan2(rel[0], -rel[1])), math.degrees(math.asin(float(np.clip(rel[2] / r, -1, 1)))), r)

    def editable(self):
        """{material name: [editable parameters]} for a parametric fit: roughness wherever there is a Principled
        BSDF, base colour where it is one constant colour (not a texture or a colour attribute)."""
        out = {}
        for m in self.materials:
            if not m['resolved'] and m['color_attribute'] is None and m['name'] == 'default':
                continue
            params = ['roughness']
            if m['resolved'] and not m['color_attribute']:
                params.insert(0, 'base_color')
            out[m['name']] = params
        return out

    def group_weights(self, lights=None):
        """[groups][3]: each light group's colour x strength relative to the white light at its own strength that
        a parametric fit is taught with. lights: {group name: {energy, color} | {strength}} overrides."""
        out = []
        for g in self.groups:
            o = (lights or {}).get(g['name'], {})
            if g['kind'] == 'light':
                k = o.get('energy', g['energy']) / max(g['energy'], 1e-12)
                out.append([c * k for c in o.get('color', g['color'])])
            elif g['kind'] == 'world':
                out.append([o.get('strength', g['strength']) / max(g['strength'], 1e-12)] * 3)
            else:
                out.append([o.get('strength', 1.0)] * 3)
        return out

    def time_of(self, fi):
        """Animation time in [0, 1] of a frame index."""
        return 0.0 if self.n_frames < 2 or fi is None else fi / (self.n_frames - 1)

    def field_frame(self, domain):
        """Centre and scale that map the visible subject into the unit ball of the feature grid."""
        fx, fy, _, _ = self.camera.intrinsics()
        depth = domain.hero[2]
        return domain.pivot.astype(np.float32), float(depth * max(0.5 * self.camera.width / fx, 0.5 * self.camera.height / fy))

    def pixel_filter(self, S):
        """Per-subsample weights of the scene's pixel filter on an S x S grid inside one pixel."""
        r = self.meta['render']
        kind, width = r.get('filter_type', 'BLACKMAN_HARRIS'), max(float(r.get('filter_width', 1.5)), 1e-3)
        o = (np.arange(S) + 0.5) / S - 0.5
        if kind == 'BOX':
            w = (np.abs(o) <= 0.5 * width + 1e-9).astype(np.float64)
        elif kind == 'GAUSSIAN':
            w = np.exp(-2.0 * (o * 6.0 / width) ** 2)
        else:
            x = o / width + 0.5
            w = 0.35875 - 0.48829 * np.cos(2 * np.pi * x) + 0.14128 * np.cos(4 * np.pi * x) - 0.01168 * np.cos(6 * np.pi * x)
            w = np.where(np.abs(o) < 0.5 * width, w, 0.0)
        if w.sum() <= 0:
            w = np.ones(S)
        w2 = np.outer(w, w)
        return (w2 / w2.sum()).astype(np.float32)
