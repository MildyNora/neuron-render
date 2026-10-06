"""Metal compute kernels: the hot path of both fitting and rendering.

Stock GPU tensor ops are an order of magnitude too slow for gather-heavy work on Apple GPUs, so the
per-sample pipeline (surface reconstruction -> analytic light hints -> multiresolution feature grid)
and the per-pixel resolve (filter -> view transform) are written directly in Metal.
"""
from __future__ import annotations

import functools

import torch

FEATURE_SET = 4            # bump when the network inputs change; fitted models record it
N_BASE_FEATURES = 59    # normal 3, view 3, reflect 3, cos/fresnel/back 3, albedo 3+3, lightmap 3, material 8, SH(view) 15, SH(reflect) 15
N_TRACE_FEATURES = 18   # what the mirror ray and the refracted ray run into (see nr_dense)
N_LIGHT_FEATURES = 11   # per light: diffuse hint, mirror masks x3, two-bounce mirror x2, through-the-glass x3, rough highlight + its visibility
N_FREQ_FEATURES = 6     # per octave: sin/cos of the view direction


MAX_GROUPS = 8             # light groups a parametric fit can separate


def n_dense(n_lights, octaves, groups=1, parametric=False, time_octaves=0, n_global=0):
    """Length of the analytic feature vector. A static fit has one light group and no time or state inputs."""
    n = (N_BASE_FEATURES - 3 + 3 * groups) + (N_TRACE_FEATURES - 6 + 6 * groups) + N_FREQ_FEATURES * octaves
    n += (N_LIGHT_FEATURES + (1 if parametric else 0)) * n_lights
    return n + (2 * time_octaves + n_global if parametric else 0)


SOURCE = r"""
#include <metal_stdlib>
using namespace metal;

#define TRI_STRIDE 24u
#define MAT_STRIDE 12u
#define MAT_FEATS 8u
#define LV 5
// A plain static fit (one light group, nothing moving, the four-ray reflection estimate) is compiled with NR_PLAIN:
// everything the fits that cover changes need is then folded away at compile time rather than branched over.
#ifdef NR_PLAIN
#define MAX_G 1
#define NR_PAR(icfg) false
#define NR_GROUPS(icfg) 1
#define NR_ENV(icfg) 0
#define NR_LOBE(icfg) 0
#define NR_FOLLOW(icfg) false
#define NR_RT(lv, l) 0
#else
#define MAX_G 8
#define NR_PAR(icfg) ((icfg)[8] != 0)
#define NR_GROUPS(icfg) ((icfg)[12])
#define NR_ENV(icfg) ((icfg)[17])
#define NR_LOBE(icfg) ((icfg)[19])
#define NR_FOLLOW(icfg) ((icfg)[20] != 0)
#define NR_RT(lv, l) ((lv)[LV * (l) + 4])
#endif
#define LIGHT_STRIDE 20u
#define MAX_FEAT 320
#define P1 2654435761u
#define P2 805459861u
#define P3 3674653429u

struct Surf { float3 x; float3 xc; float3 bary; float3 n; float3 v; float3 r; float3 albedo; float cosv; float back; uint mat; int tri; };

static inline float3 ld3(device const float* b, uint i) { return float3(b[i], b[i + 1u], b[i + 2u]); }

// Reconstruct the shading point from (triangle id, ray): the rasterizer only ever tells us *which*
// triangle is visible; everything else is recomputed here, identically for fitting and rendering.
static inline Surf nr_surface(int tri, float3 o, float3 d, device const float* T, device const float* C, int sky) {
    Surf s;
    uint b = uint(tri) * TRI_STRIDE;
    s.mat = uint(T[b + 22u]);
    s.v = -d;
    s.tri = tri;
    if (tri == sky) {
        s.x = o + d * 1.0e6f; s.n = -d; s.r = d; s.albedo = float3(0.0f); s.cosv = 1.0f; s.back = 0.0f;
        s.xc = s.x; s.bary = float3(1.0f, 0.0f, 0.0f);
        return s;
    }
    float3 v0 = ld3(T, b), v1 = ld3(T, b + 3u), v2 = ld3(T, b + 6u);
    float3 ng = ld3(T, b + 18u);
    float den = dot(ng, d);
    den = fabs(den) > 1.0e-12f ? den : 1.0e-12f;
    s.x = o + d * ((T[b + 21u] - dot(ng, o)) / den);
    float3 e1 = v1 - v0, e2 = v2 - v0, ep = s.x - v0;
    float d11 = dot(e1, e1), d12 = dot(e1, e2), d22 = dot(e2, e2), dp1 = dot(ep, e1), dp2 = dot(ep, e2);
    float inv = 1.0f / max(d11 * d22 - d12 * d12, 1.0e-30f);
    float b1 = clamp((d22 * dp1 - d12 * dp2) * inv, 0.0f, 1.0f), b2 = clamp((d11 * dp2 - d12 * dp1) * inv, 0.0f, 1.0f);
    float b0 = clamp(1.0f - b1 - b2, 0.0f, 1.0f);
    float3 n = ng;
    if (T[b + 23u] < 0.5f) n = normalize(b0 * ld3(T, b + 9u) + b1 * ld3(T, b + 12u) + b2 * ld3(T, b + 15u));
    float c = dot(n, s.v);
    s.back = c < 0.0f ? 1.0f : 0.0f;
    if (c < 0.0f) { n = -n; c = -c; }
    s.n = n; s.cosv = c;
    s.r = d - 2.0f * dot(d, n) * n;
    uint cb = uint(tri) * 9u;
    s.albedo = b0 * ld3(C, cb) + b1 * ld3(C, cb + 3u) + b2 * ld3(C, cb + 6u);
    s.xc = s.x; s.bary = float3(b0, b1, b2);
    return s;
}

// Parametric fits: learned features live in the reference pose (so they travel with a moving object), and a
// material's base colour is an input (its row of the material table) rather than baked into the corners.
static inline void nr_canon(thread Surf& s, int sky, device const float* Tref, device const float* M) {
    if (s.tri == sky) return;
    uint b = uint(s.tri) * 9u;
    s.xc = s.bary.x * ld3(Tref, b) + s.bary.y * ld3(Tref, b + 3u) + s.bary.z * ld3(Tref, b + 6u);
    s.albedo *= ld3(M, s.mat * MAT_STRIDE + 8u);
}

static inline int nr_sh3(float3 d, thread float* f, int k) {
    float x = d.x, y = d.y, z = d.z, xx = x * x, yy = y * y, zz = z * z;
    f[k++] = 0.488603f * y; f[k++] = 0.488603f * z; f[k++] = 0.488603f * x;
    f[k++] = 1.092548f * x * y; f[k++] = 1.092548f * y * z; f[k++] = 0.315392f * (3.0f * zz - 1.0f);
    f[k++] = 1.092548f * x * z; f[k++] = 0.546274f * (xx - yy);
    f[k++] = 0.590044f * y * (3.0f * xx - yy); f[k++] = 2.890611f * x * y * z; f[k++] = 0.457046f * y * (5.0f * zz - 1.0f);
    f[k++] = 0.373176f * z * (5.0f * zz - 3.0f); f[k++] = 0.457046f * x * (5.0f * zz - 1.0f);
    f[k++] = 1.445306f * z * (xx - yy); f[k++] = 0.590044f * x * (xx - 3.0f * yy);
    return k;
}

static inline float nr_soft(float ang, float width) { return 1.0f / (1.0f + exp(clamp(ang / width, -30.0f, 30.0f))); }

// Unshadowed diffuse irradiance hint from one light at point x with normal n.
static inline float nr_light_E(float3 x, float3 n, device const float* Lg, uint lb) {
    int type = int(Lg[lb]);
    float3 c = ld3(Lg, lb + 1u), nl = ld3(Lg, lb + 10u);
    if (type == 0) {
        float3 ux = ld3(Lg, lb + 4u), uy = ld3(Lg, lb + 7u);
        float hx = Lg[lb + 13u], hy = Lg[lb + 14u], E = 0.0f;
        for (int q = 0; q < 4; q++) {
            float3 p = c + ux * (hx * ((q & 1) ? 0.57735f : -0.57735f)) + uy * (hy * ((q & 2) ? 0.57735f : -0.57735f));
            float3 L = p - x;
            float d2 = max(dot(L, L), 1.0e-8f);
            float3 wl = L * rsqrt(d2);
            E += max(dot(n, wl), 0.0f) * max(-dot(wl, nl), 0.0f) / d2;
        }
        return E * hx * hy;
    }
    if (type == 3) return max(-dot(n, nl), 0.0f);
    float3 L = c - x;
    float d2 = max(dot(L, L), 1.0e-8f);
    float3 wl = L * rsqrt(d2);
    float spot = 1.0f;
    if (type == 2) {
        float cc = Lg[lb + 17u];
        spot = smoothstep(cc, cc + (1.0f - cc) * max(Lg[lb + 18u], 1.0e-3f), -dot(wl, nl));
    }
    return max(dot(n, wl), 0.0f) * spot / d2;
}

// Signed angular distance (radians, negative = inside) between a ray and one light's emitting shape.
static inline float nr_light_ang(float3 o, float3 dir, device const float* Lg, uint lb) {
    int type = int(Lg[lb]);
    float3 c = ld3(Lg, lb + 1u), nl = ld3(Lg, lb + 10u);
    float rad = Lg[lb + 15u];
    if (type == 0) {
        float den = dot(dir, nl);
        if (den > -1.0e-4f) return 1.0e3f;
        float t = dot(c - o, nl) / den;
        if (t <= 0.0f) return 1.0e3f;
        float3 q = o + t * dir - c;
        float hx = Lg[lb + 13u], hy = Lg[lb + 14u];
        float a = dot(q, ld3(Lg, lb + 4u)), b = dot(q, ld3(Lg, lb + 7u));
        float dd = max(fabs(a) - hx, fabs(b) - hy);
        if (Lg[lb + 16u] > 0.5f) dd = (length(float2(a / hx, b / hy)) - 1.0f) * min(hx, hy);
        return dd * (-den) / t;
    }
    if (type == 3) return acos(clamp(-dot(dir, nl), -1.0f, 1.0f)) - rad;
    float3 L = c - o;
    float d2 = max(dot(L, L), 1.0e-8f);
    float3 wl = L * rsqrt(d2);
    if (type == 2 && -dot(wl, nl) < Lg[lb + 17u]) return 1.0e3f;
    return acos(clamp(dot(dir, wl), -1.0f, 1.0f)) - atan(rad * rsqrt(d2));
}

// ---- a small ray tracer: secondary rays are traced exactly, so that the network is told where
//      reflections and refractions *go* and only has to learn what they *carry* ----

struct Hit { float t; int tri; float u; float v; };

static inline Hit nr_trace(float3 o, float3 d, device const float* bvh, device const int* order, device const float* T) {
    Hit h; h.t = 1.0e30f; h.tri = -1; h.u = 0.0f; h.v = 0.0f;
    float3 inv = 1.0f / float3(fabs(d.x) > 1.0e-9f ? d.x : 1.0e-9f, fabs(d.y) > 1.0e-9f ? d.y : 1.0e-9f, fabs(d.z) > 1.0e-9f ? d.z : 1.0e-9f);
    uint stack[48]; int sp = 0; stack[sp++] = 0u;
    for (int guard = 0; sp > 0 && guard < 4096; guard++) {   // bounded, whatever the input (a NaN ray visits every node)
        uint b = stack[--sp] * 8u;
        float3 t0 = (ld3(bvh, b) - o) * inv, t1 = (ld3(bvh, b + 4u) - o) * inv;
        float3 tn = min(t0, t1), tf = max(t0, t1);
        float tnear = max(max(tn.x, tn.y), max(tn.z, 0.0f)), tfar = min(min(tf.x, tf.y), min(tf.z, h.t));
        if (tnear > tfar) continue;
        int a = int(bvh[b + 3u]), c = int(bvh[b + 7u]);
        if (c > 0) {
            for (int i = 0; i < c; i++) {
                int tri = order[a + i];
                uint tb = uint(tri) * TRI_STRIDE;
                float3 v0 = ld3(T, tb), e1 = ld3(T, tb + 3u) - v0, e2 = ld3(T, tb + 6u) - v0;
                float3 p = cross(d, e2);
                float det = dot(e1, p);
                if (fabs(det) < 1.0e-14f) continue;
                float idet = 1.0f / det;
                float3 tv = o - v0;
                float u = dot(tv, p) * idet;
                if (u < 0.0f || u > 1.0f) continue;
                float3 q = cross(tv, e1);
                float v = dot(d, q) * idet;
                if (v < 0.0f || u + v > 1.0f) continue;
                float t = dot(e2, q) * idet;
                if (t > 1.0e-5f && t < h.t) { h.t = t; h.tri = tri; h.u = u; h.v = v; }
            }
        } else if (sp < 46) { stack[sp++] = uint(-c); stack[sp++] = uint(a); }
    }
    return h;
}

static inline bool nr_refract(float3 d, float3 n, float eta, thread float3& out) { // n faces against d
    float c = -dot(d, n), k = 1.0f - eta * eta * (1.0f - c * c);
    if (k < 0.0f) return false;
    out = normalize(eta * d + (eta * c - sqrt(k)) * n);
    return true;
}

// Shading normal at a traced hit (interpolated on smooth triangles), turned to face the incoming ray.
static inline float3 nr_hit_normal(Hit h, float3 dir, device const float* T) {
    uint b = uint(h.tri) * TRI_STRIDE;
    float3 n = ld3(T, b + 18u);
    if (T[b + 23u] < 0.5f) n = normalize((1.0f - h.u - h.v) * ld3(T, b + 9u) + h.u * ld3(T, b + 12u) + h.v * ld3(T, b + 15u));
    return dot(n, dir) > 0.0f ? -n : n;
}

// World position -> feature-grid coordinates in [0,1)^3. The subject sits in the unit ball; everything
// beyond (floors, backdrops, the sky at infinity) is contracted into the shell between radius 1 and 2.
// cfg: [0..2] centre, [3] 1/scale, [4..6] log-radiance of a constant sky, [7] 1 if the sky lives in the lightmap
static inline float3 nr_grid_pos(float3 x, device const float* cfg) {
    float3 q = (x - float3(cfg[0], cfg[1], cfg[2])) * cfg[3];
    float m = length(q);
    if (m > 1.0f) q *= (2.0f - 1.0f / m) / m;
    return clamp(q * 0.25f + 0.5f, 0.0f, 0.999999f);
}

static inline uint nr_cell(uint3 c, uint t, uint res1, uint size, int dense) {
    if (dense != 0) return c.x + res1 * (c.y + res1 * (c.z + res1 * t));
    return (c.x ^ (c.y * P1) ^ (c.z * P2) ^ (t * P3)) & (size - 1u);
}

// Multiresolution grid lookup. A level row is (resolution, offset, size, dense, time resolution). Levels with a
// time resolution are four-dimensional: the two time slices around tau are blended, so what the grid stores
// can change over an animation.
static inline void nr_hash(float3 x, float tau, device const float* table, device const int* lv, int L, int F, thread float* f, int k) {
    for (int l = 0; l < L; l++) {
        int res = lv[LV * l]; uint off = uint(lv[LV * l + 1]); uint size = uint(lv[LV * l + 2]); int dense = lv[LV * l + 3]; int rt = NR_RT(lv, l);
        float3 p = x * float(res);
        float3 p0 = floor(p); float3 w = p - p0; uint3 c0 = uint3(int3(p0));
        float tt = clamp(tau, 0.0f, 1.0f) * float(rt);
        float t0 = min(floor(tt), float(max(rt - 1, 0)));
        float wt = tt - t0;
        for (int c = 0; c < F; c++) f[k + l * F + c] = 0.0f;
        for (uint ts = 0u; ts < (rt > 0 ? 2u : 1u); ts++) {
            float ws = rt > 0 ? (ts != 0u ? wt : 1.0f - wt) : 1.0f;
            for (uint j = 0u; j < 8u; j++) {
                uint3 dd = uint3(j & 1u, (j >> 1) & 1u, (j >> 2) & 1u);
                float wk = (dd.x != 0u ? w.x : 1.0f - w.x) * (dd.y != 0u ? w.y : 1.0f - w.y) * (dd.z != 0u ? w.z : 1.0f - w.z) * ws;
                uint e = (off + nr_cell(c0 + dd, uint(t0) + ts, uint(res + 1), size, dense)) * uint(F);
                for (int c = 0; c < F; c++) f[k + l * F + c] += wk * table[e + uint(c)];
            }
        }
    }
}

// The baked lightmap: view-independent log-radiance of whatever sits at a world position. It is fitted
// first (fit.train_lightmap) and is what a traced reflection or refraction ray "sees" where it lands:
// texture, shadow and bounce light included, at the price of one grid lookup.
static inline float3 nr_lightmap(float3 x, device const float* cfg, device const float* lt, device const int* llv, int LL) {
    float3 p = nr_grid_pos(x, cfg);
    float3 acc = float3(0.0f);
    for (int l = 0; l < LL; l++) {
        int res = llv[LV * l]; uint off = uint(llv[LV * l + 1]); uint size = uint(llv[LV * l + 2]); int dense = llv[LV * l + 3];
        float3 q = p * float(res);
        float3 q0 = floor(q); float3 w = q - q0; uint3 c0 = uint3(int3(q0));
        for (uint j = 0u; j < 8u; j++) {
            uint3 dd = uint3(j & 1u, (j >> 1) & 1u, (j >> 2) & 1u);
            float wk = (dd.x != 0u ? w.x : 1.0f - w.x) * (dd.y != 0u ? w.y : 1.0f - w.y) * (dd.z != 0u ? w.z : 1.0f - w.z);
            uint e = (off + nr_cell(c0 + dd, 0u, uint(res + 1), size, dense)) * 3u;
            acc += wk * float3(lt[e], lt[e + 1u], lt[e + 2u]);
        }
    }
    return acc;
}

// What a ray that leaves the scene sees.
static inline float3 nr_sky(float3 x, float3 dir, device const float* cfg, device const float* lt, device const int* llv, int LL) {
    if (cfg[7] > 0.5f) return nr_lightmap(x + dir * 1.0e6f, cfg, lt, llv, LL);
    return float3(cfg[4], cfg[5], cfg[6]);
}

// Parametric fits keep one lightmap per light group, changing over time, with the surface's own base colour
// factored out (so it survives a material edit): grid features at (reference position, time), decoded by a
// linear map to log-radiance per group, plus log albedo.
static inline void nr_lightmap_p(float3 xc, float tau, float3 logalb, device const float* cfg, device const float* lt,
                                 device const int* llv, device const float* ldec, int LL, int FL, int G, thread float* out) {
    float h[96];
    nr_hash(nr_grid_pos(xc, cfg), tau, lt, llv, LL, FL, h, 0);
    int n = LL * FL;
    for (int j = 0; j < 3 * G; j++) {
        uint row = uint(j) * uint(n + 1);
        float acc = ldec[row + uint(n)];
        for (int i = 0; i < n; i++) acc += ldec[row + uint(i)] * h[i];
        out[j] = acc + logalb[j % 3];
    }
}

// Lightmap at a surface point: three numbers for a static fit, three per light group for a parametric one.
static inline void nr_lm(float3 xc, float3 alb, float tau, device const float* cfg, device const float* lt,
                         device const int* llv, device const float* ldec, device const int* icfg, thread float* out) {
    if (!NR_PAR(icfg)) {
        float3 m = nr_lightmap(xc, cfg, lt, llv, icfg[7]);
        out[0] = m.x; out[1] = m.y; out[2] = m.z;
        return;
    }
    nr_lightmap_p(xc, tau, log(alb + 0.02f), cfg, lt, llv, ldec, icfg[7], icfg[13], icfg[12], out);
}

// The world as Blender's own shader gives it, baked at fit time into an equirectangular map of log-radiance
// (environment.py): column = azimuth, row = polar angle from +Z.
static inline float3 nr_env(float3 d, device const float* env, int W) {
    int H = W / 2;
    float u = (atan2(d.y, d.x) * 0.15915494f + 0.5f) * float(W) - 0.5f;
    float v = acos(clamp(d.z, -1.0f, 1.0f)) * 0.31830989f * float(H) - 0.5f;
    float u0 = floor(u), v0 = floor(v);
    float fu = u - u0, fv = v - v0;
    int x0 = clamp((int(u0) + W) % W, 0, W - 1), x1 = (x0 + 1) % W;   // (the clamp: a degenerate ray must not read outside)
    int y0 = clamp(int(v0), 0, H - 1), y1 = clamp(int(v0) + 1, 0, H - 1);
    float3 top = mix(ld3(env, uint(y0 * W + x0) * 3u), ld3(env, uint(y0 * W + x1) * 3u), fu);
    float3 bot = mix(ld3(env, uint(y1 * W + x0) * 3u), ld3(env, uint(y1 * W + x1) * 3u), fu);
    return mix(top, bot, fv);
}

// ... and what a ray that leaves the scene sees: the baked environment when there is one (icfg[17] its width,
// icfg[18] the light group the world belongs to), else the sky as the fit knows it (cfg[8..]: a constant per
// light group, parametric fits).
static inline void nr_lm_sky(float3 x, float3 dir, float tau, device const float* cfg, device const float* lt,
                             device const int* llv, device const float* ldec, device const int* icfg,
                             device const float* env, thread float* out) {
    if (NR_ENV(icfg) > 0) {
        float3 e = nr_env(dir, env, NR_ENV(icfg));
        if (!NR_PAR(icfg)) { out[0] = e.x; out[1] = e.y; out[2] = e.z; return; }
        for (int j = 0; j < 3 * NR_GROUPS(icfg); j++) out[j] = cfg[8 + j];
        int g = 3 * icfg[18];
        out[g] = e.x; out[g + 1] = e.y; out[g + 2] = e.z;
        return;
    }
    if (cfg[7] > 0.5f) { nr_lm(x + dir * 1.0e6f, float3(0.98f), tau, cfg, lt, llv, ldec, icfg, out); return; }
    if (!NR_PAR(icfg)) { out[0] = cfg[4]; out[1] = cfg[5]; out[2] = cfg[6]; return; }
    for (int j = 0; j < 3 * NR_GROUPS(icfg); j++) out[j] = cfg[8 + j];
}

// Base colour at a traced hit, and the hit point in the reference pose.
static inline float3 nr_hit_albedo(Hit h, device const float* T, device const float* C, device const float* M, int par) {
    uint cb = uint(h.tri) * 9u;
    float3 a = (1.0f - h.u - h.v) * ld3(C, cb) + h.u * ld3(C, cb + 3u) + h.v * ld3(C, cb + 6u);
    if (par != 0) a *= ld3(M, uint(T[uint(h.tri) * TRI_STRIDE + 22u]) * MAT_STRIDE + 8u);
    return a;
}

static inline float3 nr_hit_canon(Hit h, device const float* Tref) {
    uint b = uint(h.tri) * 9u;
    return (1.0f - h.u - h.v) * ld3(Tref, b) + h.u * ld3(Tref, b + 3u) + h.v * ld3(Tref, b + 6u);
}

// The highlight one light leaves on a *rough* surface: the GGX lobe integrated over the light by
// quadrature (unshadowed, Fresnel left to the network). A blurred glow has no sharp mirror image to
// test against, so this is what tells the network where a light pools on, say, a satin floor.
static inline float nr_light_gloss(float3 x, float3 n, float3 v, float cosv, float alpha, device const float* Lg, uint lb) {
    int type = int(Lg[lb]);
    float3 c = ld3(Lg, lb + 1u), nl = ld3(Lg, lb + 10u);
    float a2 = alpha * alpha, S = 0.0f;
    if (type == 0) {
        float3 ux = ld3(Lg, lb + 4u), uy = ld3(Lg, lb + 7u);
        float hx = Lg[lb + 13u], hy = Lg[lb + 14u];
        for (int j = 0; j < 4; j++) {
            for (int i = 0; i < 4; i++) {
                float3 L = c + ux * (hx * (float(i) - 1.5f) * 0.5f) + uy * (hy * (float(j) - 1.5f) * 0.5f) - x;
                float d2 = max(dot(L, L), 1.0e-8f);
                float3 l = L * rsqrt(d2);
                float cl = -dot(l, nl);
                if (dot(n, l) <= 0.0f || cl <= 0.0f) continue;
                float nh = dot(n, normalize(l + v));
                float den = nh * nh * (a2 - 1.0f) + 1.0f;
                S += a2 / (3.14159265f * den * den) * cl / d2;
            }
        }
        S *= hx * hy * 0.25f;
    } else {
        float3 l = -nl;
        float w = 1.0f;
        if (type != 3) {
            float3 L = c - x;
            float d2 = max(dot(L, L), 1.0e-8f);
            l = L * rsqrt(d2); w = 1.0f / d2;
            if (type == 2 && -dot(l, nl) < Lg[lb + 17u]) w = 0.0f;
        }
        if (dot(n, l) > 0.0f) {
            float nh = dot(n, normalize(l + v));
            float den = nh * nh * (a2 - 1.0f) + 1.0f;
            S = a2 / (3.14159265f * den * den) * w;
        }
    }
    return S / (4.0f * max(cosv, 0.02f));
}

// Fraction of a light that a point can actually see (shadow rays: four for an area light, one otherwise).
static inline float nr_light_visible(float3 xo, device const float* Lg, uint lb, device const float* bvh,
                                     device const int* order, device const float* T) {
    int type = int(Lg[lb]);
    float3 c = ld3(Lg, lb + 1u), ux = ld3(Lg, lb + 4u), uy = ld3(Lg, lb + 7u);
    float hx = type == 0 ? Lg[lb + 13u] : 0.0f, hy = type == 0 ? Lg[lb + 14u] : 0.0f, seen = 0.0f;
    int nq = type == 0 ? 4 : 1;
    for (int q = 0; q < nq; q++) {
        float3 L = (type == 3 ? xo - ld3(Lg, lb + 10u) * 1.0e4f : c + ux * (hx * ((q & 1) ? 0.5f : -0.5f)) + uy * (hy * ((q & 2) ? 0.5f : -0.5f))) - xo;
        float dist = length(L);
        Hit h = nr_trace(xo, L / dist, bvh, order, T);
        seen += (h.tri < 0 || h.t > dist) ? 1.0f : 0.0f;
    }
    return seen / float(nq);
}

// What a traced ray sees where it lands: the lightmap there. With icfg[20] set, a surface that has no look of
// its own passes the ray on, once: a metal mirrors it and tints what it then finds, glass lets it through. So a
// mirror seen in a mirror shows what it reflects from *there*, not an average of what the camera saw in it.
static inline void nr_seen(Hit h, float3 xw, float3 dir, float tau, device const float* T, device const float* Tref,
                           device const float* C, device const float* M, device const float* bvh, device const int* order,
                           device const float* lt, device const int* llv, device const float* ldec, device const float* env,
                           device const float* cfg, device const int* icfg, thread float* out) {
    int par = NR_PAR(icfg) ? 1 : 0;
    if (NR_FOLLOW(icfg)) {
        uint hm = uint(T[uint(h.tri) * TRI_STRIDE + 22u]) * MAT_STRIDE;
        bool metal = M[hm + 1u] >= 0.5f && M[hm] <= 0.6f, glass = M[hm + 2u] > 0.5f;
        if (metal || glass) {
            float3 n = nr_hit_normal(h, dir, T);
            float3 d2 = metal ? dir - 2.0f * dot(dir, n) * n : dir;
            float3 o2 = xw + n * (metal ? 1.0e-4f : -1.0e-4f);
            Hit h2 = nr_trace(o2, d2, bvh, order, T);
            for (int k = 0; k < 3 && h2.tri >= 0 && M[uint(T[uint(h2.tri) * TRI_STRIDE + 22u]) * MAT_STRIDE + 2u] > 0.5f; k++) {
                o2 += d2 * (h2.t + 1.0e-4f);
                h2 = nr_trace(o2, d2, bvh, order, T);
            }
            if (h2.tri >= 0) nr_lm(par != 0 ? nr_hit_canon(h2, Tref) : o2 + d2 * h2.t, nr_hit_albedo(h2, T, C, M, par), tau, cfg, lt, llv, ldec, icfg, out);
            else nr_lm_sky(o2, d2, tau, cfg, lt, llv, ldec, icfg, env, out);
            if (metal) {
                float3 tint = log(nr_hit_albedo(h, T, C, M, par) + 0.02f);
                for (int j = 0; j < 3 * NR_GROUPS(icfg); j++) out[j] += tint[j % 3];
            }
            return;
        }
    }
    nr_lm(par != 0 ? nr_hit_canon(h, Tref) : xw, nr_hit_albedo(h, T, C, M, par), tau, cfg, lt, llv, ldec, icfg, out);
}

#define NR_LM_HIT(h, xw, dir, out) nr_seen(h, xw, dir, tau, T, Tref, C, M, bvh, order, lt, llv, ldec, env, cfg, icfg, out)

static inline int nr_dense(Surf s, float3 d, float tau, device const float* gvec, device const float* T, device const float* Tref,
                           device const float* C, device const float* M, device const float* Lg, device const float* bvh,
                           device const int* order, device const float* lt, device const int* llv, device const float* ldec,
                           device const float* env, device const float* cfg, device const int* icfg, thread float* f) {
    int NL = icfg[1], NF = icfg[6], LL = icfg[7], par = NR_PAR(icfg) ? 1 : 0, G3 = 3 * NR_GROUPS(icfg);
    int k = 0;
    f[k++] = s.n.x; f[k++] = s.n.y; f[k++] = s.n.z;
    f[k++] = s.v.x; f[k++] = s.v.y; f[k++] = s.v.z;
    f[k++] = s.r.x; f[k++] = s.r.y; f[k++] = s.r.z;
    f[k++] = s.cosv; f[k++] = pow(1.0f - clamp(s.cosv, 0.0f, 1.0f), 5.0f); f[k++] = s.back;
    f[k++] = s.albedo.x; f[k++] = s.albedo.y; f[k++] = s.albedo.z;
    f[k++] = log(s.albedo.x + 0.01f); f[k++] = log(s.albedo.y + 0.01f); f[k++] = log(s.albedo.z + 0.01f);
    // The lightmap here, where the mirror ray lands, and where the ray through glass lands (per light group).
    float lm[3 * MAX_G], mr[3 * MAX_G], ms[3 * MAX_G];
    for (int j = 0; j < G3; j++) { lm[j] = 0.0f; mr[j] = 0.0f; ms[j] = 0.0f; }
    if (LL > 0) nr_lm(s.xc, s.albedo, tau, cfg, lt, llv, ldec, icfg, lm);
    int klm = k;
    bool have_mr = false;
    for (int j = 0; j < G3; j++) f[k++] = lm[j];
    for (uint i = 0u; i < MAT_FEATS; i++) f[k++] = M[s.mat * MAT_STRIDE + i];
    k = nr_sh3(s.v, f, k);
    k = nr_sh3(s.r, f, k);
    float3 ph = s.v * 3.14159265f;
    for (int o = 0; o < NF; o++, ph *= 2.0f) {
        float3 sn = sin(ph), cs = cos(ph);
        f[k++] = sn.x; f[k++] = sn.y; f[k++] = sn.z; f[k++] = cs.x; f[k++] = cs.y; f[k++] = cs.z;
    }

    // Mirror path: where does the reflection go, and (one more bounce) where does *that* go?
    float3 xo = s.x + s.n * 1.0e-4f, x2 = xo, r2 = s.r;
    float vis1 = 1.0f, vis2 = 0.0f, near1 = 0.0f, E2 = 0.0f;
    // Glass path: refract in, bounce inside (total internal reflection), refract out -- and on through any
    // further glass in the way, until the ray lands on something opaque or leaves the scene.
    float npass = 0.0f, plen = 0.0f, ntir = 0.0f, vis_e = 0.0f;
    float3 xe = s.x, te = s.r;
    // Only for glossy surfaces: on a rough one a single traced ray would stamp a sharp image where
    // Cycles shows a blur, so there the network is left to learn the reflection on its own.
    float rough = M[s.mat * MAT_STRIDE];
    bool glossy = rough <= 0.3f;
    bool surface = M[s.mat * MAT_STRIDE + 7u] < 0.75f;
    if (glossy && surface) {
        Hit h1 = nr_trace(xo, s.r, bvh, order, T);
        if (h1.tri >= 0) {
            vis1 = 0.0f; near1 = 1.0f / (1.0f + h1.t);
            float3 n2 = nr_hit_normal(h1, s.r, T);
            float3 xh = xo + h1.t * s.r;
            x2 = xh + n2 * 1.0e-4f;
            r2 = s.r - 2.0f * dot(s.r, n2) * n2;
            for (int i = 0; i < NL; i++) E2 += nr_light_E(x2, n2, Lg, uint(i) * LIGHT_STRIDE);
            if (LL > 0) NR_LM_HIT(h1, xh, s.r, mr);
            vis2 = nr_trace(x2, r2, bvh, order, T).tri < 0 ? 1.0f : 0.0f;
        } else if (LL > 0) {
            nr_lm_sky(xo, s.r, tau, cfg, lt, llv, ldec, icfg, env, mr);
        }
        have_mr = LL > 0;
        float ior = M[s.mat * MAT_STRIDE + 5u] + 1.0f;
        if (M[s.mat * MAT_STRIDE + 2u] > 1.0e-3f && s.back < 0.5f) {
            float3 td = d, oi = s.x - s.n * 1.0e-4f;
            nr_refract(d, s.n, 1.0f / ior, td);
            bool inside = true;
            for (int b = 0; b < 8; b++) {
                Hit hi = nr_trace(oi, td, bvh, order, T);
                if (hi.tri < 0) {
                    if (!inside) { vis_e = 1.0f; if (LL > 0) nr_lm_sky(oi, td, tau, cfg, lt, llv, ldec, icfg, env, ms); }
                    break;
                }
                float3 nb = nr_hit_normal(hi, td, T);
                float3 xb = oi + hi.t * td;
                if (inside) {
                    plen += hi.t;
                    float3 out = td;
                    if (nr_refract(td, nb, ior, out)) { inside = false; npass += 1.0f; td = out; oi = xb - nb * 1.0e-4f; xe = oi; te = td; }
                    else { td = td - 2.0f * dot(td, nb) * nb; oi = xb + nb * 1.0e-4f; ntir += 1.0f; }
                } else {
                    uint hm = uint(T[uint(hi.tri) * TRI_STRIDE + 22u]);
                    if (M[hm * MAT_STRIDE + 2u] > 0.5f && b < 6) {   // more glass: go through it as well
                        ior = M[hm * MAT_STRIDE + 5u] + 1.0f;
                        float3 in = td;
                        nr_refract(td, nb, 1.0f / ior, in);
                        td = in; oi = xb - nb * 1.0e-4f; inside = true;
                    } else {
                        if (LL > 0) NR_LM_HIT(hi, xb, td, ms);
                        break;
                    }
                }
            }
        }
    }
    float got = npass > 0.5f ? 1.0f : 0.0f;
    // A rough metal still mirrors its surroundings, only blurred, and what its reflection carries is the average
    // over the lobe, in radiance. (Rough dielectrics reflect too little for this to pay.) By default that is four
    // rays spread around the mirror direction, for moderately rough metals only. With icfg[19] > 0 it is that
    // many rays drawn from the GGX distribution itself, for every metal that is not polished: needed once
    // things move or materials change, because the network can then no longer memorise, place by place, what
    // a crude estimate gets wrong.
    int lobe = NR_LOBE(icfg);
    bool metal = M[s.mat * MAT_STRIDE + 1u] >= 0.5f && LL > 0 && surface;
    if (lobe > 0 ? (metal && rough > 0.08f) : (metal && !glossy && rough <= 0.6f)) {
        float3 ax = lobe > 0 ? s.n : s.r;
        float3 t1 = normalize(cross(ax, fabs(ax.z) < 0.9f ? float3(0.0f, 0.0f, 1.0f) : float3(1.0f, 0.0f, 0.0f)));
        float3 t2 = cross(ax, t1);
        float spread = 1.5f * rough * rough, a2 = rough * rough * rough * rough, hits = 0.0f, nearsum = 0.0f, cnt = 0.0f;
        float acc[3 * MAX_G], one[3 * MAX_G];
        for (int j = 0; j < G3; j++) acc[j] = 0.0f;
        int nq = lobe > 0 ? lobe : 4;
        for (int q = 0; q < nq; q++) {
            float3 dir;
            if (lobe > 0) {   // a microfacet normal from the GGX distribution (a fixed, well-spread set), then mirror in it
                float xi = (float(q) + 0.5f) / float(nq), phi = 6.2831853f * fract(float(q) * 0.61803399f);
                float ct = rsqrt(1.0f + a2 * xi / (1.0f - xi)), st = sqrt(max(1.0f - ct * ct, 0.0f));
                float3 hv = st * (cos(phi) * t1 + sin(phi) * t2) + ct * s.n;
                float vh = dot(s.v, hv);
                if (vh <= 0.0f) continue;
                dir = 2.0f * vh * hv - s.v;
            } else {
                dir = normalize(s.r + spread * (((q & 1) ? 1.0f : -1.0f) * t1 + ((q & 2) ? 1.0f : -1.0f) * t2));
            }
            float below = 0.02f - dot(dir, s.n);
            if (below > 0.0f) dir = normalize(dir + s.n * below);
            Hit hq = nr_trace(xo, dir, bvh, order, T);
            if (hq.tri >= 0) { NR_LM_HIT(hq, xo + hq.t * dir, dir, one); hits += 1.0f; nearsum += 1.0f / (1.0f + hq.t); }
            else nr_lm_sky(xo, dir, tau, cfg, lt, llv, ldec, icfg, env, one);
            for (int j = 0; j < G3; j++) acc[j] += exp(one[j]);
            cnt += 1.0f;
        }
        if (cnt > 0.5f) {
            for (int j = 0; j < G3; j++) mr[j] = log(acc[j] / cnt);
            if (!glossy) { near1 = nearsum / cnt; vis1 = 1.0f - hits / cnt; }
            have_mr = true;
        }
    }
    // A metal has no look of its own, and neither has clear glass: its lightmap is an average over everything the
    // teacher's cameras happened to see in it. With icfg[20] the first guess there is what the surface shows *now*:
    // the tinted reflection, or what lies behind the glass.
    if (NR_FOLLOW(icfg) && surface) {
        float3 tint = log(s.albedo + 0.02f);
        if (M[s.mat * MAT_STRIDE + 2u] > 0.5f && got > 0.5f) { for (int j = 0; j < G3; j++) f[klm + j] = ms[j]; }
        else if (M[s.mat * MAT_STRIDE + 1u] >= 0.5f && have_mr) { for (int j = 0; j < G3; j++) f[klm + j] = mr[j] + tint[j % 3]; }
    }
    f[k++] = vis1; f[k++] = near1; f[k++] = vis2; f[k++] = log(E2 + 0.01f) * (1.0f - vis1);
    for (int j = 0; j < G3; j++) f[k++] = mr[j];
    f[k++] = 0.5f * npass; f[k++] = 1.0f / (1.0f + plen); f[k++] = exp(-plen); f[k++] = 0.5f * ntir; f[k++] = vis_e;
    for (int j = 0; j < G3; j++) f[k++] = ms[j];
    f[k++] = got * te.x; f[k++] = got * te.y; f[k++] = got * te.z;

    for (int i = 0; i < NL; i++) {
        uint lb = uint(i) * LIGHT_STRIDE;
        f[k++] = log(nr_light_E(s.x, s.n, Lg, lb) + 0.01f);
        float a1 = nr_light_ang(xo, s.r, Lg, lb);
        float m1 = glossy ? vis1 : 0.0f;   // a rough surface shows no mirror image, only the glow below
        f[k++] = m1 * nr_soft(a1, 0.02f); f[k++] = m1 * nr_soft(a1, 0.06f); f[k++] = m1 * nr_soft(a1, 0.2f);
        float a2 = nr_light_ang(x2, r2, Lg, lb);
        f[k++] = vis2 * nr_soft(a2, 0.06f); f[k++] = vis2 * nr_soft(a2, 0.2f);
        float a3 = nr_light_ang(xe, te, Lg, lb);
        float ge = got * vis_e;
        f[k++] = ge * nr_soft(a3, 0.03f); f[k++] = ge * nr_soft(a3, 0.08f); f[k++] = ge * nr_soft(a3, 0.25f);
        float S = nr_light_gloss(s.x, s.n, s.v, s.cosv, max(rough * rough, 0.1f), Lg, lb);
        float vs = glossy ? vis1 : 1.0f;
        // When things move, so do their shadows: tell the network whether this lamp is actually in view.
        bool shadowed = par != 0 && surface;
        float seen = shadowed ? nr_light_visible(xo, Lg, lb, bvh, order, T) : 1.0f;
        if (!glossy && S > 1.0e-2f) vs = shadowed ? seen : nr_light_visible(xo, Lg, lb, bvh, order, T);
        f[k++] = log(S + 1.0e-3f); f[k++] = vs;
        if (par != 0) f[k++] = seen;
    }
    if (par != 0) {
        float pt = clamp(tau, 0.0f, 1.0f) * 3.14159265f;
        for (int o = 0; o < icfg[14]; o++, pt *= 2.0f) { f[k++] = sin(pt); f[k++] = cos(pt); }
        for (int i = 0; i < icfg[15]; i++) f[k++] = gvec[i];
    }
    return k;
}

// icfg: [0] sky id, [1] lights, [2] grid levels, [3] features/level, [4] embedding width, [5] single origin,
//       [6] direction octaves, [7] lightmap levels, [8] parametric, [9] triangle rows per frame, [10] BVH nodes
//       per frame, [11] material rows per state, [12] light groups, [13] lightmap features/level,
//       [14] time octaves, [15] global-state length, [16] lamp rows per frame, [17] environment map width (0: none),
//       [18] the world's light group, [19] rays in a rough metal's reflection lobe (0: the four-ray estimate),
//       [20] 1: traced rays are followed once more off metal and through glass (see nr_seen)
// vs / vt, one row per view: (frame slot, material state) and (time in [0,1], global material state...). A batch
// may mix samples from different frames and material settings; each reads its own slice of the stacked tables.
#ifdef NR_PLAIN
#define NR_STATE \
    uint oi = icfg[5] != 0 ? 0u : uint(oidx[idx]); \
    uint fs = 0u; \
    device const float* Tf = T; \
    device const float* Mf = M; \
    device const float* gv = vt; \
    float tau = 0.0f; \
    float3 d = ld3(dir, 3u * idx); \
    Surf s = nr_surface(tri[idx], ld3(org, 3u * oi), d, Tf, C, icfg[0]);
#else
#define NR_STATE \
    uint oi = icfg[5] != 0 ? 0u : uint(oidx[idx]); \
    uint fs = uint(vs[2u * oi]), ms = uint(vs[2u * oi + 1u]); \
    device const float* Tf = T + fs * uint(icfg[9]) * TRI_STRIDE; \
    device const float* Mf = M + ms * uint(icfg[11]) * MAT_STRIDE; \
    device const float* gv = vt + oi * uint(1 + icfg[15]); \
    float tau = gv[0]; \
    float3 d = ld3(dir, 3u * idx); \
    Surf s = nr_surface(tri[idx], ld3(org, 3u * oi), d, Tf, C, icfg[0]); \
    if (icfg[8] != 0) nr_canon(s, icfg[0], Tref, Mf);
#endif

#define NR_SETUP \
    NR_STATE \
    device const float* Lf = Lg + fs * uint(icfg[16]) * LIGHT_STRIDE; \
    device const float* Bf = bvh + fs * uint(icfg[10]) * 8u; \
    device const int* Of = order + fs * uint(icfg[9] - 1);

// Grid coordinates (and log albedo) only: fitting the lightmap.
kernel void nr_pos(device float* pos, device float* alb, device const int* tri, device const float* dir,
                   device const float* org, device const int* oidx, device const int* vs, device const float* vt,
                   device const float* T, device const float* Tref, device const float* C, device const float* M,
                   device const float* cfg, device const int* icfg, uint idx [[thread_position_in_grid]]) {
    NR_STATE
    float3 p = nr_grid_pos(s.xc, cfg);
    pos[4u * idx] = p.x; pos[4u * idx + 1u] = p.y; pos[4u * idx + 2u] = p.z; pos[4u * idx + 3u] = tau;
    float3 la = log(s.albedo + 0.02f);
    alb[3u * idx] = la.x; alb[3u * idx + 1u] = la.y; alb[3u * idx + 2u] = la.z;
}

// Fitting: dense features + grid coordinates (the grid lookup is a separate, differentiable kernel pair).
kernel void nr_geo(device float* feat, device float* pos, device const int* tri, device const float* dir,
                   device const float* org, device const int* oidx, device const int* vs, device const float* vt,
                   device const float* T, device const float* Tref, device const float* C, device const float* M,
                   device const float* Lg, device const float* bvh, device const int* order, device const float* lt,
                   device const int* llv, device const float* ldec, device const float* env, device const float* cfg,
                   device const int* icfg, uint idx [[thread_position_in_grid]]) {
    NR_SETUP
    float f[MAX_FEAT];
    int n = nr_dense(s, d, tau, gv + 1, Tf, Tref, C, Mf, Lf, Bf, Of, lt, llv, ldec, env, cfg, icfg, f);
    for (int i = 0; i < n; i++) feat[idx * uint(n) + uint(i)] = f[i];
    float3 p = nr_grid_pos(s.xc, cfg);
    pos[4u * idx] = p.x; pos[4u * idx + 1u] = p.y; pos[4u * idx + 2u] = p.z; pos[4u * idx + 3u] = tau;
}

kernel void nr_hash_fwd(device float* out, device const float* pos, device const float* table, device const int* lv,
                        constant int& L, constant int& F, uint idx [[thread_position_in_grid]]) {
    float f[MAX_FEAT];
    nr_hash(ld3(pos, 4u * idx), pos[4u * idx + 3u], table, lv, L, F, f, 0);
    for (int i = 0; i < L * F; i++) out[idx * uint(L * F) + uint(i)] = f[i];
}

static inline void nr_atomic_add(device atomic_uint* p, float v) {
    uint old = atomic_load_explicit(p, memory_order_relaxed);
    for (int it = 0; it < 256; it++) {
        uint want = as_type<uint>(as_type<float>(old) + v);
        if (atomic_compare_exchange_weak_explicit(p, &old, want, memory_order_relaxed, memory_order_relaxed)) break;
    }
}

kernel void nr_hash_bwd(device atomic_uint* grad, device const float* pos, device const float* gout, device const int* lv,
                        constant int& L, constant int& F, uint idx [[thread_position_in_grid]]) {
    float3 x = ld3(pos, 4u * idx);
    float tau = pos[4u * idx + 3u];
    for (int l = 0; l < L; l++) {
        int res = lv[LV * l]; uint off = uint(lv[LV * l + 1]); uint size = uint(lv[LV * l + 2]); int dense = lv[LV * l + 3]; int rt = NR_RT(lv, l);
        float3 p = x * float(res);
        float3 p0 = floor(p); float3 w = p - p0; uint3 c0 = uint3(int3(p0));
        float tt = clamp(tau, 0.0f, 1.0f) * float(rt);
        float t0 = min(floor(tt), float(max(rt - 1, 0)));
        float wt = tt - t0;
        for (uint ts = 0u; ts < (rt > 0 ? 2u : 1u); ts++) {
            float ws = rt > 0 ? (ts != 0u ? wt : 1.0f - wt) : 1.0f;
            for (uint j = 0u; j < 8u; j++) {
                uint3 dd = uint3(j & 1u, (j >> 1) & 1u, (j >> 2) & 1u);
                float wk = (dd.x != 0u ? w.x : 1.0f - w.x) * (dd.y != 0u ? w.y : 1.0f - w.y) * (dd.z != 0u ? w.z : 1.0f - w.z) * ws;
                uint e = (off + nr_cell(c0 + dd, uint(t0) + ts, uint(res + 1), size, dense)) * uint(F);
                for (int c = 0; c < F; c++) nr_atomic_add(&grad[e + uint(c)], wk * gout[idx * uint(L * F) + uint(l * F + c)]);
            }
        }
    }
}

// Rendering: the whole network input for one shading sample, in half precision, in one pass.
kernel void nr_encode(device half* out, device const int* tri, device const float* dir, device const float* org,
                      device const int* oidx, device const int* vs, device const float* vt, device const float* T,
                      device const float* Tref, device const float* C, device const float* M, device const float* Lg,
                      device const float* bvh, device const int* order, device const float* lt, device const int* llv,
                      device const float* ldec, device const float* env, device const float* table, device const float* emb,
                      device const int* lv, device const float* cfg, device const int* icfg, uint idx [[thread_position_in_grid]]) {
    NR_SETUP
    float f[MAX_FEAT];
    int L = icfg[2], F = icfg[3], FT = icfg[4];
    int n = nr_dense(s, d, tau, gv + 1, Tf, Tref, C, Mf, Lf, Bf, Of, lt, llv, ldec, env, cfg, icfg, f);
    nr_hash(nr_grid_pos(s.xc, cfg), tau, table, lv, L, F, f, n);
    n += L * F;
    for (int c = 0; c < FT; c++) f[n + c] = emb[uint(tri[idx]) * uint(FT) + uint(c)];
    n += FT;
    for (int i = 0; i < n; i++) out[idx * uint(n) + uint(i)] = half(f[i]);
}

// ---- per-pixel resolve: pixel filter over shading groups, then Blender's view transform (baked LUT) ----

// One pixel in scene-linear radiance: filter-weighted shading groups, each the sum of its light groups scaled by
// that group's current colour and strength (rcfg[10] groups, three weights each from rcfg[11]).
static inline float3 nr_pixel(uint p, device const half* rad, device const int* px_ref, device const int* e_group,
                              device const float* e_w, device const float* px_miss, device const float* rcfg) {
    int ref = px_ref[p];
    int n = ref & 31; uint e0 = uint(ref >> 5);
    int G = int(rcfg[10]);
    float3 acc = float3(rcfg[2], rcfg[3], rcfg[4]) * px_miss[p];
    for (int k = 0; k < n; k++) {
        uint g = uint(e_group[e0 + uint(k)]) * uint(3 * G);
        float3 sum = float3(0.0f);
        for (int q = 0; q < G; q++) {
            uint b = g + uint(3 * q);
            float3 y = float3(float(rad[b]), float(rad[b + 1u]), float(rad[b + 2u]));
            sum += float3(rcfg[11 + 3 * q], rcfg[12 + 3 * q], rcfg[13 + 3 * q]) * max(exp(y) - rcfg[0], 0.0f);
        }
        acc += e_w[e0 + uint(k)] * sum * rcfg[1];
    }
    return acc;
}

static inline float3 nr_display(float3 lin, device const half* lut, device const float* rcfg) {
    int N = int(rcfg[5]); float vmax = rcfg[6], a = rcfg[7];
    float3 u = log(clamp(lin, 0.0f, vmax) / a + 1.0f) / log(vmax / a + 1.0f) * float(N - 1);
    float3 fl = clamp(floor(u), 0.0f, float(N - 2));
    float3 fr = u - fl; int3 i0 = int3(fl);
    float3 acc = float3(0.0f);
    for (int k = 0; k < 8; k++) {
        int3 dd = int3(k & 1, (k >> 1) & 1, (k >> 2) & 1);
        float w = (dd.x != 0 ? fr.x : 1.0f - fr.x) * (dd.y != 0 ? fr.y : 1.0f - fr.y) * (dd.z != 0 ? fr.z : 1.0f - fr.z);
        uint e = uint(((i0.x + dd.x) * N + (i0.y + dd.y)) * N + (i0.z + dd.z)) * 3u;
        acc += w * float3(float(lut[e]), float(lut[e + 1u]), float(lut[e + 2u]));
    }
    return acc;
}

// rcfg: [0] eps, [1] 1/radiance scale, [2..4] sky rgb, [5] lut size, [6] lut vmax, [7] lut a, [8] dither, [9] linear output,
//       [10] light groups, [11..] weight rgb per group
kernel void nr_resolve_u8(device uchar* out, device const half* rad, device const int* px_ref, device const int* e_group,
                          device const float* e_w, device const float* px_miss, device const half* lut,
                          device const float* rcfg, uint idx [[thread_position_in_grid]]) {
    float3 d = nr_display(nr_pixel(idx, rad, px_ref, e_group, e_w, px_miss, rcfg), lut, rcfg);
    uint h = idx * 747796405u + 2891336453u;
    h = ((h >> ((h >> 28u) + 4u)) ^ h) * 277803737u;
    h = (h >> 22u) ^ h;
    float noise = (float(h & 0xFFFFu) / 65536.0f - 0.5f) * 0.84f * rcfg[8];
    float3 q = clamp(d * 255.0f + 0.5f + noise, 0.0f, 255.0f);
    out[3u * idx] = uchar(q.x); out[3u * idx + 1u] = uchar(q.y); out[3u * idx + 2u] = uchar(q.z);
}

kernel void nr_resolve_f(device float* out, device const half* rad, device const int* px_ref, device const int* e_group,
                         device const float* e_w, device const float* px_miss, device const half* lut,
                         device const float* rcfg, uint idx [[thread_position_in_grid]]) {
    float3 lin = nr_pixel(idx, rad, px_ref, e_group, e_w, px_miss, rcfg);
    float3 d = rcfg[9] > 0.5f ? lin : nr_display(lin, lut, rcfg);
    out[3u * idx] = d.x; out[3u * idx + 1u] = d.y; out[3u * idx + 2u] = d.z;
}

// Bicubic (Catmull-Rom) upsampling of a display-referred frame rendered below output resolution.
static inline float nr_cubic(float x) {
    x = fabs(x);
    if (x < 1.0f) return (1.5f * x - 2.5f) * x * x + 1.0f;
    if (x < 2.0f) return ((-0.5f * x + 2.5f) * x - 4.0f) * x + 2.0f;
    return 0.0f;
}

// dims: source width, height, output width, height
kernel void nr_upsample_u8(device uchar* out, device const float* src, device const int* dims, device const float* rcfg,
                           uint idx [[thread_position_in_grid]]) {
    int sw = dims[0], sh = dims[1], dw = dims[2], dh = dims[3];
    float fx = (float(int(idx) % dw) + 0.5f) * float(sw) / float(dw) - 0.5f;
    float fy = (float(int(idx) / dw) + 0.5f) * float(sh) / float(dh) - 0.5f;
    int ix = int(floor(fx)), iy = int(floor(fy));
    float3 acc = float3(0.0f);
    float ws = 0.0f;
    for (int j = -1; j <= 2; j++) {
        float wy = nr_cubic(fy - float(iy + j));
        int yy = clamp(iy + j, 0, sh - 1);
        for (int i = -1; i <= 2; i++) {
            float w = wy * nr_cubic(fx - float(ix + i));
            uint b = uint(yy * sw + clamp(ix + i, 0, sw - 1)) * 3u;
            acc += w * float3(src[b], src[b + 1u], src[b + 2u]);
            ws += w;
        }
    }
    uint h = idx * 747796405u + 2891336453u;
    h = ((h >> ((h >> 28u) + 4u)) ^ h) * 277803737u;
    h = (h >> 22u) ^ h;
    float noise = (float(h & 0xFFFFu) / 65536.0f - 0.5f) * 0.84f * rcfg[8];
    float3 q = clamp(acc / ws * 255.0f + 0.5f + noise, 0.0f, 255.0f);
    out[3u * idx] = uchar(q.x); out[3u * idx + 1u] = uchar(q.y); out[3u * idx + 2u] = uchar(q.z);
}

// The same view transform, differentiable: display colour of y = log(radiance * scale + eps) and its
// Jacobian d(display)/dy, so the fit can minimise error where it is seen -- on the display.
kernel void nr_view(device float* disp, device float* jac, device const float* y, device const half* lut,
                    device const float* rcfg, uint idx [[thread_position_in_grid]]) {
    float3 ex = exp(ld3(y, 3u * idx));
    float3 lin = max(ex - rcfg[0], 0.0f) * rcfg[1];
    int N = int(rcfg[5]); float vmax = rcfg[6], a = rcfg[7];
    float kk = float(N - 1) / log(vmax / a + 1.0f);
    float3 lc = min(lin, vmax);
    float3 u = log(lc / a + 1.0f) * kk;
    float3 du = kk / (lc + a) * ex * rcfg[1] * float3(ex > rcfg[0]) * float3(lin < vmax);
    float3 fl = clamp(floor(u), 0.0f, float(N - 2));
    float3 fr = u - fl; int3 i0 = int3(fl);
    float3 acc = float3(0.0f), gx = float3(0.0f), gy = float3(0.0f), gz = float3(0.0f);
    for (int k = 0; k < 8; k++) {
        int3 dd = int3(k & 1, (k >> 1) & 1, (k >> 2) & 1);
        uint e = uint(((i0.x + dd.x) * N + (i0.y + dd.y)) * N + (i0.z + dd.z)) * 3u;
        float3 c = float3(float(lut[e]), float(lut[e + 1u]), float(lut[e + 2u]));
        float wx = dd.x != 0 ? fr.x : 1.0f - fr.x, wy = dd.y != 0 ? fr.y : 1.0f - fr.y, wz = dd.z != 0 ? fr.z : 1.0f - fr.z;
        float sx = dd.x != 0 ? 1.0f : -1.0f, sy = dd.y != 0 ? 1.0f : -1.0f, sz = dd.z != 0 ? 1.0f : -1.0f;
        acc += wx * wy * wz * c; gx += sx * wy * wz * c; gy += wx * sy * wz * c; gz += wx * wy * sz * c;
    }
    gx *= du.x; gy *= du.y; gz *= du.z;
    uint b = 3u * idx, j = 9u * idx;
    disp[b] = acc.x; disp[b + 1u] = acc.y; disp[b + 2u] = acc.z;
    jac[j] = gx.x; jac[j + 1u] = gx.y; jac[j + 2u] = gx.z;
    jac[j + 3u] = gy.x; jac[j + 4u] = gy.y; jac[j + 5u] = gy.z;
    jac[j + 6u] = gz.x; jac[j + 7u] = gz.y; jac[j + 8u] = gz.z;
}
"""


@functools.lru_cache(maxsize=2)
def library(plain=False):
    """The compiled kernels. plain: the build for a static fit with none of the extras (see NR_PLAIN)."""
    return torch.mps.compile_shader(('#define NR_PLAIN 1\n' if plain else '') + SOURCE)
