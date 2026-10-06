// Neuron Render, Unity runtime: the shading kernel ported from neuron_render/kernels.py (Metal) to HLSL.
// Same tables, same feature layout, same arithmetic, so a pack exported from a fit renders here what it
// renders there. Comments describing *why* live in kernels.py; this file follows it line by line.
#ifndef NEURON_COMMON
#define NEURON_COMMON

#define TRI_STRIDE 24u
#define MAT_STRIDE 12u
#define MAT_FEATS 8u
#define LV 5
#define MAX_G 8
#define LIGHT_STRIDE 20u
#define MAX_FEAT 320
#define P1 2654435761u
#define P2 805459861u
#define P3 3674653429u

StructuredBuffer<float> _Tri;        // this frame's triangle table [T + 1, 24]
StructuredBuffer<float> _TriRef;     // the reference pose
StructuredBuffer<float> _Colors;     // [T + 1, 9]
StructuredBuffer<float> _Materials;  // [M + 1, 12], current state
StructuredBuffer<float> _Lights;     // [NL rows, 20], this frame
StructuredBuffer<float> _Bvh;        // [N, 8]
StructuredBuffer<int>   _BvhOrder;
StructuredBuffer<float> _Ldec;
StructuredBuffer<float> _Env;        // equirectangular log-radiance, or 3 zeros
StructuredBuffer<float> _Cfg;
StructuredBuffer<int>   _Icfg;
StructuredBuffer<int>   _Lv;
StructuredBuffer<int>   _Llv;
StructuredBuffer<uint>  _Table;      // halves, two per uint
StructuredBuffer<uint>  _LTable;
StructuredBuffer<uint>  _Emb;
StructuredBuffer<float> _Vt;         // time in [0,1], then the global material state

float H16(StructuredBuffer<uint> b, uint i) { uint w = b[i >> 1]; return f16tof32((i & 1u) != 0u ? (w >> 16) : (w & 0xffffu)); }
float3 ld3(StructuredBuffer<float> b, uint i) { return float3(b[i], b[i + 1u], b[i + 2u]); }

struct Surf { float3 x; float3 xc; float3 bary; float3 n; float3 v; float3 r; float3 albedo; float cosv; float back; uint mat; int tri; };
struct Hit { float t; int tri; float u; float v; };

Surf nr_surface(int tri, float3 o, float3 d, int sky) {
    Surf s;
    uint b = uint(tri) * TRI_STRIDE;
    s.mat = uint(_Tri[b + 22u]);
    s.v = -d;
    s.tri = tri;
    if (tri == sky) {
        s.x = o + d * 1.0e6; s.n = -d; s.r = d; s.albedo = 0.0; s.cosv = 1.0; s.back = 0.0;
        s.xc = s.x; s.bary = float3(1.0, 0.0, 0.0);
        return s;
    }
    float3 v0 = ld3(_Tri, b), v1 = ld3(_Tri, b + 3u), v2 = ld3(_Tri, b + 6u);
    float3 ng = ld3(_Tri, b + 18u);
    float den = dot(ng, d);
    den = abs(den) > 1.0e-12 ? den : 1.0e-12;
    s.x = o + d * ((_Tri[b + 21u] - dot(ng, o)) / den);
    float3 e1 = v1 - v0, e2 = v2 - v0, ep = s.x - v0;
    float d11 = dot(e1, e1), d12 = dot(e1, e2), d22 = dot(e2, e2), dp1 = dot(ep, e1), dp2 = dot(ep, e2);
    float inv = 1.0 / max(d11 * d22 - d12 * d12, 1.0e-30);
    float b1 = clamp((d22 * dp1 - d12 * dp2) * inv, 0.0, 1.0), b2 = clamp((d11 * dp2 - d12 * dp1) * inv, 0.0, 1.0);
    float b0 = clamp(1.0 - b1 - b2, 0.0, 1.0);
    float3 n = ng;
    if (_Tri[b + 23u] < 0.5) n = normalize(b0 * ld3(_Tri, b + 9u) + b1 * ld3(_Tri, b + 12u) + b2 * ld3(_Tri, b + 15u));
    float c = dot(n, s.v);
    s.back = c < 0.0 ? 1.0 : 0.0;
    if (c < 0.0) { n = -n; c = -c; }
    s.n = n; s.cosv = c;
    s.r = d - 2.0 * dot(d, n) * n;
    uint cb = uint(tri) * 9u;
    s.albedo = b0 * ld3(_Colors, cb) + b1 * ld3(_Colors, cb + 3u) + b2 * ld3(_Colors, cb + 6u);
    s.xc = s.x; s.bary = float3(b0, b1, b2);
    return s;
}

void nr_canon(inout Surf s, int sky) {
    if (s.tri == sky) return;
    uint b = uint(s.tri) * 9u;
    s.xc = s.bary.x * ld3(_TriRef, b) + s.bary.y * ld3(_TriRef, b + 3u) + s.bary.z * ld3(_TriRef, b + 6u);
    s.albedo *= ld3(_Materials, s.mat * MAT_STRIDE + 8u);
}

int nr_sh3(float3 d, inout float f[MAX_FEAT], int k) {
    float x = d.x, y = d.y, z = d.z, xx = x * x, yy = y * y, zz = z * z;
    f[k++] = 0.488603 * y; f[k++] = 0.488603 * z; f[k++] = 0.488603 * x;
    f[k++] = 1.092548 * x * y; f[k++] = 1.092548 * y * z; f[k++] = 0.315392 * (3.0 * zz - 1.0);
    f[k++] = 1.092548 * x * z; f[k++] = 0.546274 * (xx - yy);
    f[k++] = 0.590044 * y * (3.0 * xx - yy); f[k++] = 2.890611 * x * y * z; f[k++] = 0.457046 * y * (5.0 * zz - 1.0);
    f[k++] = 0.373176 * z * (5.0 * zz - 3.0); f[k++] = 0.457046 * x * (5.0 * zz - 1.0);
    f[k++] = 1.445306 * z * (xx - yy); f[k++] = 0.590044 * x * (xx - 3.0 * yy);
    return k;
}

float nr_soft(float ang, float width) { return 1.0 / (1.0 + exp(clamp(ang / width, -30.0, 30.0))); }

float nr_light_E(float3 x, float3 n, uint lb) {
    int type = int(_Lights[lb]);
    float3 c = ld3(_Lights, lb + 1u), nl = ld3(_Lights, lb + 10u);
    if (type == 0) {
        float3 ux = ld3(_Lights, lb + 4u), uy = ld3(_Lights, lb + 7u);
        float hx = _Lights[lb + 13u], hy = _Lights[lb + 14u], E = 0.0;
        for (int q = 0; q < 4; q++) {
            float3 p = c + ux * (hx * ((q & 1) ? 0.57735 : -0.57735)) + uy * (hy * ((q & 2) ? 0.57735 : -0.57735));
            float3 L = p - x;
            float d2 = max(dot(L, L), 1.0e-8);
            float3 wl = L * rsqrt(d2);
            E += max(dot(n, wl), 0.0) * max(-dot(wl, nl), 0.0) / d2;
        }
        return E * hx * hy;
    }
    if (type == 3) return max(-dot(n, nl), 0.0);
    float3 L = c - x;
    float d2 = max(dot(L, L), 1.0e-8);
    float3 wl = L * rsqrt(d2);
    float spot = 1.0;
    if (type == 2) {
        float cc = _Lights[lb + 17u];
        spot = smoothstep(cc, cc + (1.0 - cc) * max(_Lights[lb + 18u], 1.0e-3), -dot(wl, nl));
    }
    return max(dot(n, wl), 0.0) * spot / d2;
}

float nr_light_ang(float3 o, float3 dir, uint lb) {
    int type = int(_Lights[lb]);
    float3 c = ld3(_Lights, lb + 1u), nl = ld3(_Lights, lb + 10u);
    float rad = _Lights[lb + 15u];
    if (type == 0) {
        float den = dot(dir, nl);
        if (den > -1.0e-4) return 1.0e3;
        float t = dot(c - o, nl) / den;
        if (t <= 0.0) return 1.0e3;
        float3 q = o + t * dir - c;
        float hx = _Lights[lb + 13u], hy = _Lights[lb + 14u];
        float a = dot(q, ld3(_Lights, lb + 4u)), b = dot(q, ld3(_Lights, lb + 7u));
        float dd = max(abs(a) - hx, abs(b) - hy);
        if (_Lights[lb + 16u] > 0.5) dd = (length(float2(a / hx, b / hy)) - 1.0) * min(hx, hy);
        return dd * (-den) / t;
    }
    if (type == 3) return acos(clamp(-dot(dir, nl), -1.0, 1.0)) - rad;
    float3 L = c - o;
    float d2 = max(dot(L, L), 1.0e-8);
    float3 wl = L * rsqrt(d2);
    if (type == 2 && -dot(wl, nl) < _Lights[lb + 17u]) return 1.0e3;
    return acos(clamp(dot(dir, wl), -1.0, 1.0)) - atan(rad * rsqrt(d2));
}

Hit nr_trace(float3 o, float3 d) {
    Hit h; h.t = 1.0e30; h.tri = -1; h.u = 0.0; h.v = 0.0;
    float3 inv = 1.0 / float3(abs(d.x) > 1.0e-9 ? d.x : 1.0e-9, abs(d.y) > 1.0e-9 ? d.y : 1.0e-9, abs(d.z) > 1.0e-9 ? d.z : 1.0e-9);
    uint stack[48]; int sp = 0; stack[sp++] = 0u;
    for (int guard = 0; sp > 0 && guard < 4096; guard++) {
        uint b = stack[--sp] * 8u;
        float3 t0 = (ld3(_Bvh, b) - o) * inv, t1 = (ld3(_Bvh, b + 4u) - o) * inv;
        float3 tn = min(t0, t1), tf = max(t0, t1);
        float tnear = max(max(tn.x, tn.y), max(tn.z, 0.0)), tfar = min(min(tf.x, tf.y), min(tf.z, h.t));
        if (tnear > tfar) continue;
        int a = int(_Bvh[b + 3u]), c = int(_Bvh[b + 7u]);
        if (c > 0) {
            for (int i = 0; i < c; i++) {
                int tri = _BvhOrder[a + i];
                uint tb = uint(tri) * TRI_STRIDE;
                float3 v0 = ld3(_Tri, tb), e1 = ld3(_Tri, tb + 3u) - v0, e2 = ld3(_Tri, tb + 6u) - v0;
                float3 p = cross(d, e2);
                float det = dot(e1, p);
                if (abs(det) < 1.0e-14) continue;
                float idet = 1.0 / det;
                float3 tv = o - v0;
                float u = dot(tv, p) * idet;
                if (u < 0.0 || u > 1.0) continue;
                float3 q = cross(tv, e1);
                float v = dot(d, q) * idet;
                if (v < 0.0 || u + v > 1.0) continue;
                float t = dot(e2, q) * idet;
                if (t > 1.0e-5 && t < h.t) { h.t = t; h.tri = tri; h.u = u; h.v = v; }
            }
        } else if (sp < 46) { stack[sp++] = uint(-c); stack[sp++] = uint(a); }
    }
    return h;
}

bool nr_refract(float3 d, float3 n, float eta, out float3 o) {
    float c = -dot(d, n), k = 1.0 - eta * eta * (1.0 - c * c);
    o = d;
    if (k < 0.0) return false;
    o = normalize(eta * d + (eta * c - sqrt(k)) * n);
    return true;
}

float3 nr_hit_normal(Hit h, float3 dir) {
    uint b = uint(h.tri) * TRI_STRIDE;
    float3 n = ld3(_Tri, b + 18u);
    if (_Tri[b + 23u] < 0.5) n = normalize((1.0 - h.u - h.v) * ld3(_Tri, b + 9u) + h.u * ld3(_Tri, b + 12u) + h.v * ld3(_Tri, b + 15u));
    return dot(n, dir) > 0.0 ? -n : n;
}

float3 nr_grid_pos(float3 x) {
    float3 q = (x - float3(_Cfg[0], _Cfg[1], _Cfg[2])) * _Cfg[3];
    float m = length(q);
    if (m > 1.0) q *= (2.0 - 1.0 / m) / m;
    return clamp(q * 0.25 + 0.5, 0.0, 0.999999);
}

uint nr_cell(uint3 c, uint t, uint res1, uint size, int dense) {
    if (dense != 0) return c.x + res1 * (c.y + res1 * (c.z + res1 * t));
    return (c.x ^ (c.y * P1) ^ (c.z * P2) ^ (t * P3)) & (size - 1u);
}

// which: 0 = the network's grid (_Table / _Lv), 1 = the lightmap (_LTable / _Llv)
float nr_fetch(int which, uint e) { return which == 0 ? H16(_Table, e) : H16(_LTable, e); }
int nr_lvrow(int which, int i) { return which == 0 ? _Lv[i] : _Llv[i]; }

#define NR_HASH_BODY  \
    for (int l = 0; l < L; l++) { \
        int res = nr_lvrow(which, LV * l); uint off = uint(nr_lvrow(which, LV * l + 1)); uint size = uint(nr_lvrow(which, LV * l + 2)); \
        int dense = nr_lvrow(which, LV * l + 3); int rt = nr_lvrow(which, LV * l + 4); \
        float3 p = x * float(res); \
        float3 p0 = floor(p); float3 w = p - p0; uint3 c0 = uint3(int3(p0)); \
        float tt = clamp(tau, 0.0, 1.0) * float(rt); \
        float t0 = min(floor(tt), float(max(rt - 1, 0))); \
        float wt = tt - t0; \
        for (int c = 0; c < F; c++) f[k + l * F + c] = 0.0; \
        uint nts = rt > 0 ? 2u : 1u; \
        for (uint ts = 0u; ts < nts; ts++) { \
            float ws = rt > 0 ? (ts != 0u ? wt : 1.0 - wt) : 1.0; \
            for (uint j = 0u; j < 8u; j++) { \
                uint3 dd = uint3(j & 1u, (j >> 1) & 1u, (j >> 2) & 1u); \
                float wk = (dd.x != 0u ? w.x : 1.0 - w.x) * (dd.y != 0u ? w.y : 1.0 - w.y) * (dd.z != 0u ? w.z : 1.0 - w.z) * ws; \
                uint e = (off + nr_cell(c0 + dd, uint(t0) + ts, uint(res + 1), size, dense)) * uint(F); \
                for (int c = 0; c < F; c++) f[k + l * F + c] += wk * nr_fetch(which, e + uint(c)); \
            } \
        } \
    }

void nr_hash(int which, float3 x, float tau, int L, int F, inout float f[MAX_FEAT], int k) {
 NR_HASH_BODY 
}
#define LM_FEATS 96
void nr_hash_lm(int which, float3 x, float tau, int L, int F, inout float f[LM_FEATS], int k) {
 NR_HASH_BODY 
}

// static fits: three channels of log-radiance summed over the levels
float3 nr_lightmap(float3 x, int LL) {
    float3 p = nr_grid_pos(x);
    float3 acc = 0.0;
    for (int l = 0; l < LL; l++) {
        int res = _Llv[LV * l]; uint off = uint(_Llv[LV * l + 1]); uint size = uint(_Llv[LV * l + 2]); int dense = _Llv[LV * l + 3];
        float3 q = p * float(res);
        float3 q0 = floor(q); float3 w = q - q0; uint3 c0 = uint3(int3(q0));
        for (uint j = 0u; j < 8u; j++) {
            uint3 dd = uint3(j & 1u, (j >> 1) & 1u, (j >> 2) & 1u);
            float wk = (dd.x != 0u ? w.x : 1.0 - w.x) * (dd.y != 0u ? w.y : 1.0 - w.y) * (dd.z != 0u ? w.z : 1.0 - w.z);
            uint e = (off + nr_cell(c0 + dd, 0u, uint(res + 1), size, dense)) * 3u;
            acc += wk * float3(H16(_LTable, e), H16(_LTable, e + 1u), H16(_LTable, e + 2u));
        }
    }
    return acc;
}

void nr_lightmap_p(float3 xc, float tau, float3 logalb, int LL, int FL, int G, inout float o[3 * MAX_G]) {
    float h[LM_FEATS];
    for (int z = 0; z < LM_FEATS; z++) h[z] = 0.0;
    nr_hash_lm(1, nr_grid_pos(xc), tau, LL, FL, h, 0);
    int n = LL * FL;
    for (int j = 0; j < 3 * G; j++) {
        uint row = uint(j) * uint(n + 1);
        float acc = _Ldec[row + uint(n)];
        for (int i = 0; i < n; i++) acc += _Ldec[row + uint(i)] * h[i];
        o[j] = acc + logalb[j % 3];
    }
}

void nr_lm(float3 xc, float3 alb, float tau, inout float o[3 * MAX_G]) {
    if (_Icfg[8] == 0) {
        float3 m = nr_lightmap(xc, _Icfg[7]);
        o[0] = m.x; o[1] = m.y; o[2] = m.z;
        return;
    }
    nr_lightmap_p(xc, tau, log(alb + 0.02), _Icfg[7], _Icfg[13], _Icfg[12], o);
}

float3 nr_env(float3 d, int W) {
    int H = W / 2;
    float u = (atan2(d.y, d.x) * 0.15915494 + 0.5) * float(W) - 0.5;
    float v = acos(clamp(d.z, -1.0, 1.0)) * 0.31830989 * float(H) - 0.5;
    float u0 = floor(u), v0 = floor(v);
    float fu = u - u0, fv = v - v0;
    int x0 = clamp((int(u0) + W) % W, 0, W - 1), x1 = (x0 + 1) % W;
    int y0 = clamp(int(v0), 0, H - 1), y1 = clamp(int(v0) + 1, 0, H - 1);
    float3 top = lerp(ld3(_Env, uint(y0 * W + x0) * 3u), ld3(_Env, uint(y0 * W + x1) * 3u), fu);
    float3 bot = lerp(ld3(_Env, uint(y1 * W + x0) * 3u), ld3(_Env, uint(y1 * W + x1) * 3u), fu);
    return lerp(top, bot, fv);
}

void nr_lm_sky(float3 x, float3 dir, float tau, inout float o[3 * MAX_G]) {
    if (_Icfg[17] > 0) {
        float3 e = nr_env(dir, _Icfg[17]);
        if (_Icfg[8] == 0) { o[0] = e.x; o[1] = e.y; o[2] = e.z; return; }
        for (int j = 0; j < 3 * _Icfg[12]; j++) o[j] = _Cfg[8 + j];
        int g = 3 * _Icfg[18];
        o[g] = e.x; o[g + 1] = e.y; o[g + 2] = e.z;
        return;
    }
    if (_Cfg[7] > 0.5) { nr_lm(x + dir * 1.0e6, float3(0.98, 0.98, 0.98), tau, o); return; }
    if (_Icfg[8] == 0) { o[0] = _Cfg[4]; o[1] = _Cfg[5]; o[2] = _Cfg[6]; return; }
    for (int j = 0; j < 3 * _Icfg[12]; j++) o[j] = _Cfg[8 + j];
}

float3 nr_hit_albedo(Hit h, int par) {
    uint cb = uint(h.tri) * 9u;
    float3 a = (1.0 - h.u - h.v) * ld3(_Colors, cb) + h.u * ld3(_Colors, cb + 3u) + h.v * ld3(_Colors, cb + 6u);
    if (par != 0) a *= ld3(_Materials, uint(_Tri[uint(h.tri) * TRI_STRIDE + 22u]) * MAT_STRIDE + 8u);
    return a;
}

float3 nr_hit_canon(Hit h) {
    uint b = uint(h.tri) * 9u;
    return (1.0 - h.u - h.v) * ld3(_TriRef, b) + h.u * ld3(_TriRef, b + 3u) + h.v * ld3(_TriRef, b + 6u);
}

float nr_light_gloss(float3 x, float3 n, float3 v, float cosv, float alpha, uint lb) {
    int type = int(_Lights[lb]);
    float3 c = ld3(_Lights, lb + 1u), nl = ld3(_Lights, lb + 10u);
    float a2 = alpha * alpha, S = 0.0;
    if (type == 0) {
        float3 ux = ld3(_Lights, lb + 4u), uy = ld3(_Lights, lb + 7u);
        float hx = _Lights[lb + 13u], hy = _Lights[lb + 14u];
        for (int j = 0; j < 4; j++) {
            for (int i = 0; i < 4; i++) {
                float3 L = c + ux * (hx * (float(i) - 1.5) * 0.5) + uy * (hy * (float(j) - 1.5) * 0.5) - x;
                float d2 = max(dot(L, L), 1.0e-8);
                float3 l = L * rsqrt(d2);
                float cl = -dot(l, nl);
                if (dot(n, l) <= 0.0 || cl <= 0.0) continue;
                float nh = dot(n, normalize(l + v));
                float den = nh * nh * (a2 - 1.0) + 1.0;
                S += a2 / (3.14159265 * den * den) * cl / d2;
            }
        }
        S *= hx * hy * 0.25;
    } else {
        float3 l = -nl;
        float w = 1.0;
        if (type != 3) {
            float3 L = c - x;
            float d2 = max(dot(L, L), 1.0e-8);
            l = L * rsqrt(d2); w = 1.0 / d2;
            if (type == 2 && -dot(l, nl) < _Lights[lb + 17u]) w = 0.0;
        }
        if (dot(n, l) > 0.0) {
            float nh = dot(n, normalize(l + v));
            float den = nh * nh * (a2 - 1.0) + 1.0;
            S = a2 / (3.14159265 * den * den) * w;
        }
    }
    return S / (4.0 * max(cosv, 0.02));
}

float nr_light_visible(float3 xo, uint lb) {
    int type = int(_Lights[lb]);
    float3 c = ld3(_Lights, lb + 1u), ux = ld3(_Lights, lb + 4u), uy = ld3(_Lights, lb + 7u);
    float hx = type == 0 ? _Lights[lb + 13u] : 0.0, hy = type == 0 ? _Lights[lb + 14u] : 0.0, seen = 0.0;
    int nq = type == 0 ? 4 : 1;
    for (int q = 0; q < nq; q++) {
        float3 L = (type == 3 ? xo - ld3(_Lights, lb + 10u) * 1.0e4 : c + ux * (hx * ((q & 1) ? 0.5 : -0.5)) + uy * (hy * ((q & 2) ? 0.5 : -0.5))) - xo;
        float dist = length(L);
        Hit h = nr_trace(xo, L / dist);
        seen += (h.tri < 0 || h.t > dist) ? 1.0 : 0.0;
    }
    return seen / float(nq);
}

void nr_seen(Hit h, float3 xw, float3 dir, float tau, inout float o[3 * MAX_G]) {
    int par = _Icfg[8] != 0 ? 1 : 0;
    if (_Icfg[20] != 0) {
        uint hm = uint(_Tri[uint(h.tri) * TRI_STRIDE + 22u]) * MAT_STRIDE;
        bool metal = _Materials[hm + 1u] >= 0.5 && _Materials[hm] <= 0.6, glass = _Materials[hm + 2u] > 0.5;
        if (metal || glass) {
            float3 n = nr_hit_normal(h, dir);
            float3 d2 = metal ? dir - 2.0 * dot(dir, n) * n : dir;
            float3 o2 = xw + n * (metal ? 1.0e-4 : -1.0e-4);
            Hit h2 = nr_trace(o2, d2);
            for (int k = 0; k < 3 && h2.tri >= 0 && _Materials[uint(_Tri[uint(h2.tri) * TRI_STRIDE + 22u]) * MAT_STRIDE + 2u] > 0.5; k++) {
                o2 += d2 * (h2.t + 1.0e-4);
                h2 = nr_trace(o2, d2);
            }
            if (h2.tri >= 0) nr_lm(par != 0 ? nr_hit_canon(h2) : o2 + d2 * h2.t, nr_hit_albedo(h2, par), tau, o);
            else nr_lm_sky(o2, d2, tau, o);
            if (metal) {
                float3 tint = log(nr_hit_albedo(h, par) + 0.02);
                for (int j = 0; j < 3 * _Icfg[12]; j++) o[j] += tint[j % 3];
            }
            return;
        }
    }
    nr_lm(par != 0 ? nr_hit_canon(h) : xw, nr_hit_albedo(h, par), tau, o);
}

// The analytic feature vector of one shading sample (see kernels.py nr_dense). Returns its length.
int nr_dense(Surf s, float3 d, float tau, inout float f[MAX_FEAT]) {
    int NL = _Icfg[1], NF = _Icfg[6], LL = _Icfg[7], par = _Icfg[8] != 0 ? 1 : 0, G3 = 3 * _Icfg[12];
    int k = 0, j;
    f[k++] = s.n.x; f[k++] = s.n.y; f[k++] = s.n.z;
    f[k++] = s.v.x; f[k++] = s.v.y; f[k++] = s.v.z;
    f[k++] = s.r.x; f[k++] = s.r.y; f[k++] = s.r.z;
    f[k++] = s.cosv; f[k++] = pow(1.0 - clamp(s.cosv, 0.0, 1.0), 5.0); f[k++] = s.back;
    f[k++] = s.albedo.x; f[k++] = s.albedo.y; f[k++] = s.albedo.z;
    f[k++] = log(s.albedo.x + 0.01); f[k++] = log(s.albedo.y + 0.01); f[k++] = log(s.albedo.z + 0.01);
    float lm[3 * MAX_G], mr[3 * MAX_G], ms[3 * MAX_G];
    for (j = 0; j < 3 * MAX_G; j++) { lm[j] = 0.0; mr[j] = 0.0; ms[j] = 0.0; }
    if (LL > 0) nr_lm(s.xc, s.albedo, tau, lm);
    int klm = k;
    bool have_mr = false;
    for (j = 0; j < G3; j++) f[k++] = lm[j];
    for (uint i = 0u; i < MAT_FEATS; i++) f[k++] = _Materials[s.mat * MAT_STRIDE + i];
    k = nr_sh3(s.v, f, k);
    k = nr_sh3(s.r, f, k);
    float3 ph = s.v * 3.14159265;
    for (int o = 0; o < NF; o++, ph *= 2.0) {
        float3 sn = sin(ph), cs = cos(ph);
        f[k++] = sn.x; f[k++] = sn.y; f[k++] = sn.z; f[k++] = cs.x; f[k++] = cs.y; f[k++] = cs.z;
    }

    float3 xo = s.x + s.n * 1.0e-4, x2 = xo, r2 = s.r;
    float vis1 = 1.0, vis2 = 0.0, near1 = 0.0, E2 = 0.0;
    float npass = 0.0, plen = 0.0, ntir = 0.0, vis_e = 0.0;
    float3 xe = s.x, te = s.r;
    float rough = _Materials[s.mat * MAT_STRIDE];
    bool glossy = rough <= 0.3;
    bool surface = _Materials[s.mat * MAT_STRIDE + 7u] < 0.75;
    if (glossy && surface) {
        Hit h1 = nr_trace(xo, s.r);
        if (h1.tri >= 0) {
            vis1 = 0.0; near1 = 1.0 / (1.0 + h1.t);
            float3 n2 = nr_hit_normal(h1, s.r);
            float3 xh = xo + h1.t * s.r;
            x2 = xh + n2 * 1.0e-4;
            r2 = s.r - 2.0 * dot(s.r, n2) * n2;
            for (int i = 0; i < NL; i++) E2 += nr_light_E(x2, n2, uint(i) * LIGHT_STRIDE);
            if (LL > 0) nr_seen(h1, xh, s.r, tau, mr);
            vis2 = nr_trace(x2, r2).tri < 0 ? 1.0 : 0.0;
        } else if (LL > 0) {
            nr_lm_sky(xo, s.r, tau, mr);
        }
        have_mr = LL > 0;
        float ior = _Materials[s.mat * MAT_STRIDE + 5u] + 1.0;
        if (_Materials[s.mat * MAT_STRIDE + 2u] > 1.0e-3 && s.back < 0.5) {
            float3 td = d, oi = s.x - s.n * 1.0e-4;
            float3 tmp;
            if (nr_refract(d, s.n, 1.0 / ior, tmp)) td = tmp;
            bool inside = true;
            for (int b = 0; b < 8; b++) {
                Hit hi = nr_trace(oi, td);
                if (hi.tri < 0) {
                    if (!inside) { vis_e = 1.0; if (LL > 0) nr_lm_sky(oi, td, tau, ms); }
                    break;
                }
                float3 nb = nr_hit_normal(hi, td);
                float3 xb = oi + hi.t * td;
                if (inside) {
                    plen += hi.t;
                    float3 outd;
                    if (nr_refract(td, nb, ior, outd)) { inside = false; npass += 1.0; td = outd; oi = xb - nb * 1.0e-4; xe = oi; te = td; }
                    else { td = td - 2.0 * dot(td, nb) * nb; oi = xb + nb * 1.0e-4; ntir += 1.0; }
                } else {
                    uint hm = uint(_Tri[uint(hi.tri) * TRI_STRIDE + 22u]);
                    if (_Materials[hm * MAT_STRIDE + 2u] > 0.5 && b < 6) {
                        ior = _Materials[hm * MAT_STRIDE + 5u] + 1.0;
                        float3 ind;
                        if (nr_refract(td, nb, 1.0 / ior, ind)) td = ind;
                        oi = xb - nb * 1.0e-4; inside = true;
                    } else {
                        if (LL > 0) nr_seen(hi, xb, td, tau, ms);
                        break;
                    }
                }
            }
        }
    }
    float got = npass > 0.5 ? 1.0 : 0.0;
    int lobe = _Icfg[19];
    bool metal = _Materials[s.mat * MAT_STRIDE + 1u] >= 0.5 && LL > 0 && surface;
    if (lobe > 0 ? (metal && rough > 0.08) : (metal && !glossy && rough <= 0.6)) {
        float3 ax = lobe > 0 ? s.n : s.r;
        float3 t1 = normalize(cross(ax, abs(ax.z) < 0.9 ? float3(0.0, 0.0, 1.0) : float3(1.0, 0.0, 0.0)));
        float3 t2 = cross(ax, t1);
        float spread = 1.5 * rough * rough, a2 = rough * rough * rough * rough, hits = 0.0, nearsum = 0.0, cnt = 0.0;
        float acc[3 * MAX_G], one[3 * MAX_G];
        for (j = 0; j < 3 * MAX_G; j++) { acc[j] = 0.0; one[j] = 0.0; }
        int nq = lobe > 0 ? lobe : 4;
        for (int q = 0; q < nq; q++) {
            float3 dir;
            if (lobe > 0) {
                float xi = (float(q) + 0.5) / float(nq), phi = 6.2831853 * frac(float(q) * 0.61803399);
                float ct = rsqrt(1.0 + a2 * xi / (1.0 - xi)), st = sqrt(max(1.0 - ct * ct, 0.0));
                float3 hv = st * (cos(phi) * t1 + sin(phi) * t2) + ct * s.n;
                float vh = dot(s.v, hv);
                if (vh <= 0.0) continue;
                dir = 2.0 * vh * hv - s.v;
            } else {
                dir = normalize(s.r + spread * (((q & 1) ? 1.0 : -1.0) * t1 + ((q & 2) ? 1.0 : -1.0) * t2));
            }
            float below = 0.02 - dot(dir, s.n);
            if (below > 0.0) dir = normalize(dir + s.n * below);
            Hit hq = nr_trace(xo, dir);
            if (hq.tri >= 0) { nr_seen(hq, xo + hq.t * dir, dir, tau, one); hits += 1.0; nearsum += 1.0 / (1.0 + hq.t); }
            else nr_lm_sky(xo, dir, tau, one);
            for (j = 0; j < G3; j++) acc[j] += exp(one[j]);
            cnt += 1.0;
        }
        if (cnt > 0.5) {
            for (j = 0; j < G3; j++) mr[j] = log(acc[j] / cnt);
            if (!glossy) { near1 = nearsum / cnt; vis1 = 1.0 - hits / cnt; }
            have_mr = true;
        }
    }
    if (_Icfg[20] != 0 && surface) {
        float3 tint = log(s.albedo + 0.02);
        if (_Materials[s.mat * MAT_STRIDE + 2u] > 0.5 && got > 0.5) { for (j = 0; j < G3; j++) f[klm + j] = ms[j]; }
        else if (_Materials[s.mat * MAT_STRIDE + 1u] >= 0.5 && have_mr) { for (j = 0; j < G3; j++) f[klm + j] = mr[j] + tint[j % 3]; }
    }
    f[k++] = vis1; f[k++] = near1; f[k++] = vis2; f[k++] = log(E2 + 0.01) * (1.0 - vis1);
    for (j = 0; j < G3; j++) f[k++] = mr[j];
    f[k++] = 0.5 * npass; f[k++] = 1.0 / (1.0 + plen); f[k++] = exp(-plen); f[k++] = 0.5 * ntir; f[k++] = vis_e;
    for (j = 0; j < G3; j++) f[k++] = ms[j];
    f[k++] = got * te.x; f[k++] = got * te.y; f[k++] = got * te.z;

    for (int i = 0; i < NL; i++) {
        uint lb = uint(i) * LIGHT_STRIDE;
        f[k++] = log(nr_light_E(s.x, s.n, lb) + 0.01);
        float a1 = nr_light_ang(xo, s.r, lb);
        float m1 = glossy ? vis1 : 0.0;
        f[k++] = m1 * nr_soft(a1, 0.02); f[k++] = m1 * nr_soft(a1, 0.06); f[k++] = m1 * nr_soft(a1, 0.2);
        float a2 = nr_light_ang(x2, r2, lb);
        f[k++] = vis2 * nr_soft(a2, 0.06); f[k++] = vis2 * nr_soft(a2, 0.2);
        float a3 = nr_light_ang(xe, te, lb);
        float ge = got * vis_e;
        f[k++] = ge * nr_soft(a3, 0.03); f[k++] = ge * nr_soft(a3, 0.08); f[k++] = ge * nr_soft(a3, 0.25);
        float S = nr_light_gloss(s.x, s.n, s.v, s.cosv, max(rough * rough, 0.1), lb);
        float vs = glossy ? vis1 : 1.0;
        bool shadowed = par != 0 && surface;
        float seen = shadowed ? nr_light_visible(xo, lb) : 1.0;
        if (!glossy && S > 1.0e-2) vs = shadowed ? seen : nr_light_visible(xo, lb);
        f[k++] = log(S + 1.0e-3); f[k++] = vs;
        if (par != 0) f[k++] = seen;
    }
    if (par != 0) {
        float pt = clamp(tau, 0.0, 1.0) * 3.14159265;
        for (int o = 0; o < _Icfg[14]; o++, pt *= 2.0) { f[k++] = sin(pt); f[k++] = cos(pt); }
        for (int i = 0; i < _Icfg[15]; i++) f[k++] = _Vt[1 + i];
    }
    return k;
}

#endif
