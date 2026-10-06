// Neuron Render: primary visibility.
//
// nr_rasterize   scan-converts every triangle into an id buffer on a (W*S) x (H*S) sample grid.
// nr_resolve     collapses that grid into "shading groups" (one neural evaluation per distinct
//                triangle per pixel block, like MSAA) plus, per output pixel, the list of
//                (group, filter weight) pairs needed to reconstruct it.
//
// Exact, deterministic, no dependencies. Built by neuron_render/native.py.
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    double origin[3], right[3], up[3], fwd[3];
    double fx, fy, cx, cy; // pixel units of the output image; +y is down
    double near_clip;
    int32_t width, height;
} NRCamera;

typedef struct {
    double A[3], B[3], C[3]; // edge functions, inside when all >= 0
    double wa, wb, wc;       // inverse depth as a plane over the sample grid
    int32_t minx, maxx, miny, maxy;
    int32_t id;
} STri;

#define MAX_THREADS 64
#define MAX_CHUNKS 256
#define EDGE_EPS 1e-7

// ---- a tiny work queue: threads pull row bands, so slow (efficiency) cores never stall a frame ----

typedef struct {
    atomic_int next;
    int n;
    void (*fn)(int, void*);
    void* ctx;
} Queue;

static void* queue_worker(void* arg) {
    Queue* q = (Queue*)arg;
    for (;;) {
        int c = atomic_fetch_add(&q->next, 1);
        if (c >= q->n) break;
        q->fn(c, q->ctx);
    }
    return NULL;
}

static void run_chunks(int n, int nthreads, void (*fn)(int, void*), void* ctx) {
    Queue q;
    atomic_init(&q.next, 0);
    q.n = n; q.fn = fn; q.ctx = ctx;
    if (nthreads > MAX_THREADS) nthreads = MAX_THREADS;
    if (nthreads > n) nthreads = n;
    pthread_t th[MAX_THREADS];
    for (int t = 1; t < nthreads; t++) pthread_create(&th[t], NULL, queue_worker, &q);
    queue_worker(&q);
    for (int t = 1; t < nthreads; t++) pthread_join(th[t], NULL);
}

static int chunk_count(int rows, int nthreads) {
    int n = nthreads * 4;
    if (n > MAX_CHUNKS) n = MAX_CHUNKS;
    if (n > rows) n = rows;
    return n < 1 ? 1 : n;
}

// ---- rasterization ----

static int setup_tri(const double p[3][3], int id, int W, int H, STri* t) {
    // p[k] = (sample-space x, sample-space y, inverse depth)
    double x0 = p[0][0], y0 = p[0][1], x1 = p[1][0], y1 = p[1][1], x2 = p[2][0], y2 = p[2][1];
    double area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0);
    if (!(fabs(area) > 1e-12)) return 0;
    double minx = fmin(x0, fmin(x1, x2)), maxx = fmax(x0, fmax(x1, x2));
    double miny = fmin(y0, fmin(y1, y2)), maxy = fmax(y0, fmax(y1, y2));
    if (maxx < 0 || maxy < 0 || minx > W || miny > H) return 0;
    double s = area > 0 ? 1.0 : -1.0;
    t->A[0] = s * (y1 - y2); t->B[0] = s * (x2 - x1); t->C[0] = s * (x1 * y2 - x2 * y1);
    t->A[1] = s * (y2 - y0); t->B[1] = s * (x0 - x2); t->C[1] = s * (x2 * y0 - x0 * y2);
    t->A[2] = s * (y0 - y1); t->B[2] = s * (x1 - x0); t->C[2] = s * (x0 * y1 - x1 * y0);
    double ia = 1.0 / fabs(area);
    t->wa = (t->A[0] * p[0][2] + t->A[1] * p[1][2] + t->A[2] * p[2][2]) * ia;
    t->wb = (t->B[0] * p[0][2] + t->B[1] * p[1][2] + t->B[2] * p[2][2]) * ia;
    t->wc = (t->C[0] * p[0][2] + t->C[1] * p[1][2] + t->C[2] * p[2][2]) * ia;
    double lx = floor(minx - 0.5), hx = ceil(maxx - 0.5), ly = floor(miny - 0.5), hy = ceil(maxy - 0.5);
    t->minx = (int32_t)(lx < 0 ? 0 : lx);
    t->maxx = (int32_t)(hx > W - 1 ? W - 1 : hx);
    t->miny = (int32_t)(ly < 0 ? 0 : ly);
    t->maxy = (int32_t)(hy > H - 1 ? H - 1 : hy);
    t->id = id;
    return t->minx <= t->maxx && t->miny <= t->maxy;
}

typedef struct {
    const STri* tris;
    int ntris, W, H, n_chunks;
    int32_t* ids;
    float* zbuf;
} RasterCtx;

static void raster_chunk(int chunk, void* arg) {
    const RasterCtx* j = (const RasterCtx*)arg;
    const int W = j->W;
    const int y0 = (int)((int64_t)j->H * chunk / j->n_chunks), y1 = (int)((int64_t)j->H * (chunk + 1) / j->n_chunks);
    for (int y = y0; y < y1; y++) {
        int32_t* id = j->ids + (size_t)y * W;
        float* z = j->zbuf + (size_t)y * W;
        for (int x = 0; x < W; x++) { id[x] = -1; z[x] = 0.0f; }
    }
    for (int k = 0; k < j->ntris; k++) {
        const STri* t = &j->tris[k];
        int ys = t->miny > y0 ? t->miny : y0;
        int ye = t->maxy < y1 - 1 ? t->maxy : y1 - 1;
        for (int y = ys; y <= ye; y++) {
            double py = y + 0.5, lo = t->minx, hi = t->maxx;
            int ok = 1;
            for (int e = 0; e < 3; e++) {
                double v = t->B[e] * py + t->C[e], a = t->A[e];
                if (a > 0) { double xb = -v / a - 0.5 - EDGE_EPS; if (xb > lo) lo = xb; }
                else if (a < 0) { double xb = -v / a - 0.5 + EDGE_EPS; if (xb < hi) hi = xb; }
                else if (v < -EDGE_EPS) { ok = 0; break; }
            }
            if (!ok || lo > hi) continue;
            int xs = (int)ceil(lo), xe = (int)floor(hi);
            if (xs < t->minx) xs = t->minx;
            if (xe > t->maxx) xe = t->maxx;
            int32_t* id = j->ids + (size_t)y * W;
            float* z = j->zbuf + (size_t)y * W;
            double w = t->wa * (xs + 0.5) + t->wb * py + t->wc;
            for (int x = xs; x <= xe; x++, w += t->wa) {
                float wf = (float)w;
                if (wf > z[x]) { z[x] = wf; id[x] = t->id; }
            }
        }
    }
}

// tri: T*9 world-space vertex floats. ids/zbuf: (H*S)*(W*S). Returns the number of screen triangles.
int nr_rasterize(const float* tri, int T, const NRCamera* cam, int S, int32_t* ids, float* zbuf, int nthreads) {
    const int W = cam->width * S, H = cam->height * S;
    STri* st = (STri*)malloc(sizeof(STri) * (size_t)(2 * T + 1));
    int n = 0;
    const double nearz = cam->near_clip > 1e-9 ? cam->near_clip : 1e-9;
    for (int i = 0; i < T; i++) {
        double c[3][3]; // camera space: x right, y up, z depth
        for (int k = 0; k < 3; k++) {
            double d[3];
            for (int a = 0; a < 3; a++) d[a] = (double)tri[i * 9 + k * 3 + a] - cam->origin[a];
            c[k][0] = d[0] * cam->right[0] + d[1] * cam->right[1] + d[2] * cam->right[2];
            c[k][1] = d[0] * cam->up[0] + d[1] * cam->up[1] + d[2] * cam->up[2];
            c[k][2] = d[0] * cam->fwd[0] + d[1] * cam->fwd[1] + d[2] * cam->fwd[2];
        }
        double poly[4][3];
        int np = 0;
        for (int k = 0; k < 3; k++) { // Sutherland-Hodgman against z >= near
            const double* a = c[k];
            const double* b = c[(k + 1) % 3];
            int ain = a[2] >= nearz, bin = b[2] >= nearz;
            if (ain) { memcpy(poly[np++], a, sizeof(double) * 3); }
            if (ain != bin) {
                double u = (nearz - a[2]) / (b[2] - a[2]);
                poly[np][0] = a[0] + u * (b[0] - a[0]);
                poly[np][1] = a[1] + u * (b[1] - a[1]);
                poly[np][2] = nearz;
                np++;
            }
        }
        if (np < 3) continue;
        double s[4][3];
        for (int k = 0; k < np; k++) {
            double iz = 1.0 / poly[k][2];
            s[k][0] = (cam->cx + cam->fx * poly[k][0] * iz) * S;
            s[k][1] = (cam->cy - cam->fy * poly[k][1] * iz) * S;
            s[k][2] = iz;
        }
        for (int k = 1; k + 1 < np; k++) {
            double p[3][3];
            memcpy(p[0], s[0], sizeof(double) * 3);
            memcpy(p[1], s[k], sizeof(double) * 3);
            memcpy(p[2], s[k + 1], sizeof(double) * 3);
            if (setup_tri(p, i, W, H, &st[n])) n++;
        }
    }
    RasterCtx ctx = {st, n, W, H, chunk_count(H, nthreads), ids, zbuf};
    run_chunks(ctx.n_chunks, nthreads, raster_chunk, &ctx);
    free(st);
    return n;
}

// ---- resolve ----

#define MAX_BLOCK 1024 // R*R*S*S

typedef struct {
    const int32_t* ids;
    const NRCamera* cam;
    const float* subw;
    int S, R, full, sky_id, n_chunks;
    const uint8_t* cap; // per triangle: 1 = shade once per pixel whatever the block size (textured surfaces)
    int64_t n_groups[MAX_CHUNKS], n_entries[MAX_CHUNKS];
    int32_t* g_tri;
    float* g_dir;
    int32_t* px_ref;
    int32_t* e_group;
    float* e_w;
    float* px_miss;
} ResolveCtx;

static inline int chunk_row(const ResolveCtx* j, int chunk) {
    const int rows = (j->cam->height + j->R - 1) / j->R;
    return (int)((int64_t)rows * chunk / j->n_chunks);
}

// Each chunk writes into its own worst-case slice of the output arrays (base = samples before it);
// nr_resolve compacts the slices afterwards.
static void resolve_chunk(int chunk, void* arg) {
    ResolveCtx* j = (ResolveCtx*)arg;
    const NRCamera* cam = j->cam;
    const int W = cam->width, H = cam->height, S = j->S, R = j->R, SW = W * S, S2 = S * S;
    const double inv_s = 1.0 / S;
    const int by0 = chunk_row(j, chunk), by1 = chunk_row(j, chunk + 1);
    const int64_t base = (int64_t)by0 * R * W * S2;
    int64_t ng = 0, ne = 0;
    int32_t bid[MAX_BLOCK], bpix[MAX_BLOCK];
    double bw[MAX_BLOCK], bx[MAX_BLOCK], by_[MAX_BLOCK];
    for (int by = by0; by < by1; by++) {
        for (int bxi = 0; bxi * R < W; bxi++) {
            int nb = 0, last = 0;
            const int y_end = (by + 1) * R < H ? (by + 1) * R : H, x_end = (bxi + 1) * R < W ? (bxi + 1) * R : W;
            for (int y = by * R; y < y_end; y++) {
                for (int x = bxi * R; x < x_end; x++) {
                    int pg[16], np = 0;
                    float pw[16], miss = 0.0f;
                    const int32_t* px = j->ids + (size_t)(y * S) * SW + (size_t)x * S;
                    int uniform = !j->full;
                    for (int sy = 0; uniform && sy < S; sy++)
                        for (int sx = 0; sx < S; sx++)
                            if (px[(size_t)sy * SW + sx] != px[0]) { uniform = 0; break; }
                    const int32_t pix = y * W + x;
                    if (uniform) { // the common case: one triangle (or nothing) covers the whole pixel
                        int32_t id = px[0];
                        if (id < 0 && j->sky_id >= 0) id = j->sky_id;
                        if (id < 0) miss = 1.0f;
                        else {
                            const int32_t owner = (j->cap && j->cap[id]) ? pix : -1; // -1: shared by the whole block
                            int g = -1;
                            if (nb > 0 && bid[last] == id && bpix[last] == owner) g = last;
                            else for (int k = 0; k < nb; k++) if (bid[k] == id && bpix[k] == owner) { g = k; break; }
                            if (g < 0) { g = nb++; bid[g] = id; bpix[g] = owner; bw[g] = 0; bx[g] = 0; by_[g] = 0; }
                            last = g;
                            bw[g] += 1.0; bx[g] += x + 0.5; by_[g] += y + 0.5;
                            pg[0] = g; pw[0] = 1.0f; np = 1;
                        }
                    } else {
                        for (int sy = 0; sy < S; sy++) {
                            for (int sx = 0; sx < S; sx++) {
                                int32_t id = px[(size_t)sy * SW + sx];
                                const float w = j->subw[sy * S + sx];
                                if (id < 0) {
                                    if (j->sky_id < 0) { miss += w; continue; }
                                    id = j->sky_id;
                                }
                                const int32_t owner = (j->cap && j->cap[id]) ? pix : -1;
                                int g = -1;
                                if (!j->full) {
                                    if (nb > 0 && bid[last] == id && bpix[last] == owner) g = last;
                                    else for (int k = 0; k < nb; k++) if (bid[k] == id && bpix[k] == owner) { g = k; break; }
                                }
                                if (g < 0) { g = nb++; bid[g] = id; bpix[g] = owner; bw[g] = 0; bx[g] = 0; by_[g] = 0; }
                                last = g;
                                bw[g] += w;
                                bx[g] += w * (x + (sx + 0.5) * inv_s);
                                by_[g] += w * (y + (sy + 0.5) * inv_s);
                                int e = -1;
                                for (int k = 0; k < np; k++) if (pg[k] == g) { e = k; break; }
                                if (e < 0) { e = np++; pg[e] = g; pw[e] = 0; }
                                pw[e] += w;
                            }
                        }
                    }
                    const int64_t e0 = base + ne;
                    for (int k = 0; k < np; k++) {
                        j->e_group[e0 + k] = (int32_t)(ng + pg[k]); // chunk-local; rebased when compacting
                        j->e_w[e0 + k] = pw[k];
                    }
                    j->px_ref[(size_t)y * W + x] = (int32_t)((ne << 5) | np);
                    j->px_miss[(size_t)y * W + x] = miss;
                    ne += np;
                }
            }
            for (int g = 0; g < nb; g++) {
                const double px = bx[g] / bw[g], py = by_[g] / bw[g];
                const double xc = (px - cam->cx) / cam->fx, yc = -(py - cam->cy) / cam->fy;
                double d[3];
                for (int a = 0; a < 3; a++) d[a] = cam->right[a] * xc + cam->up[a] * yc + cam->fwd[a];
                const double il = 1.0 / sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
                const int64_t gi = base + ng + g;
                j->g_tri[gi] = bid[g];
                j->g_dir[gi * 3] = (float)(d[0] * il);
                j->g_dir[gi * 3 + 1] = (float)(d[1] * il);
                j->g_dir[gi * 3 + 2] = (float)(d[2] * il);
            }
            ng += nb;
        }
    }
    j->n_groups[chunk] = ng;
    j->n_entries[chunk] = ne;
}

// Output arrays must hold W*H*S*S elements (g_dir three times that); px_ref / px_miss hold W*H.
// On return the first totals[0] groups and totals[1] entries are valid and contiguous.
int nr_resolve(const int32_t* ids, const NRCamera* cam, int S, int R, int full, int sky_id, const float* subw,
               const uint8_t* cap, int nthreads, int32_t* g_tri, float* g_dir, int32_t* px_ref, int32_t* e_group,
               float* e_w, float* px_miss, int64_t* totals) {
    if (S < 1 || S > 4 || R < 1 || R * R * S * S > MAX_BLOCK) return -1;
    ResolveCtx* j = (ResolveCtx*)malloc(sizeof(ResolveCtx));
    j->ids = ids; j->cam = cam; j->subw = subw; j->S = S; j->R = R; j->full = full; j->sky_id = sky_id; j->cap = cap;
    j->g_tri = g_tri; j->g_dir = g_dir; j->px_ref = px_ref; j->e_group = e_group; j->e_w = e_w; j->px_miss = px_miss;
    const int rows = (cam->height + R - 1) / R, W = cam->width, H = cam->height;
    j->n_chunks = chunk_count(rows, nthreads);
    run_chunks(j->n_chunks, nthreads, resolve_chunk, j);
    int64_t G = 0, E = 0;
    for (int c = 0; c < j->n_chunks; c++) {
        const int by0 = chunk_row(j, c), by1 = chunk_row(j, c + 1);
        const int64_t base = (int64_t)by0 * R * W * S * S, ng = j->n_groups[c], ne = j->n_entries[c];
        if (base != G) {
            memmove(g_tri + G, g_tri + base, sizeof(int32_t) * (size_t)ng);
            memmove(g_dir + 3 * G, g_dir + 3 * base, sizeof(float) * 3 * (size_t)ng);
        }
        if (base != E) memmove(e_w + E, e_w + base, sizeof(float) * (size_t)ne);
        for (int64_t k = 0; k < ne; k++) e_group[E + k] = (int32_t)(e_group[base + k] + G);
        const int y0 = by0 * R, y1 = by1 * R < H ? by1 * R : H;
        const int32_t shift = (int32_t)(E << 5);
        for (int64_t p = (int64_t)y0 * W; p < (int64_t)y1 * W; p++) px_ref[p] += shift;
        G += ng; E += ne;
    }
    totals[0] = G; totals[1] = E;
    free(j);
    return 0;
}

// ---- BVH ----
// Median-split BVH over triangles, for the GPU ray tracer. Node = (box min, a, box max, b): a leaf has
// b = triangle count > 0 and a = offset into order; an inner node has a = left child, b = -right child.
// The tree's shape depends only on the triangle count, so every frame of an animation gets the same
// node count and the per-frame trees can be stacked in one buffer.

typedef struct {
    const float* lo;
    const float* hi;
    const float* cen;
    int32_t* order;
    float* nodes;
    int leaf, n;
} BvhCtx;

static void bvh_select(int32_t* idx, int n, int k, const float* cen, int axis) {
    int lo = 0, hi = n - 1;
    while (lo < hi) {
        const float pivot = cen[(size_t)idx[(lo + hi) / 2] * 3 + axis];
        int i = lo, j = hi;
        while (i <= j) {
            while (cen[(size_t)idx[i] * 3 + axis] < pivot) i++;
            while (cen[(size_t)idx[j] * 3 + axis] > pivot) j--;
            if (i <= j) { int32_t t = idx[i]; idx[i] = idx[j]; idx[j] = t; i++; j--; }
        }
        if (k <= j) hi = j;
        else if (k >= i) lo = i;
        else break;
    }
}

static int bvh_build(BvhCtx* c, int a, int b) {
    const int me = c->n++;
    float mn[3] = {1e30f, 1e30f, 1e30f}, mx[3] = {-1e30f, -1e30f, -1e30f}, cmn[3] = {1e30f, 1e30f, 1e30f}, cmx[3] = {-1e30f, -1e30f, -1e30f};
    for (int i = a; i < b; i++) {
        const size_t t = (size_t)c->order[i] * 3;
        for (int k = 0; k < 3; k++) {
            if (c->lo[t + k] < mn[k]) mn[k] = c->lo[t + k];
            if (c->hi[t + k] > mx[k]) mx[k] = c->hi[t + k];
            if (c->cen[t + k] < cmn[k]) cmn[k] = c->cen[t + k];
            if (c->cen[t + k] > cmx[k]) cmx[k] = c->cen[t + k];
        }
    }
    float* nd = c->nodes + (size_t)me * 8;
    for (int k = 0; k < 3; k++) { nd[k] = mn[k] - 1e-5f; nd[4 + k] = mx[k] + 1e-5f; }
    if (b - a <= c->leaf) { nd[3] = (float)a; nd[7] = (float)(b - a); return me; }
    int axis = 0;
    if (cmx[1] - cmn[1] > cmx[axis] - cmn[axis]) axis = 1;
    if (cmx[2] - cmn[2] > cmx[axis] - cmn[axis]) axis = 2;
    const int m = (a + b) / 2;
    bvh_select(c->order + a, b - a, m - a, c->cen, axis);
    const int left = bvh_build(c, a, m);
    const int right = bvh_build(c, m, b);
    nd = c->nodes + (size_t)me * 8;
    nd[3] = (float)left; nd[7] = (float)(-right);
    return me;
}

// tri: T*9 floats. nodes must hold 2*T*8 floats, order T ints. Returns the node count.
int nr_build_bvh(const float* tri, int T, int leaf, float* nodes, int32_t* order) {
    float* buf = (float*)malloc(sizeof(float) * (size_t)T * 9);
    BvhCtx c = {buf, buf + (size_t)T * 3, buf + (size_t)T * 6, order, nodes, leaf, 0};
    for (int i = 0; i < T; i++) {
        order[i] = i;
        for (int k = 0; k < 3; k++) {
            const float v0 = tri[(size_t)i * 9 + k], v1 = tri[(size_t)i * 9 + 3 + k], v2 = tri[(size_t)i * 9 + 6 + k];
            const float lo = fminf(v0, fminf(v1, v2)), hi = fmaxf(v0, fmaxf(v1, v2));
            buf[(size_t)i * 3 + k] = lo;
            buf[(size_t)T * 3 + (size_t)i * 3 + k] = hi;
            buf[(size_t)T * 6 + (size_t)i * 3 + k] = 0.5f * (lo + hi);
        }
    }
    bvh_build(&c, 0, T);
    free(buf);
    return c.n;
}
