/* Step 6G cpu-3d, Phase 1: a compiled per-pixel COST FLOOR for a CPU 3-D path, one core.
 *
 * Not a renderer and not a prototype: a probe that does the arithmetic a CPU frame of each
 * affected family would have to do, in plain scalar C (gcc -O3 -march=native, no fast-math), so
 * the "C++/SIMD" column of the architecture comparison is a measured lower bound on cost and not a
 * guess. Pixels it produces are thrown away.
 *
 *   dmc  : W x W frame, K capsules over a checker PLANE under a cube-map SKYBOX, headlight +
 *          shadow-casting spotlight (cutoff 45, exponent 10), round-to-uint8. Modes: brute (every
 *          pixel tests every capsule) and tiled (per-tile capsule lists from projected bounding
 *          spheres; shadow rays only inside the movers' dilated union bbox).
 *   tri  : T triangles of edge ~S px rasterised into a 480x480 z-buffer with 3-attribute Gouraud
 *          interpolation and a headlight shade per covered pixel (the Fetch mover pass).
 *
 *   gcc -O3 -march=native -fno-math-errno -o cost_floor cost_floor.c -lm
 *   ./cost_floor dmc 84 8 ; ./cost_floor dmc 84 19 ; ./cost_floor tri 480 100000 6
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
static float frand(void) { return (float)rand() / (float)RAND_MAX; }
typedef struct { float x, y, z; } v3;
static inline v3 V(float x, float y, float z) { v3 r = {x, y, z}; return r; }
static inline v3 add(v3 a, v3 b) { return V(a.x + b.x, a.y + b.y, a.z + b.z); }
static inline v3 sub(v3 a, v3 b) { return V(a.x - b.x, a.y - b.y, a.z - b.z); }
static inline v3 mul(v3 a, float s) { return V(a.x * s, a.y * s, a.z * s); }
static inline float dot(v3 a, v3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static inline v3 nrm(v3 a) { float l = sqrtf(dot(a, a)); return l > 0 ? mul(a, 1.0f / l) : a; }
static inline v3 matv(const float *R, v3 v) { return V(R[0]*v.x + R[1]*v.y + R[2]*v.z, R[3]*v.x + R[4]*v.y + R[5]*v.z, R[6]*v.x + R[7]*v.y + R[8]*v.z); }
static inline v3 mattv(const float *R, v3 v) { return V(R[0]*v.x + R[3]*v.y + R[6]*v.z, R[1]*v.x + R[4]*v.y + R[7]*v.z, R[2]*v.x + R[5]*v.y + R[8]*v.z); }
static inline uint8_t q8(float c) { c = c < 0 ? 0 : (c > 1 ? 1 : c); return (uint8_t)(c * 255.0f + 0.5f); }

/* ---------------------------------------------------------------- dmc */
typedef struct { v3 pos; float R[9]; float r, hh; int x0, y0, x1, y1; } Cap;
#define INF 1e30f
#define MAXK 64
#define TILE 8

static float cap_isect(const Cap *c, v3 o, v3 d, v3 *n) {
  v3 ol = mattv(c->R, sub(o, c->pos)), dl = mattv(c->R, d);
  float a = dl.x*dl.x + dl.y*dl.y, b = ol.x*dl.x + ol.y*dl.y, cc = ol.x*ol.x + ol.y*ol.y - c->r*c->r;
  float disc = b*b - a*cc, best = INF;
  if (disc > 0 && a > 1e-12f) {
    float sq = sqrtf(disc);
    for (int s = -1; s <= 1; s += 2) { float t = (-b + s*sq) / a; float z = ol.z + t*dl.z; if (t > 0 && fabsf(z) <= c->hh && t < best) best = t; }
  }
  for (int s = -1; s <= 1; s += 2) {
    float zc = s * c->hh; v3 oc = V(ol.x, ol.y, ol.z - zc); float bb = dot(oc, dl), c2 = dot(oc, oc) - c->r*c->r, dd = bb*bb - c2;
    if (dd > 0) { float s2 = sqrtf(dd); float t1 = -bb - s2, t2 = -bb + s2; if (t1 > 0 && t1 < best) best = t1; if (t2 > 0 && t2 < best) best = t2; }
  }
  if (best < INF) { v3 pl = add(ol, mul(dl, best)); float zc = pl.z > c->hh ? c->hh : (pl.z < -c->hh ? -c->hh : pl.z); v3 nl = nrm(V(pl.x, pl.y, pl.z - zc)); *n = matv(c->R, nl); }
  return best;
}
static uint8_t *TEX, *TEX2, *SKY; static int TW = 300, SW = 800;
static inline v3 tex_bilin(const uint8_t *t, int w, float u, float v) {
  u -= floorf(u); v -= floorf(v); float fx = u * w - 0.5f, fy = v * w - 0.5f; int x0 = (int)floorf(fx), y0 = (int)floorf(fy); float ax = fx - x0, ay = fy - y0;
  int x1 = (x0 + 1) % w, y1 = (y0 + 1) % w; x0 = (x0 + w) % w; y0 = (y0 + w) % w; v3 c = V(0,0,0); float wts[4] = {(1-ax)*(1-ay), ax*(1-ay), (1-ax)*ay, ax*ay}; int xs[4] = {x0,x1,x0,x1}, ys[4] = {y0,y0,y1,y1};
  for (int k = 0; k < 4; k++) { const uint8_t *p = t + 3 * (ys[k] * w + xs[k]); c = add(c, mul(V(p[0], p[1], p[2]), wts[k] / 255.0f)); }
  return c;
}
static inline v3 sky_sample(v3 d) {
  float ax = fabsf(d.x), ay = fabsf(d.y), az = fabsf(d.z); int face; float sc, tc, ma;
  if (ax >= ay && ax >= az) { ma = ax; face = d.x > 0 ? 0 : 1; sc = d.x > 0 ? -d.z : d.z; tc = -d.y; }
  else if (ay >= az) { ma = ay; face = d.y > 0 ? 2 : 3; sc = d.x; tc = d.y > 0 ? d.z : -d.z; }
  else { ma = az; face = d.z > 0 ? 4 : 5; sc = d.z > 0 ? d.x : -d.x; tc = -d.y; }
  float u = 0.5f * (sc / ma + 1.0f), v = 0.5f * (tc / ma + 1.0f);
  return tex_bilin(SKY + (size_t)face * SW * SW * 3, SW, u, v);
}
static void run_dmc(int W, int K, int tiled, int frames, int seed) {
  srand(seed);
  int N = W * W; float *dirs = malloc(sizeof(float) * 3 * N); uint8_t *img = malloc(3 * N);
  float f = tanf(45.0f * 0.5f * (float)M_PI / 180.0f);
  for (int y = 0; y < W; y++) for (int x = 0; x < W; x++) { float sx = (2.0f*(x+0.5f)/W - 1.0f)*f, sy = (1.0f - 2.0f*(y+0.5f)/W)*f; v3 d = nrm(V(sx, sy, -1.0f)); float *p = dirs + 3*(y*W+x); p[0]=d.x; p[1]=d.y; p[2]=d.z; }
  /* camera: side view like cheetah's: looks +y, up z. xmat columns right/up/back. */
  float Rc[9] = {1,0,0, 0,0,-1, 0,1,0}; v3 camfwd = V(0, 1, 0);
  float cos_cut = cosf(45.0f * (float)M_PI / 180.0f);
  Cap caps[MAXK]; int ntile = (W + TILE - 1) / TILE; int *tl = malloc(sizeof(int) * ntile * ntile * (MAXK + 1));
  double t0 = now(); long shadow_tests = 0, prim_tests = 0;
  for (int fr = 0; fr < frames; fr++) {
    v3 cam = V(0.3f * frand() - 0.15f, -3.0f, 0.6f + 0.1f * frand()); v3 lightpos = add(cam, V(0, 3, 1.4f)); v3 lightdir = V(0, 0, -1);
    int ux0 = W, uy0 = W, ux1 = -1, uy1 = -1;
    for (int k = 0; k < K; k++) { Cap *c = &caps[k]; c->pos = add(cam, V(-0.6f + 1.2f*frand(), 2.7f + 0.6f*frand(), -0.5f + 1.0f*frand())); c->r = 0.04f + 0.06f*frand(); c->hh = 0.05f + 0.15f*frand();
      float a = 6.28f*frand(), b = 3.14f*frand(); v3 z = V(cosf(a)*sinf(b), sinf(a)*sinf(b), cosf(b)); v3 xr = nrm(V(-z.y, z.x, 0.001f)); v3 yr = V(z.y*xr.z - z.z*xr.y, z.z*xr.x - z.x*xr.z, z.x*xr.y - z.y*xr.x);
      float R[9] = {xr.x, yr.x, z.x, xr.y, yr.y, z.y, xr.z, yr.z, z.z}; memcpy(c->R, R, sizeof R);
      /* screen bbox of the bounding sphere */
      v3 rel = mattv(Rc, sub(c->pos, cam)); float rad = c->r + c->hh; float zc = -rel.z; if (zc < 0.05f) zc = 0.05f; float px = (rel.x / zc) / f, py = (rel.y / zc) / f, pr = (rad / zc) / f;
      c->x0 = (int)floorf((px - pr + 1) * 0.5f * W) - 1; c->x1 = (int)ceilf((px + pr + 1) * 0.5f * W) + 1; c->y0 = (int)floorf((1 - (py + pr)) * 0.5f * W) - 1; c->y1 = (int)ceilf((1 - (py - pr)) * 0.5f * W) + 1;
      if (c->x0 < 0) c->x0 = 0; if (c->y0 < 0) c->y0 = 0; if (c->x1 > W) c->x1 = W; if (c->y1 > W) c->y1 = W;
      if (c->x0 < ux0) ux0 = c->x0; if (c->y0 < uy0) uy0 = c->y0; if (c->x1 > ux1) ux1 = c->x1; if (c->y1 > uy1) uy1 = c->y1; }
    int m = W / 6; ux0 -= m; uy0 -= m; ux1 += m; uy1 += 2 * m; if (ux0 < 0) ux0 = 0; if (uy0 < 0) uy0 = 0; if (ux1 > W) ux1 = W; if (uy1 > W) uy1 = W;
    if (tiled) { for (int ty = 0; ty < ntile; ty++) for (int tx = 0; tx < ntile; tx++) { int *L = tl + (ty*ntile+tx)*(MAXK+1); L[0] = 0; for (int k = 0; k < K; k++) { Cap *c = &caps[k]; if (c->x1 > tx*TILE && c->x0 < (tx+1)*TILE && c->y1 > ty*TILE && c->y0 < (ty+1)*TILE) L[++L[0]] = k; } } }
    for (int y = 0; y < W; y++) for (int x = 0; x < W; x++) {
      const float *dp = dirs + 3*(y*W+x); v3 d = matv(Rc, V(dp[0], dp[1], dp[2])); v3 o = cam; float best = INF; int id = -1; v3 n = V(0,0,1);
      if (d.z < 0) { best = -o.z / d.z; id = -2; }
      if (tiled) { const int *L = tl + ((y/TILE)*ntile + x/TILE)*(MAXK+1); for (int i = 1; i <= L[0]; i++) { v3 nn; float t = cap_isect(&caps[L[i]], o, d, &nn); prim_tests++; if (t < best) { best = t; id = L[i]; n = nn; } } }
      else { for (int k = 0; k < K; k++) { v3 nn; float t = cap_isect(&caps[k], o, d, &nn); prim_tests++; if (t < best) { best = t; id = k; n = nn; } } }
      v3 col;
      if (id == -1) { col = sky_sample(d); }
      else {
        v3 p = add(o, mul(d, best)); v3 base = id == -2 ? V(1,1,1) : V(0.7f, 0.5f, 0.3f);
        if (id == -2) { float u = 0.5f*p.x - 0.5f, v = -0.5f*p.y - 0.5f; v3 c0 = tex_bilin(TEX, TW, u, v); float foot = best * (2*f/W) / fabsf(d.z); float lod = foot * TW * 0.5f; v3 c1 = tex_bilin(TEX2, TW/2, u, v); float w1 = lod > 1 ? (lod > 2 ? 1 : lod - 1) : 0; base = add(mul(c0, 1-w1), mul(c1, w1)); }
        v3 vdir = mul(d, -1); v3 res = mul(base, 0.4f); /* headlight ambient */
        { v3 L = mul(camfwd, -1); float ndl = dot(n, L); if (ndl > 0) { res = add(res, mul(base, 0.8f * ndl)); v3 H = nrm(add(L, vdir)); float ndh = dot(n, H); if (ndh > 0) res = add(res, mul(V(1,1,1), 0.1f * 0.5f * powf(ndh, 64.0f))); } }
        { v3 Lv = sub(lightpos, p); float dist = sqrtf(dot(Lv, Lv)); v3 L = mul(Lv, 1.0f/dist); float ct = dot(mul(L, -1), lightdir);
          if (ct >= cos_cut) { float att = powf(ct, 10.0f); float ndl = dot(n, L); if (ndl > 0) { float vis = 1.0f;
              int in_box = x >= ux0 && x < ux1 && y >= uy0 && y < uy1;
              if (!tiled || in_box) { v3 so = add(p, mul(n, 1e-4f)); for (int k = 0; k < K; k++) { if (k == id) continue; v3 nn; float t = cap_isect(&caps[k], so, L, &nn); shadow_tests++; if (t > 0 && t < dist - 1e-3f) { vis = 0.3f; break; } } }
              float wgt = att * vis; res = add(res, mul(base, 0.7f * ndl * wgt)); v3 H = nrm(add(L, vdir)); float ndh = dot(n, H); if (ndh > 0) res = add(res, mul(V(1,1,1), 0.3f * 0.5f * powf(ndh, 64.0f) * wgt)); } } }
        col = res;
      }
      uint8_t *q = img + 3*(y*W+x); q[0] = q8(col.x); q[1] = q8(col.y); q[2] = q8(col.z);
    }
  }
  double dt = (now() - t0) / frames; long chk = 0; for (int i = 0; i < 3*N; i++) chk += img[i];
  printf("{\"kernel\":\"dmc\",\"W\":%d,\"K\":%d,\"tiled\":%d,\"us_per_frame\":%.1f,\"ns_per_pixel\":%.2f,\"fps_1core\":%.0f,\"prim_tests_per_px\":%.2f,\"shadow_tests_per_px\":%.2f,\"chk\":%ld}\n",
         W, K, tiled, dt*1e6, dt/N*1e9, 1.0/dt, (double)prim_tests/frames/N, (double)shadow_tests/frames/N, chk % 1000);
  free(dirs); free(img); free(tl);
}

/* ---------------------------------------------------------------- tri */
static void run_tri(int W, int T, float S, int frames, int seed) {
  srand(seed);
  int N = W * W; float *zb = malloc(sizeof(float) * N); uint8_t *img = malloc(3 * N);
  float *tri = malloc(sizeof(float) * 18 * T); /* per tri: 3 x (x, y, z, nx, ny, nz) */
  double t0 = now(); long covered = 0, bboxpx = 0;
  for (int fr = 0; fr < frames; fr++) {
    for (int t = 0; t < T; t++) { float cx = frand()*W, cy = frand()*W, cz = frand(); v3 nn = nrm(V(frand()-0.5f, frand()-0.5f, frand())); for (int v = 0; v < 3; v++) { float a = 6.28f*frand(); float *p = tri + 18*t + 6*v; p[0] = cx + S*0.6f*cosf(a); p[1] = cy + S*0.6f*sinf(a); p[2] = cz + 0.01f*frand(); p[3] = nn.x + 0.1f*frand(); p[4] = nn.y + 0.1f*frand(); p[5] = nn.z; } }
    for (int i = 0; i < N; i++) zb[i] = INF; memset(img, 0, 3*N);
    for (int t = 0; t < T; t++) { const float *a = tri + 18*t, *b = a + 6, *c = a + 12;
      int x0 = (int)floorf(fminf(a[0], fminf(b[0], c[0]))), x1 = (int)ceilf(fmaxf(a[0], fmaxf(b[0], c[0]))), y0 = (int)floorf(fminf(a[1], fminf(b[1], c[1]))), y1 = (int)ceilf(fmaxf(a[1], fmaxf(b[1], c[1])));
      if (x0 < 0) x0 = 0; if (y0 < 0) y0 = 0; if (x1 >= W) x1 = W-1; if (y1 >= W) y1 = W-1; float area = (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0]); if (fabsf(area) < 1e-6f) continue; float ia = 1.0f/area;
      for (int y = y0; y <= y1; y++) for (int x = x0; x <= x1; x++) { bboxpx++; float px = x + 0.5f, py = y + 0.5f;
        float w0 = ((b[0]-px)*(c[1]-py) - (b[1]-py)*(c[0]-px)) * ia, w1 = ((c[0]-px)*(a[1]-py) - (c[1]-py)*(a[0]-px)) * ia, w2 = 1.0f - w0 - w1;
        if (w0 < 0 || w1 < 0 || w2 < 0) continue; float z = w0*a[2] + w1*b[2] + w2*c[2]; int i = y*W + x; if (z >= zb[i]) continue; zb[i] = z; covered++;
        v3 n = nrm(V(w0*a[3]+w1*b[3]+w2*c[3], w0*a[4]+w1*b[4]+w2*c[4], w0*a[5]+w1*b[5]+w2*c[5])); float ndl = n.z > 0 ? n.z : 0; float spec = 0.5f * powf(ndl, 64.0f);
        uint8_t *q = img + 3*i; q[0] = q8(0.9f*(0.1f + 0.4f*ndl) + spec); q[1] = q8(0.9f*(0.1f + 0.4f*ndl) + spec); q[2] = q8(0.9f*(0.1f + 0.4f*ndl) + spec); } }
  }
  double dt = (now() - t0) / frames; long chk = 0; for (int i = 0; i < 3*N; i++) chk += img[i];
  printf("{\"kernel\":\"tri\",\"W\":%d,\"T\":%d,\"S\":%.1f,\"ms_per_frame\":%.2f,\"ns_per_tri\":%.1f,\"covered_px_per_frame\":%.0f,\"bbox_px_per_frame\":%.0f,\"fps_1core\":%.1f,\"chk\":%ld}\n",
         W, T, S, dt*1e3, dt/T*1e9, (double)covered/frames, (double)bboxpx/frames, 1.0/dt, chk % 1000);
  free(zb); free(img); free(tri);
}

int main(int argc, char **argv) {
  if (argc < 2) { fprintf(stderr, "usage: cost_floor dmc W K | tri W T S\n"); return 1; }
  if (!strcmp(argv[1], "dmc")) {
    int W = atoi(argv[2]), K = atoi(argv[3]); if (K > MAXK) K = MAXK;
    TEX = malloc(TW*TW*3); TEX2 = malloc((TW/2)*(TW/2)*3); SKY = malloc((size_t)6*SW*SW*3);
    for (int y = 0; y < TW; y++) for (int x = 0; x < TW; x++) { int c = ((x/150) ^ (y/150)) & 1; uint8_t *p = TEX + 3*(y*TW+x); p[0] = c ? 60 : 20; p[1] = c ? 90 : 40; p[2] = c ? 120 : 60; }
    for (int y = 0; y < TW/2; y++) for (int x = 0; x < TW/2; x++) { uint8_t *p = TEX2 + 3*(y*(TW/2)+x); p[0] = 40; p[1] = 65; p[2] = 90; }
    for (size_t i = 0; i < (size_t)6*SW*SW; i++) { uint8_t *p = SKY + 3*i; int y = (i / SW) % SW; p[0] = 30 + y/8; p[1] = 60 + y/8; p[2] = 90 + y/8; if ((i * 2654435761u) % 997 == 0) p[0] = p[1] = p[2] = 255; }
    int frames = W <= 128 ? 400 : 40;
    run_dmc(W, K, 0, frames, 1); run_dmc(W, K, 1, frames, 1);
  } else if (!strcmp(argv[1], "tri")) { run_tri(atoi(argv[2]), atoi(argv[3]), (float)atof(argv[4]), 20, 1); }
  return 0;
}
