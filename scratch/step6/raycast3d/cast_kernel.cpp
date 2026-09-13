// RAYCAST-3 micro-benchmark kernel. NOT a renderer, NOT wired into stable_envs.
//
// One C++ transcription of raycast3d's nearest-hit cast, written to answer one question: how many
// nanoseconds does ONE ray-primitive intersection cost on one core when the K-primitive loop stays in
// registers instead of being paid as ~200 elementwise torch kernels over materialised (B,N)
// temporaries? The torch path measures 145-240 ns/test (raycast3d ticket 1) and 220 ns/test in the
// cpu-3d design's independent reproduction.
//
// BIT-EXACTNESS IS THE POINT, NOT THE SPEED. The frozen Step-4 digests were produced by
// cast_nearest_torch (via the Triton kernel that is bit-identical to it), so a CPU kernel is
// admissible only if it agrees to the last bit. Three rules, the same three the Triton kernel needed:
//   * expressions transcribed in the SAME association order as the torch source;
//   * no FMA contraction -- the caller compiles with -ffp-contract=off, so a*b+c rounds twice;
//   * no fast-math -- / and sqrtf stay IEEE round-to-nearest.
// The benchmark asserts this against cast_nearest_torch on the same inputs rather than claiming it.
//
// Covers SPHERE and CAPSULE. Those are 33 of ball_in_cup's 36 movers, 2 of reacher's 4, 5 of
// finger's 7 and every OGBench agent primitive but one; CYLINDER/BOX/PLANE are the same shape of
// work (a quadratic or a slab test plus a frame change) and are left out of the MICRO-benchmark on
// purpose -- adding them would change the packaging question, not the ns/test answer.
#include <torch/extension.h>
#include <cmath>
#include <cstdint>

namespace {

constexpr float INF_T = 1e30f;

// mirrors raycast3d._sphere
inline void isect_sphere(float ox, float oy, float oz, float dx, float dy, float dz,
                         float px, float py, float pz, float r,
                         float &t_out, float &nx, float &ny, float &nz) {
  const float ocx = ox - px, ocy = oy - py, ocz = oz - pz;
  const float b = ocx * dx + ocy * dy + ocz * dz;
  const float c = ocx * ocx + ocy * ocy + ocz * ocz - r * r;
  const float disc = b * b - c;
  const float sq = std::sqrt(disc < 0.0f ? 0.0f : disc);
  float t = -b - sq;
  t = (t > 0.0f) ? t : (-b + sq);
  const bool hit = (disc > 0.0f) && (t > 0.0f);
  t_out = hit ? t : INF_T;
  const float tc = t_out < 1e6f ? t_out : 1e6f;
  nx = (ocx + tc * dx) / r;
  ny = (ocy + tc * dy) / r;
  nz = (ocz + tc * dz) / r;
}

// mirrors raycast3d._capsule: barrel (both roots, in order), then the two spherical caps
inline void isect_capsule(float ox, float oy, float oz, float dx, float dy, float dz,
                          float px, float py, float pz,
                          const float *R, float r, float hh,
                          float &t_out, float &nx, float &ny, float &nz) {
  const float ax = ox - px, ay = oy - py, az = oz - pz;
  // _mtv(R, .) : world -> geom frame, column-major read of a row-major 3x3
  const float olx = R[0] * ax + R[3] * ay + R[6] * az;
  const float oly = R[1] * ax + R[4] * ay + R[7] * az;
  const float olz = R[2] * ax + R[5] * ay + R[8] * az;
  const float dlx = R[0] * dx + R[3] * dy + R[6] * dz;
  const float dly = R[1] * dx + R[4] * dy + R[7] * dz;
  const float dlz = R[2] * dx + R[5] * dy + R[8] * dz;

  const float a = dlx * dlx + dly * dly;
  const float b = olx * dlx + oly * dly;
  const float c = olx * olx + oly * oly - r * r;
  const float disc = b * b - a * c;
  const float sq = std::sqrt(disc < 0.0f ? 0.0f : disc);
  float best = INF_T;
  const float den_a = a > 1e-12f ? a : 1e-12f;
  for (int s = 0; s < 2; ++s) {
    const float sgn = s == 0 ? -1.0f : 1.0f;
    const float t = (-b + sgn * sq) / den_a;
    const float z = olz + t * dlz;
    const bool ok = (disc > 0.0f) && (a > 1e-12f) && (t > 0.0f) && (std::fabs(z) <= hh);
    if (ok && t < best) best = t;
  }
  for (int s = 0; s < 2; ++s) {
    const float zc = (s == 0 ? -1.0f : 1.0f) * hh;
    const float ocz = olz - zc;
    const float bb = olx * dlx + oly * dly + ocz * dlz;
    const float cc = olx * olx + oly * oly + ocz * ocz - r * r;
    const float dd = bb * bb - cc;
    const float s2 = std::sqrt(dd < 0.0f ? 0.0f : dd);
    for (int g = 0; g < 2; ++g) {
      const float t = -bb + (g == 0 ? -1.0f : 1.0f) * s2;
      if ((dd > 0.0f) && (t > 0.0f) && (t < best)) best = t;
    }
  }
  const float bc = best < 1e6f ? best : 1e6f;
  const float plx = olx + bc * dlx;
  const float ply = oly + bc * dly;
  const float plz = olz + bc * dlz;
  float zcl = plz < hh ? plz : hh;
  zcl = zcl > -hh ? zcl : -hh;
  const float mx = plx, my = ply, mz = plz - zcl;
  float den = std::sqrt(mx * mx + my * my + mz * mz);
  den = den > 1e-12f ? den : 1e-12f;
  const float nlx = mx / den, nly = my / den, nlz = mz / den;
  t_out = best;
  nx = R[0] * nlx + R[1] * nly + R[2] * nlz;   // _mv(R, nl)
  ny = R[3] * nlx + R[4] * nly + R[5] * nlz;
  nz = R[6] * nlx + R[7] * nly + R[8] * nlz;
}

}  // namespace

// kinds: (K,) int32, 2 = sphere, 3 = capsule.  sizes: (K,2) [r, hh].  pos: (K,3).  Rm: (K,9).
// o: (N,3) or (1,3) broadcast.  d: (N,3).  Returns (t (N,), n (N,3), idx (N,) int64).
std::vector<torch::Tensor> cast_nearest_cpu(torch::Tensor kinds, torch::Tensor sizes,
                                            torch::Tensor pos, torch::Tensor Rm,
                                            torch::Tensor o, torch::Tensor d) {
  TORCH_CHECK(d.dim() == 2 && d.size(1) == 3, "d must be (N,3)");
  const int64_t N = d.size(0);
  const int64_t K = kinds.size(0);
  const bool o_bcast = o.size(0) == 1;

  auto t_out = torch::empty({N}, d.options());
  auto n_out = torch::empty({N, 3}, d.options());
  auto i_out = torch::empty({N}, d.options().dtype(torch::kInt64));

  const int32_t *kp = kinds.data_ptr<int32_t>();
  const float *sp = sizes.data_ptr<float>();
  const float *pp = pos.data_ptr<float>();
  const float *rp = Rm.data_ptr<float>();
  const float *op = o.data_ptr<float>();
  const float *dp = d.data_ptr<float>();
  float *tq = t_out.data_ptr<float>();
  float *nq = n_out.data_ptr<float>();
  int64_t *iq = i_out.data_ptr<int64_t>();

  for (int64_t i = 0; i < N; ++i) {
    const float ox = op[o_bcast ? 0 : i * 3 + 0];
    const float oy = op[o_bcast ? 1 : i * 3 + 1];
    const float oz = op[o_bcast ? 2 : i * 3 + 2];
    const float dx = dp[i * 3 + 0], dy = dp[i * 3 + 1], dz = dp[i * 3 + 2];
    float best = INF_T, bnx = 0.0f, bny = 0.0f, bnz = 0.0f;
    int64_t bidx = 0;
    for (int64_t k = 0; k < K; ++k) {
      float t, nx, ny, nz;
      if (kp[k] == 2)
        isect_sphere(ox, oy, oz, dx, dy, dz, pp[k * 3], pp[k * 3 + 1], pp[k * 3 + 2],
                     sp[k * 2], t, nx, ny, nz);
      else
        isect_capsule(ox, oy, oz, dx, dy, dz, pp[k * 3], pp[k * 3 + 1], pp[k * 3 + 2],
                      rp + k * 9, sp[k * 2], sp[k * 2 + 1], t, nx, ny, nz);
      if (t < best) { best = t; bnx = nx; bny = ny; bnz = nz; bidx = k; }
    }
    tq[i] = best;
    nq[i * 3] = bnx; nq[i * 3 + 1] = bny; nq[i * 3 + 2] = bnz;
    iq[i] = bidx;
  }
  return {t_out, n_out, i_out};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("cast_nearest_cpu", &cast_nearest_cpu, "nearest-hit cast, scalar C++");
}
