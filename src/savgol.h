// savgol.h
// Portable, header-only Savitzky-Golay 1D smoothing.
//
// Provides two primitives:
//   - savgol_coeffs(W, P, out):   compute W-length FIR kernel (odd W>=3, 1<=P<=W-2).
//   - savgol_apply_1d(in, T, C, out, W, P): apply kernel along time axis for
//     contiguous time-major layout [T, C] (stride = C). Mirror-reflect edges.
//
// Plain C++11. No external deps. No threads, no SIMD intrinsics.

#ifndef ACESTEP_SAVGOL_H
#define ACESTEP_SAVGOL_H

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <vector>

// Compute W-length Savitzky-Golay FIR coefficients for polynomial order P.
// Preconditions (asserted in debug):
//   - W odd, W >= 3
//   - 1 <= P <= W - 2
// Method:
//   1) Build Vandermonde A[i,k] = (i - m)^k for i in [0,W), k in [0,P], m = W/2.
//   2) Form M = A^T A on doubles.
//   3) Solve M * h = e_0 via Gauss-Jordan on doubles.
//   4) out[i] = float( sum_k h[k] * A[i,k] ).
static inline void savgol_coeffs(int W, int P, float * out) {
    assert(W >= 3);
    assert((W % 2) == 1);
    assert(P >= 1);
    assert(P <= W - 2);

    const int m  = W / 2;
    const int Pp = P + 1;

    // Vandermonde A: W x Pp  (stored row-major: A[i*Pp + k])
    std::vector<double> A((size_t) W * (size_t) Pp, 0.0);
    for (int i = 0; i < W; ++i) {
        const double x = (double) (i - m);
        double       v = 1.0;
        for (int k = 0; k < Pp; ++k) {
            A[(size_t) i * (size_t) Pp + (size_t) k] = v;
            v *= x;
        }
    }

    // M = A^T A  (Pp x Pp), symmetric
    std::vector<double> M((size_t) Pp * (size_t) Pp, 0.0);
    for (int a = 0; a < Pp; ++a) {
        for (int b = 0; b < Pp; ++b) {
            double s = 0.0;
            for (int i = 0; i < W; ++i) {
                s += A[(size_t) i * (size_t) Pp + (size_t) a] *
                     A[(size_t) i * (size_t) Pp + (size_t) b];
            }
            M[(size_t) a * (size_t) Pp + (size_t) b] = s;
        }
    }

    // Augmented matrix [ M | e_0 ] for Gauss-Jordan, size Pp x (Pp+1)
    const int            cols = Pp + 1;
    std::vector<double>  G((size_t) Pp * (size_t) cols, 0.0);
    for (int r = 0; r < Pp; ++r) {
        for (int c = 0; c < Pp; ++c) {
            G[(size_t) r * (size_t) cols + (size_t) c] =
                M[(size_t) r * (size_t) Pp + (size_t) c];
        }
        G[(size_t) r * (size_t) cols + (size_t) Pp] = (r == 0) ? 1.0 : 0.0;
    }

    // Gauss-Jordan elimination with partial pivoting
    for (int col = 0; col < Pp; ++col) {
        // pivot: find row with largest |G[r,col]| for r in [col, Pp)
        int    pivot      = col;
        double pivot_abs  = std::fabs(G[(size_t) col * (size_t) cols + (size_t) col]);
        for (int r = col + 1; r < Pp; ++r) {
            const double v = std::fabs(G[(size_t) r * (size_t) cols + (size_t) col]);
            if (v > pivot_abs) {
                pivot_abs = v;
                pivot     = r;
            }
        }
        assert(pivot_abs > 0.0 && "singular matrix in savgol_coeffs");
        if (pivot != col) {
            for (int c = 0; c < cols; ++c) {
                double tmp = G[(size_t) col * (size_t) cols + (size_t) c];
                G[(size_t) col * (size_t) cols + (size_t) c] =
                    G[(size_t) pivot * (size_t) cols + (size_t) c];
                G[(size_t) pivot * (size_t) cols + (size_t) c] = tmp;
            }
        }
        // normalize pivot row
        const double inv = 1.0 / G[(size_t) col * (size_t) cols + (size_t) col];
        for (int c = 0; c < cols; ++c) {
            G[(size_t) col * (size_t) cols + (size_t) c] *= inv;
        }
        // eliminate other rows
        for (int r = 0; r < Pp; ++r) {
            if (r == col) continue;
            const double factor = G[(size_t) r * (size_t) cols + (size_t) col];
            if (factor == 0.0) continue;
            for (int c = 0; c < cols; ++c) {
                G[(size_t) r * (size_t) cols + (size_t) c] -=
                    factor * G[(size_t) col * (size_t) cols + (size_t) c];
            }
        }
    }

    // h = last column of G
    std::vector<double> h((size_t) Pp, 0.0);
    for (int k = 0; k < Pp; ++k) {
        h[(size_t) k] = G[(size_t) k * (size_t) cols + (size_t) Pp];
    }

    // out[i] = sum_k h[k] * A[i,k]
    for (int i = 0; i < W; ++i) {
        double s = 0.0;
        for (int k = 0; k < Pp; ++k) {
            s += h[(size_t) k] * A[(size_t) i * (size_t) Pp + (size_t) k];
        }
        out[i] = (float) s;
    }
}

// Mirror-reflect index: map t' in [-(W/2), T + W/2] into [0, T).
//   t'      < 0     -> -t'      (x[-k] = x[k])
//   t'      >= T    -> 2*T - 2 - t' (x[T+k] = x[T-2-k])
// Requires T >= 2 for the reflection to be well-defined at far edges; callers
// should guard T < 3 before invoking the filter.
static inline int savgol_mirror_index(int t, int T) {
    if (t < 0)  t = -t;
    if (t >= T) t = 2 * T - 2 - t;
    // In pathological cases (very large W vs tiny T) could still be out of range;
    // clamp as a defensive measure.
    if (t < 0)  t = 0;
    if (t >= T) t = T - 1;
    return t;
}

// Apply Savitzky-Golay along time axis for contiguous time-major [T, C] layout.
// in and out may alias (in == out is safe). Temporary per-channel column buffer
// is allocated internally; output is written after all reads for that channel.
//
// Guards:
//   - If T < 3 or W < 3 or W > T, no-op and copy in -> out if in != out.
//   - W must be odd (asserted in debug). Caller is responsible for parity.
static inline void savgol_apply_1d(const float * in, int T, int C,
                                   float * out, int W, int P) {
    // No-op guards
    if (T < 3 || W < 3 || W > T) {
        if (in != out) {
            std::memcpy(out, in, sizeof(float) * (size_t) T * (size_t) C);
        }
        return;
    }
    assert((W % 2) == 1 && "savgol_apply_1d: W must be odd");
    assert(P >= 1 && P <= W - 2);

    std::vector<float> coeffs((size_t) W);
    savgol_coeffs(W, P, coeffs.data());

    const int m = W / 2;

    // Temporary column buffer (one channel's worth along time).
    std::vector<float> col((size_t) T);

    for (int c = 0; c < C; ++c) {
        // Compute filtered column into `col`, reading from `in`.
        for (int t = 0; t < T; ++t) {
            double s = 0.0;
            for (int j = 0; j < W; ++j) {
                const int tt = savgol_mirror_index(t + j - m, T);
                s += (double) coeffs[(size_t) j] *
                     (double) in[(size_t) tt * (size_t) C + (size_t) c];
            }
            col[(size_t) t] = (float) s;
        }
        // Write back to `out` (safe even if in == out, since we fully read this
        // channel before writing it — cross-channel reads within the same time
        // step also remain valid since we never wrote to any prior channel's
        // time slot for this column; we only overwrite channel c here, and the
        // next channel's reads of `in` still see the original values in other
        // channels because we wrote only channel c for all t, and channel c is
        // read only through index (tt, c) — which we just fully computed from
        // the original input before writing).
        for (int t = 0; t < T; ++t) {
            out[(size_t) t * (size_t) C + (size_t) c] = col[(size_t) t];
        }
    }
}

#endif  // ACESTEP_SAVGOL_H
