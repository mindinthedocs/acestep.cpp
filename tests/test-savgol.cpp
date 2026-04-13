// test-savgol.cpp
// Standalone unit test for src/savgol.h.
// Build:  make test-savgol  (see Makefile)
// Usage:  ./test-savgol      (exit 0 on success, non-zero on fail)
#include "savgol.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static int g_fail = 0;

static void report(const char * name, bool ok, const char * detail = nullptr) {
    if (ok) {
        fprintf(stdout, "[PASS] %s\n", name);
    } else {
        fprintf(stdout, "[FAIL] %s%s%s\n", name,
                detail ? " : " : "", detail ? detail : "");
        g_fail++;
    }
}

static bool close_abs(float a, float b, float tol) {
    return std::fabs(a - b) <= tol;
}

// ---- Test 1: coefficients for (W=5, P=2) == [-3,12,17,12,-3]/35 ----
static void test_coeffs_5_2() {
    float c[5];
    savgol_coeffs(5, 2, c);
    const float ref[5] = { -3.f / 35.f, 12.f / 35.f, 17.f / 35.f,
                           12.f / 35.f, -3.f / 35.f };
    bool ok = true;
    char detail[128];
    detail[0] = 0;
    for (int i = 0; i < 5; ++i) {
        if (!close_abs(c[i], ref[i], 1e-5f)) {
            ok = false;
            snprintf(detail, sizeof(detail),
                     "i=%d got=%.8f ref=%.8f", i, c[i], ref[i]);
            break;
        }
    }
    report("savgol_coeffs(5, 2) matches [-3,12,17,12,-3]/35", ok, detail);
}

// ---- Test 2: coefficients for (W=7, P=3) == [-2,3,6,7,6,3,-2]/21 ----
static void test_coeffs_7_3() {
    float c[7];
    savgol_coeffs(7, 3, c);
    const float ref[7] = { -2.f / 21.f, 3.f / 21.f, 6.f / 21.f, 7.f / 21.f,
                           6.f / 21.f, 3.f / 21.f, -2.f / 21.f };
    bool ok = true;
    char detail[128];
    detail[0] = 0;
    for (int i = 0; i < 7; ++i) {
        if (!close_abs(c[i], ref[i], 1e-5f)) {
            ok = false;
            snprintf(detail, sizeof(detail),
                     "i=%d got=%.8f ref=%.8f", i, c[i], ref[i]);
            break;
        }
    }
    report("savgol_coeffs(7, 3) matches [-2,3,6,7,6,3,-2]/21", ok, detail);
}

// ---- Test 3: symmetry around center for several (W, P) pairs ----
static void test_symmetry() {
    const int cases[][2] = { {5, 2}, {7, 2}, {7, 3}, {9, 2}, {9, 4}, {11, 3}, {15, 4} };
    bool all_ok = true;
    char detail[128];
    detail[0] = 0;
    for (auto & wp : cases) {
        const int W = wp[0];
        const int P = wp[1];
        std::vector<float> c((size_t) W);
        savgol_coeffs(W, P, c.data());
        for (int i = 0; i < W / 2; ++i) {
            if (!close_abs(c[(size_t) i], c[(size_t) (W - 1 - i)], 1e-5f)) {
                all_ok = false;
                snprintf(detail, sizeof(detail),
                         "W=%d P=%d i=%d c[i]=%.8f c[W-1-i]=%.8f",
                         W, P, i, c[(size_t) i], c[(size_t) (W - 1 - i)]);
                break;
            }
        }
        if (!all_ok) break;
    }
    report("coefficients symmetric around center", all_ok, detail);
}

// ---- Test 4: sum of coefficients == 1.0 (DC gain) ----
static void test_sum_is_one() {
    const int cases[][2] = { {5, 2}, {7, 2}, {7, 3}, {9, 2}, {9, 4}, {11, 3}, {15, 4} };
    bool all_ok = true;
    char detail[128];
    detail[0] = 0;
    for (auto & wp : cases) {
        const int W = wp[0];
        const int P = wp[1];
        std::vector<float> c((size_t) W);
        savgol_coeffs(W, P, c.data());
        double s = 0.0;
        for (int i = 0; i < W; ++i) s += (double) c[(size_t) i];
        if (!close_abs((float) s, 1.0f, 1e-5f)) {
            all_ok = false;
            snprintf(detail, sizeof(detail),
                     "W=%d P=%d sum=%.8f", W, P, (float) s);
            break;
        }
    }
    report("coefficients sum to 1.0 (DC gain)", all_ok, detail);
}

// ---- Test 5: polynomial preservation (degree <= P) on interior ----
// Signal x[t] = a + b*t + c*t^2 with distinct (a,b,c) per channel.
// Apply SG(W=7, P=2); interior error should be ~0 (f32 roundoff).
static void test_polynomial_preservation() {
    const int T = 64;
    const int C = 3;
    const int W = 7;
    const int P = 2;

    const float params[3][3] = {
        { 1.0f,  0.5f,  0.01f   },
        { -2.0f, 0.25f, -0.005f },
        { 0.0f,  1.1f,  0.002f  },
    };

    std::vector<float> in((size_t) T * (size_t) C, 0.f);
    for (int t = 0; t < T; ++t) {
        for (int c = 0; c < C; ++c) {
            const float a = params[c][0];
            const float b = params[c][1];
            const float q = params[c][2];
            in[(size_t) t * (size_t) C + (size_t) c] =
                a + b * (float) t + q * (float) t * (float) t;
        }
    }
    std::vector<float> out((size_t) T * (size_t) C, 0.f);
    savgol_apply_1d(in.data(), T, C, out.data(), W, P);

    const int m = W / 2;
    float max_err = 0.0f;
    int   worst_t = -1, worst_c = -1;
    for (int t = m; t < T - m; ++t) {
        for (int c = 0; c < C; ++c) {
            const float got  = out[(size_t) t * (size_t) C + (size_t) c];
            const float ref  = in[(size_t) t * (size_t) C + (size_t) c];
            const float diff = std::fabs(got - ref);
            if (diff > max_err) {
                max_err = diff;
                worst_t = t;
                worst_c = c;
            }
        }
    }
    char detail[160];
    snprintf(detail, sizeof(detail),
             "max_interior_err=%.3e at t=%d c=%d", max_err, worst_t, worst_c);
    const bool ok = max_err < 1e-4f;
    report("SG(W=7,P=2) preserves degree-2 polynomial on interior", ok, detail);
}

// ---- Test 6: impulse response ----
// Input: delta at t = T/2. Filter (W=5, P=2). At the impulse location, the
// filtered output at the coefficient offsets should equal the coefficients
// (center-aligned). Also assert bounded (no NaN/Inf) everywhere.
static void test_impulse_response() {
    const int T = 33;
    const int C = 1;
    const int W = 5;
    const int P = 2;
    const int t0 = T / 2;

    std::vector<float> in((size_t) T * (size_t) C, 0.f);
    in[(size_t) t0] = 1.0f;

    std::vector<float> out((size_t) T * (size_t) C, 0.f);
    savgol_apply_1d(in.data(), T, C, out.data(), W, P);

    float coeffs[5];
    savgol_coeffs(W, P, coeffs);
    const int m = W / 2;

    bool ok = true;
    char detail[160];
    detail[0] = 0;
    // Output at (t0 + k) for k in [-m, m] should equal coeffs[m - k].
    // Proof: out[t] = sum_j coeffs[j] * in[t + j - m]. With in = delta at t0,
    // out[t] = coeffs[m + t0 - t]. So out[t0 + k] = coeffs[m - k].
    for (int k = -m; k <= m; ++k) {
        const int   t   = t0 + k;
        const float got = out[(size_t) t];
        const float ref = coeffs[m - k];
        if (!close_abs(got, ref, 1e-6f)) {
            ok = false;
            snprintf(detail, sizeof(detail),
                     "t=%d got=%.8f ref=%.8f", t, got, ref);
            break;
        }
    }

    // Zero elsewhere in the interior (away from the impulse and boundaries).
    if (ok) {
        for (int t = 0; t < T; ++t) {
            if (t >= t0 - m && t <= t0 + m) continue;
            const float v = out[(size_t) t];
            if (std::isnan(v) || std::isinf(v)) {
                ok = false;
                snprintf(detail, sizeof(detail), "non-finite at t=%d", t);
                break;
            }
            if (std::fabs(v) > 1e-6f) {
                ok = false;
                snprintf(detail, sizeof(detail),
                         "expected 0 at t=%d got=%.8f", t, v);
                break;
            }
        }
    }

    // Boundedness: run a second impulse near the boundary to exercise the
    // mirror-reflect path and ensure no NaN/Inf.
    if (ok) {
        std::vector<float> in2((size_t) T, 0.f);
        in2[0] = 1.0f;
        std::vector<float> out2((size_t) T, 0.f);
        savgol_apply_1d(in2.data(), T, 1, out2.data(), W, P);
        for (int t = 0; t < T; ++t) {
            const float v = out2[(size_t) t];
            if (std::isnan(v) || std::isinf(v) || std::fabs(v) > 10.0f) {
                ok = false;
                snprintf(detail, sizeof(detail),
                         "boundary impulse non-finite or huge at t=%d v=%.8f",
                         t, v);
                break;
            }
        }
    }

    report("impulse response matches coefficients; bounded near boundary",
           ok, detail);
}

// ---- Test 7: in-place correctness ----
// Filtering with in == out must match filtering with separate buffers, bit-identical.
static void test_in_place() {
    const int T = 48;
    const int C = 5;
    const int W = 7;
    const int P = 2;

    std::vector<float> src((size_t) T * (size_t) C);
    // Deterministic pseudo-random signal
    uint32_t rng = 0x12345678u;
    for (size_t i = 0; i < src.size(); ++i) {
        rng = rng * 1664525u + 1013904223u;
        src[i] = ((float) (rng >> 8) / (float) (1u << 24)) * 2.0f - 1.0f;
    }

    std::vector<float> out_sep((size_t) T * (size_t) C, 0.f);
    savgol_apply_1d(src.data(), T, C, out_sep.data(), W, P);

    std::vector<float> buf = src;  // copy
    savgol_apply_1d(buf.data(), T, C, buf.data(), W, P);

    bool ok = true;
    char detail[160];
    detail[0] = 0;
    for (size_t i = 0; i < buf.size(); ++i) {
        if (buf[i] != out_sep[i]) {
            ok = false;
            snprintf(detail, sizeof(detail),
                     "mismatch at i=%zu inplace=%.8f sep=%.8f diff=%.3e",
                     i, buf[i], out_sep[i],
                     std::fabs(buf[i] - out_sep[i]));
            break;
        }
    }
    report("in-place filtering matches separate-buffer filtering (bit-identical)",
           ok, detail);
}

int main(int /*argc*/, char ** /*argv*/) {
    test_coeffs_5_2();
    test_coeffs_7_3();
    test_symmetry();
    test_sum_is_one();
    test_polynomial_preservation();
    test_impulse_response();
    test_in_place();

    if (g_fail == 0) {
        fprintf(stdout, "\nAll savgol tests PASSED\n");
        return 0;
    }
    fprintf(stdout, "\n%d savgol test(s) FAILED\n", g_fail);
    return 1;
}
