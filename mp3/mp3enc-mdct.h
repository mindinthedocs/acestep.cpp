#pragma once
// mp3enc-mdct.h
// Forward MDCT for the MP3 encoder: 36 subband samples -> 18 frequency lines.
// Window and MDCT are combined into a single step (ISO 11172-3 Annex C).
// Part of mp3enc. MIT license.

#include <cmath>
#include <cstring>

#ifndef M_PI
#    define M_PI 3.14159265358979323846
#endif
// Combined window + forward MDCT for long blocks.
// in[36] = prev[18] + cur[18] (raw subband samples, no pre-windowing)
// out[18] = MDCT frequency coefficients
//
// Formula (ISO 11172-3 Annex C):
//   out[k] = sum(n=0..35) in[n] * sin(PI/36 * (n + 0.5))
//                                * cos(PI/72 * (2*n + 19) * (2*k + 1))
//
// The sin() term IS the normal block window (block_type 0).
static void mp3enc_mdct36(const float * in, float * out) {
    // The raw MDCT sum over 36 windowed samples gives values 9x what the
    // decoder's IMDCT expects. Factor 1/9 = 2/N_half compensates, where
    // N_half = 18 is the number of MDCT coefficients per subband.
    for (int k = 0; k < 18; k++) {
        float sum = 0.0f;
        for (int n = 0; n < 36; n++) {
            float w = sinf((float) M_PI / 36.0f * ((float) n + 0.5f));
            float c = cosf((float) M_PI / 72.0f * (float) (2 * n + 19) * (float) (2 * k + 1));
            sum += in[n] * w * c;
        }
        out[k] = sum * (1.0f / 9.0f);
    }
}

// Alias reduction butterfly between adjacent subbands.
// Applied after MDCT, before quantization.
// ISO 11172-3 Table B.9 coefficients (from minimp3 CC0).
//
// For each pair of adjacent bands (band, band+1):
//   mdct[band][17-i] and mdct[band+1][i] are butterflied with cs/ca.
static void mp3enc_alias_reduce(float * mdct_out) {
    for (int band = 1; band < 32; band++) {
        float * a = mdct_out + (band - 1) * 18;  // previous band
        float * b = mdct_out + band * 18;        // current band
        for (int i = 0; i < 8; i++) {
            float u   = a[17 - i];
            float d   = b[i];
            a[17 - i] = u * mp3enc_cs[i] - d * mp3enc_ca[i];
            b[i]      = d * mp3enc_cs[i] + u * mp3enc_ca[i];
        }
    }
}

// Forward MDCT for short blocks: 3 windows of 12 samples -> 6 frequency lines each.
// The 3*6 = 18 lines are interleaved by window: [w0l0, w1l0, w2l0, w0l1, ...].
// This matches minimp3's L3_imdct_short which reads at stride 3.
//
// Window positioning within the 36-sample block (ISO 11172-3 Figure C.6):
//   Window 0: samples 6..17
//   Window 1: samples 12..23
//   Window 2: samples 18..29
//
// Short block window: sin(pi/12 * (n + 0.5)) for n = 0..11
static void mp3enc_mdct_short(const float * in, float * out) {
    // Window offsets within the 36-sample input
    static const int win_offset[3] = { 6, 12, 18 };

    for (int w = 0; w < 3; w++) {
        const float * x = in + win_offset[w];

        // Forward MDCT-12 with sin(pi/12*(n+0.5)) window
        // out[k] = (2/6) * sum(n=0..11) x[n] * sin(pi/12*(n+0.5)) * cos(pi/24*(2n+7)*(2k+1))
        for (int k = 0; k < 6; k++) {
            float sum = 0.0f;
            for (int n = 0; n < 12; n++) {
                float win     = sinf((float) M_PI / 12.0f * ((float) n + 0.5f));
                float cos_val = cosf((float) M_PI / 24.0f * (float) (2 * n + 7) * (float) (2 * k + 1));
                sum += x[n] * win * cos_val;
            }
            // Interleaved storage: line k of window w goes to position w + k*3.
            // Normalization: 2/N_half = 2/6 = 1/3 (standard MDCT normalization).
            out[w + k * 3] = sum * (1.0f / 9.0f);
        }
    }
}

// Process all 32 subbands for one granule.
// sb_samples layout: prev_gr[32][18] and cur_gr[32][18] (band major).
// mdct_out[576]: output frequency lines (32 subbands * 18 lines)
static void mp3enc_mdct_granule(const float sb_prev[32][18],
                                const float sb_cur[32][18],
                                float *     mdct_out,
                                int         block_type) {
    for (int band = 0; band < 32; band++) {
        // assemble 36 samples: prev[18] + cur[18]
        float mdct_in[36];
        for (int k = 0; k < 18; k++) {
            mdct_in[k]      = sb_prev[band][k];
            mdct_in[k + 18] = sb_cur[band][k];
        }

        if (block_type == 2) {
            // Short blocks: 3 windows of MDCT-12, interleaved output
            mp3enc_mdct_short(mdct_in, mdct_out + band * 18);
        } else {
            mp3enc_mdct36(mdct_in, mdct_out + band * 18);
        }
    }

    // Alias reduction: only for long blocks (ISO 11172-3, clause 2.4.3.4)
    if (block_type != 2) {
        mp3enc_alias_reduce(mdct_out);
    }
}
