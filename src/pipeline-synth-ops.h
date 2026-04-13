#pragma once
// pipeline-synth-ops.h: ACE-Step synthesis pipeline, primitive operations
//
// Primitive operations: VAE encode/decode, DiT context building, text encoding,
// FSQ roundtrip, noise init, repaint splice. Called by the thin orchestrator.

#include "bpe.h"
#include "cond-enc.h"
#include "debug.h"
#include "dit-sampler.h"
#include "dit.h"
#include "fsq-detok.h"
#include "fsq-tok.h"
#include "philox.h"
#include "pipeline-synth.h"
#include "qwen3-enc.h"
#include "timer.h"
#include "vae-enc.h"
#include "vae.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// Internal state shared across all ops within one ace_synth_generate call.
// Declared here so ops functions can take it by reference.
struct SynthState {
    // model dimensions (from ctx)
    int Oc;
    int ctx_ch;

    // src audio encoding
    bool               have_cover;
    std::vector<float> cover_latents;        // [T_cover * 64] time-major VAE latents
    std::vector<float> noise_blend_latents;  // clean VAE latents for cover_noise_strength blending
    int                T_cover;

    // mode flags
    std::string task;
    bool        is_repaint;
    bool        is_lego_region;
    float       rs;
    float       re;
    bool        use_source_context;
    bool        have_codes;
    int         max_codes_len;

    // shared params (from reqs[0])
    AceRequest rr;
    float      duration;
    int        num_steps;
    float      guidance_scale;
    float      shift;

    // diffusion schedule
    std::vector<float> schedule;

    // SDE mode: inject fresh noise at each denoising step (vs ODE pure integration)
    bool use_sde;

    // per-batch seeds (for reproducible SDE re-noising: seed + step offset)
    std::vector<int64_t> seeds;

    // latent dimensions
    int T;
    int S;
    int enc_S;

    // region
    int repaint_t0;
    int repaint_t1;

    // repaint quality (resolved from repaint_strength via _resolve_repaint_config)
    float repaint_injection_ratio;
    int   repaint_crossfade_frames;
    float repaint_wav_cf_sec;

    // DiT instruction
    std::string instruction_str;     // main DiT instruction (set by orchestrator step 6)
    std::string nc_instruction_str;  // non-cover pass instruction (for cover_steps < 1.0 switching)

    // conditioning tensors
    std::vector<float> timbre_feats;
    int                S_ref_timbre;  // actual timbre sequence length (from VAE encode)

    // text encoding (per-batch)
    std::vector<std::vector<float>> per_enc;
    std::vector<int>                per_enc_S;
    std::vector<std::vector<float>> per_enc_nc;
    std::vector<int>                per_enc_S_nc;
    bool                            need_enc_switch;

    // stacked encoder hidden states
    int                max_enc_S;
    std::vector<float> enc_hidden;
    std::vector<float> enc_hidden_nc;
    std::vector<int>   per_enc_S_nc_final;
    std::vector<float> null_cond_vec;

    // DiT context
    std::vector<float> context;
    std::vector<float> context_silence;
    int                cover_steps;

    // noise + output
    std::vector<float> noise;
    std::vector<float> output;
    std::vector<float> repaint_src;
    std::vector<int>   per_S;

    // debug / timing
    DebugDumper dbg;
    Timer       timer;
};

// Encode src_audio -> cover_latents. Sets s.have_cover, s.cover_latents, s.T_cover.
int ops_encode_src(AceSynth * ctx, const float * src_audio, int src_len, SynthState & s);

// FSQ roundtrip on cover_latents (cover mode only).
void ops_fsq_roundtrip(AceSynth * ctx, SynthState & s);

// Savitzky-Golay 1D smoothing of source latents (cover-nofsq only). In-place on s.cover_latents.
void ops_filter_source_latents(AceSynth * ctx, SynthState & s);

// Resolve shared DiT params (steps, guidance, shift) and scan audio_codes.
int ops_resolve_params(AceSynth * ctx, const AceRequest * reqs, int batch_n, SynthState & s);

// Build flow-matching timestep schedule.
void ops_build_schedule(SynthState & s);

// Resolve T (latent frame count) and S (patch count).
int ops_resolve_T(AceSynth * ctx, SynthState & s);

// Clamp rs/re to source duration. Caller ensures a region task is active.
int ops_clamp_region(SynthState & s);

// Encode timbre from ref_audio via VAE. Sets s.timbre_feats and s.S_ref_timbre.
void ops_encode_timbre(AceSynth * ctx, const float * ref_audio, int ref_len, SynthState & s);

// Per-batch text + lyric encoding (main pass + optional non-cover pass).
// Stacks results into s.enc_hidden / s.enc_hidden_nc.
int ops_encode_text(AceSynth * ctx, const AceRequest * reqs, int batch_n, SynthState & s);

// Build DiT context tensor [batch_n, T, ctx_ch] = src_latents(64) | mask(64).
int ops_build_context(AceSynth * ctx, const AceRequest * reqs, int batch_n, SynthState & s);

// Build silence context + cover_steps for audio_cover_strength switching.
void ops_build_context_silence(AceSynth * ctx, int batch_n, SynthState & s);

// Initialise noise tensor (Philox) + cover noise blend + per_S + repaint_src buffer.
void ops_init_noise_and_repaint(AceSynth * ctx, const AceRequest * reqs, int batch_n, SynthState & s);

// Run the DiT denoising loop.
int ops_dit_generate(AceSynth * ctx, int batch_n, SynthState & s, bool (*cancel)(void *), void * cancel_data);

// VAE decode all batch items + waveform splice for repaint/lego regions.
// src_audio is the original interleaved PCM used for splice.
// Returns 0 on success, -1 on error/cancel.
int ops_vae_decode_and_splice(AceSynth *    ctx,
                              int           batch_n,
                              AceAudio *    out,
                              SynthState &  s,
                              const float * src_audio,
                              int           src_len,
                              bool (*cancel)(void *),
                              void * cancel_data);
