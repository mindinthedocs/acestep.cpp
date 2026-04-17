#pragma once
// pipeline-synth.h: ACE-Step synthesis pipeline
//
// Three module categories:
//   Resident (loaded at init, kept for the pipeline's lifetime):
//     TextEnc, CondEnc, FSQ tok, FSQ detok, BPE. All small (< 1 GB total).
//   Ephemeral (loaded and freed inside a single op):
//     VAE encoder. Used twice per job (source + timbre), always brief.
//   Lazy-shared (loaded on first use, kept until the caller releases):
//     DiT, VAE decoder. Large, reused across every group of a batch.
//
// The ops self-load their lazy-shared modules (via ace_synth_dit_load /
// ace_synth_vae_load, both idempotent), so no module sits in VRAM outside
// its exclusive usage window. The caller unloads a module once it is done
// with it (typically: unload DiT between phase 1 and phase 2, unload the
// VAE decoder after phase 2).

#include "request.h"

#include <cstdlib>

struct AceSynth;
struct AceSynthJob;

struct AceSynthParams {
    const char * text_encoder_path;  // Qwen3 text encoder GGUF (required)
    const char * dit_path;           // DiT GGUF (required)
    const char * vae_path;           // VAE GGUF (required)
    const char * adapter_path;       // adapter safetensors or directory (NULL to disable)
    float        adapter_scale;      // user scale multiplier, 1.0 by default
    bool         use_fa;             // flash attention (default: true)
    bool         clamp_fp16;         // clamp hidden states to FP16 range (default: false)
    bool         use_batch_cfg;      // batch cond+uncond in one DiT forward (default: true)
    int          vae_chunk;          // latent frames per tile (default: 256)
    int          vae_overlap;        // overlap frames per side (default: 64)
    const char * dump_dir;           // intermediate tensor dump dir (NULL = disabled)
};

// Output audio buffer. Caller must free with ace_audio_free().
struct AceAudio {
    float * samples;      // planar stereo [L0..LN, R0..RN]
    int     n_samples;    // per channel
    int     sample_rate;  // always 48000
};

void ace_synth_default_params(AceSynthParams * p);

// Load the resident modules and cache DiT metadata. DiT and VAE weights are
// not allocated here; they are brought in lazily by the ops that need them.
// Returns NULL on failure.
AceSynth * ace_synth_load(const AceSynthParams * params);

// Lazy-shared module lifecycle. Both pairs are idempotent: load returns
// true if the module is already resident, unload is a no-op if absent.
// Normally called by the ops themselves; exposed for the orchestrator to
// unload a module between phases (so the next phase sees the full VRAM).
bool ace_synth_dit_load(AceSynth * ctx);
void ace_synth_dit_unload(AceSynth * ctx);
bool ace_synth_vae_load(AceSynth * ctx);
void ace_synth_vae_unload(AceSynth * ctx);

// Phase 1: encode sources, build context, run all DiT denoising steps.
// ops_dit_generate lazy-loads the DiT right before the denoising loop,
// so VAE-encoder, text and cond passes run with no DiT in VRAM. After
// the first group of a batch the DiT stays resident (load is idempotent)
// and subsequent groups reuse it. Call ace_synth_dit_unload when the
// batch is done.
// Produces a job that carries DiT latents in RAM for later VAE decoding.
// reqs[batch_n]: each request has its own caption, lyrics, metadata, audio_codes, and seed.
//   The first request (reqs[0]) is used for shared params (mode, duration, DiT settings).
//   seed must be resolved (non-negative) before calling this function.
// src_audio / ref_audio: interleaved stereo 48kHz buffers, NULL when not applicable.
// batch_n: number of requests (1..9).
// src_latents / src_latents_T: optional pre-encoded 25Hz x 64-ch cover latents
//   (T*64 f32). When non-null, bypasses VAE encode + FSQ; requires
//   task_type=cover-nofsq. Caller retains ownership.
// cancel/cancel_data: abort callback, polled between DiT steps. NULL = never cancel.
// Returns NULL on error or cancellation.
AceSynthJob * ace_synth_job_run_dit(AceSynth *         ctx,
                                    const AceRequest * reqs,
                                    const float *      src_audio,
                                    int                src_len,
                                    const float *      ref_audio,
                                    int                ref_len,
                                    int                batch_n,
                                    bool (*cancel)(void *)  = nullptr,
                                    void *        cancel_data     = nullptr,
                                    const float * src_latents     = nullptr,
                                    int           src_latents_T   = 0);

// Phase 2: VAE decode and waveform splice.
// ops_vae_decode_and_splice lazy-loads the decoder on first use and leaves
// it resident for the subsequent jobs of the same batch. Call
// ace_synth_vae_unload when done.
// splice_src / splice_len: interleaved stereo source reused for repaint/lego wave splicing.
//   Pass NULL when the job did not carry a source audio.
// out[batch_n] allocated by caller, filled with audio buffers.
// Returns 0 on success, -1 on error or cancellation.
int ace_synth_job_run_vae(AceSynth *    ctx,
                          AceSynthJob * job,
                          const float * splice_src,
                          int           splice_len,
                          AceAudio *    out,
                          bool (*cancel)(void *) = nullptr,
                          void * cancel_data     = nullptr);

// How many output slots the job expects. Equals the batch_n passed to run_dit.
int ace_synth_job_batch_n(const AceSynthJob * job);

void ace_synth_job_free(AceSynthJob * job);

void ace_audio_free(AceAudio * audio);

// Free the pipeline. Also releases any residual DiT or VAE still loaded.
void ace_synth_free(AceSynth * ctx);
