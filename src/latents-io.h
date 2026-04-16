#pragma once
// latents-io.h: parse/serialize the 25Hz x 64-channel pre-FSQ latent tensor.
// Binary format mirrors DebugDumper::debug_dump_2d (src/debug.h:62-65):
//   [int32 ndims=2][int32 T][int32 channels=64][float32 T*64]

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

static inline bool latents_parse(const uint8_t * data, size_t size, std::vector<float> & out, int & T) {
    if (!data || size < 12) {
        return false;
    }
    int32_t ndims, d0, d1;
    std::memcpy(&ndims, data + 0, 4);
    std::memcpy(&d0, data + 4, 4);
    std::memcpy(&d1, data + 8, 4);
    if (ndims != 2 || d0 <= 0 || d1 != 64) {
        return false;
    }
    size_t expected = 12 + (size_t) d0 * 64 * sizeof(float);
    if (size != expected) {
        return false;
    }
    T = d0;
    out.resize((size_t) d0 * 64);
    std::memcpy(out.data(), data + 12, out.size() * sizeof(float));
    return true;
}

static inline void latents_serialize(const float * data, int T, std::string & out_body) {
    int32_t ndims = 2, d0 = T, d1 = 64;
    size_t payload = (size_t) T * 64 * sizeof(float);
    out_body.resize(12 + payload);
    std::memcpy(&out_body[0], &ndims, 4);
    std::memcpy(&out_body[4], &d0, 4);
    std::memcpy(&out_body[8], &d1, 4);
    std::memcpy(&out_body[12], data, payload);
}
