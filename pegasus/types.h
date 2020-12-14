#pragma once
#include <complex>
#include <vector>

#include <seal/ciphertext.h>
#include <seal/plaintext.h>
#include <seal/seal.h>

namespace gemini {

/// Ciphertext
using Ctx    = seal::Ciphertext;
/// Plaintext
using Ptx    = seal::Plaintext;
using DecKey = seal::Decryptor;
using F64    = double;
using C64    = std::complex<double>;
using U64    = uint64_t;
using I64    = int64_t;
using F64Vec = std::vector<F64>;
using C64Vec = std::vector<C64>;

size_t GetNModuli(Ctx ct);

struct Ft64 {
    static constexpr F64 CMP_EPSILON = 1e-9;

    static bool AlmostEquals(F64 const &a, F64 const &b) { return seal::util::are_close(a, b); }

    static bool AlmostEquals(F64 const &a, F64 const &b, const F64 &e) { return std::abs(a - b) < e; }
};

}  // namespace gemini

