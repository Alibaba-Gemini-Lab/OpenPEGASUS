#pragma once

#include <array>
#include <cassert>
#include <memory>
#include <vector>

namespace seal {
class SEALContext;
class SecretKey;
class PublicKey;
class Ciphertext;
class Plaintext;
class KSwitchKeys;
class Modulus;
}  // namespace seal

using RtPtr = std::shared_ptr<seal::SEALContext>;

namespace lwe {
using T = uint64_t;
using sT = int64_t;
using gT = __uint128_t;

struct params {
  static constexpr int n() { return 1024; }
};

typedef struct Ctx_st {
  std::array<T, params::n() + 1> data_;
  double scale;
} Ctx_t[1];

void SKInit(seal::SecretKey& sk_ntt, seal::SecretKey& sk_non_ntt, int hwt,
            const RtPtr rt);
double SymDec(const Ctx_t ct, const seal::SecretKey& sk_non_ntt,
              const RtPtr rt);
void GenerateHammingSecretKey(seal::SecretKey& sk, int hwt, bool is_ntt,
                              const RtPtr rt);

const T* CtData(const Ctx_t ct);
const T* CtData(const Ctx_st& ct);
T* CtData(Ctx_t ct);
T* CtData(Ctx_st& ct);
size_t CtLen(Ctx_t ct);
size_t CtLen(const Ctx_st& ct);

const T* SKData(const seal::SecretKey& sk);

}  // namespace lwe
