#pragma once
#include <gmpxx.h>
#include <seal/plaintext.h>

#include "pegasus/lwe.h"

enum class NTTDir {
  ToNTT,
  FromNTT,
};

namespace rlwe {

constexpr double Lvl0NoiseStddev = 1 << 10;
constexpr double Lvl1NoiseStddev = 1 << 10;
constexpr double Lvl2NoiseStddev = 3.20;

namespace math {

/// acc += op0* op1 using double-width precision
/// |acc| = 2 * |op0| = 2 * |op1|
void FMAU128(uint64_t *acc, const uint64_t *op0, const uint64_t *op1,
             const size_t len);

void ModSwtich(uint64_t *input, size_t n, uint64_t p, int w);

void ModRescale(uint64_t *input, size_t n, uint64_t p, int w);

struct FastMulMod {
  uint64_t cnst, p;
  uint64_t cnst_shoup;

  explicit FastMulMod(uint64_t cnst, uint64_t p);

  inline uint64_t operator()(uint64_t x) const {
    uint64_t t = lazy(x);
    return t - ((p & -static_cast<uint64_t>(t < p)) ^ p);
  }

  uint64_t lazy(uint64_t x) const;
};

}  // namespace math

using SK = seal::SecretKey;
using PK = seal::PublicKey;
using Ctx = seal::Ciphertext;
using Ptx = seal::Plaintext;
using GSWCt = seal::KSwitchKeys;
using BK = std::vector<GSWCt>;
using RepackKey = GSWCt;

// Used to swtich LWE(s_N, N, p) tto LWE(s_n, n, p)
typedef struct DecomposedLWEKSwitchKey_st {
  uint32_t decompose_base_;
  uint32_t ndigits_;        // w^D ~ p0
  std::vector<Ctx> parts_;  // ceil(N/n) * D
} DecomposedLWEKSwitchKey_t[1];

typedef struct RLWE2LWECt_st {
  rlwe::Ptx a;
  std::vector<uint64_t> b;
  double scale = 1.;
} RLWE2LWECt_t[1];

void GSW_Init(GSWCt &gsw, size_t nNormalPrimes, bool halfKey = false);

size_t GSW_NRows(GSWCt const &gsw);
const Ctx &GSW_GetRow(GSWCt const &gsw, size_t rowIdx);
Ctx &GSW_GetRow(GSWCt &gsw, size_t rowIdx);

template <size_t ctIdx>
uint64_t *GSW_GetModuli(GSWCt &gsw, size_t rowIdx, size_t modIdx);
template <size_t ctIdx>
const uint64_t *GSW_GetModuli(GSWCt const &gsw, size_t rowIdx, size_t modIdx);

size_t GetNNormalPrimes(RtPtr const rt);
size_t GetNSpecialPrimes(RtPtr const rt);

bool GSWEncrypt(GSWCt &ct, int64_t m, bool halfKey, SK const &sk,
                RtPtr const rt, double stddev = 3.20);

bool GSWEncrypt(GSWCt &ct, SK const &lwe_sk, SK const &sk, RtPtr const rt,
                double stddev = 3.20);

bool GSWEncrypt(GSWCt &ct, const Ptx &non_ntt_msg, size_t nmoduli, bool halfKey,
                SK const &sk, RtPtr const rt, double stddev = 3.20);

void ExternalProduct(Ctx &ct, GSWCt const &gsw, RtPtr const rt);

void ExternalProduct(Ctx &ct, const Ctx *ct_array, const int ndigits,
                     const uint32_t decompose_base, RtPtr const rt);

void BKInit(BK &bk, SK const &lwe_sk, SK const &rlwe_sk, const RtPtr rt);

std::vector<int64_t> LWE_SymDec(const RLWE2LWECt_t lwe_ct,
                                const rlwe::Ptx &sk_non_ntt, const RtPtr rt);

void SampleExtract(RLWE2LWECt_t lwe_ct, const rlwe::Ctx &rlwe_ct,
                   size_t extract_index, const RtPtr rlwe_rt);

void SampleExtract(RLWE2LWECt_st *lwe_ct, const rlwe::Ctx &rlwe_ct,
                   std::vector<size_t> const &extract_index,
                   const RtPtr rlwe_rt);

// RLWE(m; n, p0) -> LWE(m mod p0; n)
void SampleExtract(lwe::Ctx_t lwe_n, const rlwe::Ctx &rlwe_n,
                   size_t extract_index, const RtPtr lwe_rt_mod_p0);

// Encrypt rlwe_sk using lwe_sk
void LWEKSKeyInit(DecomposedLWEKSwitchKey_t key, uint32_t decompose_base,
                  const rlwe::Ptx &rlwe_sk_non_ntt,
                  const seal::SecretKey &lwe_sk_ntt, const RtPtr lwe_rt_mod_p0,
                  const RtPtr rlwe_rt);

// LWE(m; N, q0) -> LWE(m; n, 2^w)
void LWEKeySwitch(lwe::Ctx_t lwe_n, const RLWE2LWECt_t lwe_N,
                  const DecomposedLWEKSwitchKey_t key,
                  const RtPtr lwe_rt_mod_p0);

bool IsCtxComponentZero(Ctx const &ct, size_t idx);

bool SwitchNTTForm(uint64_t *poly, NTTDir dir, int nModuli, RtPtr const rt);

bool SwitchNTTForm(Ctx &ct, RtPtr const rt);

};  // namespace rlwe
