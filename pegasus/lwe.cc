#include "pegasus/lwe.h"

#include <gmp.h>
#include <seal/context.h>
#include <seal/randomgen.h>
#include <seal/randomtostd.h>
#include <seal/secretkey.h>
#include <seal/util/clipnormal.h>
#include <seal/util/uintarithsmallmod.h>

#include <random>

namespace lwe {

void GenerateHammingSecretKey(seal::SecretKey& sk, int hwt, bool is_ntt,
                              const RtPtr context) {
  using namespace seal;
  auto& context_data = *context->key_context_data();
  auto& parms = context_data.parms();
  auto& coeff_modulus = parms.coeff_modulus();
  size_t coeff_count = parms.poly_modulus_degree();
  size_t coeff_modulus_size = coeff_modulus.size();

  sk.data().resize(util::mul_safe(coeff_count, coeff_modulus_size));

  std::shared_ptr<UniformRandomGenerator> rng(
      parms.random_generator()->create());

  if (hwt >= coeff_count || hwt == 0)
    throw std::logic_error("sample_poly_hwt: hwt out of bound");

  RandomToStandardAdapter engine(rng);
  /* reservoir sampling */
  std::vector<int> picked(hwt);
  std::iota(picked.begin(), picked.end(), 0);

  for (size_t k = hwt; k < coeff_count; ++k) {
    std::uniform_int_distribution<size_t> dist(0, k - 1);
    size_t pos = dist(engine);  // uniform in [0, k)
    if (pos < hwt) picked[pos] = k;
  }

  /* For the picked poistions, sample {-1, 1} uniformly at random */
  std::vector<bool> rnd2(hwt);
  std::uniform_int_distribution<int> dist(0, 1);
  for (size_t i = 0; i < hwt; ++i) {
    rnd2[i] = dist(engine);
  }

  std::sort(picked.begin(), picked.end());
  uint64_t* dst_ptr = sk.data().data();
  for (size_t j = 0; j < coeff_modulus_size; j++) {
    const uint64_t neg_one = coeff_modulus[j].value() - 1;
    const uint64_t xor_one = neg_one ^ 1UL;
    std::memset(dst_ptr, 0, sizeof(*dst_ptr) * coeff_count);

    for (size_t i = 0; i < hwt; ++i) {
      dst_ptr[picked[i]] = [xor_one, neg_one](bool b) {
        return 1;
        // b = true -> c = 0xFF -> one
        // b = false -> c = 0x00 -> neg_one
        // uint64_t c = -static_cast<uint64_t>(b);
        // return (xor_one & c) ^ neg_one;
      }(rnd2[i]);
    }

    dst_ptr += coeff_count;
  }

  if (is_ntt) {
    const auto ntt_tables = context_data.small_ntt_tables();
    util::RNSIter secret_key(sk.data().data(), coeff_count);
    util::ntt_negacyclic_harvey(secret_key, coeff_modulus_size, ntt_tables);
  }

  // Set the parms_id for secret key
  sk.parms_id() = context_data.parms_id();

  for (size_t i = 0; i < hwt; ++i) {
    rnd2[i] = 0;
  }
  std::memset(picked.data(), 0, sizeof(picked[0]) * picked.size());
}

void SKInit(seal::SecretKey& sk_ntt, seal::SecretKey &sk_non_ntt, int hwt, const RtPtr rt) {
  using namespace seal;
  auto working_context = rt->key_context_data();
  if (!working_context) {
    throw std::runtime_error("SKInit: context nullptr");
  }
  const auto ntt_tables = working_context->small_ntt_tables();
  const size_t degree = working_context->parms().poly_modulus_degree();
  const size_t nmoduli = working_context->parms().coeff_modulus().size();
  if (degree != params::n()) {
    throw std::invalid_argument("SKInit: invalid runtime meta");
  }
  if (hwt < 0) {
    throw std::invalid_argument("SKInit: hwt < 0");
  }

  GenerateHammingSecretKey(sk_ntt, hwt, /*is_ntt*/ false, rt);
  sk_non_ntt.data().resize(degree);
  // copy the 1-st moduli to sk_non_ntt
  std::copy_n(sk_ntt.data().data(), degree, sk_non_ntt.data().data());
  // convert to ntt-form
  util::RNSIter secret_key(sk_ntt.data().data(), degree);
  util::ntt_negacyclic_harvey(secret_key, nmoduli, ntt_tables);
}

double SymDec(const Ctx_t ct, const seal::SecretKey& sk_non_ntt, const RtPtr rt) {
  const auto& modulus = rt->last_context_data()->parms().coeff_modulus().front();
  T half = modulus.value() >> 1;

  const T* sk_data = sk_non_ntt.data().data();
  const T* ct_data = CtData(ct);
  T acc = seal::util::dot_product_mod(sk_data, ct_data, params::n(), modulus);
  acc = seal::util::add_uint_mod(acc, ct_data[params::n()], modulus);

  sT v = acc;
  if (acc >= half) v -= modulus.value();
  return static_cast<double>(v / ct->scale);
}

const T* CtData(const Ctx_t ct) { return ct->data_.data(); }
T* CtData(Ctx_t ct) { return ct->data_.data(); }
size_t CtLen(const Ctx_t ct) { return ct->data_.size(); }
const T* CtData(const Ctx_st &ct) { return ct.data_.data(); }
size_t CtLen(const Ctx_st &ct) { return ct.data_.size(); }
T* CtData(Ctx_st &ct) { return ct.data_.data(); }

const T* SKData(seal::SecretKey const& sk) { return sk.data().data(); }

}  // namespace lwe
