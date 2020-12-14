#include "pegasus/rlwe.h"

#include <seal/seal.h>
#include <seal/util/clipnormal.h>
#include <seal/util/polyarithsmallmod.h>
#include <seal/util/rlwe.h>
#include <seal/valcheck.h>

namespace rlwe {
namespace math {

FastMulMod::FastMulMod(uint64_t cnst, uint64_t p) : cnst(cnst), p(p) {
  uint64_t cnst_128[2]{0, cnst};
  uint64_t shoup[2];
  seal::util::divide_uint128_inplace(cnst_128, p, shoup);
  cnst_shoup = shoup[0];  // cnst_shoup = cnst * 2^64 / p
}

uint64_t FastMulMod::lazy(uint64_t x) const {
  unsigned long long hw64;
  seal::util::multiply_uint64_hw64(x, cnst_shoup, &hw64);
  std::uint64_t q = static_cast<std::uint64_t>(hw64) * p;
  return (x * cnst - q);
}

/// acc += op0 * op1
inline void FMAU128(uint64_t *acc, const uint64_t *op0, const uint64_t *op1,
                    const size_t len) {
  unsigned long long wide[2];
  for (size_t i = 0; i < len; ++i, acc += 2) {
    seal::util::multiply_uint64(*op0++, *op1++, wide);
    uint64_t _wp[2] = {static_cast<uint64_t>(wide[0]),
                       static_cast<uint64_t>(wide[1])};
    auto carry = seal::util::add_uint(acc, _wp, 2, acc);
    if (carry > 0) {
      std::cerr << " FMAU128: overflow!\n";
    }
  }
}

inline uint64_t barrett_reduce_64_lazy(const uint64_t input,
                                       const seal::Modulus &modulus) {
#ifndef NDEBUG
  if (modulus.is_zero()) {
    throw std::invalid_argument("modulus");
  }
  if (input >> 63) {
    throw std::invalid_argument("input");
  }
#endif
  // Reduces input using base 2^64 Barrett reduction
  // input must be at most 63 bits

  unsigned long long tmp[2];
  const uint64_t *const_ratio = modulus.const_ratio().data();
  seal::util::multiply_uint64(input, const_ratio[1], tmp);

  // Barrett subtraction
  return input - tmp[1] * modulus.value();
}

inline uint64_t barrett_reduce_128_lazy(const uint64_t *input,
                                        const seal::Modulus &modulus) {
  using namespace seal::util;
#ifndef NDEBUG
  if (!input) {
    throw std::invalid_argument("input");
  }
  if (modulus.is_zero()) {
    throw std::invalid_argument("modulus");
  }
#endif
  unsigned long long tmp1, tmp2[2], tmp3, carry;
  const uint64_t *const_ratio = modulus.const_ratio().data();

  // Multiply input and const_ratio
  // Round 1
  multiply_uint64_hw64(input[0], const_ratio[0], &carry);

  multiply_uint64(input[0], const_ratio[1], tmp2);
  tmp3 = tmp2[1] + add_uint64(tmp2[0], carry, 0, &tmp1);

  // Round 2
  multiply_uint64(input[1], const_ratio[0], tmp2);
  carry = tmp2[1] + add_uint64(tmp1, tmp2[0], 0, &tmp1);

  // This is all we care about
  tmp1 = input[1] * const_ratio[1] + tmp3 + carry;

  // Barrett subtraction
  return input[0] - tmp1 * modulus.value();
}

}  // namespace math

bool SwitchNTTForm(uint64_t *poly, NTTDir dir, int nModuli, RtPtr const rt) {
  using namespace seal;
  using namespace seal::util;
  auto cntxt = rt->first_context_data();
  const auto &parms = cntxt->parms();
  const NTTTables *nttTables = cntxt->small_ntt_tables();
  const size_t degree = parms.poly_modulus_degree();

  RNSIter poly_iter(poly, degree);
  switch (dir) {
    case NTTDir::ToNTT:
      SEAL_ITERATE(iter(poly_iter, nttTables), nModuli,
                   [](auto I) { ntt_negacyclic_harvey(get<0>(I), get<1>(I)); });
      break;
    case NTTDir::FromNTT:
      SEAL_ITERATE(iter(poly_iter, nttTables), nModuli, [](auto I) {
        inverse_ntt_negacyclic_harvey(get<0>(I), get<1>(I));
      });
      break;
  }

  return true;
}

bool SwitchNTTForm(Ctx &ct, RtPtr const rt) {
  if (ct.size() != 2) {
    std::cerr << "SwitchNTTForm: require |ct| = 2" << std::endl;
    return false;
  }

  const size_t nModuli = ct.coeff_modulus_size();
  NTTDir dir = ct.is_ntt_form() ? NTTDir::FromNTT : NTTDir::ToNTT;
  SwitchNTTForm(ct.data(0), dir, nModuli, rt);
  SwitchNTTForm(ct.data(1), dir, nModuli, rt);

  ct.is_ntt_form() = !ct.is_ntt_form();

  return true;
}

size_t GetNNormalPrimes(RtPtr const rt) {
  return rt->first_context_data()->parms().coeff_modulus().size();
}

size_t GetNSpecialPrimes(RtPtr const rt) {
  size_t n = rt->key_context_data()->parms().coeff_modulus().size();
  return n - GetNNormalPrimes(rt);
}

void GSW_Init(GSWCt &gsw, size_t nNormalPrimes, bool halfKey) {
  if (nNormalPrimes == 0) {
    throw std::logic_error("GSW_Init: empty moduli");
  }
  gsw.data().resize(1);
  gsw.data()[0].resize(halfKey ? nNormalPrimes : 2 * nNormalPrimes);
}

size_t GSW_NRows(GSWCt const &gsw) { return gsw.size(); }

const Ctx &GSW_GetRow(GSWCt const &gsw, size_t rowIdx) {
  if (gsw.size() != 1) throw std::logic_error("GSW_GetRow: invalid gsw ct");
  const std::vector<seal::PublicKey> &rows = gsw.data(0);
  if (rows.size() <= rowIdx)
    throw std::logic_error("GSW_GetRow: row index out-of-bound");
  return rows[rowIdx].data();
}

Ctx &GSW_GetRow(GSWCt &gsw, size_t rowIdx) {
  if (gsw.size() != 1) throw std::logic_error("GSW_GetRow: invalid gsw ct");
  std::vector<seal::PublicKey> &rows = gsw.data(0);
  if (rows.size() <= rowIdx)
    throw std::logic_error("GSW_GetRow: row index out-of-bound");
  return rows[rowIdx].data();
}

uint64_t *Ct_GetModuli(Ctx &ct, size_t ctIdx, size_t modIdx) {
  if (modIdx >= ct.coeff_modulus_size())
    throw std::logic_error("Ct_GetModuli: modIdx out-of-bound");
  if (ctIdx >= ct.size())
    throw std::logic_error("Ct_GetModuli: ctIdx out-of-bound");
  return ct.data(ctIdx) + modIdx * ct.poly_modulus_degree();
}

const uint64_t *Ct_GetModuli(const Ctx &ct, size_t ctIdx, size_t modIdx) {
  if (modIdx >= ct.coeff_modulus_size())
    throw std::logic_error("Ct_GetModuli: modIdx out-of-bound");
  if (ctIdx >= ct.size())
    throw std::logic_error("Ct_GetModuli: ctIdx out-of-bound");
  return ct.data(ctIdx) + modIdx * ct.poly_modulus_degree();
}

template <size_t ctIdx>
uint64_t *GSW_GetModuli(GSWCt &gsw, size_t rowIdx, size_t modIdx) {
  Ctx &row = GSW_GetRow(gsw, rowIdx);
  return Ct_GetModuli(row, ctIdx, modIdx);
}

template <size_t ctIdx>
const uint64_t *GSW_GetModuli(const GSWCt &gsw, size_t rowIdx, size_t modIdx) {
  const Ctx &row = GSW_GetRow(gsw, rowIdx);
  return Ct_GetModuli(row, ctIdx, modIdx);
}

static void sample_poly_normal(
    std::shared_ptr<seal::UniformRandomGenerator> rng,
    const seal::EncryptionParameters &parms, const double stddev,
    uint64_t *destination) {
  using namespace seal::util;
  auto coeff_modulus = parms.coeff_modulus();
  size_t coeff_modulus_size = coeff_modulus.size();
  size_t coeff_count = parms.poly_modulus_degree();

  if (are_close(stddev, 0.0)) {
    set_zero_poly(coeff_count, coeff_modulus_size, destination);
    return;
  }

  seal::RandomToStandardAdapter engine(rng);
  ClippedNormalDistribution dist(
      0, stddev,
      global_variables::noise_distribution_width_multiplier * stddev);

  for (size_t i = 0; i < coeff_count; i++) {
    int64_t noise = static_cast<int64_t>(dist(engine));
    if (noise > 0) {
      for (size_t j = 0; j < coeff_modulus_size; j++) {
        destination[i + j * coeff_count] = static_cast<uint64_t>(noise);
      }
    } else if (noise < 0) {
      noise = -noise;
      for (size_t j = 0; j < coeff_modulus_size; j++) {
        destination[i + j * coeff_count] =
            coeff_modulus[j].value() - static_cast<uint64_t>(noise);
      }
    } else {
      for (size_t j = 0; j < coeff_modulus_size; j++) {
        destination[i + j * coeff_count] = 0;
      }
    }
  }
}

static void _do_encrypt_symmetric(SK const &secret_key, const RtPtr rt,
                                  double stddev, Ctx &destination) {
  using namespace seal::util;
  using namespace std;
  // We use a fresh memory pool with `clear_on_destruction' enabled.
  seal::MemoryPoolHandle pool =
      seal::MemoryManager::GetPool(seal::mm_prof_opt::FORCE_NEW, true);
  auto parms_id = rt->key_context_data()->parms_id();

  auto &context_data = *rt->get_context_data(parms_id);
  auto &parms = context_data.parms();
  auto &coeff_modulus = parms.coeff_modulus();
  size_t coeff_modulus_size = coeff_modulus.size();
  size_t coeff_count = parms.poly_modulus_degree();
  auto ntt_tables = context_data.small_ntt_tables();
  size_t encrypted_size = 2;

  destination.resize(rt, parms_id, encrypted_size);
  destination.is_ntt_form() = true;
  destination.scale() = 1.0;

  auto rng_error = parms.random_generator()->create();
  shared_ptr<seal::UniformRandomGenerator> rng_ciphertext;
  rng_ciphertext = seal::BlakePRNGFactory().create();

  // Generate ciphertext: (c[0], c[1]) = ([-(as+e)]_q, a)
  uint64_t *c0 = destination.data(0);
  uint64_t *c1 = destination.data(1);

  // Sample a uniformly at random
  // sample the NTT form directly
  sample_poly_uniform(rng_ciphertext, parms, c1);

  // Sample e <-- chi
  auto noise(allocate_poly(coeff_count, coeff_modulus_size, pool));
  sample_poly_normal(rng_error, parms, stddev, noise.get());

  // calculate -(a*s + e) (mod q) and store in c[0]
  for (size_t i = 0; i < coeff_modulus_size; i++) {
    dyadic_product_coeffmod(secret_key.data().data() + i * coeff_count,
                            c1 + i * coeff_count, coeff_count, coeff_modulus[i],
                            c0 + i * coeff_count);
    // Transform the noise e into NTT representation.
    ntt_negacyclic_harvey(noise.get() + i * coeff_count, ntt_tables[i]);

    add_poly_coeffmod(noise.get() + i * coeff_count, c0 + i * coeff_count,
                      coeff_count, coeff_modulus[i], c0 + i * coeff_count);
    negate_poly_coeffmod(c0 + i * coeff_count, coeff_count, coeff_modulus[i],
                         c0 + i * coeff_count);
  }

  destination.parms_id() = parms_id;
  destination.is_ntt_form() = true;
}

static bool _do_GSWEncrypt(
    GSWCt &ct, const uint64_t *non_ntt_poly, size_t nmoduli, bool halfKey,
    SK const &sk, RtPtr const rt,
    double stddev = seal::util::global_variables::noise_standard_deviation) {
  using namespace seal;
  using namespace seal::util;
  if (!is_metadata_valid_for(sk, rt)) {
    throw std::runtime_error("GSWEncrypt: invalid sk and runtime");
  }
  auto key_cntxt = rt->key_context_data();
  auto ct_cntxt = rt->first_context_data();
  const auto &parms = key_cntxt->parms();

  const auto &nttTables = key_cntxt->small_ntt_tables();
  const size_t degree = parms.poly_modulus_degree();
  const size_t nNormal = ct_cntxt->parms().coeff_modulus().size();
  const size_t nKeyRow = halfKey ? nNormal : 2 * nNormal;

  GSW_Init(ct, nNormal, halfKey);
  for (size_t j = 0; j < nKeyRow; ++j) {
    // Generate ciphertext: (c[0], c[1]) = ([-(as+e)]_q, a)
    // phase = c[0] + c[1] * s
    if (are_close(stddev, global_variables::noise_standard_deviation)) {
      encrypt_zero_symmetric(sk, rt, key_cntxt->parms_id(), /*ntt*/ true,
                             /*saveseed*/ false, GSW_GetRow(ct, j));
    } else {
      _do_encrypt_symmetric(sk, rt, stddev, GSW_GetRow(ct, j));
    }
  }

  uint64_t qk = parms.coeff_modulus().back().value();

  for (size_t j = 0; j < nNormal; ++j) {
    std::vector<uint64_t> temp_poly(degree, 0);

    const auto &qj = nttTables[j].modulus();
    // qk*m mod qj
    std::transform(non_ntt_poly, non_ntt_poly + degree, temp_poly.data(),
                   [&qj, qk](uint64_t um) {
                     int64_t m = um;
                     while (m < 0) {
                       m += qj.value();
                     }
                     return util::multiply_uint_mod(m, qk, qj);
                   });

    util::ntt_negacyclic_harvey(temp_poly.data(), nttTables[j]);
    // Each row of the GSW is a RLWE cipher

    // First L-rows are ciphers of \hat{qi} * \hat{qi}^(-1) * qk * m
    uint64_t *poly0_mod_qj = GSW_GetModuli<0>(ct, j, j);
    add_poly_coeffmod(poly0_mod_qj, temp_poly.data(), degree, qj, poly0_mod_qj);

    if (!halfKey) {
      // Second L-rows are ciphers of \hat{qi} * \hat{qi}^(-1) * qk * m
      uint64_t *poly1_mod_qj = GSW_GetModuli<1>(ct, nNormal + j, j);
      add_poly_coeffmod(poly1_mod_qj, temp_poly.data(), degree, qj,
                        poly1_mod_qj);
    }
    if (nmoduli == nNormal) non_ntt_poly += degree;
  }

  ct.parms_id() = key_cntxt->parms_id();
  return true;
}

bool GSWEncrypt(GSWCt &ct, const Ptx &non_ntt_msg, const size_t nmoduli,
                bool halfKey, SK const &sk, RtPtr const rt, double stddev) {
  size_t n = non_ntt_msg.coeff_count() / nmoduli;
  if (non_ntt_msg.is_ntt_form()) {
    throw std::logic_error("Error: require non_ntt msg");
  }

  if (n != rt->first_context_data()->parms().poly_modulus_degree()) {
    throw std::logic_error("Error: msg.length != RLWE.N");
    return false;
  }

  return _do_GSWEncrypt(ct, non_ntt_msg.data(), nmoduli, halfKey, sk, rt,
                        stddev);
}

/// GSW(m) := [Enc(0), Enc(0), ..., Enc(0)] + [([m]_q0, 0), ([m]_q1, 0), ...,
/// ([m]_qL, 0),
///                                            (0, [m]_q0), (0, [m]_q1), ...,
///                                            (0, [m]_qL)]
/// 2*L rows, each L + 1 moduli
bool GSWEncrypt(GSWCt &ct, int64_t m, bool halfKey, SK const &sk,
                RtPtr const rt, double stddev) {
  const size_t degree = rt->key_context_data()->parms().poly_modulus_degree();
  std::vector<uint64_t> temp_poly(degree, 0);
  temp_poly[0] = m;
  return _do_GSWEncrypt(ct, temp_poly.data(), 1, halfKey, sk, rt, stddev);
}

bool IsCtxComponentZero(Ctx const &ct, size_t idx) {
  if (idx >= ct.size()) {
    return false;
  }

  const size_t degree = ct.poly_modulus_degree();
  const size_t nCtModuli = ct.coeff_modulus_size();
  return !std::any_of(ct.data(idx), ct.data(idx) + nCtModuli * degree,
                      [](auto c) { return c != 0; });
}

void ExternalProduct(Ctx &ct, const Ctx *ct_array, const int ndigits,
                     const uint32_t decompose_base, RtPtr const rt) {
  using namespace seal;
  if (ct.size() != 2) {
    throw std::invalid_argument("ExternalProduct: require |ct| = 2");
  }

  if (!is_metadata_valid_for(ct, rt)) {
    throw std::invalid_argument("ExternalProduct: invalid runtime");
  }

  if (ct.is_ntt_form()) {
    throw std::invalid_argument("ExternalProduct: require non_ntt ct");
  }

  auto key_cntxt = rt->key_context_data();
  auto ct_cntxt = rt->first_context_data();
  const int n_ct_components = IsCtxComponentZero(ct, 1) ? 1 : 2;
  const size_t degree = ct_cntxt->parms().poly_modulus_degree();
  const size_t nCtModuli = ct.coeff_modulus_size();
  const size_t nSpecials = GetNSpecialPrimes(rt);

  if (nSpecials != 0) {
    throw std::invalid_argument(
        "ExternalProduct: require #special = 0 for decomposed gsw");
  }

  if (nCtModuli != 1) {
    throw std::invalid_argument(
        "ExternalProduct: require #moduli = 1 for decomposed gsw");
  }

  if (n_ct_components != 1) {
    throw std::invalid_argument(
        "ExternalProduct: require n_ct_components = 1 for decomposed gsw");
  }

  const uint64_t w = (1ULL << decompose_base);
  const uint64_t w_mod = w - 1;

  Ctx out;
  seal::Evaluator evaluator(rt);

  Ptx decomposed;
  decomposed.resize(degree);
  decomposed.parms_id() = ct_cntxt->parms_id();

  // \sum_{k} r_k * RLWE(w^k * s)
  size_t shift = 0;
  for (size_t k = 0; k < ndigits; ++k, shift += decompose_base) {
    std::transform(ct.data(0), ct.data(0) + degree, decomposed.data(),
                   [w_mod, shift](uint64_t c) -> uint64_t {
                     return (c >> shift) & w_mod;
                   });

    SwitchNTTForm(decomposed.data(), NTTDir::ToNTT, 1, rt);

    auto ct_copy{ct_array[k]};
    evaluator.multiply_plain_inplace(ct_copy, decomposed);

    if (out.size() > 0) {
      evaluator.add_inplace(out, ct_copy);
    } else {
      out = ct_copy;
    }
  }

  ct = out;
  ct.is_ntt_form() = true;
}

void ExternalProduct(Ctx &ct, GSWCt const &gsw, RtPtr const rt) {
  using namespace seal;
  if (ct.size() != 2) {
    throw std::invalid_argument("ExternalProduct: require |ct| = 2");
  }

  if (!is_metadata_valid_for(ct, rt)) {
    throw std::invalid_argument("ExternalProduct: invalid runtime");
  }

  if (ct.is_ntt_form()) {
    throw std::invalid_argument("ExternalProduct: require non_ntt ct");
  }

  auto mem_pool = MemoryManager::GetPool();
  auto key_cntxt = rt->key_context_data();
  auto ct_cntxt = rt->get_context_data(ct.parms_id());

  const auto &nttTables = key_cntxt->small_ntt_tables();
  const size_t degree = ct_cntxt->parms().poly_modulus_degree();
  const size_t nCtModuli = ct.coeff_modulus_size();
  const size_t nSpecials = GetNSpecialPrimes(rt);
  const size_t nMaxCtModuli =
      rt->first_context_data()->parms().coeff_modulus().size();
  const int n_ct_components = IsCtxComponentZero(ct, 1) ? 1 : 2;

  if (nSpecials != 1) {
    throw std::invalid_argument("ExternalProduct: require #special = 1");
  }

  /// [ct'[0]]_{qj} <- \sum_{i} [gsw[i, 0]]_{qj} * [ct[0]]_{qi} mod qj +
  /// \sum_{i} [gsw[L+i, 0]]_{qj} * [ct[1]]_{qi} [ct'[1]]_{qj} <- \sum_{i}
  /// [gsw[i, 1]]_{qj} * [ct[0]]_{qi} mod qj + \sum_{i} [gsw[L+i, 1]]_{qj} *
  /// [ct[1]]_{qi} qj includes special primes(s), qi only loop over cipher
  /// moduli.

  util::Pointer<uint64_t> lazy_mul_poly[2] = {
      util::allocate_poly(degree * 2, 1, mem_pool),
      util::allocate_poly(degree * 2, 1, mem_pool)};

  util::Pointer<uint64_t> spcl_rns_part[2] = {
      util::allocate_poly(degree, nSpecials, mem_pool),
      util::allocate_poly(degree, nSpecials, mem_pool)};

  util::Pointer<uint64_t> nrml_rns_part =
      util::allocate_poly(degree, 1, mem_pool);

  for (ssize_t j = nCtModuli + nSpecials - 1; j >= 0; --j) {
    const bool is_special = (j >= nCtModuli);
    const size_t rns_idx = is_special ? nMaxCtModuli + (j - nCtModuli) : j;

    std::fill_n(lazy_mul_poly[0].get(), degree * 2, 0);
    std::fill_n(lazy_mul_poly[1].get(), degree * 2, 0);

    // 1) Inner Product with GSW components
    for (int l = 0; l < n_ct_components; ++l) {
      std::vector<uint64_t> tmp_rns(degree);
      for (size_t i = 0; i < nCtModuli; ++i) {
        const uint64_t *ct_ptr = Ct_GetModuli(ct, l, i);
        if (nttTables[i].modulus().value() >
            nttTables[rns_idx].modulus().value()) {
          // qi > qk
          auto mod_qj = nttTables[rns_idx].modulus();
          std::transform(ct_ptr, ct_ptr + degree, tmp_rns.data(),
                         [&mod_qj](uint64_t u) {
                           return seal::util::barrett_reduce_64(u, mod_qj);
                         });
        } else {
          // qk > qi
          std::copy_n(ct_ptr, degree, tmp_rns.data());
        }
        // ntt in [0, 4qi)
        util::ntt_negacyclic_harvey_lazy(tmp_rns.data(), nttTables[rns_idx]);
        const uint64_t *ct_qi_mod_qj = tmp_rns.data();  // [ct[l]_{qi}]_{qj}

        const uint64_t *gsw_i0_qj =
            GSW_GetModuli<0>(gsw, nMaxCtModuli * l + i, rns_idx);
        const uint64_t *gsw_i1_qj =
            GSW_GetModuli<1>(gsw, nMaxCtModuli * l + i, rns_idx);
        math::FMAU128(lazy_mul_poly[0].get(), gsw_i0_qj, ct_qi_mod_qj, degree);
        math::FMAU128(lazy_mul_poly[1].get(), gsw_i1_qj, ct_qi_mod_qj, degree);
      }
    }  // l = 0, 1

    // 2) Reduction and Rescale
    // 2-1) For special rns part, add (p-1)/2 and convert to the power-basis
    // 2-2) For normal rns part, compute qk^(1) (ct mod qi - ct mod qk) mod qi
    if (is_special) {
      for (int l : {0, 1}) {
        auto acc_ptr = lazy_mul_poly[l].get();
        uint64_t *dst_ptr = spcl_rns_part[l].get();
        for (size_t d = 0; d < degree; ++d, acc_ptr += 2) {
          dst_ptr[d] = math::barrett_reduce_128_lazy(
              acc_ptr, nttTables[rns_idx].modulus());
        }

        const uint64_t half = nttTables[rns_idx].modulus().value() >> 1;
        util::inverse_ntt_negacyclic_harvey_lazy(dst_ptr, nttTables[rns_idx]);

        SEAL_ITERATE(dst_ptr, degree, [half, rns_idx, &nttTables](uint64_t &J) {
          J = util::barrett_reduce_64(J + half, nttTables[rns_idx].modulus());
        });
      }
    } else {
      const Modulus &mod_qj = nttTables[j].modulus();
      const uint64_t qk = nttTables[nMaxCtModuli].modulus().value();
      const uint64_t qj = mod_qj.value();
      const uint64_t neg_half_mod =
          qj - util::barrett_reduce_64(qk >> 1, mod_qj);
      uint64_t qj_lazy = qj << 1;  // some multiples of qi
      uint64_t inv_qk;
      if (!util::try_invert_uint_mod(qk, mod_qj, inv_qk)) {
        throw std::runtime_error("ExternalProduct: inv_qk failed");
      }

      std::vector<uint64_t> last_moduli(degree);
      math::FastMulMod mulmod_s(inv_qk, qj);

      for (int l : {0, 1}) {  // two cipher components
        const uint64_t *acc_ptr = lazy_mul_poly[l].get();
        uint64_t *dst_ptr = nrml_rns_part.get();
        // lazy reduce to [0, 2p)
        for (size_t d = 0; d < degree; ++d, acc_ptr += 2) {
          *dst_ptr++ = math::barrett_reduce_128_lazy(
              acc_ptr, nttTables[rns_idx].modulus());
        }

        uint64_t *ct_ptr = nrml_rns_part.get();
        uint64_t *ans_ptr = ct.data(l) + j * degree;
        const uint64_t *last_moduli_ptr = spcl_rns_part[l].get();
        if (qk > qj) {
          // Lazy add (p-1)/2, results in [0, 2p)
          std::transform(
              last_moduli_ptr, last_moduli_ptr + degree, last_moduli.data(),
              [neg_half_mod, &mod_qj](uint64_t u) {
                return math::barrett_reduce_64_lazy(u + neg_half_mod, mod_qj);
              });
        } else {
          // Lazy add (p-1)/2, results in [0, 2p)
          std::transform(
              last_moduli_ptr, last_moduli_ptr + degree, last_moduli.data(),
              [neg_half_mod](uint64_t u) { return u + neg_half_mod; });
        }
        // [0, 2p)
        util::inverse_ntt_negacyclic_harvey_lazy(ct_ptr, nttTables[j]);

        // qk^(-1) * ([ct]_qi - [ct]_qk) mod qi
        std::transform(ct_ptr, ct_ptr + degree, last_moduli.data(), ans_ptr,
                       [&mulmod_s, qj_lazy](uint64_t c, uint64_t v) {
                         return mulmod_s(c + qj_lazy - v);
                       });
      }
    }  // normal rns part
  }    // handle all the normal rns.
}

void BKInit(BK &bk, SK const& lwe_sk, SK const &rlwe_sk, const RtPtr rt) {
  bk.resize(lwe::params::n());
  auto working_context = rt->key_context_data();

  const lwe::T *s_data = lwe::SKData(lwe_sk);
  for (size_t i = 0; i < lwe::params::n(); ++i) {
    GSWEncrypt(bk.at(i), (int64_t)s_data[i], /*half*/ false, rlwe_sk, rt,
               Lvl1NoiseStddev);
  }
}

void LWEKSKeyInit(DecomposedLWEKSwitchKey_t key, uint32_t decompose_base,
                  const rlwe::Ptx &rlwe_sk_non_ntt,
                  const seal::SecretKey &lwe_sk_ntt, const RtPtr lwe_rt_mod_q0,
                  const RtPtr rlwe_rt) {
  using namespace seal::util;
  const size_t N = rlwe_rt->last_context_data()->parms().poly_modulus_degree();
  const size_t n =
      lwe_rt_mod_q0->last_context_data()->parms().poly_modulus_degree();
  const uint64_t rlwe_q0 =
      rlwe_rt->last_context_data()->parms().coeff_modulus()[0].value();
  const auto &mod_q0 =
      lwe_rt_mod_q0->last_context_data()->parms().coeff_modulus()[0];

  if (decompose_base > 16) {
    throw std::invalid_argument("LWEKSKeyInit: decompose_base out-of-bound");
  }

  if (n > N || rlwe_q0 != mod_q0.value() || (N % n != 0)) {
    throw std::invalid_argument("LWEKSKeyInit: meta mismatch");
  }

  if (rlwe_sk_non_ntt.coeff_count() != N) {
    throw std::invalid_argument("LWEKSKeyInit: key meta mismatch");
  }

  int ndigits =
      std::ceil(std::log2(static_cast<double>(rlwe_q0)) / decompose_base);
  if (ndigits == 0) {
    throw std::invalid_argument("LWEKSKeyInit: ndigits > 0");
  }

  const size_t key_sze = N / n;
  key->parts_.resize(key_sze * ndigits);
  key->decompose_base_ = decompose_base;
  key->ndigits_ = ndigits;

  Ptx rlwe_sk_part;
  rlwe_sk_part.resize(n);
  const uint64_t *rlwe_sk_non_ntt_ptr = rlwe_sk_non_ntt.data();
  const double stddev = Lvl0NoiseStddev;
  std::vector<uint64_t> factors(ndigits);

  uint64_t w = 1;
  for (uint64_t k = 0; k < ndigits; ++k) {
    // factors[k] = multiply_uint_mod(w, q0_div_2w, mod_q0);
    factors[k] = w;
    w <<= decompose_base;
  }

  for (auto &ct : key->parts_) {
    ct.resize(rlwe_rt, rlwe_rt->last_context_data()->parms_id(), 2);
  }

  auto gsw_iter = key->parts_.begin();
  for (size_t i = 0; i < key_sze; ++i, rlwe_sk_non_ntt_ptr += n) {
    for (size_t k = 0; k < ndigits; ++k, ++gsw_iter) {
      Ctx &ct = *gsw_iter;
      _do_encrypt_symmetric(lwe_sk_ntt, lwe_rt_mod_q0, stddev, ct);

      uint64_t *rlwe_sk_ntt_ptr = rlwe_sk_part.data();
      multiply_poly_scalar_coeffmod(rlwe_sk_non_ntt_ptr, n, factors[k], mod_q0,
                                    rlwe_sk_ntt_ptr);
      SwitchNTTForm(rlwe_sk_ntt_ptr, NTTDir::ToNTT, 1, lwe_rt_mod_q0);

      add_poly_coeffmod(ct.data(0), rlwe_sk_part.data(), n, mod_q0, ct.data(0));
    }
  }

  if (rlwe_sk_non_ntt_ptr != (rlwe_sk_non_ntt.data() + N)) {
    throw std::runtime_error("LWEKSKeyInit BUG");
  }

  if (gsw_iter != key->parts_.end()) {
    throw std::runtime_error("LWEKSKeyInit BUG");
  }

  std::fill_n(rlwe_sk_part.data(), n, 0);
}

std::vector<int64_t> LWE_SymDec(const RLWE2LWECt_t lwe_ct,
                                const rlwe::Ptx &sk_non_ntt, const RtPtr rt) {
  using namespace seal::util;
  if (!seal::is_metadata_valid_for(sk_non_ntt, rt)) {
    throw std::invalid_argument("LWE_SymDec: invalid secret key");
  }
  if (!seal::is_metadata_valid_for(lwe_ct->a, rt)) {
    throw std::invalid_argument("LWE_SymDec: invalid cipher meta");
  }
  if (sk_non_ntt.coeff_count() < lwe_ct->a.coeff_count()) {
    throw std::invalid_argument("LWE_SymDec: invalid cipher coeff_count");
  }

  const auto working_context = rt->key_context_data();
  const size_t nsp = working_context->parms().n_special_primes();
  const auto &modulus = working_context->parms().coeff_modulus();
  const size_t nmoduli = lwe_ct->b.size();
  if (nmoduli + nsp > modulus.size()) {
    throw std::invalid_argument("LWE_SymDec: invalid cipher nmoduli");
  }
  const size_t degree = lwe_ct->a.coeff_count() / nmoduli;
  if (degree != working_context->parms().poly_modulus_degree()) {
    throw std::invalid_argument("LWE_SymDec: invalid cipher degree");
  }

  std::vector<int64_t> dec(nmoduli);
  const uint64_t *sk_non_ntt_ptr = sk_non_ntt.data();
  const uint64_t *lwe_ct_ptr = lwe_ct->a.data();

  for (size_t i = 0; i < nmoduli; ++i) {
    uint64_t d =
        dot_product_mod(sk_non_ntt_ptr, lwe_ct_ptr, degree, modulus[i]);
    d = add_uint_mod(d, lwe_ct->b[i], modulus[i]);
    dec[i] = d >= (modulus[i].value() >> 1) ? -(modulus[i].value() - d) : d;
  }

  return dec;
}

void SampleExtract(RLWE2LWECt_st *lwe_ct, const rlwe::Ctx &rlwe_ct,
                   std::vector<size_t> const &extract_indices,
                   const RtPtr rlwe_rt) {
  using namespace seal::util;
  if (!seal::is_metadata_valid_for(rlwe_ct, rlwe_rt)) {
    throw std::invalid_argument("SampleExtract: invalid rlwe cipher");
  }

  if (rlwe_ct.size() != 2) {
    throw std::invalid_argument("SampleExtract: require rlwe cipher of size 2");
  }

  if (rlwe_ct.is_ntt_form()) {
    throw std::invalid_argument("SampleExtract: require non_ntt form cipher2");
  }

  if (!lwe_ct) {
    throw std::invalid_argument("SampleExtract: nullptr lwe cipher");
  }

  const auto working_context = rlwe_rt->get_context_data(rlwe_ct.parms_id());
  const size_t degree = rlwe_ct.poly_modulus_degree();
  const size_t nmoduli = rlwe_ct.coeff_modulus_size();
  const auto &modulus = working_context->parms().coeff_modulus();

  for (size_t extract_index : extract_indices) {
    lwe_ct->a.parms_id() = seal::parms_id_zero;
    lwe_ct->a.resize(degree * nmoduli);
    lwe_ct->b.resize(nmoduli);

    uint64_t *lwe_ct_ptr = lwe_ct->a.data();
    const uint64_t *rlwe_ct_ptr = rlwe_ct.data(1);
    for (size_t i = 0; i < nmoduli; ++i) {
      std::vector<uint64_t> temp(degree);
      std::copy_n(rlwe_ct_ptr, degree, lwe_ct_ptr);

      // extract the coefficient
      uint64_t qi = modulus.at(i).value();
      std::reverse(lwe_ct_ptr, lwe_ct_ptr + extract_index + 1);
      std::reverse(lwe_ct_ptr + extract_index + 1, lwe_ct_ptr + degree);
      std::transform(lwe_ct_ptr + extract_index + 1, lwe_ct_ptr + degree,
                     lwe_ct_ptr + extract_index + 1,
                     [qi](uint64_t v) { return qi - v; });

      rlwe_ct_ptr += degree;
      lwe_ct_ptr += degree;
    }

    rlwe_ct_ptr = rlwe_ct.data(0) + extract_index;
    for (size_t i = 0; i < nmoduli; ++i) {
      lwe_ct->b[i] = rlwe_ct_ptr[0];
      rlwe_ct_ptr += degree;
    }
    lwe_ct++;
  }
}

// RLWE(m(X); N, Q) -> LWE(m_k; N, Q) under the same key.
void SampleExtract(RLWE2LWECt_t lwe_ct, const rlwe::Ctx &rlwe_ct,
                   size_t extract_index, const RtPtr rlwe_rt) {
  SampleExtract(lwe_ct, rlwe_ct, std::vector<size_t>(1, extract_index),
                rlwe_rt);
}

/// LWE(N, q0) -> LWE(n, q0)
void LWEKeySwitch(lwe::Ctx_t lwe_n, const RLWE2LWECt_t lwe_N,
                  const DecomposedLWEKSwitchKey_t key,
                  const RtPtr lwe_rt_mod_q0) {
  using namespace seal::util;

  const size_t N = lwe_N->a.coeff_count();
  const size_t n = lwe::params::n();
  const size_t key_sze = N / n;
  const size_t ndigits = key->ndigits_;

  const auto working_context = lwe_rt_mod_q0->last_context_data();

  if (key_sze * ndigits != key->parts_.size()) {
    std::cerr << key_sze << "*" << ndigits << " != " << key->parts_.size() << "\n";
    throw std::invalid_argument("LWEKeySwitch: invalid key size.");
  }

  if (lwe_N->b.size() != 1) {
    throw std::invalid_argument("LWEKeySwitch: invalid lwe_N cipher");
  }

  if (working_context->parms().poly_modulus_degree() != n) {
    throw std::invalid_argument("LWEKeySwitch: mismatch lwe_rt_mod_q0");
  }

  Ctx trivial(lwe_rt_mod_q0, working_context->parms_id());
  trivial.resize(2);
  trivial.is_ntt_form() = false;

  Ctx accum(lwe_rt_mod_q0, working_context->parms_id());
  accum.is_ntt_form() = false;
  accum.resize(2);

  auto mod_p0 = working_context->parms().coeff_modulus().front();
  const int64_t p0 = mod_p0.value();

  seal::Evaluator evaluator(lwe_rt_mod_q0);
  for (int i = 0; i < N / n; ++i) {
    std::copy_n(lwe_N->a.data(i * n), n, trivial.data(0));

    std::reverse(trivial.data(0) + 1, trivial.data(0) + n);
    std::transform(trivial.data(0) + 1, trivial.data(0) + n,
                   trivial.data(0) + 1,
                   [p0](uint64_t u) { return u > 0 ? p0 - u : 0; });

    std::fill_n(trivial.data(1), n, 0);
    trivial.is_ntt_form() = false;

    ExternalProduct(trivial, &(key->parts_.at(i * ndigits)), ndigits,
                    key->decompose_base_, lwe_rt_mod_q0);

    SwitchNTTForm(trivial, lwe_rt_mod_q0);

    try {
      evaluator.add_inplace(accum, trivial);
    } catch (std::logic_error err) {
      std::cout << "LWEKeySwitch logic_error at " << i << "-th key\n";
    }
  }

  accum.data(0)[0] = add_uint_mod(lwe_N->b[0], accum.data(0)[0], mod_p0);
  SampleExtract(lwe_n, accum, 0, lwe_rt_mod_q0);
}

// RLWE(m(x); n, q0) -> LWE(m_k; n, q0)
void SampleExtract(lwe::Ctx_t lwe_n, const rlwe::Ctx &rlwe_n,
                   size_t extract_index, const RtPtr lwe_rt_mod_q0) {
  using namespace seal::util;

  if (rlwe_n.size() != 2) {
    throw std::invalid_argument(
        "SampleExtract: require rlwe_n cipher of size 2");
  }

  if (!seal::is_metadata_valid_for(rlwe_n, lwe_rt_mod_q0)) {
    throw std::invalid_argument("SampleExtract: invalid rlwe_n cipher");
  }

  if (rlwe_n.is_ntt_form()) {
    throw std::invalid_argument("SampleExtract: require non_ntt cipher");
  }

  const auto working_context = lwe_rt_mod_q0->last_context_data();
  const size_t n = working_context->parms().poly_modulus_degree();

  if (n != lwe::params::n()) {
    throw std::invalid_argument("SampleExtract: LWE.n mismatch");
  }

  if (extract_index >= n) {
    throw std::invalid_argument("SampleExtract: extract_index out of bound");
  }

  uint64_t *lwe_ct_ptr = lwe::CtData(lwe_n);
  std::copy_n((const uint64_t *)rlwe_n.data(1), n, lwe_ct_ptr);
  lwe_ct_ptr[n] = rlwe_n.data(0)[extract_index];

  const auto mod_q0 =
      lwe_rt_mod_q0->last_context_data()->parms().coeff_modulus().front();
  const int64_t q0 = mod_q0.value();

  // arrange vector and negate
  std::reverse(lwe_ct_ptr, lwe_ct_ptr + extract_index + 1);
  std::reverse(lwe_ct_ptr + extract_index + 1, lwe_ct_ptr + n);
  std::transform(lwe_ct_ptr + extract_index + 1, lwe_ct_ptr + n,
                 lwe_ct_ptr + extract_index + 1,
                 [q0](uint64_t v) { return v == 0 ? 0 : q0 - v; });
}

}  // namespace rlwe
