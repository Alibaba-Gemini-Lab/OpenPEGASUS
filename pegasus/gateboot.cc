#include "pegasus/gateboot.h"

#include <seal/kswitchkeys.h>
#include <seal/util/polyarithsmallmod.h>
#include <seal/util/rlwe.h>

#include <random>
constexpr size_t ACC_DEGREE = 2;

namespace rlwe {

static void addCtxInplace(Ctx &accum, const Ctx &operand, const RtPtr runtime) {
  using namespace seal::util;
  if (accum.coeff_modulus_size() > operand.coeff_modulus_size()) {
    throw std::logic_error("addCtxInplace: accum.nmoduli > operand.nmoduli");
  }

  if (!is_metadata_valid_for(accum, runtime) ||
      !is_metadata_valid_for(operand, runtime)) {
    throw std::invalid_argument("addCtxInplace: invalid cipher meta");
  }

  if (accum.is_ntt_form() != operand.is_ntt_form()) {
    throw std::invalid_argument("addCtxInplace: ntt form mismatch");
  }

  if (accum.size() != operand.size()) {
    throw std::invalid_argument("addCtxInplace: cipher size mismatch");
  }

  const auto working_context = runtime->get_context_data(accum.parms_id());
  const auto &modulus = working_context->parms().coeff_modulus();
  const size_t degree = accum.poly_modulus_degree();
  const size_t nmoduli = accum.coeff_modulus_size();
  const size_t ct_sze = accum.size();

  for (size_t k = 0; k < ct_sze; ++k) {
    uint64_t *accum_ptr = accum.data(k);
    const uint64_t *operand_ptr = operand.data(k);
    for (size_t cm = 0; cm < nmoduli; ++cm) {
      const auto &mod_qi = modulus[cm];

      std::transform(accum_ptr, accum_ptr + degree, operand_ptr, accum_ptr,
                     [&mod_qi](uint64_t u, uint64_t v) {
                       return add_uint_mod(u, v, mod_qi);
                     });
      accum_ptr += degree;
      operand_ptr += degree;
    }
  }
}

void LWEGateBooter::SetPostMultiplier(const double multiplier) {
  if (multiplier != post_multiplier) {
    const size_t degree =
        rlwe_rt->last_context_data()->parms().poly_modulus_degree();
    post_multiplier = multiplier;
    InitLUT(LUTs["Abs"], degree, [](double e) { return std::abs(e); });
    InitLUT(LUTs["Sigmoid"], degree,
            [](double e) { return 1. / (1. + std::exp(-e)); });
    InitLUT(LUTs["Tanh"], degree, [](double e) { return std::tanh(e); });
    InitLUT(LUTs["Sign"], degree, [](double e) { return std::signbit(e); });
    InitLUT(LUTs["IsNegative"], degree,
            [](double e) { return std::sqrt(std::abs(e)) < 0.60 ? 1. : 0.; });
    InitLUT(LUTs["AbsLog"], degree,
            [](double e) { return std::log(std::abs(e)); });
    InitLUT(LUTs["AbsSqrt"], degree,
            [](double e) { return std::sqrt(std::abs(e)); });
    InitLUT(LUTs["InvAbsSqrt"], degree,
            [](double e) { return 1. / std::sqrt(std::abs(e) + 1e-6); });
    InitLUT(LUTs["Inverse"], degree, [](double e) { return 1. / (1e-20 + e); });
    InitLUT(LUTs["LeakyReLU"], degree,
            [](double e) { return std::signbit(e) ? 0.01 * e : e; });
    InitLUT(LUTs["ReLU"], degree,
            [](double e) { return std::signbit(e) ? 0.0 : e; });
    InitLUT(LUTs["Identity"], degree, [](double e) { return e; });
    InitLUT(LUTs["Clip"], degree, [](double e) {
      return std::abs(e) < 5.0 ? e : (e < 0 ? -5.0 : 5.0);
    });
    InitLUT(LUTs["Exponent"], degree, [](double e) {
      return e <= -5.0 ? 0.0 : (e >= 5.0 ? 148. : std::exp(e));
    });
  }
}

LWEGateBooter::LWEGateBooter(const double msg_scale, const BK &bk,
                             const RtPtr rlwe_rt, const double post_multiplier)
    : msg_scale(msg_scale), bk(bk), rlwe_rt(rlwe_rt) {
  if (bk.size() != lwe::params::n()) {
    throw std::invalid_argument(
        "LWEGateBooter: mismatch LWE.n and boot_key size");
  }

  SetPostMultiplier(post_multiplier);
}

void LWEGateBooter::InitLUT(std::vector<uint64_t> &lut, size_t len,
                            std::function<double(double)> func) const {
  const uint64_t q0 =
      rlwe_rt->last_context_data()->parms().coeff_modulus()[0].value();
  const double scale = msg_scale;
  const double step_sze =
      static_cast<double>(q0) / (ACC_DEGREE * len * msg_scale);  // 1/2N

  auto mod_q0 = [scale, q0](double d) -> uint64_t {
    bool sign = d < 1e-20;
    int64_t sd = std::floor(std::abs(d) * scale);
    return sd > 0 ? (sign ? q0 - sd : sd) : 0;
  };

  lut.resize(len);

  /*
   * Enc(sign(m) * X^{|m| mod N}) * sign(m) * LUT(m) * -X^{N - |m| mod N}
   */
  int32_t half = len >> 1;
  lut[0] = mod_q0(func(0.) * post_multiplier);

  // [0, N) --> [0, N/2) caps [-N/2, -1]
  // Negative half
  // The flip -X{N - |m| mod N} becomes X^{N - |m| mod N}
  double step = -step_sze;
  for (int32_t v = 1; v <= half; ++v, step -= step_sze) {
    lut[v] = mod_q0(func(step) * post_multiplier);
  }

  // Positive half
  step = step_sze * (half - 1);
  for (int32_t v = half + 1; v < len; ++v, step -= step_sze) {
    lut[v] = mod_q0(-func(step) * post_multiplier);
  }
}

void LWEGateBooter::IsZeroThenTag(RLWE2LWECt_t lwe_N_ct,
                                  const lwe::Ctx_t lwe_n_ct,
                                  const double t) const {
  const size_t degree =
      rlwe_rt->last_context_data()->parms().poly_modulus_degree();
  std::vector<uint64_t> lut;
  InitLUT(lut, degree,
          [t](const double u) { return std::abs(u) < 0.1 ? t : 0.; });
  Apply(lwe_N_ct, lwe_n_ct, lut);
}

void LWEGateBooter::MulConstant(RLWE2LWECt_t lwe_N_ct,
                                const lwe::Ctx_t lwe_n_ct,
                                const double v) const {
  const size_t degree =
      rlwe_rt->last_context_data()->parms().poly_modulus_degree();
  std::vector<uint64_t> lut;
  InitLUT(lut, degree, [v](const double u) { return u * v; });
  Apply(lwe_N_ct, lwe_n_ct, lut);
}

void LWEGateBooter::GreaterThan(RLWE2LWECt_t lwe_N_ct,
                                const lwe::Ctx_t lwe_n_ct,
                                const double v) const {
  const size_t degree =
      rlwe_rt->last_context_data()->parms().poly_modulus_degree();
  std::vector<uint64_t> lut;
  InitLUT(lut, degree, [v](const double u) {
    return u > v ? 1. : std::abs(u - v) < 1e-4 ? 1 : 0.;
  });
  Apply(lwe_N_ct, lwe_n_ct, lut);
}

#define LWEGATEBOOT_FUNC(Name, tag)                                          \
  void LWEGateBooter::Name(RLWE2LWECt_t lwe_N_ct, const lwe::Ctx_t lwe_n_ct) \
      const {                                                                \
    auto kv = LUTs.find(tag);                                                \
    Apply(lwe_N_ct, lwe_n_ct, kv->second);                                   \
  }

LWEGATEBOOT_FUNC(Abs, "Abs")
LWEGATEBOOT_FUNC(Sigmoid, "Sigmoid")
LWEGATEBOOT_FUNC(Tanh, "Tanh")
LWEGATEBOOT_FUNC(Sign, "Sign")
LWEGATEBOOT_FUNC(IsNegative, "IsNegative")
LWEGATEBOOT_FUNC(AbsLog, "AbsLog")
LWEGATEBOOT_FUNC(AbsSqrt, "AbsSqrt")
LWEGATEBOOT_FUNC(InvAbsSqrt, "InvAbsSqrt")
LWEGATEBOOT_FUNC(Inverse, "Inverse")
LWEGATEBOOT_FUNC(LeakyReLU, "LeakyReLU")
LWEGATEBOOT_FUNC(ReLU, "ReLU")
LWEGATEBOOT_FUNC(Identity, "Identity")
LWEGATEBOOT_FUNC(Clip, "Clip")
LWEGATEBOOT_FUNC(Exponent, "Exponent")

void LWEGateBooter::InitTestVector(Ctx &acc, lwe::T b,
                                   const std::vector<uint64_t> &LUT) const {
  using namespace seal::util;
  const size_t nmoduli = acc.coeff_modulus_size();
  const size_t degree = acc.poly_modulus_degree();
  auto seal_context = rlwe_rt->get_context_data(acc.parms_id());
  auto ntt_tables = seal_context->small_ntt_tables();

  bool odd = (b / degree) & 1;  // b = [0, k*degree)
  b %= degree;

  if (LUT.size() != degree) {
    throw std::invalid_argument("InitTestVector: invalid LUT length");
  }

  auto rns_ptr = acc.data(0);
  for (size_t cm = 0; cm < nmoduli; ++cm, rns_ptr += degree) {
    const auto &modulus = ntt_tables[cm].modulus();
    const uint64_t qi = modulus.value();
    std::copy_n(LUT.data(), degree, rns_ptr);

    auto pool = seal::MemoryManager::GetPool();
    // LUT * X^b mod X^N + 1, b \in [0, 2N)
    uint64_t coeff, exponent;

    if (odd) {
      coeff = qi - 1;
      exponent = b;
    } else {
      coeff = 1;
      exponent = b;
    }

    negacyclic_multiply_poly_mono_coeffmod(rns_ptr, degree, coeff, exponent,
                                           modulus, rns_ptr, pool);
  }

  set_zero_poly(degree, nmoduli, acc.data(1));

  acc.is_ntt_form() = false;
  acc.scale() = 1.;
}

void LWEGateBooter::MultiplyXaiMinusOne(Ctx &ct, uint64_t ai) const {
  using namespace seal::util;
  if (!seal::is_metadata_valid_for(ct, rlwe_rt)) {
    throw std::logic_error("MultiplyXaiMinusOne:  invalid ct");
  }

  if (ct.is_ntt_form()) {
    throw std::invalid_argument("MultiplyXaiMinusOne: require non-ntt");
  }

  const size_t degree = ct.poly_modulus_degree();
  const uint64_t ai_mod = ai & (degree - 1);
  const auto &modulus =
      rlwe_rt->get_context_data(ct.parms_id())->parms().coeff_modulus();
  const size_t nmoduli = modulus.size();
  const bool even = 0 == ((ai / degree) & 1);

  auto pool = seal::MemoryManager::GetPool();
  std::vector<uint64_t> temp(degree);

  for (size_t k = 0; k < ct.size(); ++k) {
    if (rlwe::IsCtxComponentZero(ct, k)) continue;

    uint64_t *ct_rns_ptr = ct.data(k);

    // P(X) * (X^{ai} - 1)
    for (size_t cm = 0; cm < nmoduli; ++cm, ct_rns_ptr += degree) {
      const auto &mod_qi = modulus[cm];
      std::copy_n((const uint64_t *)ct_rns_ptr, degree, temp.data());
      std::rotate(temp.rbegin(), temp.rbegin() + ai_mod,
                  temp.rend());  // RHS rotated

      // ai \in [0, N), P(X) * X^{ai}
      if (even) {
        // negate [0, ai_mod)
        std::transform(
            temp.data(), temp.data() + ai_mod, temp.data(),
            [&mod_qi](uint64_t k) { return negate_uint_mod(k, mod_qi); });
        // ai \in [N, 2N), P(X) * -X^{N-ai}
      } else {
        // negate [ai_mod, degree)
        std::transform(
            temp.begin() + ai_mod, temp.end(), temp.data() + ai_mod,
            [&mod_qi](uint64_t k) { return negate_uint_mod(k, mod_qi); });
      }

      sub_poly_coeffmod(temp.data(), ct_rns_ptr, degree, modulus[cm],
                        ct_rns_ptr);
    }
  }
}

#define RANDOM_ROUNDING 0
// 0 <= x < srcMod -> round(x / srcMod * targetMod)
static uint64_t ModSwitch(uint64_t x, uint64_t srcMod, uint64_t targetMod) {
#if RANDOM_ROUNDING
  if (x >= srcMod) throw std::invalid_argument("ModSwitch: x >= p");

  uint64_t mask = targetMod - 1;
  uint64_t half = srcMod >> 1;
  int64_t sx = x;
  if (x >= half) sx -= srcMod;

  double y = static_cast<double>(x) / srcMod;
  double u = y * targetMod;
  double fu = std::floor(u);
  uint64_t ret = static_cast<uint64_t>(fu);

  double r = u - std::floor(u);
  std::uniform_real_distribution<double> uniform(0., 1.);
  std::mt19937 rdv(std::time(0));
  double h = uniform(rdv);
  if (h <= r) {
    ret += 1;
  }
  return ret & mask;
#else
  if (x >= srcMod) throw std::invalid_argument("ModSwitch: x >= p");
  uint64_t mask = targetMod - 1;
  uint64_t half = srcMod >> 1;
  int64_t sx = x;
  if (x >= half) sx -= srcMod;
  double y = static_cast<double>(x) / srcMod;
  return ((uint64_t)std::round(y * targetMod)) & mask;
#endif
}

void LWEGateBooter::Apply(RLWE2LWECt_t lwe_N_ct, const lwe::Ctx_t lwe_n_ct,
                          const std::vector<uint64_t> &LUT) const {
  const auto working_context = rlwe_rt->last_context_data();
  const size_t n = lwe::params::n();
  const size_t N = working_context->parms().poly_modulus_degree();
  const uint64_t q0 = working_context->parms().coeff_modulus()[0].value();
  const size_t new_mod_sze = ACC_DEGREE * N;

  if (bk.size() != n) {
    throw std::invalid_argument("oWEAbs: boot_key size mismatch n");
  }

  Ctx acc_non_ntt;  // 0s
  acc_non_ntt.resize(rlwe_rt, working_context->parms_id(), 2);

  uint64_t bmod = ModSwitch(lwe::CtData(lwe_n_ct)[n], q0, new_mod_sze);
  // ACC_0 = LUT_poly * X^{b'} where b' = round(2N * b/p)
  InitTestVector(acc_non_ntt, bmod, LUT);
  bool allzero = true;
  for (int i = 0; i < n; ++i) {
    uint64_t ai = ModSwitch(lwe::CtData(lwe_n_ct)[i], q0, new_mod_sze);
    if (ai == 0) continue;
    allzero = false;

    Ctx multed{acc_non_ntt};
    // ACC * (X^{ai} - 1) in the non-ntt form
    MultiplyXaiMinusOne(multed, ai);

    ExternalProduct(multed, bk.at(i), rlwe_rt);

    addCtxInplace(acc_non_ntt, multed, rlwe_rt);
  }

  if (allzero) {
    throw std::runtime_error("LUT");
  }

  // RLWE(|m|; N, q0) -> LWE(|m|; N, q0)
  SampleExtract(lwe_N_ct, acc_non_ntt, 0, rlwe_rt);
  lwe_N_ct->scale = lwe_n_ct->scale;
}

}  // namespace rlwe
