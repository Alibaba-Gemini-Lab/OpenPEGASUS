#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <seal/seal.h>
#include <seal/util/polyarithsmallmod.h>

#include <Eigen/Dense>
#include <array>
#include <fstream>
#include <random>
#include <thread>

#include "ThreadPool.h"
#include "chevb_approximator.h"
#include "gateboot.h"
#include "kmean.h"
#include "linear_transform.h"
#include "lwe.h"
#include "repack.h"
#include "rlwe.h"
#include "runtime.h"
#include "timer.h"

namespace gemini {

#define CHECK_AND_ABORT(state)                                                \
  do {                                                                        \
    auto st = state;                                                          \
    if (!st.IsOk()) {                                                         \
      std::cerr << __LINE__ << " " << #state << " " << st.Msg() << std::endl; \
      exit(1);                                                                \
    }                                                                         \
  } while (0)

class PegasusRunTime {
 public:
  struct Parms {
    int lvl0_lattice_dim;  // n_{lwe}
    int lvl1_lattice_dim;  // n_{lut}
    int lvl2_lattice_dim;  // n_{ckks}
    int nslots;            // 1 <= nslots <= n_{ckks} / 2
    int nlevels;           // number of levels on CKKS
    double scale = 1.;
    double s2c_multiplier = 1.;
  };

  static constexpr int KS_DC_BASE = 7;
  static constexpr int SECRET_KEY_HW = 64;
  constexpr int numBitsP0() const { return 46; }

  explicit PegasusRunTime(Parms parms, size_t num_threads)
      : parms_(parms), num_threads_(std::max<size_t>(1, num_threads)) {
    std::string JSON = level2Params(parms.nlevels);

    runtime_ = gemini::RunTime::Create(JSON);
    runtime_->ShowContext(std::cout);
    setUpRuntime();
    setUpSecretKeys();
    setUpPublicKeys();
    setUpFunctors();

    printf("n_{lwe} = %d, n_{lut} = %d, n_{ckks} = %d\n",
           parms_.lvl0_lattice_dim, parms_.lvl1_lattice_dim,
           parms_.lvl2_lattice_dim);
    printf("|msg| < %f, scale = 2^%f, extra_scale = 2^%f, nslots = %d, #threads = "
        "%zd\n",
        MsgRange(), std::log2(parms_.scale), std::log2(ExtraScaling()),
        parms_.nslots, num_threads_);
  }
  
  void ResetLT(size_t nslots, double s2c_multiplier) {
    using namespace gemini;
    using namespace seal;
    seal_boot::LinearTransformer::Parms ltParams;
    ltParams.nslots = nslots;
    ltParams.s2c_lvl_start = 2;
    ltParams.c2s_lvl_start = 0;  // no C2S is needed
    ltParams.s2cMultiplier = s2c_multiplier;
    ltParams.c2sMultiplier = 1.;

    linearTransformer_.reset(
        new seal_boot::LinearTransformer(ltParams, runtime_));
    parms_.nslots = nslots;
    parms_.s2c_multiplier = s2c_multiplier;
  }

  double MsgRange() const { return (1L << numBitsP0()) * 0.2 / parms_.scale; }

  double ExtraScaling() const {
    return (1L << numBitsP0()) * 0.125 / parms_.scale;
  }

  template <typename VecType>
  Status EncodeThenEncrypt(const VecType &vec, Ctx &out) const {
    Ptx ptx;
    CHECK_STATUS(runtime_->Encode(vec, parms_.scale, &ptx));
    CHECK_STATUS(runtime_->Encrypt(ptx, &out));
    return Status::Ok();
  }

  template <typename VecType>
  Status DecryptThenDecode(Ctx const &in, VecType &out) const {
    Ptx ptx;
    CHECK_STATUS(runtime_->Decrypt(in, &ptx));
    CHECK_STATUS(runtime_->Decode(ptx, parms_.nslots, &out));
    return Status::Ok();
  }

  Status SlotsToCoeffs(Ctx *ct, int nct) const {
    ThreadPool pool(num_threads_);
    const size_t work_load = (nct + num_threads_ - 1) / num_threads_;
    for (int w = 0; w < num_threads_; ++w) {
      size_t start = w * work_load;
      size_t end = std::min<size_t>(start + work_load, nct);
      if (end > start) {
        pool.enqueue(
            [&](size_t s, size_t e) {
              for (size_t i = s; i < e; ++i) {
                CHECK_AND_ABORT(SlotsToCoeffs(ct[i]));
              }
            },
            start, end);
      }
    }
    return Status::Ok();
  }

  Status SlotsToCoeffs(Ctx &out) const {
    size_t level = GetNModuli(out) - 1;
    size_t depth = linearTransformer_->depth();
    if (level < depth) {
      return Status::NotReady("SlotsToCoeffs require more levels");
    }
    // We keep only one moduli after S2C
    runtime_->DropModuli(&out, level - linearTransformer_->s2c_lvl_start());
    Ctx s2c;
    CHECK_STATUS(linearTransformer_->SlotsToCoeffs(out, &s2c));
    out = s2c;
    return Status::Ok();
  }

  Status RotateLeft(Ctx &out, size_t offset) const {
    CHECK_STATUS(runtime_->RotateLeft(&out, offset));
    return Status::Ok();
  }

  Status Add(Ctx &a, Ctx const &b) const {
    CHECK_STATUS(runtime_->Add(&a, b));
    return Status::Ok();
  }

  Status Sub(Ctx &a, Ctx const &b) const {
    if (&a == &b) {
      return Status::NotReady("Sub itself is not supported");
    }
    CHECK_STATUS(runtime_->Sub(&a, b));
    return Status::Ok();
  }

  Status Square(Ctx &a) const {
    CHECK_STATUS(runtime_->Mul(&a, a));
    return Status::Ok();
  }

  Status RelinThenRescale(Ctx &a) const {
    CHECK_STATUS(runtime_->Relin(&a));
    F64 scale_up = parms_.scale * runtime_->GetModulusPrime(GetNModuli(a) - 1);
    scale_up = std::round(scale_up / a.scale());
    if (scale_up >= 1.) {
      CHECK_STATUS(runtime_->MulScalar(&a, 1., scale_up));
    }
    CHECK_STATUS(runtime_->RescaleNext(&a));
    return Status::Ok();
  }

  Status ExtraAllCoefficients(const Ctx &in, std::vector<lwe::Ctx_st> &lwe_ct) {
    std::vector<rlwe::RLWE2LWECt_st> lwe_N_ct(parms_.nslots);
    if (!in.is_ntt_form()) {
      rlwe::SampleExtract(lwe_N_ct.data(), in, extract_indices_,
                          runtime_->SEALRunTime());
    } else {
      auto copy{in};
      rlwe::SwitchNTTForm(copy, runtime_->SEALRunTime());
      rlwe::SampleExtract(lwe_N_ct.data(), copy, extract_indices_,
                          runtime_->SEALRunTime());
    }
    lwe_ct.resize(parms_.nslots);

    ThreadPool pool(num_threads_);
    const size_t work_load = (parms_.nslots + num_threads_ - 1) / num_threads_;
    for (int w = 0; w < num_threads_; ++w) {
      size_t start = w * work_load;
      size_t end = std::min<size_t>(start + work_load, parms_.nslots);

      if (end > start) {
        pool.enqueue(
            [&](size_t s, size_t e) {
              for (size_t i = s; i < e; ++i) {
                rlwe::LWEKeySwitch(&lwe_ct[i], &lwe_N_ct[i], lvl2Tolvl0_,
                                   lvl0_runtime_);
                lwe_ct[i].scale = in.scale();
              }
            },
            start, end);
      }
    }
    return Status::Ok();
  }

  void MulScalarLWECt(lwe::Ctx_st &out, const lwe::Ctx_st &a,
                      uint64_t scalar) const {
    const auto &q0 =
        lvl0_runtime_->first_context_data()->parms().coeff_modulus()[0];
    using namespace seal::util;
    multiply_poly_scalar_coeffmod(CtData(&a), lwe::params::n() + 1, scalar, q0,
                                  CtData(&out));
    out.scale = a.scale;
  }

  void AddLWECt(lwe::Ctx_st &out, lwe::Ctx_st const &a,
                lwe::Ctx_st const &b) const {
    using namespace lwe;
    using namespace seal::util;
    const auto &q0 =
        lvl0_runtime_->first_context_data()->parms().coeff_modulus()[0];
    if (!seal::util::are_close(a.scale, b.scale)) {
      throw std::invalid_argument("AddLWECt scale mismatch");
    }
    add_poly_coeffmod(CtData(&a), CtData(&b), lwe::params::n() + 1, q0,
                      CtData(&out));
    out.scale = a.scale;
  }

  void SubLWECt(lwe::Ctx_st &out, lwe::Ctx_st const &a,
                lwe::Ctx_st const &b) const {
    using namespace lwe;
    using namespace seal::util;
    if (&a == &b) {
      throw std::runtime_error("SubLWECt self-substraction");
    }
    if (!seal::util::are_close(a.scale, b.scale)) {
      throw std::invalid_argument("SubLWECt scale mismatch");
    }
    const auto &q0 =
        lvl0_runtime_->first_context_data()->parms().coeff_modulus()[0];
    sub_poly_coeffmod(CtData(&a), CtData(&b), lwe::params::n() + 1, q0,
                      CtData(&out));
    out.scale = a.scale;
  }

  // out = 2*min(a, b)
  inline void Min2(lwe::Ctx_st &out, lwe::Ctx_st const &a,
                   lwe::Ctx_st const &b) const {
    lwe::Ctx_st sub;
    AddLWECt(out, a, b);
    SubLWECt(sub, a, b);
    Abs(&sub);
    SubLWECt(out, out, sub);
  }

  void doMinElement(lwe::Ctx_st &out,
                    const std::vector<lwe::Ctx_st *> &ct_array,
                    const size_t start, const size_t n) const {
    if (n == 2) {
      Min2(out, *ct_array[start], *ct_array[start + 1]);
    } else {
      lwe::Ctx_st t0, t1;
      doMinElement(t0, ct_array, start, n >> 1);
      doMinElement(t1, ct_array, start + (n >> 1), n >> 1);
      Min2(out, t0, t1);
    }
  }

  void MinElement(lwe::Ctx_st &out,
                  const std::vector<lwe::Ctx_st *> &ct_array) const {
    if (ct_array.empty()) {
      throw std::runtime_error("MinElement nullptr");
    }

    if (!gemini::IsTwoPower(ct_array.size())) {
      throw std::runtime_error("MinElement needs 2^d input wires currently");
    }

    doMinElement(out, ct_array, 0, ct_array.size());
  }

#define DEFINE_LUT(FNAME)                                              \
  inline void FNAME(lwe::Ctx_t lvl0_ct) const {                        \
    F64 out_scale = lvl0_ct->scale * lutFunctor_->GetPostMultiplier(); \
    rlwe::RLWE2LWECt_t lvl1_ct;                                        \
    lutFunctor_->FNAME(lvl1_ct, lvl0_ct);                              \
    rlwe::LWEKeySwitch(lvl0_ct, lvl1_ct, lvl1Tolvl0_, lvl0_runtime_);  \
    lvl0_ct->scale = out_scale;                                        \
  }

#define DEFINE_OUTPUT_BOUNDED_LUT(FNAME)                               \
  inline void FNAME(lwe::Ctx_t lvl0_ct) const {                        \
    F64 out_scale = lvl0_ct->scale * lutFunctor_->GetPostMultiplier(); \
    rlwe::RLWE2LWECt_t lvl1_ct;                                        \
    lutFunctor_->FNAME(lvl1_ct, lvl0_ct);                              \
    rlwe::LWEKeySwitch(lvl0_ct, lvl1_ct, lvl1Tolvl0_, lvl0_runtime_);  \
    lvl0_ct->scale = out_scale;                                        \
  }

#define DEFINE_MT(FNAME)                                               \
  inline void FNAME(lwe::Ctx_st *lvl0_ct, int num_wires) const {       \
    ThreadPool pool(num_threads_);                                     \
    const size_t work_load =                                           \
        (parms_.nslots + num_threads_ - 1) / num_threads_;             \
    for (int w = 0; w < num_threads_; ++w) {                           \
      size_t start = w * work_load;                                    \
      size_t end = std::min<size_t>(start + work_load, parms_.nslots); \
      if (end > start) {                                               \
        pool.enqueue(                                                  \
            [&](size_t s, size_t e) {                                  \
              for (size_t i = s; i < e; ++i) {                         \
                FNAME(lvl0_ct + i);                                    \
              }                                                        \
            },                                                         \
            start, end);                                               \
      }                                                                \
    }                                                                  \
  }

  inline void MulConstant(lwe::Ctx_t lvl0_ct, double v) const {
    F64 out_scale = lvl0_ct->scale * lutFunctor_->GetPostMultiplier();
    rlwe::RLWE2LWECt_t lvl1_ct;
    lutFunctor_->MulConstant(lvl1_ct, lvl0_ct, v);
    rlwe::LWEKeySwitch(lvl0_ct, lvl1_ct, lvl1Tolvl0_, lvl0_runtime_);
    lvl0_ct->scale = out_scale;
  }

  DEFINE_LUT(Abs)
  DEFINE_LUT(AbsLog)
  DEFINE_LUT(LeakyReLU)
  DEFINE_LUT(ReLU)
  DEFINE_LUT(AbsSqrt)
  DEFINE_OUTPUT_BOUNDED_LUT(Sigmoid)
  DEFINE_OUTPUT_BOUNDED_LUT(Tanh)
  DEFINE_OUTPUT_BOUNDED_LUT(Sign)
  DEFINE_OUTPUT_BOUNDED_LUT(IsNegative)

  DEFINE_MT(AbsLog)
  DEFINE_MT(LeakyReLU)
  DEFINE_MT(ReLU)
  DEFINE_MT(AbsSqrt)
  DEFINE_MT(Sigmoid)
  DEFINE_MT(Tanh)
  DEFINE_MT(Sign)
  DEFINE_MT(IsNegative)

  double DecryptLWE(lwe::Ctx_st const &lwe_ct) const {
    return lwe::SymDec(&lwe_ct, lvl0_sk_non_ntt_, lvl0_runtime_);
  }

  inline const size_t num_threads() const { return num_threads_; }

 protected:
  std::string level2Params(int level_left) const {
    std::stringstream ss;
    ss << "{\"log2PolyDegree\":" << (int)std::log2(parms_.lvl2_lattice_dim)
       << ",";
    ss << "\"nSpecialPrimes\":1,\"seed\":0,";
    ss << "\"moduliArray\":[" << std::to_string(numBitsP0()) << ",";

    level_left = std::max(1, level_left);
    std::string norm_sze = std::to_string(numBitsP0());
    for (int i = 1; i < level_left; ++i) {
      ss << norm_sze << ",";
    }

    // Final is the special modulus
    ss << 59 << "]";
    ss << "}";
    return ss.str();
  }
  void setUpRuntime() {
    using namespace seal;
    auto lvl2_runtime = runtime_->SEALRunTime();
    const auto &modulus =
        lvl2_runtime->key_context_data()->parms().coeff_modulus();
    EncryptionParameters parms(seal::scheme_type::CKKS);

    // Level-1 RGSW works with 2 moduli
    std::vector<Modulus> lvl1_modulus{modulus.front(), modulus.back()};
    parms.set_coeff_modulus(lvl1_modulus);
    parms.set_poly_modulus_degree(parms_.lvl1_lattice_dim);
    parms.set_galois_generator(5);
    lvl1_runtime_ = SEALContext::Create(parms, true, sec_level_type::none);

    // Level-0 LWE works with 1 modulus
    std::vector<Modulus> lvl0_modulus{modulus.front()};
    parms.set_poly_modulus_degree(parms_.lvl0_lattice_dim);
    parms.set_coeff_modulus(lvl0_modulus);
    lvl0_runtime_ = SEALContext::Create(parms, true, sec_level_type::none);
  }

  void setUpSecretKeys() {
    // non-ntt form of the CKKS secret key
    auto lvl2_runtime = runtime_->SEALRunTime();
    auto const &lvl2_sk = runtime_->SEALSecretKey();
    lvl2_sk_non_ntt_.data().resize(parms_.lvl2_lattice_dim);
    std::copy_n((const uint64_t *)lvl2_sk.data().data(),
                parms_.lvl2_lattice_dim, lvl2_sk_non_ntt_.data().data());
    rlwe::SwitchNTTForm(lvl2_sk_non_ntt_.data().data(), NTTDir::FromNTT, 1,
                        lvl2_runtime);
    // Generate sk_{lut}
    lwe::GenerateHammingSecretKey(lvl1_sk_ntt_, SECRET_KEY_HW, /*is_ntt*/ true,
                                  lvl1_runtime_);
    lvl1_sk_non_ntt_.data().resize(parms_.lvl1_lattice_dim);
    std::copy_n((const uint64_t *)lvl1_sk_ntt_.data().data(),
                parms_.lvl1_lattice_dim, lvl1_sk_non_ntt_.data().data());
    rlwe::SwitchNTTForm(lvl1_sk_non_ntt_.data().data(), NTTDir::FromNTT, 1,
                        lvl1_runtime_);
    // Generate sk_{lwe}
    lwe::SKInit(lvl0_sk_ntt_, lvl0_sk_non_ntt_, SECRET_KEY_HW, lvl0_runtime_);
  }

  void setUpPublicKeys() {
    // gemini::RpKeyInit(repackKey_, std::pow(2., numBitsP0()),
    //                   runtime_->SEALSecretKey(), lvl1_sk_non_ntt_, runtime_);

    rlwe::BKInit(lutEvalKey_, lvl0_sk_non_ntt_, lvl1_sk_ntt_, lvl1_runtime_);
    rlwe::LWEKSKeyInit(lvl2Tolvl0_, KS_DC_BASE, lvl2_sk_non_ntt_.data(),
                       lvl0_sk_ntt_, lvl0_runtime_, runtime_->SEALRunTime());
    rlwe::LWEKSKeyInit(lvl1Tolvl0_, KS_DC_BASE, lvl1_sk_non_ntt_.data(),
                       lvl0_sk_ntt_, lvl0_runtime_, lvl1_runtime_);
  }

  void setUpFunctors() {
    using namespace gemini;
    using namespace seal;
    seal_boot::LinearTransformer::Parms ltParams;
    ltParams.nslots = parms_.nslots;
    ltParams.s2c_lvl_start = 2;
    ltParams.c2s_lvl_start = 0;  // no C2S is needed
    ltParams.s2cMultiplier = parms_.s2c_multiplier;
    ltParams.c2sMultiplier = 1.;

    sinFunctor_.reset(new seal_boot::ChevbApproximator(runtime_));
    linearTransformer_.reset(
        new seal_boot::LinearTransformer(ltParams, runtime_));

    lutFunctor_.reset(
        new rlwe::LWEGateBooter(parms_.scale, lutEvalKey_, lvl1_runtime_, 1.));
    output_bounded_lutFunctor_.reset(new rlwe::LWEGateBooter(
        parms_.scale, lutEvalKey_, lvl1_runtime_, ExtraScaling()));
    extract_indices_.resize(parms_.nslots);
    const size_t log2N = (size_t)std::log2(parms_.lvl2_lattice_dim);
    for (size_t i = 0; i < parms_.nslots; ++i) {
      extract_indices_[i] = seal::util::reverse_bits(i, log2N - 1);
    }
  }

 private:
  struct Parms parms_;
  const size_t num_threads_;
  std::shared_ptr<gemini::RunTime> runtime_{nullptr};
  std::shared_ptr<gemini::seal_boot::ChevbApproximator> sinFunctor_{nullptr};
  std::shared_ptr<gemini::seal_boot::LinearTransformer> linearTransformer_{
      nullptr};

  // Secret keys
  seal::SecretKey lvl0_sk_ntt_, lvl1_sk_ntt_;
  seal::SecretKey lvl2_sk_non_ntt_, lvl1_sk_non_ntt_;
  lwe::SK_t lvl0_sk_non_ntt_;

  // Public keys
  rlwe::BK lutEvalKey_;
  gemini::RpKey repackKey_;
  rlwe::DecomposedLWEKSwitchKey_t lvl2Tolvl0_;
  rlwe::DecomposedLWEKSwitchKey_t lvl1Tolvl0_;

  // LUT-related
  std::vector<size_t> extract_indices_;
  std::shared_ptr<rlwe::LWEGateBooter> lutFunctor_{nullptr};
  std::shared_ptr<rlwe::LWEGateBooter> output_bounded_lutFunctor_{nullptr};

  // Runtime
  typedef std::shared_ptr<seal::SEALContext> SEALRunTimePtr;
  SEALRunTimePtr lvl1_runtime_;
  SEALRunTimePtr lvl0_runtime_;
};
}  // namespace gemini

void MockKmean(const int numPoints, const int numFeatures, const int numCenters,
               gemini::PegasusRunTime &runtime);

int run_kmean(int argc, char *argv[]) {
  using namespace gemini;
  const int n_features = 16;
  for (int n : {4096}) {
    PegasusRunTime::Parms pp;
    pp.lvl0_lattice_dim = lwe::params::n();
    pp.lvl1_lattice_dim = n;
    pp.lvl2_lattice_dim = 1 << 16;
    pp.nlevels = 4;
    pp.scale = std::pow(2., 38);

    for (int n_points : {256, 1024}) {
      for (int n_centers: {2, 4, 8}) {
        pp.nslots = n_points;
        pp.s2c_multiplier = 1. / n_centers;
        PegasusRunTime pg_rt(pp, /*num_threads*/ 2);
        MockKmean(n_points, n_features, n_centers, pg_rt);
      }
    }
  }

  return 0;
}

int run_lut(int argc, char *argv[]) {
  using namespace gemini;
  PegasusRunTime::Parms pp;
  pp.lvl0_lattice_dim = lwe::params::n();
  pp.lvl1_lattice_dim = argc > 1 ? std::atoi(argv[1]) : 2048;
  pp.lvl2_lattice_dim = 1 << 16;
  pp.nslots = 1024;
  pp.nlevels = 4;
  pp.scale = std::pow(2., 40);

  PegasusRunTime pg_rt(pp, /*num_threads*/ 16);
  printf("msg range [-8., 8]\n");
  for (int f = 0; f < 7; ++f) {
    F64Vec slots(pp.nslots);
    {
      std::mt19937 rdv(std::time(0));
      std::uniform_real_distribution<double> uniform(-8., 8.);
      std::generate_n(slots.begin(), pp.nslots, [&]() { return uniform(rdv); });
    }

    Ctx ckks_ct;
    CHECK_AND_ABORT(pg_rt.EncodeThenEncrypt(slots, ckks_ct));
    CHECK_AND_ABORT(pg_rt.SlotsToCoeffs(ckks_ct));

    std::vector<lwe::Ctx_st> lwe_ct;
    double s2c_ks_time{0.};
    {
      AutoTimer timer(&s2c_ks_time);
      CHECK_AND_ABORT(pg_rt.ExtraAllCoefficients(ckks_ct, lwe_ct));
    }

    double accum_s2c_err{0.};
    for (int i = 0; i < pp.nslots; ++i) {
      accum_s2c_err += std::abs(slots[i] - pg_rt.DecryptLWE(lwe_ct[i]));
    }
    printf("\nS2C.error 2^%f\t%f sec\n", std::log2(accum_s2c_err / pp.nslots),
           s2c_ks_time / 1000.);

    std::string tag;
    std::function<double(double)> target_func;
    double lut_time{0.};

    AutoTimer timer(&lut_time);
    switch (f) {
      case 0: {
        pg_rt.LeakyReLU(lwe_ct.data(), lwe_ct.size());
        tag = "LeakyReLU";
        target_func = [](double e) { return std::signbit(e) ? 0.01 * e : e; };
      } break;

      case 1: {
        pg_rt.AbsLog(lwe_ct.data(), lwe_ct.size());
        tag = "AbsLog";
        target_func = [](double e) -> double { return std::log(std::abs(e)); };
      } break;
      case 2: {
        pg_rt.ReLU(lwe_ct.data(), lwe_ct.size());
        tag = "ReLU";
        target_func = [](double e) -> double { return std::max(0., e); };
      } break;
      case 3: {
        pg_rt.AbsSqrt(lwe_ct.data(), lwe_ct.size());
        tag = "AbsSqrt";
        target_func = [](double e) -> double { return std::sqrt(std::abs(e)); };
      } break;
      case 4: {
        printf("-----------------\n");
        pg_rt.Tanh(lwe_ct.data(), lwe_ct.size());
        tag = "Tanh";
        target_func = [](double e) -> double { return std::tanh(e); };
      } break;
      case 5: {
        pg_rt.Sign(lwe_ct.data(), lwe_ct.size());
        tag = "Sign";
        target_func = [](double e) -> double { return std::signbit(e); };
      } break;
      case 6: {
        pg_rt.Sigmoid(lwe_ct.data(), lwe_ct.size());
        tag = "Sigmoid";
        target_func = [](double e) { return 1. / (1. + std::exp(-e)); };
      } break;
    }
    timer.stop();

    double accum_lut_err{0.};
    for (int i = 0; i < pp.nslots; ++i) {
      double gnd = target_func(slots[i]);
      double cmp = pg_rt.DecryptLWE(lwe_ct[i]);
      accum_lut_err += std::abs(gnd - cmp);
    }

    printf("LUT (%s).error 2^%f\t%f sec\n", tag.c_str(),
           std::log2(accum_lut_err / pp.nslots), lut_time / 1000.);
  }

  return 0;
}

int main(int argc, char *argv[]) { return run_kmean(argc, argv); }

void MockKmean(const int n_points, const int n_features, const int n_centers, gemini::PegasusRunTime &rt) {
  using namespace gemini;
  using namespace seal;
  constexpr double msg_range = 1.;
  printf("Kmean #points = %d, #features = %d, #centers = %d\n", n_points, n_features, n_centers);
  // printf("uniform in (%f, %f)\n", -msg_range, msg_range);

  MatrixType data_points(n_points, n_features);
  MatrixType centers(n_centers, n_features);

  data_points.Random(-msg_range, msg_range);
  centers.Random(-msg_range, msg_range);
  for (int j = 0; j < n_centers; ++j) {
    centers.Set(j, j, 2. * (std::rand() & 1 ? msg_range : -msg_range));
    centers.Set(j, n_features - j - 1, 2. * (std::rand() & 1 ? msg_range : -msg_range));
  }
  MatrixType ground_distance(n_points, n_centers);
  ComputeDistance(ground_distance, data_points, centers);

  std::vector<Ctx> enc_data_points(n_features);
  std::vector<Ctx> enc_centers(n_features);

  for (int c = 0; c < n_features; ++c) {
    CHECK_AND_ABORT(
        rt.EncodeThenEncrypt(data_points.GetColumn(c), enc_data_points[c]));
  }

  for (int c = 0; c < n_features; ++c) {
    CHECK_AND_ABORT(rt.EncodeThenEncrypt(centers.GetColumn(c), enc_centers[c]));
  }

  double distance_time{0.};
  std::vector<Ctx> enc_distances(n_centers);  // n * K

  for (int k = 0; k < n_centers; ++k) {
    AutoTimer timer(&distance_time);
    auto _enc_centers(enc_centers);
    if (k > 0) {
      for (auto &enc_center : _enc_centers) {
        CHECK_AND_ABORT(rt.RotateLeft(enc_center, k));
      }
    }

    for (int c = 0; c < n_features; ++c) {
      Ctx temp(enc_data_points.at(c));
      CHECK_AND_ABORT(rt.Sub(temp, _enc_centers.at(c)));
      CHECK_AND_ABORT(rt.Square(temp));
      if (enc_distances[k].size()) {
        CHECK_AND_ABORT(rt.Add(enc_distances[k], temp));
      } else {
        enc_distances[k] = temp;
      }
    }
    CHECK_AND_ABORT(rt.RelinThenRescale(enc_distances[k]));
  }

  double extract_time = 0.;
  // wires[k][i] is the distance between
  // point{i} and center{j = i + k mod K}
  std::vector<lwe::Ctx_st> wires[n_centers];
  {
    AutoTimer timer(&extract_time);
    rt.SlotsToCoeffs(enc_distances.data(), enc_distances.size());
    for (int j = 0; j < n_centers; ++j) {
      CHECK_AND_ABORT(rt.ExtraAllCoefficients(enc_distances[j], wires[j]));
    }
  }

  std::atomic<int> miss_assign{0};
  std::atomic<int> close_assign{0};
  std::atomic<int> correct_assign{0};

  auto worker = [&](int start, int end) {
    for (int i = start; i < end; ++i) {
      // wires[k][i] is the distance between
      // point{i} and center{j = i + k mod K}
      std::vector<lwe::Ctx_st *> distances(n_centers);
      for (int j = 0; j < n_centers; ++j) {
        int k = std::abs(j - i) % n_centers;
        if (j < i && k > 0) k = n_centers - k;
        distances[j] = &wires[k][i];
      }

      lwe::Ctx_st min_elt;
      rt.MinElement(min_elt, distances);

      std::vector<lwe::Ctx_st> indicator(n_centers);
      rt.MulConstant(&min_elt, 1. / n_centers);

      for (int j = 0; j < n_centers; ++j) {
        rt.SubLWECt(indicator[j], *distances[j], min_elt);
      }

      for (int j = 0; j < n_centers; ++j) {
        rt.IsNegative(&indicator[j]);
      }

      double gnd_min = ground_distance.Get(i, 0);
      int gnd_index = 0;
      for (int j = 1; j < n_centers; ++j) {
        if (ground_distance.Get(i, j) < gnd_min) {
          gnd_min = ground_distance.Get(i, j);
          gnd_index = j;
        }
      }
      gnd_min = std::sqrt(gnd_min);

      if (rt.DecryptLWE(indicator[gnd_index]) > 0.9) {
        correct_assign.fetch_add(1);
      } else {
        bool is_close = false;
        for (int j = 0; j < n_centers && !is_close; ++j) {
          if (j == gnd_index) continue;
          if (rt.DecryptLWE(indicator[j]) > 0.9) {
            double d = std::sqrt(ground_distance.Get(i, j));
            if (std::abs(d - gnd_min) < gnd_min / 1000.) {
              is_close = true;
            }
          } 
        }

        if (is_close) {
          close_assign.fetch_add(1);
        } else {
          miss_assign.fetch_add(1);
        }
      }
    }
  };

  double min_index_time{0.};
  {
    AutoTimer timer(&min_index_time);
    size_t num_threads = rt.num_threads();
    const int work_load =
        static_cast<int>((n_points + num_threads - 1) / num_threads);

    ThreadPool pool(num_threads);
    for (size_t i = 0; i < num_threads; ++i) {
      int start = i * work_load;
      int end = std::min(start + work_load, n_points);
      pool.enqueue(worker, start, end);
    }
  }

  printf("distance %f s, extract %f s, min_index %f s\n", distance_time / 1000., extract_time / 1000., min_index_time / 1000.);
  printf("#miss %d, #correct %d #close %d\n", miss_assign.load(), correct_assign.load(), close_assign.load());
}
