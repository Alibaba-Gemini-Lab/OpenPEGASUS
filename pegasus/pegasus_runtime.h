#include "pegasus/chevb_approximator.h"
#include "pegasus/contrib/ThreadPool.h"
#include "pegasus/gateboot.h"
#include "pegasus/linear_transform.h"
#include "pegasus/lwe.h"
#include "pegasus/repack.h"
#include "pegasus/rlwe.h"
#include "pegasus/runtime.h"

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
    int lvl2_lattice_dim;  // n_ckks}
    int nslots;            // 1 <= nslots <= n_{ckks} / 2
    int nlevels;           // number of levels on CKKS
    double scale = 1.;
    double s2c_multiplier = 1.;
    bool enable_repacking = false;
  };

  static constexpr int KS_DC_BASE = 7;
  static constexpr int SECRET_KEY_HW = 64;
  constexpr int numBitsP0() const { return 45; }

  explicit PegasusRunTime(Parms parms, size_t num_threads);

  double MsgRange() const { return (1L << numBitsP0()) * 0.25 / parms_.scale; }

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
  Status DecryptThenDecode(Ctx const &in, int nslots, VecType &out) const {
    if (nslots == 0 || !IsTwoPower(nslots) ||
        nslots > runtime_->MaximumNSlots()) {
      return Status::ArgumentError("DecryptThenDecode invalid nslots");
    }
    Ptx ptx;
    CHECK_STATUS(runtime_->Decrypt(in, &ptx));
    CHECK_STATUS(runtime_->Decode(ptx, nslots, &out));
    return Status::Ok();
  }

  template <typename VecType>
  Status DecryptThenDecode(Ctx const &in, VecType &out) const {
    return DecryptThenDecode(in, parms_.nslots, out);
  }

  Status SlotsToCoeffs(Ctx *ct, int nct) const;

  Status SlotsToCoeffs(Ctx &out) const;

  Status Repack(Ctx &out, std::vector<lwe::Ctx_st> const &wires) const;

  Status RotateLeft(Ctx &out, size_t offset) const;

  Status Add(Ctx &a, Ctx const &b) const;

  Status Sub(Ctx &a, Ctx const &b) const;

  Status Square(Ctx &a) const;

  Status RelinThenRescale(Ctx &a) const;

  Status ExtraAllCoefficients(const Ctx &in, std::vector<lwe::Ctx_st> &lwe_ct);

  void MulScalarLWECt(lwe::Ctx_st &out, const lwe::Ctx_st &a,
                      uint64_t scalar) const;

  void AddLWECt(lwe::Ctx_st &out, lwe::Ctx_st const &a,
                lwe::Ctx_st const &b) const;

  void SubLWECt(lwe::Ctx_st &out, lwe::Ctx_st const &a,
                lwe::Ctx_st const &b) const;

  void MulConstant(lwe::Ctx_t lvl0_ct, double v) const;

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

#undef DEFINE_MT
#undef DEFINE_LUT

  inline const size_t num_threads() const { return num_threads_; }

  double DecryptLWE(lwe::Ctx_st const &lwe_ct) const;

 protected:
  std::string level2Params(int level_left) const;

  void setUpRuntime();

  void setUpSecretKeys();

  void setUpPublicKeys();

  void setUpFunctors();

 private:
  struct Parms parms_;
  const size_t num_threads_;
  std::shared_ptr<gemini::RunTime> runtime_{nullptr};
  std::shared_ptr<gemini::BetterSine> sinFunctor_{nullptr};
  std::shared_ptr<gemini::LinearTransformer> linearTransformer_{nullptr};

  // Secret keys
  seal::SecretKey lvl0_sk_ntt_, lvl0_sk_non_ntt_;
  seal::SecretKey lvl1_sk_ntt_, lvl1_sk_non_ntt_;
  // the lvl2_sk_ntt is generated in gemini::RunTime
  seal::SecretKey lvl2_sk_non_ntt_;

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
