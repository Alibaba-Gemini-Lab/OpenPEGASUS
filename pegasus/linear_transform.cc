#include "pegasus/linear_transform.h"

#include <deque>
#include <vector>

#include "pegasus/rotgroup_helper.h"
#include "pegasus/runtime.h"
#include "pegasus/types.h"

namespace gemini {
#define OPT_USE_MONT 1  // Use Montgomery Reduction

#if OPT_USE_MONT
#define CtxPtxMul(rt, ct, pt) rt->MulPlainMontgomery(ct, pt)
#define CtxPtxFMA(rt, acc, ct, pt) rt->FMAMontgomery(acc, ct, pt)
#define EncodePtx(rt, vec, scale, pt) rt->EncodeMontgomery(vec, scale, pt)
#else
#define CtxPtxMul(rt, ct, pt) rt->MulPlain(ct, pt)
#define CtxPtxFMA(rt, acc, ct, pt) rt->FMA(acc, ct, pt)
#define EncodePtx(rt, vec, scale, pt) rt->Encode(vec, scale, pt)
#endif

static int GetDiagonalRotatedScaled(C64Mat const &M, long diagIndex,
                                    long lhsRot, F64 multiplier, C64Vec *diag);

static int GetDiagonalScaled(C64Mat const &M, long diagIndex, F64 multiplier,
                             C64Vec *diag) {
  return GetDiagonalRotatedScaled(M, diagIndex, 0, multiplier, diag);
}

static int GetDiagonalRotated(C64Mat const &M, long diagIndex, long lhsRot,
                              C64Vec *diag) {
  return GetDiagonalRotatedScaled(M, diagIndex, lhsRot, 1., diag);
}

static int GetDiagonal(C64Mat const &M, long diagIndex, C64Vec *diag) {
  return GetDiagonalRotatedScaled(M, diagIndex, 0, 1., diag);
}

std::vector<C64Mat> LinearTransformer::ImplBase::BuildV0DecompMatrix(
    long nslots) {
  struct {
    void BuildPartialMat(C64Mat &A, size_t start_row, size_t start_col,
                         size_t size) {
      const size_t hsize = size / 2;
      const size_t qsize = 4 * size;
      const int gen = 5;
      const C64 omega = std::exp(C64(0., 2. * MM_PI / qsize));

      for (size_t i = 0, genpow = 1; i < hsize; i++) {
        A.coeffRef(start_row + i, start_col + i) = 1.;
        A.coeffRef(start_row + hsize + i, start_col + i) = 1.;

        C64 w = std::pow(omega, genpow);

        A.coeffRef(start_row + i, start_col + hsize + i) = w;
        A.coeffRef(start_row + hsize + i, start_col + hsize + i) = -w;

        genpow *= gen;
        genpow %= qsize;
      }
    }
  } helper;

  size_t log2n = Log2(nslots);
  std::vector<C64Mat> components(log2n);
  for (size_t i = 0; i < log2n; ++i) {
    components[i].resize(nslots, nslots);
    for (size_t j = 0; j < (1UL << i); ++j) {
      size_t size = nslots / (1UL << i);
      helper.BuildPartialMat(components[i], size * j, size * j, size);
    }
  }

  return components;
}

static size_t ProperRadix(int log2n) {
  if (!(log2n & 1) && log2n >= 2) {
    return log2n / 2;
  }
  return log2n;
}

struct LinearTransformer::RadixImpl : public LinearTransformer::ImplBase {
  /// Number of packing slots
  const size_t nslots_;
  /// log2(nslots)
  const size_t log2n_;
  /// log2(radix)
  const size_t log2r_;
  /// Radix for decomposite the transform matrix.
  const size_t radix_;
  /// log_{radix}(nslots), should be 1, or 2.
  const size_t logrn_;
  const size_t c2s_lvl_start_;
  const size_t s2c_lvl_start_;
  inline bool enable_s2c() const { return s2c_lvl_start_ > 0; }
  inline bool enable_c2s() const { return c2s_lvl_start_ > 0; }
  /// Extra multiplier used in CoeffsToSlots. The coeffients will multiplyed by
  /// this multiplier.
  const F64 c2sMultiplier_;
  /// Extra multiplier used in SlotsToCoeffs. The values in slots will
  /// multiplyed by this multiplier.
  const F64 s2cMultiplier_;
  /// CKKS-encoded matrix for S2C and C2S
  std::deque<std::vector<Ptx>> polyV0_;
  std::deque<std::vector<Ptx>> polyInvV0_;

  std::shared_ptr<RunTime> rt_;

  void Init() {
    std::vector<C64Mat> V0Decomp = BuildV0DecompMatrix(nslots_);
    assert(V0Decomp.size() == log2n_);
    std::vector<C64Mat> invV0Decomp(log2n_);
    for (size_t i = 0; i < log2n_; i++) {
      // V0^{-1} = (1/n) * V0^T by definition.
      // We decomposite V0 into `log2(n)` matrices, and thus the divisor is 1/2.
      invV0Decomp[i] = V0Decomp[i].transpose().unaryExpr(
          [](C64 const &v) -> C64 { return std::conj(v) * 0.5; });
    }

    for (size_t i = 0; i < log2n_; i += log2r_) {
      C64Mat &invV0 = invV0Decomp[i];
      for (size_t j = 1; j < log2r_; j++) {
        // Multiplication to the right
        invV0 = invV0Decomp[i + j] * invV0;  // V0^{-1} scaled by 1/(2*radix_).
      }
    }

    if (enable_s2c()) {
      polyV0_.resize(logrn_);
    }
    if (enable_c2s()) {
      polyInvV0_.resize(logrn_);
    }

    if (radix_ > BSGS_THESHOLD_0) {
      ComputePolysStep0_BSGS(invV0Decomp[0]);
    } else {
      ComputePolysStep0(invV0Decomp[0]);
    }
    if (radix_ > BSGS_THESHOLD_1) {
      ComputePolysStep1_BSGS(invV0Decomp);
    } else {
      ComputePolysStep1(invV0Decomp);
    }
  }

  Status ComputePolysStep0_BSGS(C64Mat const &invV0) {
    const F64 v0Scale = static_cast<F64>(radix_ * s2cMultiplier_);
    C64Mat V0 = invV0.transpose().unaryExpr(
        [v0Scale](C64 const &v) { return std::conj(v) * v0Scale; });

    C64Vec diag(nslots_);
    const int gap = nslots_ / radix_;
    const size_t bs = FloorSqrt(radix_);

    if (enable_s2c()) {
      polyV0_[0].resize(radix_);
      F64 s2c_scale = static_cast<F64>(rt_->GetModulusPrime(s2c_lvl_start_));
      for (size_t k = 0; k < radix_; k += bs) {
        const size_t lhs_rot = nslots_ - k * gap;  // left-hand-side rotation
        for (size_t j = k, diag_idx = k * gap; j < radix_;
             ++j, diag_idx += gap) {
          GetDiagonalRotated(V0, diag_idx, lhs_rot, &diag);
          CHECK_STATUS(EncodePtx(rt_, diag, s2c_scale, &polyV0_[0][j]));
        }
      }
    }

    if (enable_c2s()) {
      polyInvV0_[0].resize(radix_);
      F64 c2s_scale = static_cast<F64>(rt_->GetModulusPrime(c2s_lvl_start_));
      for (size_t k = 0; k < radix_; k += bs) {
        const size_t lhs_rot = nslots_ - k * gap;  // left-hand-side rotation
        for (size_t j = k, diag_idx = k * gap; j < radix_;
             ++j, diag_idx += gap) {
          GetDiagonalRotatedScaled(invV0, diag_idx, lhs_rot, c2sMultiplier_,
                                   &diag);
          CHECK_STATUS(EncodePtx(rt_, diag, c2s_scale, &polyInvV0_[0][j]));
        }
      }
    }
    return Status::Ok();
  }

  Status ComputePolysStep0(C64Mat const &invV0) {
    const F64 v0Scale = static_cast<F64>(radix_ * s2cMultiplier_);
    C64Mat V0 = invV0.transpose().unaryExpr(
        [v0Scale](C64 const &v) { return std::conj(v) * v0Scale; });

    F64 s2c_scale = 1., c2s_scale = 1.;
    if (enable_s2c()) {
      polyV0_[0].resize(radix_);
      s2c_scale = static_cast<F64>(rt_->GetModulusPrime(s2c_lvl_start_));
    }

    if (enable_c2s()) {
      polyInvV0_[0].resize(radix_);
      c2s_scale = static_cast<F64>(rt_->GetModulusPrime(c2s_lvl_start_));
    }

    C64Vec diag(nslots_);
    const int gap = nslots_ / radix_;

    if (enable_s2c()) {
      for (size_t j = 0; j < radix_; ++j) {
        GetDiagonal(V0, j * gap, &diag);
        CHECK_STATUS(EncodePtx(rt_, diag, s2c_scale, &polyV0_[0][j]));
      }
    }

    if (enable_c2s()) {
      for (size_t j = 0; j < radix_; ++j) {
        GetDiagonalScaled(invV0, j * gap, c2sMultiplier_, &diag);
        CHECK_STATUS(EncodePtx(rt_, diag, c2s_scale, &polyInvV0_[0][j]));
      }
    }
    return Status::Ok();
  }

  Status ComputePolysStep1_BSGS(std::vector<C64Mat> const &invV0) {
    if (invV0.size() != log2n_) {
      throw std::length_error("ComputePolysStep1_BSGS: invV0 size mismatch");
    }

    if (logrn_ == 1) return Status::Ok();

    const size_t n_comp = 2 * radix_ - 1;
    const size_t bs = FloorSqrt(n_comp);

    C64Vec diag(nslots_);
    for (size_t i = 1, comp_idx = log2r_; i < logrn_; ++i, comp_idx += log2r_) {
      const size_t gap = nslots_ / std::pow(radix_, i + 1);
      F64 v0Scale = static_cast<F64>(radix_ * s2cMultiplier_);
      C64Mat V0 = invV0[comp_idx].transpose().unaryExpr(
          [v0Scale](C64 const &v) { return std::conj(v) * v0Scale; });

      if (enable_s2c()) {
        F64 s2c_scale =
            static_cast<F64>(rt_->GetModulusPrime(s2c_lvl_start_ - i));
        polyV0_[i].resize(n_comp);
        for (size_t k = 0; k < n_comp; k += bs) {
          const long lhs_rot = static_cast<long>(nslots_ - k * gap);

          long diag_idx = static_cast<long>((k + 1 - radix_) * gap);
          for (size_t j = k; j < n_comp; ++j, diag_idx += gap) {
            GetDiagonalRotated(V0, diag_idx, lhs_rot, &diag);
            CHECK_STATUS(EncodePtx(rt_, diag, s2c_scale, &polyV0_[i][j]));
          }
        }
      }

      if (enable_c2s()) {
        F64 c2s_scale =
            static_cast<F64>(rt_->GetModulusPrime(c2s_lvl_start_ - i));
        polyInvV0_[i].resize(n_comp);
        for (size_t k = 0; k < n_comp; k += bs) {
          const long lhs_rot = static_cast<long>(nslots_ - k * gap);

          long diag_idx = static_cast<long>((k + 1 - radix_) * gap);
          for (size_t j = k; j < n_comp; ++j, diag_idx += gap) {
            GetDiagonalRotatedScaled(invV0[comp_idx], diag_idx, lhs_rot,
                                     c2sMultiplier_, &diag);
            CHECK_STATUS(EncodePtx(rt_, diag, c2s_scale, &polyInvV0_[i][j]));
          }
        }
      }
    }
    return Status::Ok();
  }

  Status ComputePolysStep1(std::vector<C64Mat> const &invV0) {
    if (logrn_ == 1) return Status::Ok();

    C64Vec diag(nslots_);
    const int n_comp = 2 * radix_ - 1;
    for (size_t i = 1, comp_idx = log2r_; i < logrn_; ++i, comp_idx += log2r_) {
      const long gap = nslots_ / std::pow(radix_, i + 1);
      const F64 v0Scale = static_cast<F64>(radix_ * s2cMultiplier_);
      C64Mat V0 = invV0[comp_idx].transpose().unaryExpr(
          [v0Scale](C64 const &v) { return std::conj(v) * v0Scale; });

      if (enable_s2c()) {
        F64 s2c_scale =
            static_cast<F64>(rt_->GetModulusPrime(s2c_lvl_start_ - i));
        polyV0_[i].resize(n_comp);
        for (int j = 0; j < n_comp; ++j) {
          GetDiagonal(V0, static_cast<long>((-radix_ + 1 + j) * gap), &diag);
          CHECK_STATUS(EncodePtx(rt_, diag, s2c_scale, &polyV0_[i][j]));
        }
      }

      if (enable_c2s()) {
        F64 c2s_scale =
            static_cast<F64>(rt_->GetModulusPrime(c2s_lvl_start_ - i));
        polyInvV0_[i].resize(n_comp);
        for (int j = 0; j < n_comp; ++j) {
          GetDiagonalScaled(invV0[comp_idx],
                            static_cast<long>((-radix_ + 1 + j) * gap),
                            c2sMultiplier_, &diag);
          CHECK_STATUS(EncodePtx(rt_, diag, c2s_scale, &polyInvV0_[i][j]));
        }
      }
    }
    return Status::Ok();
  }

  Status ApplyLinearTransStep0(Ctx *ct, std::vector<Ptx> const &diag) const {
    if (radix_ > BSGS_THESHOLD_0) {
      return ApplyLinearTransStep0_BSGS(ct, diag);
    }

    CHECK_BOOL(diag.size() != radix_,
               Status::InternalError("ApplyLinearTransStep0"));

    const size_t gap = nslots_ / radix_;
    Ctx origin{*ct}, rot;

    CHECK_STATUS(CtxPtxMul(rt_, ct, diag[0]));
    for (size_t i = 1; i < radix_; ++i) {
      rot = origin;
      CHECK_STATUS(rt_->RotateLeft(&rot, i * gap));
      CHECK_STATUS(CtxPtxFMA(rt_, ct, rot, diag[i]));
    }
    return Status::Ok();
  }

  Status ApplyLinearTransStep1(
      Ctx *ct, std::deque<std::vector<Ptx>> const &diags) const {
    if (radix_ > BSGS_THESHOLD_1) {
      return ApplyLinearTransStep1_BSGS(ct, diags);
    }

    CHECK_BOOL(diags.size() != logrn_,
               Status::InternalError("ApplyLinearTransStep1"));
    CHECK_BOOL(diags[0].size() != radix_,
               Status::InternalError("ApplyLinearTransStep1"));

    if (logrn_ == 1) return Status::Ok();

    for (size_t i = 1; i < logrn_; ++i) {
      CHECK_BOOL(diags[i].size() != 2 * radix_ - 1,
                 Status::InternalError("ApplyLinearTransStep1"));
    }

    Ctx origin, rot;
    for (size_t i = 1, gap = nslots_ / (radix_ * radix_); i < logrn_;
         ++i, gap /= radix_) {
      CHECK_STATUS(rt_->RotateRight(ct, static_cast<long>((radix_ - 1) * gap)));
      origin = *ct;
      CHECK_STATUS(CtxPtxMul(rt_, ct, diags[i][0]));
      for (size_t j = 1; j < 2 * radix_ - 1; ++j) {
        Ctx rot = origin;
        CHECK_STATUS(rt_->RotateLeft(&rot, j * gap));
        CHECK_STATUS(CtxPtxFMA(rt_, ct, rot, diags[i][j]));
      }
    }
    return Status::Ok();
  }

  Status ApplyLinearTransStep0_BSGS(Ctx *ct,
                                    std::vector<Ptx> const &diag) const {
    CHECK_BOOL(ct == nullptr, Status::ArgumentError(
                                  "ApplyLinearTransStep0_BSGS: null pointer"));
    CHECK_BOOL(diag.size() != radix_,
               Status::InternalError("ApplyLinearTransStep0_BSGS"));

    const size_t bs = FloorSqrt(radix_);
    const size_t gs = CeilDiv(radix_, bs);
    const size_t gap = nslots_ / radix_;

    std::vector<Ctx> rotated_ct(bs, *ct);
    for (size_t j = 1; j < bs; ++j) {  // baby-step
      CHECK_STATUS(rt_->RotateLeft(&rotated_ct[j], j * gap));
    }

    for (size_t i = 0; i < gs && i * bs < radix_; ++i) {
      const size_t offset = i * bs;

      Ctx inner{rotated_ct[0]};
      CHECK_STATUS(CtxPtxMul(rt_, &inner, diag[offset]));
      for (size_t j = 1; j < bs && j + offset < radix_; j++) {
        /// inner += rotated_ct[j] * diags[offset + j]
        CHECK_STATUS(CtxPtxFMA(rt_, &inner, rotated_ct[j], diag[offset + j]));
      }

      if (i == 0) {
        *ct = inner;
      } else {  // giant-step
        CHECK_STATUS(rt_->RotateLeft(&inner, i * bs * gap));
        CHECK_STATUS(rt_->Add(ct, inner));
      }
    }
    return Status::Ok();
  }

  Status ApplyLinearTransStep1_BSGS(
      Ctx *ct, std::deque<std::vector<Ptx>> const &diags) const {
    CHECK_BOOL(diags.size() != logrn_,
               Status::InternalError("ApplyLinearTransStep1_BSGS"));
    CHECK_BOOL(diags[0].size() != radix_,
               Status::InternalError("ApplyLinearTransStep1_BSGS"));
    if (logrn_ == 1) return Status::Ok();

    for (size_t i = 1; i < logrn_; ++i) {
      CHECK_BOOL(diags[i].size() != 2 * radix_ - 1,
                 Status::InternalError("ApplyLinearTransStep1_BSGS"));
    }

    const size_t n_comp = diags[1].size();
    const size_t bs = FloorSqrt(n_comp);
    const size_t gs = CeilDiv(n_comp, bs);

    for (size_t i = 1; i < logrn_; ++i) {
      size_t gap = nslots_ / std::pow(radix_, i + 1);
      Ctx temp{*ct};
      CHECK_STATUS(rt_->RotateRight(&temp, (radix_ - 1) * gap));
      std::vector<Ctx> rotated_ct(bs, temp);

      for (size_t j = 1; j < bs; ++j) {  // baby-step
        CHECK_STATUS(rt_->RotateLeft(&rotated_ct[j], j * gap));
      }

      for (size_t j = 0; j < gs && j * bs < n_comp; ++j) {
        const size_t offset = j * bs;
        Ctx inner{rotated_ct[0]};
        CHECK_STATUS(CtxPtxMul(rt_, &inner, diags[i].at(offset)));

        for (size_t k = 1; k < bs && offset + k < n_comp; k++) {
          // inner += rotated_ct[k] * diags[i][offset + k]
          CHECK_STATUS(
              CtxPtxFMA(rt_, &inner, rotated_ct[k], diags[i].at(offset + k)));
        }

        if (j == 0) {
          *ct = inner;
        } else {  // giant-step
          CHECK_STATUS(rt_->RotateLeft(&inner, j * bs * gap));
          CHECK_STATUS(rt_->Add(ct, inner));
        }
      }
    }
    return Status::Ok();
  }

 public:
  /// When radix > BSGS_THESHOLD_0, to use BSGS algorithm in Step0.
  static constexpr size_t BSGS_THESHOLD_0 = 4;
  /// When radix > BSGS_THESHOLD_1, to use BSGS algorithm in Step1.
  static constexpr size_t BSGS_THESHOLD_1 = 2;

  explicit RadixImpl(LinearTransformer::Parms parms,
                     std::shared_ptr<RunTime> rt)
      : nslots_(parms.nslots),
        log2n_(Log2(static_cast<U64>(nslots_))),
        log2r_(ProperRadix(log2n_)),
        radix_(static_cast<size_t>(1) << log2r_),
        logrn_(log2n_ / log2r_),
        c2s_lvl_start_(parms.c2s_lvl_start),
        s2c_lvl_start_(parms.s2c_lvl_start),
        c2sMultiplier_(std::pow(parms.c2sMultiplier, 1. / logrn_)),
        s2cMultiplier_(std::pow(parms.s2cMultiplier, 1. / logrn_)),
        rt_(rt) {
    if ((1UL << log2n_) != nslots_) {
      throw std::invalid_argument(
          "LinearTransformer: nslots should be 2-power number.");
    }

    if (nslots_ < 2) {
      throw std::invalid_argument("LinearTransformer: nslots should >= 2.");
    }

    if (std::pow(radix_, logrn_) != nslots_) {
      throw std::invalid_argument(
          "LinearTransformer: this nslots is not supported.");
    }

    size_t max_n_moduli = rt->MaximumNModuli();
    if (enable_s2c()) {
      if (s2c_lvl_start_ > max_n_moduli || s2c_lvl_start_ < logrn_) {
        throw std::invalid_argument(
            "LinearTransformer: invalid s2c_lvl_start.");
      }
    }
    if (enable_c2s()) {
      if (c2s_lvl_start_ > max_n_moduli || c2s_lvl_start_ < logrn_) {
        throw std::invalid_argument(
            "LinearTransformer: invalid c2s_lvl_start.");
      }
    }

    if (enable_c2s() || enable_s2c()) {
      Init();
    }
  }

  ~RadixImpl() {}

  Status CoeffsToSlots(Ctx const &in, Ctx *t0, Ctx *t1) const override {
    if (!enable_c2s()) {
      return Status::NotReady("CoeffsToSlots is not ready");
    }
    if (GetNModuli(in) != c2s_lvl_start_ + 1) {
      std::cerr << "WARN: CoeffsToSlots in.level != c2s_lvl_start" << std::endl;
    }
    CHECK_BOOL(t0 == nullptr,
               Status::ArgumentError("CoeffsToSlots: Null pointer"));
    CHECK_BOOL(t1 == nullptr,
               Status::ArgumentError("CoeffsToSlots: Null pointer"));
    CHECK_STATUS_INTERNAL(applyInvV0(in, t0), "CoeffsToSlots fail");
    // Seperate the real part and the image part.
    // t0 + t1*j + (t0 - t1j) = 2t0
    // t0 - t1*j - (t0 + t1j) = -2t1*j
    *t1 = *t0;
    CHECK_STATUS(rt_->Conjugate(t1));
    CHECK_STATUS(rt_->AddSub(t0, t1));
    CHECK_STATUS(rt_->MulImageUnit(t1));
    return Status::Ok();
  }

  Status CoeffsToSlots(Ctx const &in, Ctx *t0t1) const override {
    if (!enable_c2s()) {
      return Status::NotReady("CoeffsToSlots is not ready");
    }
    if (GetNModuli(in) != c2s_lvl_start_ + 1) {
      std::cerr << "WARN: CoeffsToSlots in.level != c2s_lvl_start" << std::endl;
    }
    CHECK_BOOL(t0t1 == nullptr,
               Status::ArgumentError("CoeffsToSlots: Null pointer"));
    CHECK_BOOL(nslots_ > rt_->MaximumNSlots(),
               Status::ArgumentError(
                   "Optimized CoeffsToSlots requires a smaller nslots."));

    CHECK_STATUS_INTERNAL(applyInvV0(in, t0t1),
                          "CoeffsToSlots fail: applyInvV0");
    // [a + bj | 0] -> [a - bj | 0] - > [b + aj | 0] -> [a + bj | b + aj]
    Ctx temp{*t0t1};
    CHECK_STATUS_INTERNAL(rt_->Conjugate(&temp),
                          "CoeffsToSlots fail: Conjugate");
    CHECK_STATUS_INTERNAL(rt_->MulImageUnit(&temp),
                          "CoeffsToSlots fail: MulImageUnit");
    CHECK_STATUS_INTERNAL(rt_->RotateRight(&temp, nslots_ / 2),
                          "CoeffsToSlots fail: RotateRight");
    CHECK_STATUS_INTERNAL(rt_->Add(t0t1, temp), "CoeffsToSlots fail: Add");

    // [a + bj | b + aj] -> [a - bj | b - aj]
    // [2a | 2b]
    temp = *t0t1;
    CHECK_STATUS_INTERNAL(rt_->Conjugate(&temp),
                          "CoeffsToSlots fail: Conjugate");
    CHECK_STATUS_INTERNAL(rt_->Add(t0t1, temp), "CoeffsToSlots fail: Add");

    return Status::Ok();
  }

  Status applyV0(Ctx *out) const {
    // a + bj -> V0 * (a + bj)
    assert(!polyV0_.empty());
    CHECK_STATUS(ApplyLinearTransStep1(out, polyV0_));
    for (int i = 1; i < logrn_; ++i) {
      CHECK_STATUS(rt_->RescaleNext(out));
    }
    CHECK_STATUS(ApplyLinearTransStep0(out, polyV0_[0]));
    CHECK_STATUS(rt_->RescaleNext(out));
    return Status::Ok();
  }

  Status applyInvV0(Ctx const &in, Ctx *out) const {
    // V0^{-1} * in := t0 + t1*j
    // where `t0` is the first half of coeffients, and `t1` is the second half.
    assert(!polyInvV0_.empty());
    *out = in;
    CHECK_STATUS(ApplyLinearTransStep0(out, polyInvV0_[0]));
    CHECK_STATUS(rt_->RescaleNext(out));
    CHECK_STATUS(ApplyLinearTransStep1(out, polyInvV0_));
    for (int i = 1; i < logrn_; ++i) {
      CHECK_STATUS(rt_->RescaleNext(out));
    }
    return Status::Ok();
  }

  Status SlotsToCoeffs(Ctx const &t0, const Ctx &t1, Ctx *out) const override {
    if (!enable_s2c()) {
      return Status::NotReady("SlotsToCoeffs is not ready");
    }
    if (GetNModuli(t0) != s2c_lvl_start_ + 1) {
      std::cerr << "WARN: SlotsToCoeffs ct.level != s2c_lvl_start" << std::endl;
    }
    CHECK_BOOL(out == nullptr,
               Status::ArgumentError("SlotsToCoeffs: Null pointer"));
    CHECK_BOOL(t0.parms_id() != t1.parms_id(),
               Status::ArgumentError("SlotsToCoeffs: t0 t1 param mismatch"));
    // t0, t1 -> t0 + t1*1j
    *out = t1;
    CHECK_STATUS_INTERNAL(rt_->MulImageUnit(out),
                          "SlotsToCoeffs fail: MulImageUnit");
    CHECK_STATUS_INTERNAL(rt_->Add(out, t0), "SlotsToCoeffs fail: Add");
    CHECK_STATUS_INTERNAL(applyV0(out), "SlotsToCoeffs fail: applyV0");

    return Status::Ok();
  }

  Status SlotsToCoeffs(Ctx const &t0t1, Ctx *out) const override {
    if (!enable_s2c()) {
      return Status::NotReady("SlotsToCoeffs is not ready");
    }
    if (GetNModuli(t0t1) != s2c_lvl_start_ + 1) {
      std::cerr << "WARN: SlotsToCoeffs ct.level != s2c_lvl_start" << std::endl;
    }
    CHECK_BOOL(out == nullptr,
               Status::ArgumentError("SlotsToCoeffs: Null pointer"));

    // [t0 | t1] -> [t1 | t0] -> [t0 + t1*1j | t1 + t0*1j]
    *out = t0t1;
    CHECK_STATUS_INTERNAL(rt_->RotateRight(out, nslots_ / 2),
                          "SlotsToCoeffs fail: RotateRight");
    CHECK_STATUS_INTERNAL(rt_->MulImageUnit(out),
                          "SlotsToCoeffs fail: MulImageUnit");
    CHECK_STATUS_INTERNAL(rt_->Add(out, t0t1), "SlotsToCoeffs fail: Add");
    CHECK_STATUS_INTERNAL(applyV0(out), "SlotsToCoeffs fail: applyV0");

    return Status::Ok();
  }
};

int GetDiagonalRotatedScaled(const C64Mat &A, long diag_idx, long lhs,
                             F64 multiplier, C64Vec *diag) {
  if (!diag) return 0;
  const long ncols = A.cols();
  const long nrows = A.rows();
  if (ncols != nrows) {
    throw std::length_error("GetDiagonal: require square matrix");
  }
  diag->resize(nrows);
  diag_idx %= ncols;  // diag_idx \in (-ncols, ncols)
  lhs %= ncols;       // lhs \in (-ncols, ncols)

  F64 min = std::numeric_limits<F64>::max();
  for (long row = 0; row < nrows; row++) {
    long col = row + diag_idx;
    if (col < 0) col += ncols;
    if (col >= ncols) col -= ncols;
    long dst = row - lhs + ncols;
    if (dst >= ncols) dst -= ncols;

    C64 c = A.coeff(row, col);
    diag->at(dst) = c;

    c.real(std::abs(c.real()));
    c.imag(std::abs(c.imag()));
    if (!Ft64::AlmostEquals(c.real(), 0.)) min = std::min(min, c.real());
    if (!Ft64::AlmostEquals(c.imag(), 0.)) min = std::min(min, c.imag());
  }

  if (!Ft64::AlmostEquals(multiplier, 1.)) {
    for (auto &v : *diag) {
      v.real(v.real() * multiplier);
      v.imag(v.imag() * multiplier);
    }
    min *= multiplier;
  }
  return std::log2(std::abs(min));
}

LinearTransformer::LinearTransformer(Parms parms, std::shared_ptr<RunTime> rt)
    : parms_(parms) {
  impl_ = std::make_shared<RadixImpl>(parms_, rt);
}

LinearTransformer::~LinearTransformer() {}

Status LinearTransformer::CoeffsToSlots(const Ctx &in, Ctx *t0, Ctx *t1) const {
  return impl_->CoeffsToSlots(in, t0, t1);
}

Status LinearTransformer::SlotsToCoeffs(const Ctx &t0, const Ctx &t1,
                                        Ctx *out) const {
  return impl_->SlotsToCoeffs(t0, t1, out);
}

Status LinearTransformer::CoeffsToSlots(const Ctx &in, Ctx *t0t1) const {
  return impl_->CoeffsToSlots(in, t0t1);
}

Status LinearTransformer::SlotsToCoeffs(const Ctx &t0t1, Ctx *out) const {
  return impl_->SlotsToCoeffs(t0t1, out);
}

size_t LinearTransformer::NSlots() const { return parms_.nslots; }

size_t LinearTransformer::LogRadixN() const {
  return dynamic_cast<RadixImpl *>(impl_.get())->logrn_;
}

size_t LinearTransformer::c2s_lvl_start() const {
  return dynamic_cast<RadixImpl *>(impl_.get())->c2s_lvl_start_;
}

size_t LinearTransformer::s2c_lvl_start() const {
  return dynamic_cast<RadixImpl *>(impl_.get())->s2c_lvl_start_;
}

F64 LinearTransformer::C2SMultipler() const { return parms_.c2sMultiplier; }

F64 LinearTransformer::S2CMultipler() const { return parms_.s2cMultiplier; }
}  // namespace gemini

