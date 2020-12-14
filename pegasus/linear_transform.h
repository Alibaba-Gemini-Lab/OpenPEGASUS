#pragma once
#include <memory>

#include <Eigen/Sparse>

#include "pegasus/common.h"

namespace gemini {
class RunTime;
using C64Mat = Eigen::SparseMatrix<C64>;

/**
 * Linear transform in bootstrapping, e.g., CoeffsToSlots and SlotsToCoeffs.
 * These two transforms will cancel out each other, i.e.,  SlotsToCoeffs(CoeffsToSlots(m)) = m.
 */
class LinearTransformer {
 public:
  struct Parms {
    int nslots;
    /// nbits precision used in CoeffsToSlots
    int c2sPrecision;  
    /// nbits precision used in SlotsToCoeffs
    int s2cPrecision;
    int s2c_lvl_start;
    int c2s_lvl_start;
    /// Extra multipler used in C2S
    F64 c2sMultiplier;
    /// Extra multipler used in S2C
    F64 s2cMultiplier;
  };

  explicit LinearTransformer(Parms parms, std::shared_ptr<RunTime> rt);

  ~LinearTransformer();

  /**
   * Move the coeffients of the input ciphertext into the packing slots.
   *
   * @note Suppose Enc(a0 + a1X + a2X^2 + ... + a_{n-1}X^{n-1}).
   * @note Then t0 encodes the first half [a0, a1, a2, .., a_{n/2-1}] in its slots.
   * @note And t1 encodes the second half [a_{n/2}, ..., a_{n-1}] in its slots.
   */
  Status CoeffsToSlots(const Ctx &in, Ctx *t0, Ctx *t1) const;


  /**
   * The inverse function of CoeffsToSlots.
   *
   * @note t0 contains the first-half coefficients in its slots.
   * @note t1 contains the second-half coefficients in its slots.
   */
  Status SlotsToCoeffs(const Ctx &t0, const Ctx &t1, Ctx *out) const;

  /**
   * Optimized version, can be used when nslots < rt->MaximumNSlots().
   */
  Status CoeffsToSlots(const Ctx &in, Ctx *t0t1) const;
  Status SlotsToCoeffs(const Ctx &t0t1, Ctx *out) const;

  /// Return the number of slots.
  size_t NSlots() const;

  size_t LogRadixN() const;
  size_t depth() const { return LogRadixN(); }

  size_t c2s_lvl_start() const;
  size_t s2c_lvl_start() const;

  F64 C2SMultipler() const;

  F64 S2CMultipler() const;

 private:
  struct ImplBase {
    virtual ~ImplBase(){};
    virtual Status CoeffsToSlots(const Ctx &in, Ctx *t0, Ctx *t1) const = 0;
    virtual Status SlotsToCoeffs(const Ctx &t0, const Ctx &t1, Ctx *out) const = 0;
    virtual Status CoeffsToSlots(const Ctx &in, Ctx *t0t1) const = 0;
    virtual Status SlotsToCoeffs(const Ctx &t0t1, Ctx *out) const = 0;

    std::vector<C64Mat> BuildV0DecompMatrix(long nslots);
  };

  struct RadixImpl;

  Parms parms_;
  std::shared_ptr<ImplBase> impl_;
};

}  // namespace gemini
