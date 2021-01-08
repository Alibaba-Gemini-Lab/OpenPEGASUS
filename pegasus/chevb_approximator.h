#pragma once
#include <memory>
#include "pegasus/common.h"

namespace gemini {
class RunTime;

struct ChebyPoly {
  ChebyPoly() {}

  explicit ChebyPoly(size_t num_coeffs)
      : lead_(false), max_degree_(num_coeffs - 1) {
    coeffs_.resize(num_coeffs, 0.);
  }

  size_t degree() const { return coeffs_.empty() ? 0 : coeffs_.size() - 1; }
  size_t &max_degree() { return max_degree_; }
  size_t max_degree() const { return max_degree_; }
  bool &lead() { return lead_; }
  bool lead() const { return lead_; }
  F64 *data() { return coeffs_.data(); }
  const F64 *data() const { return coeffs_.data(); }
  inline F64 operator[](size_t u) const { return coeffs_.at(u); }
  inline F64 &operator[](size_t u) { return coeffs_.at(u); }

 private:
  bool lead_;
  size_t max_degree_;
  F64Vec coeffs_;
};

class SinApproximatorBase {
 public:
  explicit SinApproximatorBase(std::shared_ptr<RunTime> rt);

  virtual ~SinApproximatorBase();

  virtual Status Apply(Ctx &x, F64 msg_scale) = 0;

  virtual F64 interval_bound() const = 0;

 protected:
  // out = in0 * in1 + in2
  Status MulThenAdd(Ctx &out, const Ctx &in0, const Ctx &in1,
                    const Ctx &in2) const;
  // out = plain * ct0 + ct1
  Status MulThenAdd(Ctx &out, const Ctx &ct0, const F64 plain,
                    const Ctx &ct1) const;
  Status SetToZero(Ctx &ct) const;
  Status AlignModulus(Ctx &lhs, Ctx &rhs) const;
  void InitZeroLike(Ctx &ct, Ctx const &src) const;
  std::shared_ptr<RunTime> rt_;
};

/**
  The algorithm are based on Jean-Philippe Bossuat et al's paper
  Efficient Bootstrapping for Approximate Homomorphic Encryption with Non-Sparse Keys
  The codes are majorly taken from 
  https://github.com/ldsec/lattigo/blob/master/ckks/polynomial_evaluation.go
*/
class BetterSine : public SinApproximatorBase {
 public:
  explicit BetterSine(std::shared_ptr<RunTime> rt);

  ~BetterSine();

  Status Apply(Ctx &enc_x, F64 msg_scale) override;

  F64 interval_bound() const override;

  static size_t Depth();

 private:
  static constexpr size_t num_double_angles();
  static constexpr size_t cheby_degree();

  // Compute the n-th Cheby basis T_{n}
  Status ComputePowerBasisCheby(U64 n, std::vector<Ctx> &power_basis);

  // Evaluate ChebyPoly given the precomputed Cheby basis.
  Status EvalChebyPoly(F64 target_scale, const ChebyPoly &cheby_poly,
                       std::function<const Ctx &(size_t)> cheby_base_getter,
                       Ctx &out) const;

  // Evaluate ChebyPoly given the precomputed Cheby basis using the BGGS
  // algorithm Baby steps: T_{0}, T_{1}, ..., T_{2^l - 1} are precomputed. Giant
  // steps: T_{2^l}, T_{2^(l+1)}, ..., T_{2^(m-1)} are precomputed.
  Status RecurseCheby(
      F64 target_scale, const U64 m, const U64 l, const ChebyPoly &cheby_poly,
      const std::function<const Ctx &(size_t)> &cheby_base_getter,
      Ctx &out) const;

  ChebyPoly chevb_poly_;
};

}  // namespace gemini

