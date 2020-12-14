#include <unordered_map>
#include "pegasus/rlwe.h"
namespace rlwe {

class LWEGateBooter {
 public:
  explicit LWEGateBooter(const double msg_scale, const BK &bk, const RtPtr rlwe_rt, const double post_multiplier = 1.);

  void Abs(RLWE2LWECt_t lwe_N, const lwe::Ctx_t lwe_n) const;

  void Sigmoid(RLWE2LWECt_t lwe_N, const lwe::Ctx_t lwe_n) const;

  void Tanh(RLWE2LWECt_t lwe_N, const lwe::Ctx_t lwe_n) const;

  void AbsLog(RLWE2LWECt_t lwe_N, const lwe::Ctx_t lwe_n) const;

  void IsZeroThenTag(RLWE2LWECt_t lwe_N, const lwe::Ctx_t lwe_n, double tag) const;

  // f(x) = sqrt(|x|)
  void AbsSqrt(RLWE2LWECt_t lwe_N, const lwe::Ctx_t lwe_n) const;

  // f(x) = 1/sqrt(|x + e|)
  void InvAbsSqrt(RLWE2LWECt_t lwe_N, const lwe::Ctx_t lwe_n) const;

  void Inverse(RLWE2LWECt_t lwe_N, const lwe::Ctx_t lwe_n) const;

  void Sign(RLWE2LWECt_t lwe_N, const lwe::Ctx_t lwe_n) const;

  void IsNegative(RLWE2LWECt_t lwe_N, const lwe::Ctx_t lwe_n) const;

  void LeakyReLU(RLWE2LWECt_t lwe_N, const lwe::Ctx_t lwe_n) const;

  void ReLU(RLWE2LWECt_t lwe_N, const lwe::Ctx_t lwe_n) const;

  void Identity(RLWE2LWECt_t lwe_N, const lwe::Ctx_t lwe_n) const;

  void Clip(RLWE2LWECt_t lwe_N, const lwe::Ctx_t lwe_n) const;

  void Exponent(RLWE2LWECt_t lwe_N, const lwe::Ctx_t lwe_n) const;

  void MulConstant(RLWE2LWECt_t lwe_N, const lwe::Ctx_t lwe_n, const double v) const;

  void GreaterThan(RLWE2LWECt_t lwe_N, const lwe::Ctx_t lwe_n, const double v) const;

  double GetPostMultiplier() const { return post_multiplier; }

 private:
  void SetPostMultiplier(double multiplier);

  void InitLUT(std::vector<uint64_t> &lut, size_t len, std::function<double(double)> func) const;

  void Apply(RLWE2LWECt_t lwe_N_ct, const lwe::Ctx_t lwe_n_ct, const std::vector<uint64_t> &LUT) const;

  // X^{b} * LUT
  void InitTestVector(Ctx &acc, lwe::T b, const std::vector<uint64_t> &LUT) const;

  // ct * (X^{ai} - 1)
  void MultiplyXaiMinusOne(Ctx &ct, uint64_t ai) const;

 private:
  const double msg_scale;
  double post_multiplier = 0.;
  const BK &bk;
  const RtPtr rlwe_rt;

  std::unordered_map<std::string, std::vector<uint64_t>> LUTs;
};

};  // namespace rlwe
