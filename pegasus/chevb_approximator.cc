#include "pegasus/chevb_approximator.h"

#include <cassert>
#include <vector>

#include "pegasus/runtime.h"
#include "pegasus/types.h"

namespace gemini {

#define PS_CHECK_OK(state)              \
  do {                                  \
    auto __st__ = state;                \
    if (!__st__.IsOk()) {               \
      std::cerr << #state << std::endl; \
      exit(0);                          \
    }                                   \
  } while (0)

#define CHECK_WRAP_ERROR(state)                                     \
  do {                                                              \
    auto __st__ = state;                                            \
    if (!__st__.IsOk()) {                                           \
      std::cerr << __FILE__ << " " << __LINE__ << " " #state << " " \
                << std::endl;                                       \
      return __st__;                                                \
    }                                                               \
  } while (0)

template <typename T>
inline U64 CeilLog2(T x) {
  return static_cast<U64>(std::ceil(std::log2(x)));
}

SinApproximatorBase::SinApproximatorBase(std::shared_ptr<RunTime> rt)
    : rt_(rt) {
  assert(rt_);
}

SinApproximatorBase::~SinApproximatorBase() {}

Status SinApproximatorBase::AlignModulus(Ctx &lhs, Ctx &rhs) const {
  size_t lhs_sze = GetNModuli(lhs);
  size_t rhs_sze = GetNModuli(rhs);
  if (lhs_sze == rhs_sze) {
    return Status::Ok();
  } else if (lhs_sze > rhs_sze) {
    return rt_->DropModuli(&lhs, lhs_sze - rhs_sze);
  } else {
    return rt_->DropModuli(&rhs, rhs_sze - lhs_sze);
  }
}

Status SinApproximatorBase::SetToZero(Ctx &ct) const {
  if (ct.size() > 0) {
    size_t degree = ct.poly_modulus_degree();
    size_t num_moduli = GetNModuli(ct);
    size_t num_elts = seal::util::mul_safe(degree, num_moduli);
    for (size_t k = 0; k < ct.size(); ++k) {
      std::fill_n(ct.data(k), num_elts, 0);
    }
  }
  return Status::Ok();
}

void SinApproximatorBase::InitZeroLike(Ctx &ct, Ctx const &src) const {
  auto cntxt = rt_->SEALRunTime();
  ct.resize(cntxt, src.parms_id(), 2);
  rt_->EncryptZero(&ct);
}

BetterSine::~BetterSine() {}

F64 BetterSine::interval_bound() const { return 21.; }
constexpr size_t BetterSine::cheby_degree() { return 52; }
constexpr size_t BetterSine::num_double_angles() { return 2; }
BetterSine::BetterSine(std::shared_ptr<RunTime> rt)
    : SinApproximatorBase(rt), chevb_poly_(cheby_degree() + 1) {
  // clang-format off
   F64 coeff[] = {0.396630860065802,0.489058736172184,0.773196629989976,0.475397084983612,0.709525350643808,0.449140821658229,0.594705229593757,0.412294534952308,0.427439511035637,0.367561929174841,0.232208791634096,0.318035600768340,0.079305793654738,0.266861799233011,0.072025604761203,0.216928037007623,0.253979483539041,
0.170614239058407,0.464264909744561,0.129634585987730,0.370313383271902,0.094980328050199,-0.067541174068716,0.066957051507958,-0.211063445511307,0.045296234971776,0.275290604453928,0.029312569959899,0.231309682493780,0.018076286280100,-0.399391502544781,0.010573156613113,0.368674389341062,
0.005832430529784,-0.192668987864232,0.003012530760228,0.088204844153296,0.001443722408541,-0.026714429777709,0.000634336236256,0.008910970862054,0.000251434837090,-0.001555591752478,0.000087878330716,0.000539473642650,0.000026166722370,-0.000025325204286,0.000006270912238,0.000025087173609,
0.000001084466507,0.000001298642174,0.000000102331139,0.000000673201927};
  // clang-format on
  F64 cnst_scale = std::pow(0.5 / M_PI, 1. / (1 << num_double_angles()));
  std::transform(coeff, coeff + (sizeof(coeff) / sizeof(coeff[0])),
                 chevb_poly_.data(), [&](F64 u) { return cnst_scale * u; });
  chevb_poly_.lead() = true;
}

size_t BetterSine::Depth() {
  return CeilLog2(cheby_degree()) + num_double_angles();
}

Status BetterSine::ComputePowerBasisCheby(U64 n,
                                          std::vector<Ctx> &power_basis) {
  if (n < 1) {
    return Status::Ok();
  }

  if (power_basis.at(n - 1).size() == 0) {
    U64 a = (n + 1) / 2;
    U64 b = n >> 1;
    U64 c = a > b ? a - b : b - a;
    assert(a > 0 && b > 0 && c < 2);

    PS_CHECK_OK(ComputePowerBasisCheby(a, power_basis));
    PS_CHECK_OK(ComputePowerBasisCheby(b, power_basis));
    PS_CHECK_OK(ComputePowerBasisCheby(c, power_basis));

    const Ctx &Ta = power_basis.at(a - 1);
    const Ctx &Tb = power_basis.at(b - 1);
    Ctx *Tn = &power_basis.at(n - 1);

    CHECK_WRAP_ERROR(rt_->MulRelin(Tn, Ta, Tb));
    CHECK_WRAP_ERROR(rt_->Add(Tn, *Tn));

    if (c == 0) {
      CHECK_WRAP_ERROR(rt_->SubScalar(Tn, 1.));
      CHECK_WRAP_ERROR(rt_->RescaleNext(Tn));
    } else {
      // Dynamic rescaling
      // Computing Tn = 2*T_{a}*T_{b} - T_{1}
      const Ctx &Tc = power_basis[1 - 1];
      if (GetNModuli(*Tn) > GetNModuli(Tc)) {
        CHECK_WRAP_ERROR(rt_->RescaleNext(Tn));
        Tn->scale() = Tc.scale();
        CHECK_WRAP_ERROR(rt_->Sub(Tn, Tc));
      } else {
        Ctx scaled_Tc{Tc};
        CHECK_WRAP_ERROR(
            rt_->MulScalar(&scaled_Tc, 1., Tn->scale() / Tc.scale()));
        CHECK_WRAP_ERROR(rt_->Sub(Tn, scaled_Tc));
        CHECK_WRAP_ERROR(rt_->RescaleNext(Tn));
      }
    }
  }
  return Status::Ok();
}

void SplitCoeff(ChebyPoly const &p, U64 num_coeffs, ChebyPoly &q,
                ChebyPoly &r) {
  // p(x) = q(x) * T_{d} + r(x) where deg(r(x)) < num_coeffs
  r = ChebyPoly(num_coeffs);

  if (p.max_degree() == p.degree()) {
    r.max_degree() = num_coeffs - 1;
  } else {
    r.max_degree() = p.max_degree() - (p.degree() - num_coeffs + 1);
  }

  std::copy_n(p.data(), num_coeffs, r.data());

  q = ChebyPoly(p.degree() - num_coeffs + 1);
  q.max_degree() = p.max_degree();
  q.lead() = p.lead();
  q[0] = p[num_coeffs];

  for (U64 i = num_coeffs + 1, j = 1; i < p.degree() + 1; ++i, ++j) {
    q[i - num_coeffs] = 2. * p[i];
    r[num_coeffs - j] -= p[i];
  }
}

Status BetterSine::Apply(Ctx &ct, F64 msg_scale) {
  if (GetNModuli(ct) <= Depth() + 1) {
    return Status::ArgumentError("level not enough");
  }
  if (!ct.is_ntt_form()) {
    return Status::ArgumentError("require ntt form");
  }

  const F64 r = std::pow(2., num_double_angles());
  const F64 q0 = static_cast<F64>(rt_->GetModulusPrime(0));

  F64 target_scale = ct.scale() *= std::round(q0 / msg_scale);
  const size_t output_level = GetNModuli(ct) - 1 - Depth();

  for (size_t i = 1; i <= num_double_angles(); ++i) {
    U64 qi = rt_->GetModulusPrime(output_level + i);
    target_scale *= qi;
    target_scale = std::sqrt(target_scale);
  }
  ct.scale() = target_scale;

  CHECK_STATUS(rt_->SubScalar(&ct, 0.25 / interval_bound()));

  // Cheby polynomial p(x) of degree d. m = ceil(log(d + 1)), l = floor(m / 2)
  // baby steps: T_{0}, T_{1}, ..., T_{2^l - 1}
  // giant steps: T_{2^l}, T_{2^{l+1}}, ..., T_{2^{m}}
  // Init: T_{0} = 1, T_{1} = x
  // Relation: T_{n} = 2*T_{a}*T_{b}-T_{c} for n=a+b, c=abs(a-b)
  const U64 m = CeilLog2(cheby_degree());
  const U64 l = m / 2;

  std::vector<Ctx> power_basis((1 << m));
  // compute T_{2}, T_{3}, ..., T_{2^l - 1}
  power_basis[0] = ct;
  for (U64 i = 2; i < (1ULL << l); ++i) {
    CHECK_WRAP_ERROR(ComputePowerBasisCheby(i, power_basis));
  }
  // compute T_{2^l}, T_{2^{l+1}}, ..., T_{2^m}
  for (U64 i = l; i <= m; ++i) {
    CHECK_WRAP_ERROR(ComputePowerBasisCheby(1 << i, power_basis));
  }

  auto cheby_base_getter = [&power_basis](size_t u) -> const Ctx & {
    if (u == 0) {
      std::cerr << "cheby_base_getter: index should > 0";
      exit(1);
    }

    if (u > power_basis.size()) {
      std::cerr << "cheby_base_getter: index out-of-bound";
      exit(1);
    }

    if (power_basis[u - 1].size() == 0) {
      std::cerr << "cheby_base_getter: get zero power_basis";
      exit(1);
    }

    return power_basis.at(u - 1);
  };

  CHECK_WRAP_ERROR(
      RecurseCheby(target_scale, m, l, chevb_poly_, cheby_base_getter, ct));

  if (!seal::util::are_close(target_scale, ct.scale())) {
    std::cerr << "RecurseCheby scale changed\n";
    exit(1);
  }

  F64 theta = std::pow(0.5 / M_PI, 1. / r);
  for (size_t i = 0; i < num_double_angles(); ++i) {
    theta *= theta;
    CHECK_WRAP_ERROR(rt_->MulRelin(&ct, ct, ct));
    CHECK_WRAP_ERROR(rt_->Add(&ct, ct));
    CHECK_WRAP_ERROR(rt_->SubScalar(&ct, theta));
    CHECK_WRAP_ERROR(rt_->RescaleNext(&ct));
  }

  ct.scale() /= std::round(q0 / msg_scale);

  return Status::Ok();
}

Status BetterSine::EvalChebyPoly(
    F64 target_scale, const ChebyPoly &cheby_poly,
    std::function<const Ctx &(size_t)> cheby_base_getter, Ctx &out) const {
  const size_t degree = cheby_poly.degree();
  auto round_abs = [](double x) { return std::abs(std::round(x)); };

  if (degree == 0) {
    const Ctx &T1 = cheby_base_getter(1);
    if (round_abs(cheby_poly[0] * target_scale) > 1.) {
      InitZeroLike(out, T1);
      out.scale() = target_scale;
      CHECK_WRAP_ERROR(rt_->AddScalar(&out, cheby_poly[0]));
    } else {
      out.release();
    }
    return Status::Ok();
  }

  // Compute Enc(T_d(x)) = \sum_{i=0}^{d} cheby_poly[i]*Enc(T_{i}(x))
  const Ctx &Td = cheby_base_getter(degree);
  const F64 cipher_scale =
      target_scale * rt_->GetModulusPrime(GetNModuli(Td) - 1);
  bool is_zero = true;

  // Skip cheby_poly[0] here
  for (long i = degree; i > 0; --i) {
    if (round_abs(cheby_poly[i] * target_scale) > 1.) {
      Ctx Ti{cheby_base_getter(static_cast<size_t>(i))};
      const F64 scalar_scale = cipher_scale / Ti.scale();
      CHECK_WRAP_ERROR(rt_->DropModuli(&Ti, GetNModuli(Ti) - GetNModuli(Td)));
      CHECK_WRAP_ERROR(rt_->MulScalar(&Ti, cheby_poly[i], scalar_scale));

      if (out.size() > 0) {
        CHECK_WRAP_ERROR(rt_->Add(&out, Ti));
      } else {
        out = Ti;
      }
      is_zero = false;
    }
  }

  // Finally adding cheby_poly[0]
  if (round_abs(cheby_poly[0] * cipher_scale) > 1.) {
    if (out.size() > 0) {
      CHECK_WRAP_ERROR(rt_->AddScalar(&out, cheby_poly[0]));
    } else {
      InitZeroLike(out, Td);
      out.scale() = cipher_scale;
      CHECK_WRAP_ERROR(rt_->AddScalar(&out, cheby_poly[0]));
    }
    is_zero = false;
  }

  if (is_zero) {
    out.release();
    return Status::Ok();
  } else {
    CHECK_WRAP_ERROR(rt_->RescaleNext(&out));
    return Status::Ok();
  }
}

Status BetterSine::RecurseCheby(
    F64 target_scale, const U64 m, const U64 l, const ChebyPoly &cheby_poly,
    std::function<const Ctx &(size_t)> const &cheby_base_getter,
    Ctx &out) const {
  if (cheby_poly.degree() < (1u << l)) {
    // Optimization for level consumption
    // See https://eprint.iacr.org/2020/1203.pdf
    if (cheby_poly.lead() && l > 1) {
      const long gap = (1L << m) - (1L << (l - 1));
      if (static_cast<long>(cheby_poly.max_degree()) > gap) {
        U64 mm = CeilLog2(cheby_poly.degree() + 1);
        U64 ll = mm / 2;
        CHECK_WRAP_ERROR(RecurseCheby(target_scale, mm, ll, cheby_poly,
                                      cheby_base_getter, out));
        return Status::Ok();
      }
    }

    try {
      CHECK_WRAP_ERROR(
          EvalChebyPoly(target_scale, cheby_poly, cheby_base_getter, out));

      if (out.size() && !seal::util::are_close(target_scale, out.scale())) {
        std::cerr << "EvalChebyPoly on " << cheby_poly.degree()
                  << " scale error\n";
        std::cerr << "Expect " << target_scale << " got " << out.scale()
                  << "\n";
      }

    } catch (std::logic_error e) {
      std::cerr << "EvalChebyPoly error: " << std::string(e.what()) << "\n";
      exit(1);
    }

    return Status::Ok();
  }

  const U64 split_degree = 1UL << (m - 1);
  const Ctx &Td = cheby_base_getter(split_degree);
  if (Td.size() == 0) {
    std::cerr << "power_basis[" << split_degree << "] is not computed";
    exit(1);
  }

  ChebyPoly quotient, remainder;
  SplitCoeff(cheby_poly, split_degree, quotient, remainder);

  size_t level = GetNModuli(Td) - 2;
  if (quotient.lead() && quotient.max_degree() >= split_degree) {
    level += 1;
  }
  const F64 qi = static_cast<F64>(rt_->GetModulusPrime(level));
  const F64 _scale = target_scale * qi / Td.scale();

  // Dirty hack for level optimization.
  Ctx q_ct, r_ct;
  CHECK_WRAP_ERROR(
      RecurseCheby(target_scale, m - 1, l, remainder, cheby_base_getter, r_ct));

  CHECK_WRAP_ERROR(
      RecurseCheby(_scale, m - 1, l, quotient, cheby_base_getter, q_ct));

  if (r_ct.size() && !seal::util::are_close(target_scale, r_ct.scale())) {
    // TODO fix this mismatch scaling factor to improve the accuracy
    r_ct.scale() = target_scale;
  }

  if (q_ct.size() && !seal::util::are_close(_scale, q_ct.scale())) {
    std::cerr << "q(x) scale not matched\n";
    printf("%f != %f\n", q_ct.scale(), _scale);
    exit(1);
  }

  // Compute out = q(X) * T(x) + r(X)
  // Note that q(X) or r(X) could be zero.
  if (q_ct.size() && r_ct.size()) {
    // assert(std::min(GetNModuli(q_ct), GetNModuli(Td)) == level + 1);
    CHECK_WRAP_ERROR(rt_->MulRelin(&out, q_ct, Td));
    // Dynamic Rescaling
    if (GetNModuli(out) <= GetNModuli(r_ct)) {
      CHECK_WRAP_ERROR(rt_->MulScalar(&r_ct, 1., out.scale() / r_ct.scale()));
      CHECK_WRAP_ERROR(rt_->Add(&out, r_ct));
      CHECK_WRAP_ERROR(rt_->RescaleNext(&out));
    } else {
      CHECK_WRAP_ERROR(rt_->RescaleNext(&out));
      if (!seal::util::are_close(out.scale(), r_ct.scale())) {
        out.scale() = r_ct.scale();
      }
      CHECK_WRAP_ERROR(rt_->Add(&out, r_ct));
    }
  } else {
    if (q_ct.size()) {
      // Case 1: quotient term is non-zero
      CHECK_WRAP_ERROR(rt_->MulRelin(&out, q_ct, Td));
      CHECK_WRAP_ERROR(rt_->RescaleNext(&out));
    } else if (r_ct.size()) {
      // Case 2: remainder term is non-zero
      out = r_ct;
    } else {
      // Case 3: both terms are zero
      out.release();
    }
  }

  return Status::Ok();
}

}  // namespace gemini
