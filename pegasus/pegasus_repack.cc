#include <seal/util/common.h>
#include <seal/util/polyarithsmallmod.h>

#include <chrono>
#include <iostream>
#include <random>
#include <string>

#include "pegasus/chevb_approximator.h"
#include "pegasus/gateboot.h"
#include "pegasus/linear_transform.h"
#include "pegasus/runtime.h"

#define BETTER_REPACK 1

class AutoTimer {
 public:
  using Time_t = std::chrono::nanoseconds;
  using Clock = std::chrono::high_resolution_clock;
  explicit AutoTimer(double *ret) : ret_(ret) { stamp_ = Clock::now(); }

  AutoTimer(double *ret, std::string const &tag_)
      : verbose(true), tag(tag_), ret_(ret) {
    stamp_ = Clock::now();
  }

  void reset() { stamp_ = Clock::now(); }

  void stop() {
    if (ret_) *ret_ += (Clock::now() - stamp_).count() / 1.0e6;
    if (verbose && ret_) std::cout << tag << " " << (*ret_) << "\n";
  }

  ~AutoTimer() { stop(); }

 protected:
  bool verbose = false;
  std::string tag;
  double *ret_ = nullptr;
  Clock::time_point stamp_;
};

void GenerateHammingSecretKey(seal::SecretKey &sk, int hwt,
                              std::shared_ptr<seal::SEALContext> context) {
  using namespace seal;
  auto &context_data = *context->key_context_data();
  auto &parms = context_data.parms();
  auto &coeff_modulus = parms.coeff_modulus();
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
  uint64_t *dst_ptr = sk.data().data();
  for (size_t j = 0; j < coeff_modulus_size; j++) {
    const std::uint64_t neg_one = coeff_modulus[j].value() - 1;
    const std::uint64_t xor_one = neg_one ^ 1UL;
    std::memset(dst_ptr, 0, sizeof(*dst_ptr) * coeff_count);
    for (size_t i = 0; i < hwt; ++i) {
      dst_ptr[picked[i]] = [xor_one, neg_one](bool b) {
        // b = true -> c = 0xFF -> one
        // b = false -> c = 0x00 -> neg_one
        uint64_t c = -static_cast<uint64_t>(b);
        return (xor_one & c) ^ neg_one;
      }(rnd2[i]);
    }
    dst_ptr += coeff_count;
  }

  util::RNSIter secret_key(sk.data().data(), coeff_count);
  auto ntt_tables = context_data.small_ntt_tables();
  util::ntt_negacyclic_harvey(secret_key, coeff_modulus_size, ntt_tables);

  // Set the parms_id for secret key
  sk.parms_id() = context_data.parms_id();

  for (size_t i = 0; i < hwt; ++i) {
    rnd2[i] = 0;
  }
  std::memset(picked.data(), 0, sizeof(picked[0]) * picked.size());
}
namespace gemini {

typedef struct RpKey {
  using T = Ctx;
  // rlwe::GSWCt key_;
  // rlwe::GSWCt key_;
  T key_;
  std::vector<T> rotated_keys_;
  double scale_ = 1.;

  const T &key() const { return key_; }
  const T &rotated_key(size_t idx) const {
    return idx == 0 ? key_ : rotated_keys_.at(idx - 1);
  }
  T &key() { return key_; }

  double &scale() { return scale_; }
  double scale() const { return scale_; }

} RpKey;

Status RpKeyInit(RpKey &rpk, const double scale, const seal::SecretKey &rlwe_sk,
                 const seal::SecretKey &rgsw_sk_non_ntt,
                 const std::shared_ptr<RunTime> runtime) {
  auto rlwe_rt = runtime->SEALRunTime();
  const size_t nslots = runtime->MaximumNSlots();
  const size_t NN = rgsw_sk_non_ntt.data().coeff_count();
  if (nslots < NN) {
    throw std::invalid_argument("RpKeyInit: require N >= 2*N'");
  }
  auto cast_to_double = [](uint64_t u) { return u > 1 ? -1. : (double)u; };

  if (!seal::is_metadata_valid_for(rlwe_sk, rlwe_rt)) {
    throw std::invalid_argument("RpKeyInit: invalid rlwe_sk");
  }

  std::vector<double> slots(NN, 0.);
  for (size_t i = 0; i < NN; ++i) {
    slots[i] = cast_to_double(rgsw_sk_non_ntt.data()[i]);
  }

  seal::Plaintext ptx;
  ptx.parms_id() = rlwe_rt->first_parms_id();
  runtime->Encode(slots, scale, &ptx);

  runtime->Encrypt(ptx, &rpk.key());
  rpk.scale() = scale;

  const size_t g = CeilSqrt(NN);
  rpk.rotated_keys_.resize(g - 1, rpk.key());
  for (size_t j = 1; j < g; ++j) {
    CHECK_STATUS_INTERNAL(runtime->RotateLeft(&rpk.rotated_keys_[j - 1], j),
                          "RotateLeft");
  }

  auto Montgomerize = [&ptx, runtime](Ctx &ctx) {
    if (ptx.coeff_count() !=
        ctx.poly_modulus_degree() * ctx.coeff_modulus_size()) {
      throw std::invalid_argument("ptx length mismatch");
    }

    for (size_t i = 0; i < ctx.size(); ++i) {
      std::copy_n(ctx.data(i), ptx.coeff_count(), ptx.data());
      runtime->Montgomerize(&ptx);
      std::copy_n(ptx.data(), ptx.coeff_count(), ctx.data(i));
    }
  };

  Montgomerize(rpk.key_);
  for (auto &ctx : rpk.rotated_keys_) {
    Montgomerize(ctx);
  }

  std::fill_n(slots.data(), slots.size(), 0.);  // clean up secret material
  return Status::Ok();
}

/// Wrap an LWE array to a matrix like object
struct LWECtArrayWrapper {
  explicit LWECtArrayWrapper(const std::vector<rlwe::RLWE2LWECt_st> &array,
                             uint64_t p0, double multiplier)
      : lwe_N_ct_array_(array),
        p0_(p0),
        p0half_(p0 >> 1),
        multiplier_(multiplier) {
    if (lwe_N_ct_array_.empty()) {
      throw std::invalid_argument("LWECtArrayWrapper: Empty array");
    }
    const size_t N = array.front().a.coeff_count();
    for (const auto &lwe_N_ct : lwe_N_ct_array_) {
      if ((lwe_N_ct.a.coeff_count() != N) || lwe_N_ct.b.size() != 1) {
        throw std::invalid_argument(
            "LWECtArrayWrapper: inconsistent LWE cipher");
      }
    }
  }

  size_t rows() const { return lwe_N_ct_array_.size(); }
  size_t cols() const { return lwe_N_ct_array_.front().a.coeff_count(); }

  inline double preprocess(uint64_t u) const {
    int64_t su = u >= p0half_ ? u - p0_ : u;
    return static_cast<double>(su) * multiplier_;
  }

  inline double operator()(size_t r, size_t c) const {
    return preprocess(*lwe_N_ct_array_.at(r).a.data(c));
  }
  inline double Get(size_t r, size_t c) const { return (*this)(r, c); }

  void GetDiagnoal(F64Vec &diag, size_t diagIdx) const {
    const size_t ncols = cols();
    const size_t nrows = rows();
    const size_t colMask = (ncols - 1);
    const size_t nrepeat = ncols / nrows;

    diag.resize(ncols);
    auto iter = diag.begin();

    for (size_t j = 0; j < nrepeat; ++j) {
      const size_t offset = j * nrows;
      for (size_t k = 0; k < nrows; ++k) {
        *iter++ = Get(k, (offset + diagIdx + k) & colMask);
      }
    }
  }

  void GetLastColumn(F64Vec &column) const {
    const size_t ncols = cols();
    const size_t nrows = rows();
    column.resize(ncols);
    for (size_t i = 0; i < nrows; ++i) {
      column[i] = preprocess(lwe_N_ct_array_[i].b[0]);
    }

    for (size_t i = nrows; i < ncols; i += nrows) {
      for (size_t k = 0; k < nrows; ++k) column[i + k] = column[k];
    }
  }

  const std::vector<rlwe::RLWE2LWECt_st> &lwe_N_ct_array_;
  const uint64_t p0_, p0half_;
  const double multiplier_;
};

template <class MatrixLike>
Status GetMatrixDiagnoal(F64Vec &diag, size_t diagIndex,
                         const MatrixLike &matrix,
                         const std::shared_ptr<RunTime> runtime) {
  const size_t nrows = matrix.rows();
  if (nrows == 0 || !IsTwoPower(nrows)) {
    return Status::ArgumentError(
        "GetMatrixDiagnoal: Invalid number of LWE cipher to pack");
  }
  const size_t ncols = matrix.cols();
  const size_t colMask = (ncols - 1);
  const size_t nrepeat = ncols / nrows;

  assert(IsTwoPower(ncols));
  if (nrows > ncols) {
    return Status::ArgumentError(
        "GetMatrixDiagnoal: Too many LWE ciphers to pack");
  }

  if (ncols > runtime->MaximumNSlots()) {
    return Status::ArgumentError(
        "GetMatrixDiagnoal: Invalid LWE cipher to pack");
  }

  if (diagIndex >= nrows) {
    return Status::ArgumentError("GetMatrixDiagnoal: Invalid diagonal index");
  }

  diag.resize(ncols);
  auto iter = diag.begin();
  for (size_t j = 0; j < nrepeat; ++j) {
    const size_t offset = j * nrows;
    for (size_t k = 0; k < nrows; ++k) {
      *iter++ =
          static_cast<double>(matrix(k, offset + diagIndex + k & colMask));
    }
  }
  return Status::Ok();
}

/**
 * Input
 *     [a0, b0, c0, ... | a1, b1, c1, ... | .... ]
 *      |<-- stride -->
 *      |<---------------  nslots -------------->
 *
 * Output
 *     [\sum ai, \sum bi, \sum ci, ...]
 *
 * Requirement
 *     nslots/stride is a 2-exponent value.
 */
Status SumStridedVectors(Ctx &enc_vec, size_t stride, size_t nslots,
                         const std::shared_ptr<RunTime> runtime) {
  const size_t vecLength = nslots / stride;
  if (vecLength == 0 || !IsTwoPower(vecLength)) {
    return Status::ArgumentError("SumStridedVectors: Invalid stride");
  }
  const size_t nsteps = static_cast<size_t>(Log2(vecLength));

  // Almost the TotalSum in HElib with a stride multiplier.
  for (size_t i = 0; i < nsteps; ++i) {
    auto copy{enc_vec};
    CHECK_STATUS_INTERNAL(runtime->RotateLeft(&copy, (1U << i) * stride),
                          "SumStridedVectors: RotateLeft failed");
    CHECK_STATUS_INTERNAL(runtime->Add(&enc_vec, copy),
                          "SumStridedVectors: Add failed");
  }
  return Status::Ok();
}

Status MultiplyPlainMatrixCipherVector(Ctx &out,
                                       const std::shared_ptr<RunTime> runtime) {
  using MatType =
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
  size_t nrows = 32;
  size_t ncols = 512;
  MatType mat = MatType::Random(nrows, ncols);
  MatType vec = MatType::Random(ncols, 1);
  MatType mul = mat * vec;

  const size_t g = CeilSqrt(nrows);
  const size_t h = CeilDiv(nrows, g);
  std::vector<F64Vec> rotated_vec(g);

  for (size_t k = 0; k < g; ++k) {
    rotated_vec[k].resize(vec.size());
    std::rotate_copy(vec.data(), vec.data() + k, vec.data() + vec.size(),
                     rotated_vec[k].data());
  }

  F64Vec diag;
  std::vector<double> accum(ncols, 0.);

  for (size_t k = 0; k < h && g * k < nrows; ++k) {
    std::vector<double> inner(ncols, 0.);
    for (size_t j = 0, diagIdx = g * k; j < g && diagIdx < nrows;
         ++j, ++diagIdx) {
      // Obtain the diagonal from LWE matrix
      CHECK_STATUS_INTERNAL(GetMatrixDiagnoal(diag, diagIdx, mat, runtime),
                            "MultiplyPlainMatrixCipherVector");
      // RHS rotated by g * k
      std::rotate(diag.rbegin(), diag.rbegin() + g * k, diag.rend());

      for (size_t i = 0; i < ncols; ++i) {
        inner[i] += diag[i] * rotated_vec.at(j)[i];  // LHS-rotated vector
      }
    }

    if (k > 0) {
      // LHS rotated
      std::rotate(inner.begin(), inner.begin() + g * k, inner.end());
    }

    for (size_t i = 0; i < ncols; ++i) {
      accum[i] += inner[i];
    }
  }

  for (size_t j = nrows; j < ncols; j += nrows) {
    for (size_t i = 0; i < nrows; ++i) {
      accum[i] += accum[i + j];
    }
  }

  for (size_t i = 0; i < nrows; ++i) std::cout << accum[i] << " ";
  std::cout << "\n";
  for (size_t i = 0; i < nrows; ++i) std::cout << mul(i) << " ";
  std::cout << "\n";

  return Status::Ok();
}

#if BETTER_REPACK
Status Repack(Ctx &out, double bound,
              const std::vector<rlwe::RLWE2LWECt_st> &lwe_N_ct_array,
              const RpKey &rpk, const std::shared_ptr<RunTime> runtime) {
  auto seal_rt = runtime->SEALRunTime();
  if (lwe_N_ct_array.empty()) {
    Status::ArgumentError("Repack: empty LWE cipher array");
  }

  const uint64_t p0 = runtime->GetModulusPrime(0);
  LWECtArrayWrapper lweMatrix(lwe_N_ct_array, p0, 1. / bound);

  const size_t Nhalf = runtime->MaximumNSlots();
  if (lweMatrix.cols() > Nhalf) {
    return Status::ArgumentError("Repack: #LWE ciphers > N/2");
  }

  Ptx ptx;
  ptx.parms_id() = rpk.key().parms_id();

  const size_t nrows = lweMatrix.rows();
  const size_t ncols = lweMatrix.cols();
  const size_t g = CeilSqrt(nrows);
  const size_t h = CeilDiv(nrows, g);

  std::vector<double> diag(lweMatrix.cols(), 0.);

  for (size_t k = 0; k < h && g * k < nrows; ++k) {
    Ctx inner;
    for (size_t j = 0, diagIdx = g * k; j < g && diagIdx < nrows;
         ++j, ++diagIdx) {
      // Obtain the diagonal from LWE matrix
      // lweMatrix.GetDiagnoal(diag, diagIdx);
      CHECK_STATUS_INTERNAL(
          GetMatrixDiagnoal(diag, diagIdx, lweMatrix, runtime),
          "MultiplyPlainMatrixCipherVector");
      // RHS rotated by g * k
      std::rotate(diag.rbegin(), diag.rbegin() + g * k, diag.rend());
      CHECK_STATUS_INTERNAL(runtime->Encode(diag, 1., &ptx), "Encode");

      if (j > 0) {
        CHECK_STATUS_INTERNAL(
            runtime->FMAMontgomery(&inner, rpk.rotated_key(j), ptx), "FMA");
      } else {
        inner = rpk.rotated_key(0);
        CHECK_STATUS_INTERNAL(runtime->MulPlainMontgomery(&inner, ptx),
                              "MulPlain");
      }
    }

    if (k > 0) {
      CHECK_STATUS_INTERNAL(runtime->RotateLeft(&inner, g * k), "RotateLeft");
      CHECK_STATUS_INTERNAL(runtime->Add(&out, inner), "Add");
    } else {
      out = inner;
    }
  }

  CHECK_STATUS_INTERNAL(SumStridedVectors(out, nrows, ncols, runtime),
                        "SumStridedVectors");

  // runtime->RescaleNext(&out);
  // std::cout << "to drop " << out.coeff_modulus_size() << "th \n";
  // runtime->RescaleByFactor(&out, rpk.scale());
  CHECK_STATUS_INTERNAL(runtime->RescaleNext(&out), "RescaleNext");
  out.scale() = 1.;
  lweMatrix.GetLastColumn(diag);

  ptx.parms_id() = out.parms_id();
  CHECK_STATUS_INTERNAL(runtime->Encode(diag, 1., &ptx), "Encode Last Column");
  CHECK_STATUS_INTERNAL(runtime->AddPlain(&out, ptx), "AddPlain");

  return Status::Ok();
}

#else
Status Repack(Ctx &out, const std::vector<rlwe::RLWE2LWECt_st> &lwe_N_ct_array,
              const RpKey &rpk, const std::shared_ptr<RunTime> runtime) {
  auto seal_rt = runtime->SEALRunTime();
  if (lwe_N_ct_array.empty()) {
    Status::ArgumentError("Repack: empty LWE cipher array");
  }

  const size_t NN = lwe_N_ct_array.front().a.coeff_count();
  const size_t Nhalf = runtime->MaximumNSlots();
  if (NN > Nhalf) {
    return Status::ArgumentError("Repack: NN > N/2");
  }

  for (const auto &lwe_N_ct : lwe_N_ct_array) {
    if ((lwe_N_ct.a.coeff_count() != NN) || lwe_N_ct.b.size() != 1) {
      return Status::ArgumentError("Repack: inconsistent input");
    }
  }

  const size_t n_ct = lwe_N_ct_array.size();
  const size_t modMask = (NN - 1);  // nslots is a 2-power value

  if (n_ct > NN) {
    return Status::ArgumentError("Repack: too many lwe_N cipher to pack");
  }

  const double bound = seal_boot::ChevbApproximator::IntervalBound();
  const uint64_t p0 = runtime->GetModulusPrime(0);
  const uint64_t p0h = p0 >> 1;

  auto preprocessor = [bound, p0, p0h](uint64_t u) -> double {
    int64_t su = u >= p0h ? u - p0 : u;
    return static_cast<double>(su) / bound;
  };

  Ptx ptx;
  ptx.parms_id() = rpk.key().parms_id();

  const size_t g = CeilSqrt(NN);
  const size_t h = CeilDiv(NN, g);

  std::vector<double> slots(NN, 0.);

  for (size_t k = 0; k < h && g * k < NN; ++k) {
    Ctx inner;

    for (size_t j = 0, diag = g * k; j < g && diag < NN; ++j, ++diag) {
      std::fill_n(slots.begin(), NN, 0.);

      // Obtain the diagonal from LWE matrix
      for (size_t i = 0; i < n_ct; ++i) {
        const size_t coeff_idx = (diag + i) & modMask;
        // RHS rotated g * k steps
        slots.at((i + g * k) & modMask) =
            preprocessor(*lwe_N_ct_array.at(i).a.data(coeff_idx));
      }

      CHECK_STATUS_INTERNAL(runtime->Encode(slots, 1., &ptx), "Encode");

      if (j > 0) {
        CHECK_STATUS_INTERNAL(
            runtime->FMAMontgomery(&inner, rpk.rotated_key(j), ptx), "FMA");
      } else {
        inner = rpk.rotated_key(0);
        CHECK_STATUS_INTERNAL(runtime->MulPlainMontgomery(&inner, ptx),
                              "MulPlain");
      }
    }

    if (k > 0) {
      CHECK_STATUS_INTERNAL(runtime->RotateLeft(&inner, g * k), "RotateLeft");
      CHECK_STATUS_INTERNAL(runtime->Add(&out, inner), "Add");
    } else {
      out = inner;
    }
  }

  runtime->RescaleByFactor(&out, rpk.scale());
  out.scale() = 1.;

  for (int i = 0; i < n_ct; ++i) {
    slots[i] = preprocessor(lwe_N_ct_array[i].b[0]);
  }
  std::fill_n(slots.begin() + n_ct, slots.size() - n_ct, 0.);

  ptx.parms_id() = out.parms_id();
  CHECK_STATUS_INTERNAL(runtime->Encode(slots, 1., &ptx), "Encode 2");
  CHECK_STATUS_INTERNAL(runtime->AddPlain(&out, ptx), "AddPlain");

  return Status::Ok();
}
#endif
}  // namespace gemini

using MatType = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;

void matmul(int dim, std::shared_ptr<gemini::RunTime> rt) {
  using namespace gemini;
  size_t max_nslots = rt->MaximumNSlots();
  if (dim * dim * dim > max_nslots) {
    return;
  }

  std::random_device rdv;
  std::uniform_real_distribution<double> uniform(-10., 10);
  std::vector<double> origin(dim * dim, 0.);
  std::generate_n(origin.data(), origin.size(), [&]() { return uniform(rdv); });
  std::iota(origin.begin(), origin.end(), 1.);

  // for (int r = 0; r < dim; ++r) {
  //   for (int c = 0; c < dim; ++c) {
  //     // origin[r * dim + c] = 1. * c;
  //     std::cout << origin[r * dim + c] << " ";
  //   }
  //   std::cout << "\n";
  // }
  // std::cout << "\n";

  Ptx ptx;
  Ctx ctx;
  rt->Encode(origin, static_cast<double>(1L << 40), &ptx);
  rt->Encrypt(ptx, &ctx);

  Ctx accum;
  Ptx maskPoly;
  const double mask_scale =
      std::floor(std::sqrt(rt->GetModulusPrime(ctx.coeff_modulus_size() - 1)));
  {
    std::vector<double> mask(origin.size(), 0.);
    maskPoly.parms_id() = ctx.parms_id();
    std::fill_n(mask.data(), dim, 1.);
    rt->Encode(mask, mask_scale, &maskPoly);
  }

  auto galois_tool = rt->SEALRunTime()->first_context_data()->galois_tool();

  double time{0.};
  for (int i = 0; i < dim; ++i) {
    AutoTimer timer(&time);
    ptx = maskPoly;
    seal::util::ConstRNSIter maskPolyIter(maskPoly.data(),
                                          ctx.poly_modulus_degree());
    seal::util::RNSIter ptxIter(ptx.data(), ctx.poly_modulus_degree());
    if (i > 0) {
      galois_tool->apply_galois_ntt(maskPolyIter, ctx.coeff_modulus_size(),
                                    galois_tool->get_elt_from_step(dim * i),
                                    ptxIter);
    }

    auto masked{ctx};
    auto st = rt->MulPlain(&masked, ptx);
    if (!st.IsOk()) std::cout << st.Msg() << "\n";

    int nsteps = Log2(dim * dim);
    // O(2 * g * log2(g))
    for (int i = 0; i < nsteps; ++i) {
      auto temp{masked};
      rt->RotateLeft(&temp, 1 << i);
      rt->Add(&masked, temp);
    }

    rt->MulPlain(&masked, ptx);

    if (accum.size()) {
      rt->Add(&accum, masked);
    } else {
      accum = masked;
    }
  }
  rt->RescaleByFactor(&accum, mask_scale * mask_scale);
  std::cout << "matmul " << dim << "*" << dim << " took " << time / 1000.
            << "sec\n";

  // rt->Decrypt(accum, &ptx);
  // std::vector<double> slots;
  // rt->Decode(ptx, dim * dim, &slots);
  // for (double v : slots) std::cout << v << " ";
  // std::cout << "\n";

  // MatType LHS = MatType::Random(dim, dim);
  // MatType RHS = MatType::Random(dim, dim);
  //
  // std::vector<double> lhs_vector(dim * dim * dim, 0.);
  // auto lhs_ptr = lhs_vector.data();
  // for (size_t c = 0; c < dim; ++c) {
  //   for (size_t r = 0; r < dim; ++r) {
  //     //LHS(r, c) = (double) r;
  //     for (size_t rep = 0; rep < dim; ++rep) {
  //       *lhs_ptr++ = LHS(r, c);
  //     }
  //   }
  // }
  //
  // std::vector<double> rhs_vector(dim * dim * dim, 0.);
  // auto rhs_ptr = rhs_vector.data();
  // for (size_t r = 0; r < dim; ++r) {
  //   for (size_t rep = 0; rep < dim; ++rep) {
  //     for (size_t c = 0; c < dim; ++c) {
  //       *rhs_ptr++ = RHS(r, c);
  //       std::cout << RHS(r, c) << " ";
  //     }
  //     std::cout << "\n";
  //   }
  // }
  // std::cout << "\n";
  //
  // MatType Ground = LHS * RHS;
  //
  // Ptx ptx;
  // Ctx lhs, rhs;
  // rt->Encode(lhs_vector, std::pow(2., 40), &ptx);
  // rt->Encrypt(ptx, &lhs);
  // rt->Encode(rhs_vector, std::pow(2., 40), &ptx);
  // rt->Encrypt(ptx, &rhs);
  //
  // rt->Mul(&lhs, rhs);
  // rt->Relin(&lhs);
  //
  // const int nsteps = Log2(dim);
  // for (int i = 0; i < nsteps; ++i) {
  //   Ctx copy{lhs};
  //   rt->RotateLeft(&copy, dim * dim * (1 << i));
  //   rt->Add(&lhs, copy);
  // }
  // rt->RescaleByFactor(&lhs, std::pow(2., 40));
  //
  // std::vector<double> decrypted;
  // rt->Decrypt(lhs, &ptx);
  // rt->Decode(ptx, dim * dim * dim, &decrypted);
  //
  // // std::cout << LHS << "\n\n";
  // // std::cout << Ground << "\n\n";
  //
  // auto iter = decrypted.begin();
  // for (size_t r = 0; r < dim; ++r) {
  //   for (size_t rep = 0; rep < dim; ++rep) {
  //     for (size_t c = 0; c < dim; ++c) {
  //       std::cout << *iter++ << " ";
  //       /#<{(|rhs_ptr++ = RHS(r, c);
  //     }
  //     std::cout << "\n";
  //   }
  // }
  // std::cout << "\n";
}

class BitonicSorter {
 public:
  explicit BitonicSorter(const rlwe::DecomposedLWEKSwitchKey_t ks_,
                         const RtPtr lvl0_rt_, const rlwe::LWEGateBooter &lut_)
      : ks(ks_), lvl0_rt(lvl0_rt_), lut(lut_) {}

  void sort(lwe::Ctx_st *lwe_ct, size_t length, bool accend) const {
    bitonic_sort(lwe_ct, 0, length, accend);
  }

 private:
  const rlwe::DecomposedLWEKSwitchKey_st *ks;
  const RtPtr lvl0_rt;
  const rlwe::LWEGateBooter &lut;

  void bitonic_sort(lwe::Ctx_st *lwe_ct, int start, int length,
                    bool accend) const {
    if (length > 1) {
      size_t m = length / 2;
      bitonic_sort(lwe_ct, start, m, !accend);
      bitonic_sort(lwe_ct, start + m, length - m, accend);
      bitonic_merge(lwe_ct, start, length, accend);
    }
  }

  void CtxAdd(lwe::Ctx_t add, const lwe::Ctx_t c0, const lwe::Ctx_t c1) const {
    using namespace seal;
    using namespace lwe;
    auto modulus = lvl0_rt->last_context_data()->parms().coeff_modulus()[0];
    util::add_poly_coeffmod(CtData(c0), CtData(c1), params::n() + 1, modulus,
                            CtData(add));
  }

  void CtxSub(lwe::Ctx_t sub, const lwe::Ctx_t c0, const lwe::Ctx_t c1) const {
    using namespace seal;
    using namespace lwe;
    auto modulus = lvl0_rt->last_context_data()->parms().coeff_modulus()[0];
    util::sub_poly_coeffmod(CtData(c0), CtData(c1), params::n() + 1, modulus,
                            CtData(sub));
  }

  // c_i, c_j -> max(), min() if accend = true
  // c_i, c_j -> min(), max() if accend = false
  void cmp_swap(lwe::Ctx_st *lwe_ct, int i, int j, bool accend) const {
    lwe::Ctx_t sum, sub;
    CtxAdd(sum, lwe_ct + i, lwe_ct + j);
    CtxSub(sub, lwe_ct + i, lwe_ct + j);

    rlwe::RLWE2LWECt_st lwe_N;
    lut.Abs(&lwe_N, sub);
    lwe::Ctx_t lwe_abs, lwe_max, lwe_min;
    LWEKeySwitch(lwe_abs, &lwe_N, ks, lvl0_rt);

    CtxAdd(lwe_max, sum, lwe_abs);
    CtxSub(lwe_min, sum, lwe_abs);

    if (accend) {
      lwe_ct[i] = lwe_max[0];
      lwe_ct[j] = lwe_min[0];
    } else {
      lwe_ct[j] = lwe_max[0];
      lwe_ct[i] = lwe_min[0];
    }
  }

  inline int greatestPowerOfTwoLessThan(int n) const {
    int k = 1;
    while (k < n) k = k << 1;
    return k >> 1;
  }

  void bitonic_merge(lwe::Ctx_st *lwe_ct, int lo, int n, bool accend) const {
    if (n > 1) {
      int m = greatestPowerOfTwoLessThan(n);
      for (int i = lo; i < lo + n - m; i++) {
        cmp_swap(lwe_ct, i, i + m, accend);
      }
      bitonic_merge(lwe_ct, lo, m, accend);
      bitonic_merge(lwe_ct, lo + m, n - m, accend);
    }
  }
};

void ShowLUTAccuracyEfficiencyTradeOff(int argc, char *argv[]) {
  using namespace gemini;
  using namespace seal;

  const size_t n = lwe::params::n();
  const size_t log2N = argc > 1 ? std::atoi(argv[1]) : 10;
  const size_t NN = argc > 2 ? (1 << std::atoi(argv[2])) : 2048;
  const int nslots = argc > 3 ? std::atoi(argv[3]) : 8;
  const size_t N = 1 << log2N;

  if (NN >= N) {
    throw std::invalid_argument("require NN < N");
  }

  const size_t n_mod_w_bits = 45;
  std::string JSON = "{\"log2PolyDegree\":" + std::to_string(log2N) +
                     ",\"nSpecialPrimes\":1,\"seed\":0,\"moduliArray\":[\
                       45, 45, 45, 45, 59]}";
  const double scale = std::pow(2., 36);  // Best: scale * m ~ 2^w
  const double msg_interval = std::floor((1L << n_mod_w_bits) * 0.25 / scale);

  auto runtime = RunTime::Create(JSON);
  runtime->ShowContext(std::cout);
  const size_t max_n_moduli = runtime->MaximumNModuli();
  const double rpkScale =
      static_cast<double>(runtime->GetModulusPrime(max_n_moduli - 1));

  seal_boot::LinearTransformer::Parms ltParams = {.nslots = nslots,
                                                  .c2sPrecision = n_mod_w_bits,
                                                  .s2cPrecision = n_mod_w_bits,
                                                  .c2sMultiplier = 1.0,
                                                  .s2cMultiplier = 1.};
  seal_boot::BetterSine sin_approximator(runtime);
  seal_boot::LinearTransformer LTer(ltParams, runtime);

  rlwe::BK bk;  // encryption of lwe_sk under rlwe_sk, i.e, GSW(s_n; s_N, N, Q).
  gemini::RpKey rpk;  // encryption of rlwe_sk under rlwe_sk

  // LWE stuffs
  RtPtr rlwe_rt = runtime->SEALRunTime();
  RtPtr rgsw_rt, lwe_rt_mod_p0;
  lwe::runtime_t lwe_rt_mod_2w;

  seal::SecretKey lwe_sk_ntt, rgsw_sk;
  seal::SecretKey rlwe_sk_non_ntt, rgsw_sk_non_ntt;

  lwe::SK_t lwe_sk_non_ntt;
  rlwe::DecomposedLWEKSwitchKey_t lvl2_to_lvl0_KS;
  // rlwe::LWEKSwitchKey_t lvl1_to_lvl0_KS;
  rlwe::DecomposedLWEKSwitchKey_t lvl1_to_lvl0_KS;

  {
    seal::EncryptionParameters parms(seal::scheme_type::CKKS);
    const auto &rlwe_modulus =
        rlwe_rt->key_context_data()->parms().coeff_modulus();
    std::vector<Modulus> rgsw_modulus{rlwe_modulus.front(),
                                      rlwe_modulus.back()};
    parms.set_coeff_modulus(rgsw_modulus);
    parms.set_poly_modulus_degree(NN);
    parms.set_galois_generator(5);
    rgsw_rt = SEALContext::Create(parms, true, sec_level_type::none);

    seal::SecretKey const &rlwe_sk = runtime->SEALSecretKey();
    rlwe_sk_non_ntt.data().resize(N);
    std::copy_n((const uint64_t *)rlwe_sk.data().data(), N,
                rlwe_sk_non_ntt.data().data());
    rlwe::SwitchNTTForm(rlwe_sk_non_ntt.data().data(), false, 1, rlwe_rt);

    // RLWE secret key is identical to RGSW secret key with 2 moduli
    // The 2nd moduli is the special moduli.
    GenerateHammingSecretKey(rgsw_sk, 64, rgsw_rt);

    rgsw_sk_non_ntt.data().resize(NN);
    std::copy_n((const uint64_t *)rgsw_sk.data().data(), NN,
                rgsw_sk_non_ntt.data().data());
    rlwe::SwitchNTTForm(rgsw_sk_non_ntt.data().data(), false, 1, rgsw_rt);

    gemini::RpKeyInit(rpk, rpkScale, rlwe_sk, rgsw_sk_non_ntt, runtime);
  }

  {
    // Create LWE Runtime and setup keys
    const std::vector<Modulus> &rlwe_modulus = rlwe_rt->key_context_data()->parms().coeff_modulus();
    // std::vector<Modulus> lwe_modulus{rlwe_modulus.front(),
    // rlwe_modulus.back()};
    std::vector<Modulus> lwe_modulus{rlwe_modulus.front()};

    EncryptionParameters parms(scheme_type::CKKS);
    parms.set_galois_generator(5);
    parms.set_poly_modulus_degree(n);
    parms.set_coeff_modulus(lwe_modulus);
    lwe_rt_mod_p0 = SEALContext::Create(parms, true, sec_level_type::none);
    lwe::RunTimeInit(lwe_rt_mod_2w, n, n_mod_w_bits);
    lwe::SKInit(lwe_sk_ntt, lwe_sk_non_ntt, lwe_rt_mod_p0);

    // Encrypt lwe_secret using rgsw secret
    rlwe::BKInit(bk, lwe_sk_non_ntt, rgsw_sk, rgsw_rt);

    rlwe::LWEKSKeyInit(lvl2_to_lvl0_KS, 7, rlwe_sk_non_ntt.data(), lwe_sk_ntt,
                       lwe_rt_mod_p0, rlwe_rt);
    rlwe::LWEKSKeyInit(lvl1_to_lvl0_KS, 7, rgsw_sk_non_ntt.data(), lwe_sk_ntt,
                       lwe_rt_mod_p0, rgsw_rt);
  }

  std::vector<double> ground_vector(nslots);
  {
    std::uniform_real_distribution<double> uniform(-msg_interval, msg_interval);
    std::random_device rdv;
    std::generate_n(ground_vector.data(), nslots, [&]() { return std::round(uniform(rdv)); });
  }

  Ptx ptx;
  Ctx ctx;
  runtime->Encode(ground_vector, scale, &ptx);
  runtime->Encrypt(ptx, &ctx);

  Ctx enc_coeff;

  runtime->DropModuli(&ctx, ctx.coeff_modulus_size() - LTer.LogRadixN() - 1);

  double s2cTime{0.};
  {
    AutoTimer timer(&s2cTime);
    LTer.SlotsToCoeffs(ctx, &enc_coeff);
  }
  std::cout << "S2C took " << s2cTime << "ms\n";
  runtime->KeepLastModuli(&enc_coeff);
}

int main(int argc, char *argv[]) {
  using namespace gemini;
  using namespace seal;

  const size_t n = lwe::params::n();
  const size_t log2N = argc > 1 ? std::atoi(argv[1]) : 10;
  const size_t NN = argc > 2 ? (1 << std::atoi(argv[2])) : 2048;
  const int nslots = argc > 3 ? std::atoi(argv[3]) : 64;
  const size_t N = 1 << log2N;

  if (NN >= N) {
    throw std::invalid_argument("require NN < N");
  }

  const size_t n_mod_w_bits = 45;

  // additional noise 2^w * sigma / (sqrt(N * 12) * p')

  std::string JSON = "{\"log2PolyDegree\":" + std::to_string(log2N) +
                     ",\"nSpecialPrimes\":1,\"seed\":0,\"moduliArray\":[\
                       45, 45, 45, 59]}";

                       //45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 59]}";

  const double scale = std::pow(2., 36);  // Best: scale * m ~ 2^w
  const double msg_interval = std::floor((1L << n_mod_w_bits) * 0.125 / scale);

  const double gateBootPostMult = 0.5 / M_PI;

  auto runtime = RunTime::Create(JSON);
  runtime->ShowContext(std::cout);
  const size_t max_n_moduli = runtime->MaximumNModuli();
  const double rpkScale = static_cast<double>(runtime->GetModulusPrime(max_n_moduli - 1));
  // std::pow(2., n_mod_w_bits);
  // std::cout << "additional noise " << std::log2(std::pow(2., n_mod_w_bits +
  // 40 - 59) * std::sqrt(NN) / std::sqrt(12.)) << "bits\n";
  std::cout << "|msg| " << msg_interval << "\n";

  seal_boot::LinearTransformer::Parms ltParams = {.nslots = nslots,
                                                  .c2sPrecision = 45,
                                                  .s2cPrecision = 45,
                                                  .c2sMultiplier = 1.0,
                                                  .s2cMultiplier = 1.0 };

  seal_boot::BetterSine sin_approximator(runtime);
  seal_boot::LinearTransformer LTer(ltParams, runtime);

  rlwe::BK bk;  // encryption of lwe_sk under rlwe_sk, i.e, GSW(s_n; s_N, N, Q).
  gemini::RpKey rpk;  // encryption of rlwe_sk under rlwe_sk

  // LWE stuffs
  RtPtr rlwe_rt = runtime->SEALRunTime();
  RtPtr rgsw_rt, lwe_rt_mod_p0;
  lwe::runtime_t lwe_rt_mod_2w;

  seal::SecretKey lwe_sk_ntt, rgsw_sk;
  seal::SecretKey rlwe_sk_non_ntt, rgsw_sk_non_ntt;

  lwe::SK_t lwe_sk_non_ntt;
  rlwe::DecomposedLWEKSwitchKey_t lvl2_to_lvl0_KS;
  rlwe::DecomposedLWEKSwitchKey_t lvl1_to_lvl0_KS;

  {
    seal::EncryptionParameters parms(seal::scheme_type::CKKS);
    const auto &rlwe_modulus =
        rlwe_rt->key_context_data()->parms().coeff_modulus();
    std::vector<Modulus> rgsw_modulus{rlwe_modulus.front(),
                                      rlwe_modulus.back()};
    parms.set_coeff_modulus(rgsw_modulus);
    parms.set_poly_modulus_degree(NN);
    parms.set_galois_generator(5);
    rgsw_rt = SEALContext::Create(parms, true, sec_level_type::none);

    seal::SecretKey const &rlwe_sk = runtime->SEALSecretKey();
    rlwe_sk_non_ntt.data().resize(N);
    std::copy_n((const uint64_t *)rlwe_sk.data().data(), N,
                rlwe_sk_non_ntt.data().data());
    rlwe::SwitchNTTForm(rlwe_sk_non_ntt.data().data(), false, 1, rlwe_rt);

    // RLWE secret key is identical to RGSW secret key with 2 moduli
    // The 2nd moduli is the special moduli.
    GenerateHammingSecretKey(rgsw_sk, 64, rgsw_rt);

    rgsw_sk_non_ntt.data().resize(NN);
    std::copy_n((const uint64_t *)rgsw_sk.data().data(), NN,
                rgsw_sk_non_ntt.data().data());
    rlwe::SwitchNTTForm(rgsw_sk_non_ntt.data().data(), false, 1, rgsw_rt);

    gemini::RpKeyInit(rpk, rpkScale, rlwe_sk, rgsw_sk_non_ntt, runtime);
  }

  {
    // Create LWE Runtime and setup keys
    const std::vector<Modulus> &rlwe_modulus =
        rlwe_rt->key_context_data()->parms().coeff_modulus();
    // std::vector<Modulus> lwe_modulus{rlwe_modulus.front(),
    // rlwe_modulus.back()};
    std::vector<Modulus> lwe_modulus{rlwe_modulus.front()};

    EncryptionParameters parms(scheme_type::CKKS);
    parms.set_galois_generator(5);
    parms.set_poly_modulus_degree(n);
    parms.set_coeff_modulus(lwe_modulus);
    lwe_rt_mod_p0 = SEALContext::Create(parms, true, sec_level_type::none);
    lwe::RunTimeInit(lwe_rt_mod_2w, n, n_mod_w_bits);
    lwe::SKInit(lwe_sk_ntt, lwe_sk_non_ntt, lwe_rt_mod_p0);

    // Encrypt lwe_secret using rgsw secret
    rlwe::BKInit(bk, lwe_sk_non_ntt, rgsw_sk, rgsw_rt);

    rlwe::LWEKSKeyInit(lvl2_to_lvl0_KS, 7, rlwe_sk_non_ntt.data(), lwe_sk_ntt,
                       lwe_rt_mod_p0, rlwe_rt);
    rlwe::LWEKSKeyInit(lvl1_to_lvl0_KS, 7, rgsw_sk_non_ntt.data(), lwe_sk_ntt,
                       lwe_rt_mod_p0, rgsw_rt);
  }

  std::vector<double> ground_vector(nslots);

  {
    std::uniform_real_distribution<double> uniform(-msg_interval, msg_interval);
    // std::mt19937 rdv;
    const double double_NN = 2. * NN;
    std::random_device rdv;
    // std::iota(ground_vector.begin(), ground_vector.end(), 1.);
    std::generate_n(ground_vector.data(), nslots, [&]() { return uniform(rdv); });
  }

  Ptx ptx;
  Ctx ctx;
  runtime->Encode(ground_vector, scale, &ptx);
  runtime->Encrypt(ptx, &ctx);

  Ctx enc_coeff;

  runtime->DropModuli(&ctx, ctx.coeff_modulus_size() - LTer.LogRadixN() - 1);

  double s2cTime{0.};
  {
    AutoTimer timer(&s2cTime);
    LTer.SlotsToCoeffs(ctx, &enc_coeff);
  }
  std::cout << "S2C took " << s2cTime << "ms\n";

  runtime->KeepLastModuli(&enc_coeff);

  rlwe::LWEGateBooter gateBooter(scale, bk, rgsw_rt);

  std::vector<size_t> extract_indices(nslots);
  for (size_t i = 0; i < nslots; ++i) {
    extract_indices[i] = seal::util::reverse_bits(i, log2N - 1);
  }

  {
    runtime->Decrypt(enc_coeff, &ptx);
    std::vector<double> temp;
    runtime->ConventionalForm(ptx, &temp, true);
    for (size_t i = 0; i < nslots; ++i) {
      ground_vector[i] = temp.at(extract_indices[i]);
    }
  }

  std::vector<rlwe::RLWE2LWECt_st> lwe_N_ct_array(nslots);
  std::vector<lwe::Ctx_st> lwe_n_ct_array(nslots);

  double extractTime{0.}, ksTime{0.}, gateBootTime{0.};

  rlwe::SwitchNTTForm(enc_coeff, rlwe_rt);

  auto lwe_N_ct_ptr = lwe_N_ct_array.data();
  auto lwe_n_ct_ptr = lwe_n_ct_array.data();

  {
    AutoTimer timer(&extractTime);
    rlwe::SampleExtract(lwe_N_ct_array.data(), enc_coeff, extract_indices,
                        rlwe_rt);
  }

  double error_ks{1e-30}, error_lut{1e-30};

  // auto target_func = [](F64 x) { return 1. / (1. + std::exp(-x)); };
  auto target_func = [](F64 x) { return std::log(std::abs(x)); };
  // auto target_func = [](F64 x) { return x; };

  for (size_t i = 0; i < lwe_N_ct_array.size(); ++i) {
    { /* LWE_N -> LWE_n */
      AutoTimer timer(&ksTime);
      rlwe::LWEKeySwitch(lwe_n_ct_ptr, lwe_N_ct_ptr, lvl2_to_lvl0_KS,
                         lwe_rt_mod_p0);
    }

    double ks_output = lwe::SymDec(lwe_n_ct_ptr, lwe_sk_non_ntt, lwe_rt_mod_p0) / scale;
    double e = std::fabs(ground_vector[i] - ks_output);
    error_ks += e;

    AutoTimer timer(&gateBootTime);
    gateBooter.AbsLog(lwe_N_ct_ptr, lwe_n_ct_ptr);
    {
      // double gnd = std::sqrt(std::abs(ks_output)) *
      // gateBooter.GetPostMultiplier();
      double gnd = target_func(ks_output) * gateBooter.GetPostMultiplier();
      double lut_output =
          rlwe::LWE_SymDec(lwe_N_ct_ptr, rgsw_sk_non_ntt.data(), rgsw_rt)[0] /
          scale;
      error_lut += std::abs(gnd - lut_output);

      if (i < 16) {
        printf("%.7f %.7f, KS %.7f LUT %.7f\n", ground_vector[i], target_func(ground_vector[i]), 
               ks_output, lut_output);
      }
    }

    ++lwe_N_ct_ptr;
    ++lwe_n_ct_ptr;
  }
  error_ks /= nslots;
  error_lut /= nslots;
  printf("average error: ks 2^%f, LUT 2^%f\n", std::log2(error_ks), std::log2(error_lut));

  printf("N = %zu, NN = %zu, n = %zu, ndigits = %d.\n", N, NN, n,
         lvl2_to_lvl0_KS->ndigits_);
  printf("SampleExtract = %f ms per. (%d slots). GateBoot %f ms per.\n",
         extractTime / lwe_N_ct_array.size(), nslots,
         (gateBootTime / lwe_n_ct_array.size()));

  // gemini::Ctx repacked;
  // gemini::Status stat;
  // std::cout << "Before repack " << lwe_N_ct_array[0].b.size() << " moduli\n";
  //
  // double repackTime{0.}, sinTime{0.};
  // {
  //   AutoTimer timer(&repackTime);
  //   stat = gemini::Repack(repacked, sin_approximator.interval_bound(),
  //                         lwe_N_ct_array, rpk, runtime);
  //   if (!stat.IsOk()) {
  //     std::cout << stat.Msg() << "\n";
  //     return 0;
  //   }
  // }
  //
  // std::cout << "After repack " << repacked.coeff_modulus_size() << "
  // moduli\n"; std::cout << "Repack Took " << repackTime << "ms\n";
  // {
  //   AutoTimer timer(&sinTime);
  //   repacked.scale() = scale;
  //   stat = sin_approximator.Apply(repacked, scale);
  //   if (!stat.IsOk()) {
  //     std::cerr << "SinError " << stat.Msg() << "\n";
  //     exit(1);
  //   }
  //   repacked.scale() = scale;
  // }
  //
  // std::cout << "After Sin " << repacked.coeff_modulus_size() << " moduli\n";
  // std::cout << "Sin took " << sinTime << "ms\n";
  //
  // {
  //   std::vector<double> slots;
  //   stat = runtime->Decrypt(repacked, &ptx);
  //   if (!stat.IsOk()) std::cerr << stat.Msg() << "\n";
  //   stat = runtime->Decode(ptx, nslots, &slots);
  //   if (!stat.IsOk()) std::cerr << stat.Msg() << "\n";
  //
  //   double accum_err = 0.;
  //   for (int i = 0; i < lwe_N_ct_array.size(); ++i) {
  //     // double gnd = ground_vector[i];
  //     double gnd = target_func(ground_vector[i]);
  //     if (i < 16) {
  //       std::cout << gnd << "->" << slots[i] << "\n";
  //     }
  //     double e = std::abs(gnd - slots[i]);
  //     accum_err += e;
  //   }
  //   accum_err /= nslots;
  //   printf("average error 2^%f\n", std::log2(accum_err));
  // }
  return 0;
}

