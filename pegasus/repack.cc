#include "pegasus/repack.h"

#include <Eigen/Dense>
namespace gemini {
Status RpKeyInit(RpKey &rpk, const F64 scale, const seal::SecretKey &rlwe_sk,
                 const seal::SecretKey &rgsw_sk_non_ntt,
                 const std::shared_ptr<RunTime> runtime) {
  auto rlwe_rt = runtime->SEALRunTime();
  const size_t nslots = runtime->MaximumNSlots();
  const size_t NN = rgsw_sk_non_ntt.data().coeff_count();
  if (nslots < NN) {
    throw std::invalid_argument("RpKeyInit: require N >= 2*N'");
  }
  auto cast_to_double = [](U64 u) -> F64 {
    return u > 1 ? -1. : static_cast<F64>(u);
  };

  if (!seal::is_metadata_valid_for(rlwe_sk, rlwe_rt)) {
    throw std::invalid_argument("RpKeyInit: invalid rlwe_sk");
  }

  F64Vec slots(NN, 0.);
  for (size_t i = 0; i < NN; ++i) {
    slots[i] = cast_to_double(rgsw_sk_non_ntt.data()[i]);
  }

  Ptx ptx;
  ptx.parms_id() = rlwe_rt->first_parms_id();
  runtime->Encode(slots, scale, &ptx);
  runtime->Encrypt(ptx, &rpk.key());
  rpk.scale() = scale;

  const size_t g = CeilSqrt(NN);
  rpk.rotated_keys_.resize(g - 1);
  for (size_t j = 1; j < g; ++j) {
    std::rotate(slots.begin(), slots.begin() + 1, slots.end());
    CHECK_STATUS(runtime->Encode(slots, scale, &ptx));
    CHECK_STATUS(runtime->Encrypt(ptx, &rpk.rotated_keys_.at(j - 1)));
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

  seal::util::seal_memzero(
      slots.data(),
      slots.size() * sizeof(slots[0]));  // clean up secret material
  return Status::Ok();
}

/// Wrap an LWE array to a matrix like object

LWECtArrayWrapper::LWECtArrayWrapper(const std::vector<lwe::Ctx_st> &array,
                                     uint64_t p0, double multiplier)
    : lwe_n_ct_array_(array),
      p0_(p0),
      p0half_(p0 >> 1),
      multiplier_(multiplier) {
  if (lwe_n_ct_array_.empty()) {
    throw std::invalid_argument("LWECtArrayWrapper: Empty array");
  }
}

void LWECtArrayWrapper::GetLastColumn(F64Vec &column) const {
  const size_t ncols = cols();
  const size_t nrows = rows();
  column.resize(ncols);
  for (size_t i = 0; i < nrows; ++i) {
    column[i] = Get(i, ncols);
  }

  for (size_t i = nrows; i < ncols; i += nrows) {
    for (size_t k = 0; k < nrows; ++k) column[i + k] = column[k];
  }
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

  for (size_t i = 0; i < nsteps; ++i) {
    auto copy{enc_vec};
    CHECK_STATUS_INTERNAL(runtime->RotateLeft(&copy, (1U << i) * stride),
                          "SumStridedVectors: RotateLeft failed");
    CHECK_STATUS_INTERNAL(runtime->Add(&enc_vec, copy),
                          "SumStridedVectors: Add failed");
  }
  return Status::Ok();
}

Status Repack(Ctx &out, double bound,
              const std::vector<lwe::Ctx_st> &lwe_n_ct_array, const RpKey &rpk,
              const std::shared_ptr<RunTime> runtime) {
  auto seal_rt = runtime->SEALRunTime();
  if (lwe_n_ct_array.empty()) {
    Status::ArgumentError("Repack: empty LWE cipher array");
  }

  const uint64_t p0 = runtime->GetModulusPrime(0);
  LWECtArrayWrapper lweMatrix(lwe_n_ct_array, p0, 1. / bound);

  const size_t Nhalf = runtime->MaximumNSlots();
  if (lweMatrix.cols() > Nhalf) {
    return Status::ArgumentError("Repack: too many LWE ciphers to pack");
  }

  Ptx ptx;
  ptx.parms_id() = rpk.key().parms_id();

  const size_t nrows = lweMatrix.rows();
  const size_t ncols = lweMatrix.cols();
  const size_t min_n = std::min(nrows, ncols);

  const size_t g = CeilSqrt(min_n);
  const size_t h = CeilDiv(min_n, g);

  std::vector<double> diag(lweMatrix.cols(), 0.);

  // Baby-Steps-Giant-Steps
  for (size_t k = 0; k < h && g * k < min_n; ++k) {
    Ctx inner;
    for (size_t j = 0, diag_idx = g * k; j < g && diag_idx < min_n;
         ++j, ++diag_idx) {
      // Obtain the diagonal from LWE matrix
      CHECK_STATUS_INTERNAL(
          GetTilingDiagonal(diag, diag_idx, lweMatrix, runtime),
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

  if (nrows < ncols) {
    // Sum-Columns
    CHECK_STATUS_INTERNAL(SumStridedVectors(out, nrows, ncols, runtime),
                          "SumStridedVectors");
  }

  runtime->RescaleNext(&out);
  out.scale() = 1.;

  lweMatrix.GetLastColumn(diag);

  ptx.parms_id() = out.parms_id();
  CHECK_STATUS_INTERNAL(runtime->Encode(diag, 1., &ptx), "Encode Last Column");
  CHECK_STATUS_INTERNAL(runtime->AddPlain(&out, ptx), "AddPlain");

  return Status::Ok();
}
}  // namespace gemini
