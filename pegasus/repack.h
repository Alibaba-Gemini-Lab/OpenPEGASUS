#include "pegasus/rlwe.h"
#include "pegasus/runtime.h"
namespace gemini {

typedef struct RpKey {
  using T = Ctx;
  T key_;
  // optimization, store the baby steps of the key.
  std::vector<T> rotated_keys_;
  F64 scale_ = 1.;

  const T &key() const { return key_; }
  const T &rotated_key(size_t idx) const {
    if (idx > rotated_keys_.size()) {
      throw std::invalid_argument("Repacking key index out-of-bound");
    }
    return idx == 0 ? key_ : rotated_keys_.at(idx - 1);
  }
  T &key() { return key_; }

  F64 &scale() { return scale_; }
  F64 scale() const { return scale_; }

} RpKey;

Status RpKeyInit(RpKey &rpk, const F64 scale, const seal::SecretKey &rlwe_sk,
                 const seal::SecretKey &rgsw_sk_non_ntt,
                 const std::shared_ptr<RunTime> runtime);

/// Wrap an LWE array to a matrix like object
struct LWECtArrayWrapper {
 public:
  LWECtArrayWrapper(const std::vector<lwe::Ctx_st> &array, U64 p0,
                    F64 multiplier);
  size_t rows() const { return lwe_n_ct_array_.size(); }
  size_t cols() const { return lwe::CtLen(lwe_n_ct_array_.front()) - 1; }

  inline F64 operator()(size_t r, size_t c) const {
    if (c > cols()) {
      throw std::invalid_argument("LWECtArrayWrapper Get(r, c) out-of-index");
    }
    uint64_t u = lwe::CtData(lwe_n_ct_array_.at(r))[c];
    return preprocess(u);
  }

  inline F64 Get(size_t r, size_t c) const { return (*this)(r, c); }
  void GetLastColumn(F64Vec &column) const;

 private:
  inline F64 preprocess(U64 u) const {
    I64 su = u >= p0half_ ? u - p0_ : u;
    return static_cast<F64>(su) * multiplier_;
  }
  const std::vector<lwe::Ctx_st> &lwe_n_ct_array_;
  const U64 p0_, p0half_;
  const F64 multiplier_;
};

template <class MatrixLike>
Status GetTilingDiagonal(F64Vec &diag, size_t diag_idx,
                         const MatrixLike &matrix,
                         const std::shared_ptr<RunTime> runtime) {
  const size_t nrows = matrix.rows();
  const size_t ncols = matrix.cols();

  if (nrows == 0 || !IsTwoPower(nrows)) {
    return Status::ArgumentError("GetTilingDiagonal: Invalid nrows");
  }

  if (ncols == 0 || !IsTwoPower(ncols)) {
    return Status::ArgumentError("GetTilingDiagonal: Invalid ncols");
  }

  const size_t col_mask = (ncols - 1);
  const size_t row_mask = (nrows - 1);
  const size_t max_n = std::max(nrows, ncols);
  const size_t min_n = std::min(nrows, ncols);

  if (max_n > runtime->MaximumNSlots()) {
    return Status::ArgumentError("GetTilingDiagonal: not enough slots");
  }

  if (diag_idx >= min_n) {
    return Status::ArgumentError("GetTilingDiagonal: Invalid diagonal index");
  }

  diag.resize(max_n);
  auto iter = diag.begin();

  for (size_t r = 0; r < max_n; ++r) {
    *iter++ = static_cast<F64>(matrix(r & row_mask, (r + diag_idx) & col_mask));
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
                         const std::shared_ptr<RunTime> runtime);

Status Repack(Ctx &out, F64 bound,
              const std::vector<lwe::Ctx_st> &lwe_n_ct_array, const RpKey &rpk,
              const std::shared_ptr<RunTime> runtime);

}  // namespace gemini

