#pragma once
#include <memory>
#if GEMINI_USE_GMP
#include <gmpxx.h>
#endif
#include <array>
#include <iosfwd>
#include <tuple>

#include "pegasus/common.h"

namespace gemini {

class RunTime {
 public:
  struct JSONField {
    static std::string Log2PolyDegree() {
      return std::string("log2PolyDegree");
    }
    static std::string ModuliArray() { return std::string("moduliArray"); }
    static std::string NSpecialPrimes() {
      return std::string("nSpecialPrimes");
    }
    static std::string DecryptionOnly() {
      return std::string("decryptionOnly");
    }
    static std::string Seed() { return std::string("seed"); }
    static std::string CompressKey() { return std::string("compressKey"); }
  };

  enum class Supports { kEncryption, kDecryption, kRelin, kRotation };

  enum class Key { kSecretKey, kPublicKey, kRelinKey, kRotationKey };

  struct CtxHeader {
    union {
      char is_ntt : 1;
      uint32_t degree;
    };
    uint32_t moduli_index;
    F64 scale;
    uint64_t pid[4];
  };

  void ShowContext(std::ostream& os) const;

  Status Encrypt(Ptx const& pt, Ctx* out) const;
  Status SymEncrypt(Ptx const& pt, Ctx* out) const;
  Status Decrypt(Ctx const& ct, Ptx* out) const;
  /// Create a ciphertext that encrypts zero.
  Status EncryptZero(Ctx* out) const;
  /// Encrypt a monic polynomial. The monic polynomial is t[0] * X^{t[1]}.
  Status EncryptMonicPoly(std::tuple<F64, U64> const& monicPoly,
                          Ctx* out) const;
  /// Encode a complex vector with the specified precision factor.
  Status Encode(C64Vec const& vec, const F64 precision, Ptx* out) const;
  /// Encode a complex scalar with the specified precision factor.
  Status Encode(C64 const scalar, const F64 precision, Ptx* out) const;
  /// Encode a real vector with the specified precision factor.
  Status Encode(F64Vec const& vec, const F64 precision, Ptx* out) const;
  /// Encode a real scalar with the specified precision factor.
  Status Encode(F64 const scalar, const F64 precision, Ptx* out) const;
  /**
   * Encode vector and then Montgomerize the plaintext. @see Montgomerize
   * @tparam VecType the type of vector. VecType is either C64Vec or F64Vec.
   * @see C64Vec
   * @see F64Vec
   */
  template <class VecType>
  Status EncodeMontgomery(VecType const& vec, const F64 precision,
                          Ptx* out) const {
    static_assert(std::is_same<VecType, C64Vec>::value ||
                      std::is_same<VecType, F64Vec>::value,
                  "EncodeMontgomery: invalid vector type");
    CHECK_STATUS_INTERNAL(Encode(vec, precision, out),
                          "EncodeMontgomery: fail");
    return Montgomerize(out);
  }
  Status Decode(Ptx const& pt, size_t nslots, C64Vec* out) const;
  Status Decode(Ptx const& pt, size_t nslots, F64Vec* out) const;

  /// Sum two ciphertexts inplace.
  Status Add(Ctx* lhs, Ctx const& rhs) const;
  /// Sum a ciphertext and plaintext inplace.
  Status AddPlain(Ctx* lhs, Ptx const& rhs) const;
  /// Sum a ciphertext and plaintext inplace.
  Status AddScalar(Ctx* lhs, F64 const rhs) const;
  /// Subtraction
  Status Sub(Ctx* lhs, Ctx const& rhs) const;
  /// Subtract a plaintext from a ciphertext inplace.
  Status SubPlain(Ctx* lhs, Ptx const& rhs) const;
  /// Sum a plain scalar from a ciphertext inplace.
  Status SubScalar(Ctx* lhs, F64 const rhs) const;

  /**
   * Sum and subtract two ciphertexts inplace.
   * @note lhs = lhs + rhs
   * @note rhs = rhs - lhs
   */
  Status AddSub(Ctx* lhs, Ctx* rhs) const;
  /// Flip the sign of the ciphertext inplace.
  Status Negate(Ctx* op) const;
  /// Multiply two ciphertexts inplace (without relinearization).
  Status Mul(Ctx* lhs, Ctx const& rhs) const;
  /// Multiply a ciphertext with a plaintext inplace.
  Status MulPlain(Ctx* lhs, Ptx const& rhs) const;
  /// Multiply a ciphertext with a scalar inplace.
  Status MulScalar(Ctx* lhs, F64 const rhs, F64 const precision) const;
  /// Multiply the ciphertext with the image-unit, i.e., 1*j, inplace.
  Status MulImageUnit(Ctx* op) const;
  /**
   * Fused Multiply-Accumulate
   * @note accum = accum + lhs * rhs
   */
  Status FMA(Ctx* accum, Ctx const& lhs, Ptx const& rhs) const;

  /// Relinearization.
  Status Relin(Ctx* ct) const;
  /// Multiply-Relin-then-Rescale
  Status MulRelinRescale(Ctx* lhs, Ctx const& rhs) const;
  Status MulRelin(Ctx* out, Ctx const& lhs, Ctx const& rhs) const;
  /**
   * Rescale the scale of ciphertext by the specified number of bits.
   * @note ct.scale = ct.scale / 2^{nbits}
   *
   * @param ct Ciphertext to rescale.
   * @param nbits Number of bits to rescale. If nbits = 0, use the last moduli
   * for rescaling.
   */
  Status RescaleByBits(Ctx* ct, int nbits) const;
  /**
   * Rescale the scale of ciphertext by the specified factor.
   * @note ct.scale = ct.scale / factor
   *
   * @param ct Ciphertext to rescale.
   * @param factor The amount factor to rescale.
   */
  Status RescaleByFactor(Ctx* ct, F64 factor) const;
  Status RescaleNext(Ctx* ct) const;
  /// Copy the first n moduli from src to dst.
  Status AlignModuliCopy(Ctx* dst, Ctx const& src,
                         const int nModuliToCopy) const;
  /// Align the moduli of two.
  Status AlignModuli(Ctx* ct0, Ctx* ct1) const;
  /// Align the moduli of ct0 to ct1. Return error if ct0.nmoduli < ct1.nmoduli
  Status AlignModuli(Ctx* ct0, const Ctx& ct1) const;
  /// Drop n last moduli.
  Status DropModuli(Ctx* ct, int toDrop) const;
  /// Drop all modulo but keep the last one.
  Status KeepLastModuli(Ctx* ct) const;
  /// Raise the modulo of the ciphertext to the maxmimum modulo. @see
  /// MaximumNModuli
  Status ModulusUp(Ctx* ct) const;
  /// Compute the conjugation inplace.
  Status Conjugate(Ctx* compx) const;
  /// Left hande side rotation.
  Status RotateLeft(Ctx* vec, size_t offset) const;
  /// Right hande side rotation.
  Status RotateRight(Ctx* vec, size_t offset) const;
  /// Convert the plaintext for Montgomery reduction.
  Status Montgomerize(Ptx* rhs) const;
  /// The Montgomerized multiplication with plaintext. @see MulPlain
  Status MulPlainMontgomery(Ctx* lhs, Ptx const& rhs_mont) const;
  /// The Montgomerized Fused Multiply-Accumulate. @see FMA
  Status FMAMontgomery(Ctx* accum, Ctx const& lhs, Ptx const& rhs_mont) const;
  /// Ciphertext assignment according to
  Status Assign(Ctx* out, seal::parms_id_type pid, Ctx const& in) const;

#if GEMINI_USE_GMP
  /**
   * Convert the coeffients of plaintext to BigNumber presentation.
   * @param negative If true, the coeffients are given in [-p/2, p/2) range.
   *    If false, the coeffients are given in [0, p) range.
   */
  Status ConventionalForm(Ptx const& pt, std::vector<mpz_class>* coeff,
                          bool negative);
  /// Convert the coeffients of plaintext to real number presentation.
  Status ConventionalForm(Ptx const& pt, F64Vec* coeff, bool negative);
#endif

  /// Return the specified moduli. Index starts from 0.
  U64 GetModulusPrime(size_t index) const;
  /// Return the maxmimum number of modulo.
  size_t MaximumNModuli() const;
  /// Return the maxmimum number of values for encoding.
  size_t MaximumNSlots() const;

  static std::shared_ptr<RunTime> Create(std::string const& config);

  bool IsSupported(Supports f) const;

  Status SaveKey(Key key, std::ostream&) const;
  Status LoadKey(Key key, std::istream&);
  std::vector<size_t> GaloisKeyIndices() const;
  Status SaveGaloisKey(size_t keyIndex, std::ostream&) const;
  Status LoadGaloisKey(size_t keyIndex, std::istream&);

  Status LoadCtx(Ctx* ct, std::istream&) const;
  Status SaveCtx(Ctx const& ct, std::ostream&) const;
  Status LoadOneModuli(Ctx* ct, size_t modIdx, std::istream&) const;
  Status SaveOneModuli(Ctx const& ct, size_t modIdx, std::ostream&) const;

  // Chimera related APIs
  std::shared_ptr<seal::SEALContext> SEALRunTime();
  const seal::SecretKey& SEALSecretKey() const;

  ~RunTime();

 private:
  /// Empty runtime without any keys
  RunTime(int log2PolyDegree, std::vector<int> const& moduliBits,
          int nSpecialPrimes);
  /// 0s seed indicate to use random seed, if decryptionOnly = true, then only
  /// generate the secret key
  RunTime(int log2PolyDegree, std::vector<int> const& moduliBits,
          int nSpecialPrimes, std::array<uint64_t, 8> const& seed,
          bool decryptionOnly, bool isCompresskKey);

  RunTime(RunTime const& rt) = delete;
  RunTime(RunTime&& rt) = delete;
  RunTime& operator=(RunTime& rt) = delete;

  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace gemini

