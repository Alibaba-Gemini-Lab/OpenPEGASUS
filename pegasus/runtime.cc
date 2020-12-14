#include "pegasus/runtime.h"

#include <base64/base64.h>
#include <rapidjson/document.h>
#include <seal/serialization.h>
#include <seal/util/iterator.h>
#include <seal/util/polyarithsmallmod.h>
#include <seal/util/uintarithsmallmod.h>

#include <fstream>
#include <nonstd/optional.hpp>

#include "pegasus/rotgroup_helper.h"
#include "pegasus/status.h"

namespace gemini {

bool RU128(F64 f, U64 u128[2]) {
  constexpr F64 two_pow_64 = 4. * static_cast<F64>(1L << 62);
  constexpr F64 two_pow_128 = two_pow_64 * two_pow_64;
  f = std::fabs(f);
  if (f >= two_pow_128) {
    return false;
  }

  if (f >= two_pow_64) {
    u128[0] = static_cast<U64>(std::fmod(f, two_pow_64));
    u128[1] = static_cast<U64>(f / two_pow_64);
  } else {
    u128[0] = std::round(f);
    u128[1] = 0;
  }

  return true;
}

std::vector<uint32_t> GaloisKeysForFasterRotation(
    size_t nslots, std::shared_ptr<seal::SEALContext> context) {
  auto gtool = context->key_context_data()->galois_tool();
  auto galois_elts = gtool->get_elts_all();
  const size_t g = CeilSqrt(nslots);
  const size_t h = CeilDiv(nslots, g);
  for (size_t k = 1; k < h; ++k) {
    galois_elts.emplace_back(gtool->get_elt_from_step(g * k));
  }

  // remove duplication
  std::sort(galois_elts.begin(), galois_elts.end());
  galois_elts.erase(std::unique(galois_elts.begin(), galois_elts.end()),
                    galois_elts.end());

  return galois_elts;
}

#if GEMINI_USE_GMP
#define GEMINI_ENABLE_CONVENTIONAL_FORM 1
#include "pegasus/contrib/safe_ptr.h"

struct ConventionalFormPrecomp {
  size_t nmoduli;
  size_t nbits_moduli_product;
  size_t nbits_modulus_shoup;
  size_t nshift_modulus_shoup;

  mpz_class moduli_product;
  mpz_class modulus_shoup;
  std::vector<mpz_class> lifting_integers;

  using Moduli = std::vector<seal::Modulus>;
  explicit ConventionalFormPrecomp(Moduli const& moduli)
      : nmoduli(moduli.size()) {
    mpz_init_set_ui(moduli_product.get_mpz_t(), 1UL);
    for (auto const& modulus : moduli) {
      mpz_mul_ui(moduli_product.get_mpz_t(), moduli_product.get_mpz_t(),
                 modulus.value());
    }
    nbits_moduli_product = mpz_sizeinbase(moduli_product.get_mpz_t(), 2);
    nshift_modulus_shoup =
        nbits_moduli_product + 65 + (size_t)std::log2(nmoduli);

    mpz_init2(modulus_shoup.get_mpz_t(), nshift_modulus_shoup);
    mpz_ui_pow_ui(modulus_shoup.get_mpz_t(), 2, nshift_modulus_shoup);
    mpz_tdiv_q(modulus_shoup.get_mpz_t(), modulus_shoup.get_mpz_t(),
               moduli_product.get_mpz_t());
    nbits_modulus_shoup = mpz_sizeinbase(modulus_shoup.get_mpz_t(), 2);

    mpz_class quotient, current_modulus;
    lifting_integers.resize(nmoduli);
    for (size_t j = 0; j < nmoduli; ++j) {
      mpz_set_ui(current_modulus.get_mpz_t(), moduli[j].value());
      mpz_divexact(quotient.get_mpz_t(), moduli_product.get_mpz_t(),
                   current_modulus.get_mpz_t());
      mpz_init2(lifting_integers[j].get_mpz_t(), nbits_moduli_product);
      mpz_invert(lifting_integers[j].get_mpz_t(), quotient.get_mpz_t(),
                 current_modulus.get_mpz_t());
      lifting_integers[j] *= quotient;
    }
  }

  ~ConventionalFormPrecomp() {}
};
#else
#define GEMINI_ENABLE_CONVENTIONAL_FORM 0
#endif

bool GetHeader(RunTime::CtxHeader* header, Ctx const& ct) {
  if (!header) return false;
  if (ct.size() != 2) return false;

  bool ntt = ct.is_ntt_form();
  uint32_t deg = ct.poly_modulus_degree();
  if (ntt) deg |= 1;

  F64 scale = ct.scale();
  auto pid = ct.parms_id();

  header->degree = deg;
  header->scale = scale;
  std::memcpy(header->pid, &pid, sizeof(pid));
  return true;
}

/**
 *SEAL's runtime implementation
 */
struct RunTime::Impl {
  struct FastCKKSEncoder {
    explicit FastCKKSEncoder(std::shared_ptr<seal::SEALContext> context)
        : context_(context) {
      auto& context_data = *context_->first_context_data();
      degree_ = context_data.parms().poly_modulus_degree();
      max_nslots_ = degree_ >> 1;
      logn_ = seal::util::get_power_of_two(degree_);
      m_ = degree_ << 1;

      rotGroup_ = RotGroupHelper::get(degree_, degree_ >> 1);

      const F64 angle = 2 * M_PI / m_;
      roots_.resize(m_ + 1);
      for (size_t j = 0; j < m_; ++j) {
        roots_[j] = std::polar<F64>(1.0, angle * j);
      }
      roots_[m_] = roots_[0];
    }

    template <typename T>
    Status Encode(const std::vector<T>& vec, const F64 scale, Ptx* out) {
      CHECK_BOOL(out == nullptr, Status::ArgumentError("Encode: nULL pointer"));

      if (out->parms_id() == seal::parms_id_zero) {
        out->parms_id() = context_->first_parms_id();
      }

      const size_t input_size = vec.size();
      CHECK_BOOL(!IsValidLength(input_size),
                 Status::ArgumentError("Invalid length to pack"));

      auto mem_pool = seal::MemoryManager::GetPool();
      auto conj_values = seal::util::allocate<C64>(input_size, mem_pool);

      std::transform(vec.cbegin(), vec.cend(), conj_values.get(),
                     [](T const& v) -> C64 { return static_cast<C64>(v); });

      Pack(conj_values.get(), input_size);  // invFFT

      F64* r_start = reinterpret_cast<F64*>(conj_values.get());
      F64* r_end = r_start + input_size * 2;
      F64 sn = scale / input_size;
      std::transform(r_start, r_end, r_start,
                     [sn](F64 v) -> F64 { return v * sn; });

      int total_nbits = context_->get_context_data(out->parms_id())
                            ->total_coeff_modulus_bit_count();
      F64 upper_limit = static_cast<F64>(1UL << std::min(62, total_nbits));
      bool any_large = std::any_of(r_start, r_end, [upper_limit](F64 v) {
        return std::fabs(v) >= upper_limit;
      });
      CHECK_BOOL(any_large,
                 Status::ArgumentError("Encode: scale out of bound"));

      auto coeffients = seal::util::allocate<I64>(degree_, mem_pool);
      RoundCoeffients(conj_values.get(), input_size, coeffients.get());
      CHECK_BOOL(!ApplyNTT(coeffients.get(), degree_, out),
                 Status::InternalError("ApplyNTT error"));
      out->scale() = scale;

      return Status::Ok();
    }

    Status Decode(Impl& rt, const Ptx& in, size_t length, C64Vec* out) {
      using namespace seal;
      auto context_data_ptr = context_->get_context_data(in.parms_id());
      if (!context_data_ptr) {
        return Status::ArgumentError("invalid plaintext");
      }

      const size_t gap = max_nslots_ / length;
      std::vector<F64> coeffients;
      CHECK_STATUS(rt.ConventionalForm(in, &coeffients, true, gap));

      if (coeffients.size() != length * 2) {
        throw std::length_error("Decode fail");
      }

      out->resize(length);
      for (size_t i = 0; i < length; ++i) {
        out->at(i).real(coeffients.at(i));
        out->at(i).imag(coeffients.at(length + i));
      }

      Unpack(out->data(), length);
      return Status::Ok();
    }

    void RoundCoeffients(const C64* array, size_t nslots, I64* dst) const {
      if (!array || !IsValidLength(nslots)) {
        throw std::invalid_argument("RoundCoeffients: invalid_argument");
      }

      const int gap = max_nslots_ / nslots;
      if (gap != 1) {
        std::fill(dst, dst + degree_, 0UL);
      }

      I64* real_part = dst;
      I64* imag_part = dst + max_nslots_;

      for (size_t i = 0; i < nslots; ++i, real_part += gap, imag_part += gap) {
        *real_part = static_cast<I64>(std::round(array[i].real()));
        *imag_part = static_cast<I64>(std::round(array[i].imag()));
      }
    }

    bool ApplyNTT(const I64* coeffients, size_t length, Ptx* out) const {
      if (!out) {
        return false;
      }
      const auto pid = out->parms_id();
      const auto context_data = context_->get_context_data(pid);
      const size_t nmoduli = context_data->parms().coeff_modulus().size();
      const auto& small_ntt_tables = context_data->small_ntt_tables();
      if (!coeffients || length != degree_ || pid == seal::parms_id_zero) {
        return false;
      }

      out->parms_id() = seal::parms_id_zero;  // stop the warning in resize()
      out->resize(length * nmoduli);

      U64* dst_ptr = out->data();
      for (size_t cm = 0; cm < nmoduli; ++cm, dst_ptr += degree_) {
        const auto& modulus = small_ntt_tables[cm];  //.modulus();
        const U64 p = modulus.modulus().value();
        std::transform(coeffients, coeffients + length, dst_ptr,
                       [&modulus, p](I64 v) -> U64 {
                         bool sign = v < 0;
                         U64 vv = seal::util::barrett_reduce_64(
                             (U64)std::abs(v), modulus.modulus());
                         return vv > 0 ? (sign ? p - vv : vv) : 0;
                       });

        seal::util::ntt_negacyclic_harvey(dst_ptr, modulus);
      }
      out->parms_id() = pid;
      return true;
    }

    inline bool IsValidLength(size_t len) const {
      return !(len <= 0 || len > max_nslots_ || max_nslots_ % len != 0);
    }

    inline C64 C64Mul(C64 const& a, const C64& b) const {
      F64 x = a.real(), y = a.imag(), u = b.real(), v = b.imag();
      return C64(x * u - y * v, x * v + y * u);
    }

    inline void Invbutterfly(C64* x0, C64* x1, C64 const& w) const {
      C64 u = *x0;
      C64 v = *x1;
      *x0 = u + v;
      *x1 = C64Mul(u - v, w);
    }

    inline void Butterfly(C64* x0, C64* x1, C64 const& w) const {
      C64 u = *x0;
      C64 v = C64Mul(*x1, w);
      *x0 = u + v;
      *x1 = u - v;
    }

    void Pack(C64* vals, size_t n) const {
      if (!vals || !IsValidLength(n)) {
        throw std::invalid_argument("pack invalid argument");
      }

      for (size_t len = n / 2, h = 1; len > 2; len >>= 1, h <<= 1) {
        const size_t quad = len << 3;
        const size_t gap = m_ / quad;
        C64* x0 = vals;
        C64* x1 = x0 + len;
        for (size_t i = 0; i < h; ++i) {
          const size_t* rot = rotGroup_.data();
          size_t idx;
          for (size_t j = 0; j < len; j += 4) {
            idx = (quad - (*rot++ & (quad - 1))) * gap;
            Invbutterfly(x0++, x1++, roots_[idx]);

            idx = (quad - (*rot++ & (quad - 1))) * gap;
            Invbutterfly(x0++, x1++, roots_[idx]);

            idx = (quad - (*rot++ & (quad - 1))) * gap;
            Invbutterfly(x0++, x1++, roots_[idx]);

            idx = (quad - (*rot++ & (quad - 1))) * gap;
            Invbutterfly(x0++, x1++, roots_[idx]);
          }

          x0 += len;
          x1 += len;
        }
      }  // main loop

      {  // len = 2, h = n / 4, quad = 16
        C64* x0 = vals;
        C64* x1 = x0 + 2;

        const size_t idx0 = (16 - (rotGroup_[0] & 15)) * (m_ / 16);
        const size_t idx1 = (16 - (rotGroup_[1] & 15)) * (m_ / 16);
        for (size_t i = 0; i < n / 4; ++i) {
          Invbutterfly(x0++, x1++, roots_[idx0]);
          Invbutterfly(x0, x1, roots_[idx1]);
          x0 += 3;
          x1 += 3;
        }
      }

      {  // len = 1
        C64* x0 = vals;
        C64* x1 = x0 + 1;

        const long idx = (8 - (rotGroup_[0] & 7)) * (m_ / 8);
        for (size_t i = 0; i < n / 2; ++i) {
          Invbutterfly(x0, x1, roots_[idx]);
          x0 += 2;
          x1 += 2;
        }
      }

      RevBinPermute(vals, n);
    }

    void Unpack(C64* vals, size_t n) const {
      if (!vals || !IsValidLength(n)) {
        throw std::invalid_argument("unpack invalid argument");
      }

      RevBinPermute(vals, n);

      {  // len = 1, h = n / 2, quad = 8
        C64* x0 = vals;
        C64* x1 = x0 + 1;

        const long idx = (rotGroup_[0] & 7) * (m_ / 8);
        for (size_t i = 0; i < n / 2; ++i) {
          Butterfly(x0, x1, roots_[idx]);
          x0 += 2;
          x1 += 2;
        }
      }

      {  // len = 2, h = n / 4, quad = 16
        C64* x0 = vals;
        C64* x1 = x0 + 2;

        const size_t idx0 = (rotGroup_[0] & 15) * (m_ / 16);
        const size_t idx1 = (rotGroup_[1] & 15) * (m_ / 16);
        for (size_t i = 0; i < n / 4; ++i) {
          Butterfly(x0++, x1++, roots_[idx0]);
          Butterfly(x0, x1, roots_[idx1]);
          x0 += 3;
          x1 += 3;
        }
      }

      for (size_t len = 4, h = n / 8; len < n; len <<= 1, h >>= 1) {
        const long quad = (len << 3) - 1;  // mod 8 * len
        const long gap = m_ / (quad + 1);

        C64* x0 = vals;
        C64* x1 = x0 + len;

        for (size_t i = 0; i < h; ++i) {
          const size_t* rot = rotGroup_.data();
          long idx;
          for (size_t j = 0; j < len; j += 4) {
            idx = ((*rot++ & quad)) * gap;
            Butterfly(x0++, x1++, roots_[idx]);

            idx = ((*rot++ & quad)) * gap;
            Butterfly(x0++, x1++, roots_[idx]);

            idx = ((*rot++ & quad)) * gap;
            Butterfly(x0++, x1++, roots_[idx]);

            idx = ((*rot++ & quad)) * gap;
            Butterfly(x0++, x1++, roots_[idx]);
          }
          x0 += len;
          x1 += len;
        }
      }
    }

    template <typename T>
    void RevBinPermute(T* array, size_t length) const {
      if (length <= 2) return;
      for (size_t i = 1, j = 0; i < length; ++i) {
        size_t bit = length >> 1;
        for (; j >= bit; bit >>= 1) {
          j -= bit;
        }
        j += bit;

        if (i < j) {
          std::swap(array[i], array[j]);
        }
      }
    }

    C64Vec roots_;
    std::vector<size_t> rotGroup_;
    size_t max_nslots_, logn_, degree_, m_;
    std::shared_ptr<seal::SEALContext> context_{nullptr};
  };

  const int log2PolyDegree_;
  const size_t polyDegree_;
  const size_t maxNSlots_;
  const size_t maxNModuli_;

  void initBase(seal::EncryptionParameters parms) {
    using namespace seal;
    auto sec_level = sec_level_type::none;
    context_ = SEALContext::Create(parms, true, sec_level);

    evaluator_.reset(new Evaluator(context_));
    s_encoder_.reset(new CKKSEncoder(context_));
    f_encoder_.reset(new FastCKKSEncoder(context_));

    const auto& cntxt_data =
        context_->get_context_data(context_->first_parms_id());
    const auto& modulus = cntxt_data->parms().coeff_modulus();
    for (const seal::Modulus& moduli : modulus) {
      U64 primeInv = [](U64 prime) {
        U64 inv = 1;
        for (int i = 0; i < 63; ++i) {
          inv *= prime;
          prime *= prime;
        }
        return inv;  // prime^{-1} mod 2^64
      }(moduli.value());
      primeInvMod64_.push_back(primeInv);
    }
  }

  explicit Impl(size_t log2PolyDegree, std::vector<int> const& moduliBits,
                int nSpecialPrimes)
      : log2PolyDegree_(log2PolyDegree),
        polyDegree_(1UL << log2PolyDegree),
        maxNSlots_(polyDegree_ / 2),
        maxNModuli_(moduliBits.size() - nSpecialPrimes) {
    using namespace seal;
    EncryptionParameters parms(scheme_type::CKKS);
    parms.set_poly_modulus_degree(polyDegree_);
    parms.set_coeff_modulus(CoeffModulus::Create(polyDegree_, moduliBits));
    parms.set_n_special_primes(nSpecialPrimes);

    initBase(parms);
  }

  void GenerateHammingSecretKey(seal::SecretKey& sk, int hwt) {
    using namespace seal;
    auto& context_data = *context_->key_context_data();
    auto& parms = context_data.parms();
    auto& coeff_modulus = parms.coeff_modulus();
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
    uint64_t* dst_ptr = sk.data().data();
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

  explicit Impl(size_t log2PolyDegree, std::vector<int> const& moduliBits,
                int nSpecialPrimes, std::array<uint64_t, 8> const& seed,
                bool decryptionOnly, bool isCompresskKey)
      : log2PolyDegree_(log2PolyDegree),
        polyDegree_(1UL << log2PolyDegree),
        maxNSlots_(polyDegree_ / 2),
        maxNModuli_(moduliBits.size() - nSpecialPrimes) {
    using namespace seal;
    EncryptionParameters parms(scheme_type::CKKS);
    parms.set_poly_modulus_degree(polyDegree_);
    parms.set_coeff_modulus(CoeffModulus::Create(polyDegree_, moduliBits));
    parms.set_n_special_primes(nSpecialPrimes);
    parms.set_galois_generator(5);

    if (std::any_of(seed.cbegin(), seed.cend(),
                    [](uint64_t s) { return s != 0; })) {
      parms.set_random_generator(
          std::make_shared<seal::BlakePRNGFactory>(seed));
    }

    initBase(parms);

    sk_ = SecretKey();
    GenerateHammingSecretKey(*sk_, 64);
    KeyGenerator keygen(context_, *sk_);
    // KeyGenerator keygen(context_);
    // sk_ = nonstd::optional<SecretKey>(keygen.secret_key());

    decryptor_.reset(new Decryptor(context_, *sk_));

    if (!decryptionOnly) {
      pk_ = nonstd::optional<PublicKey>(keygen.public_key());
      if (nSpecialPrimes > 0) {
        if (isCompresskKey) {
          rlk_serial_ =
              nonstd::optional<Serializable<RelinKeys>>(keygen.relin_keys());
          rok_serial_ =
              nonstd::optional<Serializable<GaloisKeys>>(keygen.galois_keys());
        } else {
          rlk_ = nonstd::optional<RelinKeys>(keygen.relin_keys_local());
          rok_ = nonstd::optional<GaloisKeys>(keygen.galois_keys_local(
              GaloisKeysForFasterRotation(128, context_)));
        }
      }
      encryptor_.reset(new Encryptor(context_, *pk_));
    }
  }

  size_t MaximumNSlots() const { return maxNSlots_; }

  size_t MaximumNModuli() const { return maxNModuli_; }

  bool IsSupported(RunTime::Supports f) const {
    switch (f) {
      case RunTime::Supports::kRelin:
        return rlk_ ? true : false;
      case RunTime::Supports::kRotation:
        return rok_ ? true : false;
      case RunTime::Supports::kDecryption:
        return sk_ ? true : false;
      case RunTime::Supports::kEncryption:
        return pk_ ? true : false;
    }
    return false;
  }

  template <class SealObj>
  Status saveObj(SealObj const& obj, std::ostream& os) const {
    try {
      obj.save(os);
    } catch (std::exception e) {
      return Status::InternalError("saveObj failed. " + std::string(e.what()));
    }
    return Status::Ok();
  }

  template <class SealObj>
  Status loadObj(SealObj& obj, std::istream& is) const {
    try {
      obj.load(context_, is);
    } catch (std::exception e) {
      return Status::InternalError("loadObj failed." + std::string(e.what()));
    }
    return Status::Ok();
  }

  Status SaveKey(RunTime::Key obj, std::ostream& os) const {
    switch (obj) {
      case RunTime::Key::kSecretKey: {
        if (!sk_)
          return Status::InternalError("SaveKey: secret key is not set yet");
        return saveObj(*sk_, os);
      }
      case RunTime::Key::kPublicKey: {
        if (!pk_)
          return Status::InternalError("SaveKey: public key is not set yet");
        return saveObj(*pk_, os);
      }
      case RunTime::Key::kRelinKey: {
        if (!rlk_ && !rlk_serial_)
          return Status::InternalError("SaveKey: relin key is not set yet");
        if (rlk_serial_)
          return saveObj(*rlk_serial_, os);
        else
          return saveObj(*rlk_, os);
      }
      default:
        return Status::ArgumentError("SaveKey: not supported key type");
    }
  }

  Status LoadKey(RunTime::Key obj, std::istream& is) {
    switch (obj) {
      case RunTime::Key::kSecretKey: {
        if (!sk_) sk_ = seal::SecretKey();
        auto st = loadObj(*sk_, is);
        if (st.IsOk()) {
          decryptor_.reset(new seal::Decryptor(context_, *sk_));
        }
        return st;
      }
      case RunTime::Key::kPublicKey: {
        if (!pk_) pk_ = seal::PublicKey();
        auto st = loadObj(*pk_, is);
        if (st.IsOk()) {
          encryptor_.reset(new seal::Encryptor(context_, *pk_));
        }
        return st;
      }
      case RunTime::Key::kRelinKey: {
        if (!rlk_) rlk_ = seal::RelinKeys();
        return loadObj(*rlk_, is);
      }
      default:
        return Status::ArgumentError("SaveKey: not supported key type");
    }
  }

  Status LoadCtx(Ctx* ct, std::istream& is) const {
    CHECK_BOOL(ct == nullptr, Status::ArgumentError("LoadCtx: nullptr"));

    auto st = LoadOneModuli(ct, 0, is);
    if (!st.IsOk()) return st;

    size_t nmoduli = GetNModuli(*ct);
    for (size_t cm = 1; cm < nmoduli; ++cm) {
      auto st = LoadOneModuli(ct, cm, is);
      if (!st.IsOk()) return st;
    }
    return Status::Ok();
  }

  Status SaveCtx(Ctx const& ct, std::ostream& os) const {
    size_t nmoduli = GetNModuli(ct);
    for (size_t cm = 0; cm < nmoduli; ++cm) {
      auto st = SaveOneModuli(ct, cm, os);
      os << std::endl;
      if (!st.IsOk()) return st;
    }
    return Status::Ok();
  }

  Status LoadOneModuli(Ctx* ct, size_t modIdx, std::istream& is) const {
    RunTime::CtxHeader header;
    is.read(reinterpret_cast<char*>(&header), sizeof(header));

    bool is_ntt = (header.degree & 1);
    header.degree &= (~1);
    seal::parms_id_type pid;
    std::memcpy(&pid, header.pid, sizeof(pid));

    auto dat = context_->get_context_data(pid);
    if (!dat) {
      return Status::ArgumentError("LoadOneModuli: invalid ct header: pid");
    }

    if (header.degree != dat->parms().poly_modulus_degree()) {
      return Status::ArgumentError("LoadOneModuli: invalid ct header: degree");
    }

    if (header.moduli_index >= dat->parms().coeff_modulus().size() ||
        modIdx != header.moduli_index) {
      return Status::ArgumentError(
          "LoadOneModuli: invalid ct header: moduli_index");
    }

    ct->resize(context_, pid, 2);
    ct->is_ntt_form() = is_ntt;
    ct->scale() = header.scale;

    std::string line;
    std::getline(is, line);
    auto raw = base64::Decode(line);
    std::stringstream ss{raw};

    const uint32_t offset = header.moduli_index * header.degree;
    const uint32_t bytes = sizeof(uint64_t) * header.degree;
    ss.read(reinterpret_cast<char*>(ct->data(0) + offset), bytes);
    ss.read(reinterpret_cast<char*>(ct->data(1) + offset), bytes);

    return Status::Ok();
  }

  Status SaveOneModuli(Ctx const& ct, size_t modIdx, std::ostream& os) const {
    RunTime::CtxHeader header;
    if (!GetHeader(&header, ct)) {
      return Status::InternalError("SaveOneModuli: get header failed");
    }

    header.moduli_index = modIdx;
    const uint32_t deg = ct.poly_modulus_degree();
    const uint32_t bytes = deg * sizeof(uint64_t);

    std::stringstream ss;
    ss.write(reinterpret_cast<const char*>(ct.data(0) + modIdx * deg), bytes);
    ss.write(reinterpret_cast<const char*>(ct.data(1) + modIdx * deg), bytes);
    auto raw = ss.str();
    auto base64 =
        base64::Encode((unsigned char const*)raw.c_str(), raw.length());

    os.write(reinterpret_cast<const char*>(&header), sizeof(header));
    os.write(reinterpret_cast<const char*>(base64.c_str()), base64.length());
    return Status::Ok();
  }

  std::vector<size_t> GaloisKeyIndices() const {
    std::vector<size_t> indices;
    if (!rok_ && !rok_serial_) return indices;

    throw std::runtime_error("GaloisKeyIndices not ready");
    // bool serial = rok_serial_ != nonstd::nullopt;
    // const auto& keys = serial ? rok_serial_->object().data() : rok_->data();
    // size_t n = keys.size();
    //
    // for (size_t idx = 0; idx < n; ++idx) {
    //   const auto& key = keys.at(idx);
    //   if (!key.empty()) {
    //     indices.push_back(idx);
    //   }
    // }

    return indices;
  }

  Status SaveGaloisKey(size_t keyIndex, std::ostream& os) const {
    CHECK_BOOL(rok_ == nonstd::nullopt && rok_serial_ == nonstd::nullopt,
               Status::NotReady("Rotation key is not set yet"));
    throw std::runtime_error("GaloisKeyIndices not ready");
    // bool serial = (rok_serial_ != nonstd::nullopt);
    // const uint64_t keys_dim0 =
    //     serial ? rok_serial_->object().data().size() : rok_->data().size();
    // CHECK_BOOL(keyIndex >= keys_dim0,
    //            Status::ArgumentError("SaveGaloisKey: invalid key index"));
    // const auto& key =
    //     serial ? rok_serial_->object().data(keyIndex) : rok_->data(keyIndex);
    // CHECK_BOOL(key.empty(),
    //            Status::ArgumentError("SaveGaloisKey: empty key index"));
    //
    // const uint64_t keys_dim1 = static_cast<uint64_t>(key.size());
    // os.write(reinterpret_cast<const char*>(&keys_dim0), sizeof(uint64_t));
    // os.write(reinterpret_cast<const char*>(&keys_dim1), sizeof(uint64_t));
    //
    // try {
    //   for (size_t j = 0; j < keys_dim1; j++) {
    //     key[j].save(os);
    //   }
    // } catch (std::exception e) {
    //   return Status::InternalError("SaveGaloisKey failed " +
    //                                std::string(e.what()));
    // }
    return Status::Ok();
  }

  Status LoadGaloisKey(size_t keyIndex, std::istream& is) {
    if (rok_ == nonstd::nullopt) {
      rok_ = seal::GaloisKeys();
      rok_->parms_id() = context_->key_context_data()->parms_id();
    }

    uint64_t keys_dim0;
    uint64_t keys_dim1;
    is.read(reinterpret_cast<char*>(&keys_dim0), sizeof(uint64_t));
    is.read(reinterpret_cast<char*>(&keys_dim1), sizeof(uint64_t));

    CHECK_BOOL(keyIndex >= keys_dim0,
               Status::ArgumentError("LoadGaloisKey: invalid key index"));

    auto& keys = rok_->data();
    if (keys.empty()) {
      keys.resize(keys_dim0);
    } else if (keys.size() != keys_dim0) {
      return Status::InternalError("LoadGaloisKey: invalid key to load");
    }

    auto& key = keys.at(keyIndex);
    CHECK_BOOL(!key.empty(), Status::ArgumentError(
                                 "LoadGaloisKey: key index already loaded"));
    key.resize(keys_dim1);

    try {
      for (size_t j = 0; j < keys_dim1; j++) {
        key[j].load(context_, is);
      }
    } catch (std::exception e) {
      return Status::InternalError("LoadGaloisKey failed " +
                                   std::string(e.what()));
    }
    return Status::Ok();
  }

  void ShowContext(std::ostream& ss) const {
    auto& context_data = *context_->key_context_data();

    /*
    Which scheme are we using?
    */
    std::string scheme_name;
    switch (context_data.parms().scheme()) {
      case seal::scheme_type::BFV:
        scheme_name = "BFV";
        break;
      case seal::scheme_type::CKKS:
        scheme_name = "CKKS";
        break;
      default:
        throw std::invalid_argument("unsupported scheme");
    }
    ss << "/" << std::endl;
    ss << "| Encryption parameters :" << std::endl;
    ss << "|   scheme: " << scheme_name << std::endl;
    ss << "|   poly_modulus_degree: "
       << context_data.parms().poly_modulus_degree() << std::endl;

    /*
    Print the size of the true (product) coefficient modulus.
    */
    auto coeff_modulus = context_data.parms().coeff_modulus();
    std::size_t coeff_mod_count = coeff_modulus.size();
    ss << "|   #moduli: " << coeff_mod_count << std::endl;
    ss << "|   #special_primes: " << context_data.parms().n_special_primes()
       << std::endl;
    ss << "|   coeff_modulus size: ";
    ss << context_data.total_coeff_modulus_bit_count() << " (";
    for (std::size_t i = 0; i < coeff_mod_count - 1; i++) {
      ss << coeff_modulus[i].bit_count() << " + ";
    }
    ss << coeff_modulus.back().bit_count();
    ss << ") bits" << std::endl;

    /*
    For the BFV scheme print the plain_modulus parameter.
    */
    if (context_data.parms().scheme() == seal::scheme_type::BFV) {
      ss << "|   plain_modulus: "
         << context_data.parms().plain_modulus().value() << std::endl;
    }

    ss << "\\" << std::endl;
  }

  inline Status Encrypt(Ptx const& pt, Ctx* out) const {
    CHECK_BOOL(encryptor_ == nullptr,
               Status::NotReady("Encryption key is not set yet"));
    CHECK_BOOL(out == nullptr, Status::ArgumentError("Encrypt: Null pointer"));

    try {
      encryptor_->encrypt(pt, *out);
    } catch (std::invalid_argument e) {
      return Status::InternalError("Encrypt fail: " + std::string(e.what()));
    }

    return Status::Ok();
  }

  inline Status SymEncrypt(Ptx const& pt, Ctx* out) const {
    CHECK_BOOL(sk_ == nonstd::nullopt,
               Status::NotReady("Symmetric key is not set yet"));
    CHECK_BOOL(encryptor_ == nullptr,
               Status::NotReady("Symmetric encrytor is not set yet"));
    CHECK_BOOL(out == nullptr, Status::ArgumentError("Encrypt: Null pointer"));

    try {
      encryptor_->encrypt_symmetric(pt, *out);
    } catch (std::invalid_argument e) {
      return Status::InternalError("Encrypt fail: " + std::string(e.what()));
    }

    return Status::Ok();
  }

  inline Status EncryptZero(Ctx* out) const {
    CHECK_BOOL(out == nullptr,
               Status::ArgumentError("EncryptZero: Null pointer"));

    try {
      encryptor_->encrypt_zero(*out);
    } catch (std::invalid_argument e) {
      return Status::InternalError("EncryptZero fail: " +
                                   std::string(e.what()));
    }

    return Status::Ok();
  }

  inline Status EncryptMonicPoly(std::tuple<F64, U64> const& tuple,
                                 Ctx* out) const {
    CHECK_BOOL(out == nullptr,
               Status::ArgumentError("EncryptMonicPoly: Null pointer"));
    F64 v = std::round(std::get<0>(tuple));
    if (Ft64::AlmostEquals(v, 0.)) {
      encryptor_->encrypt_zero(*out);
      return Status::Ok();
    }

    auto pid = out->parms_id();
    if (pid == seal::parms_id_zero) pid = context_->first_parms_id();

    const auto context_data = context_->get_context_data(pid);
    const size_t degree = polyDegree_;

    int total_nbits = context_data->total_coeff_modulus_bit_count() + 2;
    F64 upper_limit = static_cast<F64>(1L << (std::min(62, total_nbits)));
    CHECK_BOOL(
        v >= upper_limit,
        Status::ArgumentError("EncryptMonicPoly: coefficients outs of bound"));
    CHECK_BOOL(std::get<1>(tuple) >= degree,
               Status::ArgumentError("EncryptMonicPoly: degree outs of bound"));

    std::vector<I64> coeffients(degree, 0);
    coeffients[std::get<1>(tuple)] = static_cast<I64>(v);
    Ptx ptx;
    ptx.parms_id() = pid;
    CHECK_BOOL(!f_encoder_->ApplyNTT(coeffients.data(), degree, &ptx),
               Status::InternalError("EncryptMonicPoly: ApplyNTT fail"));
    CHECK_STATUS_INTERNAL(Encrypt(ptx, out), "EncryptMonicPoly fail");
    return Status::Ok();
  }

  inline Status Decrypt(Ctx const& ct, Ptx* out) const {
    CHECK_BOOL(decryptor_ == nullptr,
               Status::NotReady("Decryption key is not set yet"));
    CHECK_BOOL(out == nullptr, Status::ArgumentError("Decrypt: Null pointer"));

    try {
      decryptor_->decrypt(ct, *out);
    } catch (std::invalid_argument e) {
      return Status::InternalError("Decrypt fail: " + std::string(e.what()));
    }

    return Status::Ok();
  }

  inline Status Encode(C64Vec const& vec, const F64 scale, Ptx* out) const {
    return f_encoder_->Encode<C64>(vec, scale, out);
  }

  inline Status Encode(C64 const scalar, const F64 scale, Ptx* out) const {
    CHECK_BOOL(out == nullptr, Status::ArgumentError("Encode: Null pointer"));
    auto pid = out->parms_id() == seal::parms_id_zero
                   ? context_->first_parms_id()
                   : out->parms_id();
    try {
      s_encoder_->encode(scalar, pid, scale, *out);
    } catch (std::logic_error e) {
      return Status::InternalError("Encode fail: " + std::string(e.what()));
    }
    return Status::Ok();
  }

  inline Status Encode(F64Vec const& vec, const F64 scale, Ptx* out) const {
    return f_encoder_->Encode<F64>(vec, scale, out);
  }

  inline Status Encode(F64 const scalar, const F64 scale, Ptx* out) const {
    CHECK_BOOL(out == nullptr, Status::ArgumentError("Decode: Null pointer"));
    auto pid = out->parms_id() == seal::parms_id_zero
                   ? context_->first_parms_id()
                   : out->parms_id();
    try {
      s_encoder_->encode(scalar, pid, scale, *out);
    } catch (std::logic_error e) {
      return Status::InternalError("Decode fail: " + std::string(e.what()));
    }
    return Status::Ok();
  }

  inline Status Decode(Ptx const& in, size_t length, C64Vec* out) const {
    return f_encoder_->Decode(*const_cast<Impl*>(this), in, length, out);
  }

  inline Status Decode(Ptx const& pt, size_t length, F64Vec* out) const {
    CHECK_BOOL(out == nullptr, Status::ArgumentError("Decode: Null pointer"));
    C64Vec temp(length);
    CHECK_STATUS(Decode(pt, length, &temp));
    out->resize(length);
    std::transform(temp.cbegin(), temp.cend(), out->data(),
                   [](C64 const& v) -> F64 { return v.real(); });
    return Status::Ok();
  }

  inline Status Add(Ctx* lhs, Ctx const& rhs) const {
    constexpr auto SAE = &Status::ArgumentError;
    CHECK_BOOL(lhs == nullptr, SAE("Add: Null pointer"));
    CHECK_BOOL(!IsMetaValid(*lhs), SAE("Add: invalid lhs"));
    CHECK_BOOL(!IsMetaValid(rhs), SAE("Add: invalid rhs"));
    CHECK_BOOL(lhs->size() != rhs.size(), SAE("Add: size mismatch"));
    CHECK_BOOL(!Ft64::AlmostEquals(lhs->scale(), rhs.scale()),
               SAE("Add: scale mismatch"));

    size_t nmodulus_lhs = GetNModuli(*lhs);
    size_t nmodulus_rhs = GetNModuli(rhs);
    if (nmodulus_lhs > nmodulus_rhs)
      evaluator_->mod_switch_to_inplace(*lhs, rhs.parms_id());

    auto cntxt_data = context_->get_context_data(lhs->parms_id());
    const auto& moduli = cntxt_data->parms().coeff_modulus();
    for (size_t k = 0; k < rhs.size(); ++k) {
      U64* lhs_ptr = lhs->data(k);
      const U64* rhs_ptr = rhs.data(k);
      for (const auto& modulus : moduli) {
        seal::util::add_poly_coeffmod(lhs_ptr, rhs_ptr, polyDegree_, modulus,
                                      lhs_ptr);
        lhs_ptr += polyDegree_;
        rhs_ptr += polyDegree_;
      }
    }

    return Status::Ok();
  }

  inline Status AddPlain(Ctx* lhs, Ptx const& rhs) const {
    CHECK_BOOL(lhs == nullptr, Status::ArgumentError("AddPlain: Null pointer"));
    try {
      evaluator_->add_plain_inplace(*lhs, rhs);
    } catch (std::invalid_argument e) {
      return Status::InternalError("AddPlain: invalid argument " +
                                   std::string(e.what()));
    } catch (std::logic_error e) {
      return Status::InternalError("AddPlain: logic error " +
                                   std::string(e.what()));
    }

    return Status::Ok();
  }

  inline Status AddScalar(Ctx* lhs, F64 const scalar) const {
    using namespace seal;
    using namespace seal::util;
    CHECK_BOOL(lhs == nullptr,
               Status::ArgumentError("AddScalar: Null pointer"));
    const F64 scaled_scalar = std::fabs(lhs->scale() * scalar);
    const bool sign = std::signbit(scalar);
    if (Ft64::AlmostEquals(scaled_scalar, 0.)) {
      return Status::Ok();  // noting to do
    }

    auto cntxt_dat = context_->get_context_data(lhs->parms_id());
    U64 scalar128[2];
    if (!RU128(scaled_scalar, scalar128)) {
      std::cout << "scalar " << std::log2(scaled_scalar) << " bits\n";
      return Status::NotReady("AddScalar: scaled scalar out-of-bound");
    }

    ConstNTTTablesIter tables_iter(cntxt_dat->small_ntt_tables());
    size_t poly_degree = lhs->poly_modulus_degree();
    const size_t num_moduli = GetNModuli(*lhs);

    RNSIter poly_iter(lhs->data(0), poly_degree);

    SEAL_ITERATE(iter(poly_iter, tables_iter), num_moduli, [&](auto J) {
      CoeffIter poly = std::get<0>(J);
      const NTTTables& ntt_table = std::get<1>(J);
      const Modulus& modulus = ntt_table.modulus();
      U64 add_operand = barrett_reduce_128(scalar128, modulus);

      if (lhs->is_ntt_form()) {
        // NOTE(juhou): Hack of NTT([scalar, 0, 0, ... 0]) = scalar * w_0
        add_operand = multiply_uint_mod(
            add_operand, ntt_table.get_from_root_powers(0), modulus);
      }

      if (sign) {
        add_operand = negate_uint_mod(add_operand, modulus);
      }

      std::transform(poly, poly + poly_degree, poly, [&](U64 u) -> U64 {
        return add_uint_mod(u, add_operand, modulus);
      });
    });
    return Status::Ok();
  }

  inline Status Sub(Ctx* lhs, Ctx const& rhs) const {
    constexpr auto SAE = &Status::ArgumentError;
    CHECK_BOOL(lhs == nullptr, SAE("Sub: Null pointer"));
    CHECK_BOOL(!IsMetaValid(*lhs), SAE("Sub: invalid lhs"));
    CHECK_BOOL(!IsMetaValid(rhs), SAE("Sub: invalid rhs"));
    CHECK_BOOL(lhs->size() != rhs.size(), SAE("Sub: size mismatch"));
    CHECK_BOOL(!Ft64::AlmostEquals(lhs->scale(), rhs.scale()),
               SAE("Sub: scale mismatch"));

    size_t nmodulus_lhs = GetNModuli(*lhs);
    size_t nmodulus_rhs = GetNModuli(rhs);
    CHECK_BOOL(nmodulus_lhs > nmodulus_rhs,
               SAE("Sub: lhs.coeff_mod_count() > rhs.coeff_mod_count()"));

    auto cntxt_data = context_->get_context_data(lhs->parms_id());
    const auto& moduli = cntxt_data->parms().coeff_modulus();
    for (size_t k = 0; k < rhs.size(); ++k) {
      U64* lhs_ptr = lhs->data(k);
      const U64* rhs_ptr = rhs.data(k);
      for (const auto& modulus : moduli) {
        seal::util::sub_poly_coeffmod(lhs_ptr, rhs_ptr, polyDegree_, modulus,
                                      lhs_ptr);
        lhs_ptr += polyDegree_;
        rhs_ptr += polyDegree_;
      }
    }

    return Status::Ok();
  }

  inline Status SubPlain(Ctx* lhs, Ptx const& rhs) const {
    CHECK_BOOL(lhs == nullptr, Status::ArgumentError("SubPlain: Null pointer"));
    try {
      evaluator_->sub_plain_inplace(*lhs, rhs);
    } catch (std::invalid_argument e) {
      return Status::InternalError("SubPlain invalid argument: " +
                                   std::string(e.what()));
    } catch (std::logic_error e) {
      return Status::InternalError("SubPlain logic error: " +
                                   std::string(e.what()));
    }
    return Status::Ok();
  }

  inline Status SubScalar(Ctx* lhs, F64 const rhs) const {
    return AddScalar(lhs, -rhs);
  }

  inline Status AddSub(Ctx* lhs, Ctx* rhs) const {
    CHECK_BOOL(lhs == nullptr or rhs == nullptr,
               Status::ArgumentError("AddSub: Null pointer"));
    CHECK_BOOL(!IsMetaValid(*lhs),
               Status::ArgumentError("AddSub: invalid lhs"));
    CHECK_BOOL(!IsMetaValid(*rhs),
               Status::ArgumentError("AddSub: invalid rhs"));
    CHECK_BOOL(lhs->size() != rhs->size(),
               Status::ArgumentError("AddSub: size mismatch"));
    CHECK_BOOL(lhs->parms_id() != rhs->parms_id(),
               Status::ArgumentError("AddSub: parms_id mismatch"));
    CHECK_BOOL(lhs == rhs,
               Status::ArgumentError("AddSub: lhs and rhs can't not be same"));

    struct AddSubHelper {
      explicit AddSubHelper(U64 p) : p(p) {}
      inline void operator()(U64* x, U64* y) {
        U64 u = *x + *y;
        U64 v = *y - *x + p;
        *x = u < p ? u : u - p;
        *y = v < p ? v : v - p;
      }
      const U64 p;
    };

    const auto& cntxt_data = context_->get_context_data(lhs->parms_id());
    const auto& modulus = cntxt_data->parms().coeff_modulus();

    const size_t ct_size = lhs->size();
    for (size_t i = 0; i < ct_size; ++i) {
      U64* lhs_op = lhs->data(i);
      U64* rhs_op = rhs->data(i);
      for (const auto& moduli : modulus) {
        AddSubHelper addsub(moduli.value());
        for (uint64_t d = 0; d < polyDegree_; ++d) {
          addsub(lhs_op++, rhs_op++);
        }
      }  // Note: SEAL uses continus memory for the moduli.
    }
    return Status::Ok();
  }

  inline Status Negate(Ctx* ct) const {
    CHECK_BOOL(ct == nullptr, Status::ArgumentError("Negate: Null pointer"));
    try {
      evaluator_->negate_inplace(*ct);
    } catch (std::invalid_argument e) {
      return Status::InternalError("Negate invalid_argument: " +
                                   std::string(e.what()));
    } catch (std::logic_error e) {
      return Status::InternalError("Negate logic_error: " +
                                   std::string(e.what()));
    }
    return Status::Ok();
  }

  seal::parms_id_type GetParmsIdFromModulusIndex(size_t index) const {
    if (index >=
        context_->first_context_data()->parms().coeff_modulus().size()) {
      throw std::invalid_argument("GetParmsIdFromModulusIndex");
    }
    auto cntxt_dat = context_->last_context_data();
    for (size_t i = 0; i < index; ++i) {
      assert(cntxt_dat);
      cntxt_dat = cntxt_dat->prev_context_data();
    }
    return cntxt_dat->parms_id();
  }

  Status DropModulus(Ctx& ct, size_t num_to_drops) {
    using namespace seal;
    using namespace seal::util;
    if (GetNModuli(ct) > num_to_drops) {
      return Status::ArgumentError("DropModulus: too many to drop");
    }
    if (num_to_drops == 0) {
      return Status::Ok();  // nothing to do
    }

    const size_t target_num_moduli = GetNModuli(ct) - num_to_drops;
    auto target_pid = GetParmsIdFromModulusIndex(target_num_moduli - 1);
    auto cntxt_dat = context_->get_context_data(target_pid);
    const int nbits = static_cast<int>(std::log2(std::abs(ct.scale())));

    if (nbits > cntxt_dat->total_coeff_modulus_bit_count() + 2) {
      return Status::InvalidFormat("DropModulus: scale out-of-bound");
    }

    Ctx tmp;
    tmp.resize(context_, target_pid, ct.size());

    const size_t poly_degree = ct.poly_modulus_degree();
    SEAL_ITERATE(iter(ct, tmp), ct.size(), [&](auto I) {
      set_poly(std::get<0>(I), poly_degree, target_num_moduli, std::get<1>(I));
    });

    tmp.is_ntt_form() = ct.is_ntt_form();
    tmp.scale() = ct.scale();
    ct = tmp;
    return Status::Ok();
  }

  inline Status Mul(Ctx* lhs, Ctx const& rhs) const {
    constexpr auto SAE = &Status::ArgumentError;
    CHECK_BOOL(lhs == nullptr, SAE("Mul: Null pointer"));
    CHECK_BOOL(!IsMetaValid(*lhs), SAE("Mul: invalid lhs"));
    CHECK_BOOL(!IsMetaValid(rhs), SAE("Mul: invalid rhs"));
    CHECK_BOOL(lhs->size() != 2, SAE("Mul: lhs.size() != 2"));
    CHECK_BOOL(rhs.size() != 2, SAE("Mul: rhs.size() != 2"));

    size_t nmodulus_lhs = GetNModuli(*lhs);
    size_t nmodulus_rhs = GetNModuli(rhs);

    if (nmodulus_lhs > nmodulus_rhs) {
      CHECK_STATUS(DropModuli(lhs, nmodulus_lhs - nmodulus_rhs));
    }

    CHECK_BOOL(lhs->parms_id() != rhs.parms_id(),
               SAE("Mul: parms_id mismatch"));

    try {
      if (lhs != &rhs) {
        evaluator_->multiply_inplace(*lhs, rhs);
      } else {
        evaluator_->square_inplace(*lhs);
      }
    } catch (std::invalid_argument e) {
      return Status::InternalError("Mul: invalid_argument: " +
                                   std::string(e.what()));
    } catch (std::logic_error e) {
      return Status::InternalError("Mul: logic_error " + std::string(e.what()));
    }

    return Status::Ok();
  }

  inline Status MulPlain(Ctx* lhs, Ptx const& rhs) const {
    CHECK_BOOL(lhs == nullptr, Status::ArgumentError("MulPlain: Null pointer"));
    CHECK_BOOL(!IsMetaValid(*lhs),
               Status::ArgumentError("MulPlain: invalid ct"));
    CHECK_BOOL(!lhs->is_ntt_form() or !rhs.is_ntt_form(),
               Status::ArgumentError("MulPlain: require NTT-form input"));

    const auto& cntxt_data = context_->get_context_data(lhs->parms_id());
    const auto& modulus = cntxt_data->parms().coeff_modulus();
    const size_t n_moduli = modulus.size();
    const size_t degree = cntxt_data->parms().poly_modulus_degree();
    CHECK_BOOL(n_moduli > rhs.coeff_count() / degree,
               Status::ArgumentError("MulPlain: ct and pt modulus mismatch"));

    const size_t ct_sze = lhs->size();
    for (size_t i = 0; i < ct_sze; ++i) {
      U64* op0 = lhs->data(i);
      const U64* op1 = rhs.data();
      for (size_t cm = 0; cm < n_moduli; ++cm) {
        seal::util::dyadic_product_coeffmod(op0, op1, degree, modulus[cm], op0);
        op0 += degree;
        op1 += degree;
      }
    }
    lhs->scale() *= rhs.scale();
    return Status::Ok();
  }

  /// dst = op0 * op1 using Montgomery Reduction
  void _DyadicProduct(const U64* op0, const U64* op1, size_t len,
                      seal::Modulus const& modulus, const U64 primeInv,
                      U64* dst) const {
    if (!op0 || !op1 || len < 1 || !dst) {
      return;
    }
    const U64 prime = modulus.value();
    const U64 tbl[2]{prime, 0};
    assert(prime * primeInv == 1);

    std::transform(
        op0, op0 + len, op1, dst, [tbl, prime, primeInv](U64 u, U64 v) -> U64 {
          U64 R;
          unsigned long long mul128[2], H;
          seal::util::multiply_uint64(u, v, mul128);
          R = mul128[0] * primeInv;  // mod 2^64
          seal::util::multiply_uint64_hw64(R, prime, &H);
          U64 r = static_cast<uint64_t>(mul128[1] - H) + prime;  // mod 2^64
          r -= tbl[r < prime];
          return r;
        });
  }

  inline Status MulPlainMontgomery(Ctx* lhs, Ptx const& rhs_mont) const {
    CHECK_BOOL(lhs == nullptr,
               Status::ArgumentError("MulPlainMontgomery: Null pointer"));
    const auto& cntxt_data = context_->get_context_data(lhs->parms_id());
    const auto& modulus = cntxt_data->parms().coeff_modulus();
    const size_t n_moduli = modulus.size();
    const size_t degree = cntxt_data->parms().poly_modulus_degree();
    CHECK_BOOL(n_moduli > rhs_mont.coeff_count() / degree,
               Status::ArgumentError(
                   "MulPlainMontgomery: ct and pt modulus mismatch"));
    for (size_t i = 0; i < lhs->size(); ++i) {
      U64* ct_ptr = lhs->data(i);
      const U64* pt_ptr = rhs_mont.data();
      for (size_t cm = 0; cm < n_moduli;
           ++cm, ct_ptr += degree, pt_ptr += degree) {
        const U64 primeInv =
            GetMontgomeryPrecomp(cm);  // primeInv = prime^{-1} mod 2^64
        _DyadicProduct(ct_ptr, pt_ptr, degree, modulus[cm], primeInv, ct_ptr);
      }
    }
    lhs->scale() *= rhs_mont.scale();
    return Status::Ok();
  }

  inline Status MulScalar(Ctx* lhs, F64 const _scalar, F64 precision) const {
    CHECK_BOOL(lhs == nullptr,
               Status::ArgumentError("MulScalar: Null pointer"));
    CHECK_BOOL(precision <= 0.,
               Status::ArgumentError("MulScalar: Negative precision"));

    F64 f_scalar = std::round(_scalar * precision);
    int nbits = static_cast<int>(std::log2(std::abs(f_scalar)));
    if (nbits >= 63) {  // slow case
      Ptx ptx;
      Encode(_scalar, precision, &ptx);
      CHECK_STATUS_INTERNAL(MulPlain(lhs, ptx), "MulScalar fail");
    } else {
      const auto& cntxt_dat = context_->get_context_data(lhs->parms_id());
      const auto& moduli = cntxt_dat->parms().coeff_modulus();
      const uint64_t n_moduli = moduli.size();

      CHECK_BOOL(nbits > cntxt_dat->total_coeff_modulus_bit_count() + 2,
                 Status::ArgumentError("MulScalar: scale out of bound"));

      const U64 scalar = static_cast<U64>(std::abs(f_scalar));
      bool sign = _scalar < 0;
      for (size_t k = 0; k < lhs->size(); ++k) {
        U64* poly = lhs->data(k);
        for (size_t cm = 0; cm < n_moduli; ++cm, poly += polyDegree_) {
          U64 multiplier = seal::util::barrett_reduce_64(scalar, moduli[cm]);
          if (sign) {
            multiplier = moduli[cm].value() - multiplier;
          }
          seal::util::multiply_poly_scalar_coeffmod(
              poly, polyDegree_, multiplier, moduli[cm], poly);
        }
      }
      lhs->scale() *= precision;
    }
    return Status::Ok();
  }

  inline Status MulImageUnit(Ctx* ct) const {
    CHECK_BOOL(ct == nullptr,
               Status::ArgumentError("MulImageUnit: Null pointer"));
    CHECK_BOOL(!IsMetaValid(*ct),
               Status::ArgumentError("MulImageUnit: invalid ct"));
    const bool is_ntt = ct->is_ntt_form();
    const auto& cntxt_dat = context_->get_context_data(ct->parms_id());
    const auto& ntt_tables = cntxt_dat->small_ntt_tables();
    const auto& modulus = cntxt_dat->parms().coeff_modulus();
    const size_t n_moduli = modulus.size();
    const size_t degree = polyDegree_;

    for (size_t k = 0; k < ct->size(); ++k) {
      for (size_t cm = 0; cm < n_moduli; ++cm) {
        uint64_t* result = ct->data(k) + cm * degree;

        if (is_ntt) {
          seal::util::inverse_ntt_negacyclic_harvey(result, ntt_tables[cm]);
        }

        // Perform: poly * X^{N/2} mod X^N + 1.
        //          a_i * X^j -> a_i * X^{i + N/2} for 0 <= i < N/2
        //          a_j * X^i -> -a_j * X^{j - N/2} for N/2 <= j < N
        uint64_t p = modulus[cm].value();
        uint64_t* ai = result;
        uint64_t* aj = result + degree / 2;
        for (size_t i = 0; i < degree / 2; ++i, ++ai, ++aj) {
          if (*aj > 0) {
            *aj = p - *aj;  // i.e. -x1 mod p
          }
          std::swap(*ai, *aj);
        }

        if (is_ntt) {
          seal::util::ntt_negacyclic_harvey(result, ntt_tables[cm]);
        }
      }
    }
    return Status::Ok();
  }

  inline bool IsMetaValid(Ctx const& ct) const {
    return seal::is_metadata_valid_for(ct, context_);
  }

  inline Status checkForFMA(Ctx* accum, Ctx const& lhs, Ptx const& rhs) const {
    constexpr auto SAE = &Status::ArgumentError;
    CHECK_BOOL(accum == nullptr, SAE("FMA: Null pointer"));
    CHECK_BOOL(!IsMetaValid(*accum), SAE("FMA: Invalid accum"));
    CHECK_BOOL(!IsMetaValid(lhs), SAE("FMA: Invalid lhs"));
    CHECK_BOOL(!accum->is_ntt_form(), SAE("FMA: require NTT-form of accum"));
    CHECK_BOOL(!lhs.is_ntt_form(), SAE("FMA: require NTT-form of lhs"));
    CHECK_BOOL(!rhs.is_ntt_form(), SAE("FMA: require NTT-form of rhs"));
    CHECK_BOOL(
        accum->size() != lhs.size() or accum->parms_id() != lhs.parms_id(),
        SAE("FMA: accum and lhs parms_id mismatch"));
    return Status::Ok();
  }

  inline Status FMA(Ctx* accum, Ctx const& lhs, Ptx const& rhs) const {
    CHECK_STATUS(checkForFMA(accum, lhs, rhs));
    const auto& cntxt_data = context_->get_context_data(lhs.parms_id());
    const auto& modulus = cntxt_data->parms().coeff_modulus();
    const size_t n_moduli = modulus.size();
    const size_t degree = polyDegree_;
    const F64 new_scale = lhs.scale() * rhs.scale();
    CHECK_BOOL(n_moduli > rhs.coeff_count() / degree,
               Status::ArgumentError("FMA: ct and pt modulus mismatch"));
    CHECK_BOOL(!Ft64::AlmostEquals(accum->scale(), new_scale),
               Status::ArgumentError("FMA: scale mismatch"));

    for (size_t i = 0; i < lhs.size(); ++i) {
      U64* accum_ptr = accum->data(i);
      const U64* op0 = lhs.data(i);
      const U64* op1 = rhs.data();
      for (size_t cm = 0; cm < n_moduli; ++cm) {
        // wide = op0 * op1 with double-precision
        // wide[0] is low-64 and wide[1] is high-64.
        unsigned long long wide[2];
        for (size_t d = 0; d < degree; ++d) {
          seal::util::multiply_uint64(*op0++, *op1++, wide);
          // add_uint64() return the carry which is added to the high-64.
          wide[1] += seal::util::add_uint64(*accum_ptr, wide[0], &wide[0]);
          // lazy reduction
          *accum_ptr++ = seal::util::barrett_reduce_128(wide, modulus[cm]);
        }
      }
    }
    accum->scale() = new_scale;
    return Status::Ok();
  }

  Status FMAMontgomery(Ctx* accum, Ctx const& lhs, Ptx const& rhs_mont) const {
    constexpr auto SAE = &Status::ArgumentError;
    const F64 new_scale = lhs.scale() * rhs_mont.scale();
    const auto& cntxt_data = context_->get_context_data(lhs.parms_id());
    const auto& modulus = cntxt_data->parms().coeff_modulus();
    const size_t n_moduli = modulus.size();
    const size_t degree = polyDegree_;

    CHECK_STATUS_INTERNAL(checkForFMA(accum, lhs, rhs_mont),
                          "FMAMontgomery: check fail");
    CHECK_BOOL(!Ft64::AlmostEquals(accum->scale(), new_scale),
               SAE("FMAMontgoery: scale mismatch"));
    CHECK_BOOL(n_moduli > rhs_mont.coeff_count() / degree,
               SAE("MulPlainMontgomery: lhs and rhs parms_id mismatch"));

    for (size_t i = 0; i < lhs.size(); ++i) {
      for (size_t cm = 0; cm < n_moduli; ++cm) {
        U64* accum_ptr = accum->data(i) + cm * degree;
        const U64* op0 = lhs.data(i) + cm * degree;
        const U64* op1 = rhs_mont.data() + cm * degree;

        const U64 prime = modulus[cm].value();
        const U64 primeInv =
            GetMontgomeryPrecomp(cm);  // primeInv = prime^{-1} mod 2^64
        const U64 tbl[2]{prime, 0};
        assert(prime * primeInv == 1);

        unsigned long long wide[2], H;
        for (size_t d = 0; d < degree; ++d) {
          // Montgomery reduction
          seal::util::multiply_uint64(*op0++, *op1++, wide);
          U64 R = wide[0] * primeInv;
          seal::util::multiply_uint64_hw64(R, prime, &H);
          U64 r = static_cast<U64>(wide[1] - H) + prime;
          r -= tbl[r < prime];
          r += *accum_ptr;
          *accum_ptr++ = (r - tbl[r < prime]);
        }
      }
    }
    accum->scale() = new_scale;
    return Status::Ok();
  }

  Status Relin(Ctx* ct) const {
    CHECK_BOOL(ct == nullptr, Status::ArgumentError("Relin: Null pointer"));
    CHECK_BOOL(!IsSupported(RunTime::Supports::kRelin),
               Status::InternalError("Relin: not supported"));
    try {
      evaluator_->relinearize_inplace(*ct, *rlk_);
    } catch (std::invalid_argument e) {
      return Status::InternalError("Relin invalid_argument: " +
                                   std::string(e.what()));
    } catch (std::logic_error e) {
      return Status::InternalError("Relin logic_error: " +
                                   std::string(e.what()));
    }
    return Status::Ok();
  }

  Status MulRelinRescale(Ctx* lhs, const Ctx& rhs) const {
    CHECK_BOOL(lhs == nullptr,
               Status::ArgumentError("MulRelinRescale: Null pointer"));

    F64 scale = std::min(
        rhs.scale(), static_cast<F64>(GetModulusPrime(GetNModuli(*lhs) - 1)));
    CHECK_STATUS_INTERNAL(Mul(lhs, rhs), "MulRelinRescale: Mul fail");
    CHECK_STATUS_INTERNAL(Relin(lhs), "MulRelinRescale: Relin fail");
    CHECK_STATUS_INTERNAL(RescaleByFactor(lhs, scale),
                          "MulRelinRescale: Rescale fail");

    return Status::Ok();
  }

  Status MulRelin(Ctx* out, const Ctx& lhs, const Ctx& rhs) const {
    CHECK_BOOL(out == nullptr, Status::ArgumentError("MulRelin: Null pointer"));
    if (out == &lhs) {
      CHECK_STATUS_INTERNAL(Mul(out, rhs), "MulRelin: Mul fail");
    } else if (out == &rhs) {
      CHECK_STATUS_INTERNAL(Mul(out, lhs), "MulRelin: Mul fail");
    } else {
      if (GetNModuli(lhs) > GetNModuli(rhs)) {
        *out = lhs;
        CHECK_STATUS_INTERNAL(Mul(out, rhs), "MulRelin: Mul fail");
      } else {
        *out = rhs;
        CHECK_STATUS_INTERNAL(Mul(out, lhs), "MulRelin: Mul fail");
      }
    }
    CHECK_STATUS_INTERNAL(Relin(out), "MulRelin: Relin fail");
    return Status::Ok();
  }

  Status RescaleNext(Ctx* ct) const {
    CHECK_BOOL(ct == nullptr,
               Status::ArgumentError("RescaleNext: Null pointer"));
    CHECK_BOOL(
        GetNModuli(*ct) == 1,
        Status::ArgumentError("RescaleNext: already in the lowest level"));
    CHECK_STATUS_INTERNAL(rescaleNextPrime(ct), "RescaleByFactor: fail");
    return Status::Ok();
  }

  Status rescaleNextPrime(Ctx* ct) const {
    try {
      evaluator_->rescale_to_next_inplace(*ct);
    } catch (std::invalid_argument e) {
      return Status::InternalError("RescaleNextPrime invalid_argument: " +
                                   std::string(e.what()));
    } catch (std::logic_error e) {
      return Status::InternalError("RescaleNextPrime logic_error: " +
                                   std::string(e.what()));
    }

    return Status::Ok();
  }

  Status RescaleByFactor(Ctx* ct, F64 factor) const {
    CHECK_BOOL(ct == nullptr,
               Status::ArgumentError("RescaleByFactor: Null pointer"));
    CHECK_BOOL(
        GetNModuli(*ct) == 1,
        Status::ArgumentError("RescaleByFactor: already in the lowest level"));
    CHECK_BOOL(factor < 0,
               Status::ArgumentError("RescaleByFactor: factor <= 0"));
    CHECK_BOOL(Ft64::AlmostEquals(factor, 0.),
               Status::ArgumentError("RescaleByFactor: factor <= 0"));

    F64 next_prime = static_cast<F64>(GetModulusPrime(GetNModuli(*ct) - 1));
    F64 scaleUp = next_prime / factor;
    F64 targetScale = ct->scale() / factor;

    CHECK_BOOL(std::round(scaleUp) < 1.,
               Status::ArgumentError("RescaleByFactor: factor > moduli"));

    if (!Ft64::AlmostEquals(scaleUp, 1.)) {  // scaleUp > 1
      CHECK_STATUS_INTERNAL(MulScalar(ct, 1., scaleUp),
                            "RescaleByFactor: MulScalar fail");
    }

    CHECK_STATUS_INTERNAL(rescaleNextPrime(ct), "RescaleByFactor: fail");

    ct->scale() = targetScale;

    return Status::Ok();
  }

  Status RescaleByBits(Ctx* ct, int nbits) const {
    CHECK_BOOL(nbits > 60, Status::ArgumentError("RescaleByBits: nbits > 60"));
    CHECK_BOOL(nbits <= 0, Status::ArgumentError("RescaleByBits: nbits <= 0"));
    CHECK_STATUS_INTERNAL(RescaleByFactor(ct, static_cast<F64>(1L << nbits)),
                          "RescaleByBits: fail");
    return Status::Ok();
  }

  Status DropModuli(Ctx* ct, int nToDrop) const {
    assert(nToDrop >= 0);
    assert(ct && GetNModuli(*ct) > nToDrop);

    auto pid = ct->parms_id();
    for (int i = 0; i < nToDrop; ++i) {
      auto next = context_->get_context_data(pid)->next_context_data();
      pid = next->parms_id();
    }

    try {
      evaluator_->mod_switch_to_inplace(*ct, pid);
    } catch (std::logic_error e) {
      return Status::InternalError("KeepLastModuli logic_error: " +
                                   std::string(e.what()));
    }
    return Status::Ok();
  }

  Status AlignModuliCopy(Ctx* out, Ctx const& src, const int nModuli) const {
    CHECK_BOOL(out == nullptr,
               Status::ArgumentError("AlignModuliCopy: Null pointer"));
    CHECK_BOOL((int)GetNModuli(src) < nModuli,
               Status::ArgumentError("AlignModuliCopy: nModuli out-of-bound"));

    auto context_dat = context_->last_context_data();
    for (int i = 1; i < nModuli; ++i) {
      context_dat = context_dat->prev_context_data();
      CHECK_BOOL(context_dat == nullptr,
                 Status::InternalError("AlignModuliCopy"));
    }

    if (src.scale() > 0) {
      int moduliSze = context_dat->total_coeff_modulus_bit_count();
      CHECK_BOOL(
          (int)std::log2(src.scale()) > moduliSze,
          Status::ArgumentError("AlignModuliCopy: #moduli < src.scale()."));
    }

    out->resize(context_, context_dat->parms_id(), src.size());
    const size_t degree = context_dat->parms().poly_modulus_degree();
    const size_t nbytes = sizeof(src.data()[0]) * degree;

    for (size_t k = 0; k < src.size(); ++k) {
      auto src_ptr = src.data(k);
      auto dst_ptr = out->data(k);

      for (int cm = 0; cm < nModuli; ++cm) {
        std::memcpy(dst_ptr, src_ptr, nbytes);
        src_ptr += degree;
        dst_ptr += degree;
      }
    }

    out->scale() = src.scale();
    out->is_ntt_form() = src.is_ntt_form();
    return Status::Ok();
  }

  Status AlignModuli(Ctx* ct0, Ctx const& ct1) const {
    CHECK_BOOL(ct0 == nullptr,
               Status::ArgumentError("AlignModuli: Null pointer"));

    size_t nm0 = GetNModuli(*ct0);
    size_t nm1 = GetNModuli(ct1);
    CHECK_BOOL(nm0 < nm1,
               Status::ArgumentError("AlignModuli: lhs.nmoduli < rhs.nmoduli"));

    if (nm0 == nm1) return Status::Ok();
    CHECK_STATUS_INTERNAL(DropModuli(ct0, nm0 - nm1), "AlignModuli fail");
    return Status::Ok();
  }

  Status AlignModuli(Ctx* ct0, Ctx* ct1) const {
    CHECK_BOOL(ct0 == nullptr,
               Status::ArgumentError("AlignModuli: Null pointer"));
    CHECK_BOOL(ct1 == nullptr,
               Status::ArgumentError("AlignModuli: Null pointer"));

    size_t nm0 = GetNModuli(*ct0);
    size_t nm1 = GetNModuli(*ct1);

    if (nm0 > nm1) {
      CHECK_STATUS_INTERNAL(DropModuli(ct0, nm0 - nm1), "AlignModuli fail");
    } else if (nm0 < nm1) {
      CHECK_STATUS_INTERNAL(DropModuli(ct1, nm1 - nm0), "AlignModuli fail");
    }
    return Status::Ok();
  }

  Status KeepLastModuli(Ctx* ct) const {
    CHECK_BOOL(ct == nullptr,
               Status::ArgumentError("KeepLastModuli: Null pointer"));
    size_t n = GetNModuli(*ct);
    if (n > 1)
      CHECK_STATUS_INTERNAL(DropModuli(ct, n - 1), "KeepLastModuli fail");
    return Status::Ok();
  }

  inline Status RotateLeft(Ctx* vec, size_t offset) const {
    CHECK_BOOL(!IsSupported(RunTime::Supports::kRotation),
               Status::InternalError("Rotation: not supported"));
    CHECK_BOOL(vec == nullptr,
               Status::ArgumentError("RotateLeft: Null pointer"));

    try {
      evaluator_->rotate_vector_inplace(*vec, offset, *rok_);
    } catch (std::invalid_argument e) {
      return Status::InternalError("RotateLeft invalid_argument: " +
                                   std::string(e.what()));
    } catch (std::logic_error e) {
      return Status::InternalError("RotateLeft logic_error: " +
                                   std::string(e.what()));
    }
    return Status::Ok();
  }

  inline Status RotateRight(Ctx* vec, size_t offset) const {
    CHECK_BOOL(!IsSupported(RunTime::Supports::kRotation),
               Status::InternalError("Rotation: not supported"));
    CHECK_BOOL(vec == nullptr,
               Status::ArgumentError("RotateRight: Null pointer"));
    try {
      evaluator_->rotate_vector_inplace(*vec, -static_cast<I64>(offset), *rok_);
    } catch (std::invalid_argument e) {
      return Status::InternalError("RotateRight invalid_argument: " +
                                   std::string(e.what()));
    } catch (std::logic_error e) {
      return Status::InternalError("RotateRight logic_error: " +
                                   std::string(e.what()));
    }
    return Status::Ok();
  }

  inline Status Conjugate(Ctx* compx) const {
    CHECK_BOOL(!IsSupported(RunTime::Supports::kRotation),
               Status::InternalError("Conjugate: not supported"));
    CHECK_BOOL(compx == nullptr,
               Status::ArgumentError("Conjugate: Null pointer"));
    try {
      U64 conj =
          context_->first_context_data()->galois_tool()->get_elt_from_step(0);
      evaluator_->apply_galois_inplace(*compx, conj, *rok_);
    } catch (std::invalid_argument e) {
      return Status::InternalError("Conjugate invalid_argument: " +
                                   std::string(e.what()));
    } catch (std::logic_error e) {
      return Status::InternalError("Conjugate logic_error: " +
                                   std::string(e.what()));
    }
    return Status::Ok();
  }

  /// For 0 < x < prime, return its Montgomery form x*2^64 mod prime.
  inline Status Montgomerize(Ptx* pt) const {
    CHECK_BOOL(pt == nullptr,
               Status::ArgumentError("Montgomerize: Null pointer"));
    CHECK_BOOL(!pt->is_ntt_form(),
               Status::ArgumentError("Montgomerize: require NTT input"));
    const auto& cntxt_data = context_->get_context_data(pt->parms_id());
    const auto& modulus = cntxt_data->parms().coeff_modulus();
    const size_t degree = polyDegree_;

    U64* pt_ptr = pt->data();
    for (const auto& moduli : modulus) {
      const U64 prime = moduli.value();
      const U64 r0 = moduli.const_ratio()[0];  // r0 = hi64(2^128 / prime)
      const U64 r1 = moduli.const_ratio()[1];  // r1 = lo64(2^128 / prime)
      // lazy Montgomery form, i.e., in the range [0, 2p).
      std::transform(
          pt_ptr, pt_ptr + degree, pt_ptr, [r0, r1, prime](U64 a) -> U64 {
            unsigned long long hi;  // SEAL requires explicit ULL type.
            seal::util::multiply_uint64_hw64(a, r0, &hi);
            return -((a * r1) + static_cast<U64>(hi)) * prime;
          });
      pt_ptr += degree;
    }
    return Status::Ok();
  }

  inline Status ModulusUp(Ctx* ct) const {
    CHECK_BOOL(ct == nullptr, Status::ArgumentError("ModulusUp: Null pointer"));
    auto dst_pid = context_->first_parms_id();
    if (ct->parms_id() == dst_pid) {
      return Status::Ok();
    }

    auto dat = context_->get_context_data(dst_pid);
    const auto& moduli = dat->parms().coeff_modulus();
    const auto& ntt_tables = dat->small_ntt_tables();
    const size_t degree = polyDegree_;
    const size_t nmoduli = moduli.size();
    const size_t ct_sze = ct->size();
    const bool is_ntt_form = ct->is_ntt_form();

    Ctx temp;
    temp.resize(context_, dst_pid, ct_sze);

    std::vector<U64> non_ntt_input(degree);
    for (size_t k = 0; k < ct_sze; ++k) {
      U64* dst_ptr = temp.data(k);
      std::memcpy(non_ntt_input.data(), ct->data(k), sizeof(U64) * degree);
      std::memcpy(dst_ptr, non_ntt_input.data(), sizeof(U64) * degree);
      if (is_ntt_form) {
        seal::util::inverse_ntt_negacyclic_harvey(non_ntt_input.data(),
                                                  ntt_tables[0]);
      }

      for (size_t i = 1; i < nmoduli; ++i) {
        const auto& mod_qi = moduli[i];
        dst_ptr += degree;
        // ct mod q0 -> (ct mod q0) mod qi
        std::transform(non_ntt_input.cbegin(), non_ntt_input.cend(), dst_ptr,
                       [&mod_qi](U64 v) {
                         return seal::util::barrett_reduce_64(v, mod_qi);
                       });
        if (is_ntt_form) {
          seal::util::ntt_negacyclic_harvey(dst_ptr, ntt_tables[i]);
        }
      }
    }

    temp.scale() = ct->scale();
    temp.is_ntt_form() = is_ntt_form;
    *ct = temp;
    return Status::Ok();
  }

  ~Impl() {
#if GEMINI_ENABLE_CONVENTIONAL_FORM
    for (auto kv : convFormPrecomp_) {
      if (kv.second) delete kv.second;
      kv.second = nullptr;
    }
    convFormPrecomp_.clear();
#endif
  }

  inline bool IsValidLength(size_t len) const {
    return ((len > 0) and (len < maxNSlots_) and (maxNSlots_ % len == 0));
  }

  inline U64 GetMontgomeryPrecomp(size_t moduliIdx) const {
    if (moduliIdx >= primeInvMod64_.size()) return 0UL;
    return primeInvMod64_[moduliIdx];
  }

  std::shared_ptr<seal::SEALContext> context_;
  std::unique_ptr<seal::Encryptor> encryptor_{nullptr};
  std::unique_ptr<seal::Decryptor> decryptor_{nullptr};
  std::unique_ptr<seal::Evaluator> evaluator_{nullptr};

  nonstd::optional<seal::SecretKey> sk_{nonstd::nullopt};
  nonstd::optional<seal::PublicKey> pk_{nonstd::nullopt};
  nonstd::optional<seal::RelinKeys> rlk_{nonstd::nullopt};
  nonstd::optional<seal::GaloisKeys> rok_{nonstd::nullopt};
  nonstd::optional<seal::Serializable<seal::RelinKeys>> rlk_serial_{
      nonstd::nullopt};
  nonstd::optional<seal::Serializable<seal::GaloisKeys>> rok_serial_{
      nonstd::nullopt};

  std::vector<U64> primeInvMod64_;

  std::unique_ptr<seal::CKKSEncoder> s_encoder_{nullptr};
  std::unique_ptr<FastCKKSEncoder> f_encoder_{nullptr};

#if GEMINI_ENABLE_CONVENTIONAL_FORM
  Status ConventionalForm(Ptx const& pt, std::vector<F64>* out, bool neg,
                          size_t gap = 1) {
    CHECK_BOOL(out == nullptr,
               Status::ArgumentError("ConventionalForm: Null pointer"));
    std::vector<mpz_class> big_coeff;
    CHECK_STATUS(ConventionalForm(pt, &big_coeff, neg, gap));

    mpf_t bigf;
    mpf_init(bigf);
    const uint64_t scale = (uint64_t)pt.scale();

    out->resize(big_coeff.size());
    for (size_t i = 0; i < big_coeff.size(); i++) {
      mpf_set_z(bigf, big_coeff[i].get_mpz_t());
      mpf_div_ui(bigf, bigf, scale);
      out->at(i) = mpf_get_d(bigf);
    }
    mpf_clear(bigf);
    return Status::Ok();
  }

  Status Assign(Ctx* out, seal::parms_id_type pid, Ctx const& in) const {
    CHECK_BOOL(out == nullptr, Status::ArgumentError("Assign: Null pointer"));
    out->is_ntt_form() = false;  // resize() requires is_ntt_form = false
    out->resize(context_, pid, in.size());
    out->is_ntt_form() = in.is_ntt_form();

    for (size_t k = 0; k < in.size(); ++k) {
      U64* out_ptr = out->data(k);
      const U64* in_ptr = in.data(k);
      for (size_t cm = 0; cm < GetNModuli(*out); ++cm) {
        std::memcpy(out_ptr, in_ptr, sizeof(U64) * polyDegree_);
        in_ptr += polyDegree_;
        out_ptr += polyDegree_;
      }
    }

    return Status::Ok();
  }

  Status ConventionalForm(Ptx const& pt, std::vector<mpz_class>* out, bool neg,
                          size_t gap = 1) {
    CHECK_BOOL(out == nullptr,
               Status::ArgumentError("ConventionalForm: Null pointer"));
    CHECK_BOOL(pt.parms_id() == seal::parms_id_zero,
               Status::ArgumentError("ConventionalForm: invalid pt"));
    const auto& cntxt_data = context_->get_context_data(pt.parms_id());
    size_t degree = polyDegree_;
    size_t nmoduli = pt.coeff_count() / degree;
    precompGuard_.lock_shared();  // R-lock
    auto val = convFormPrecomp_.find(nmoduli);
    if (val != convFormPrecomp_.end()) {
      precompGuard_.unlock_shared();
      return _toConventionalForm(pt, val->second, out, neg, gap);
    }

    precompGuard_.unlock_shared();
    precompGuard_.lock();  // W-lock
    val = convFormPrecomp_.find(nmoduli);
    if (val != convFormPrecomp_.end()) {  // double-check
      precompGuard_.unlock();
      return _toConventionalForm(pt, val->second, out, neg, gap);
    }

    auto precompObj =
        new ConventionalFormPrecomp(cntxt_data->parms().coeff_modulus());
    convFormPrecomp_.insert({nmoduli, precompObj});
    precompGuard_.unlock();
    return _toConventionalForm(pt, precompObj, out, neg, gap);
  }

  Status _toConventionalForm(Ptx const& pt, ConventionalFormPrecomp* precomp,
                             std::vector<mpz_class>* out, bool neg,
                             size_t gap = 1) {
    CHECK_BOOL(precomp == nullptr,
               Status::InternalError("ConventionalForm: Null precomp"));
    CHECK_BOOL(out == nullptr,
               Status::ArgumentError("ConventionalForm: Null pointer"));
    CHECK_BOOL(pt.parms_id() == seal::parms_id_zero,
               Status::ArgumentError("ConventionalForm: invalid pt"));

    const auto& cntxt_data = context_->get_context_data(pt.parms_id());
    const auto& ntt_tables = cntxt_data->small_ntt_tables();
    size_t degree = polyDegree_;
    CHECK_BOOL(gap < 1, Status::ArgumentError("ConventionalForm: invalid gap"));
    CHECK_BOOL((degree % gap) != 0,
               Status::ArgumentError("ConventionalForm: invalid gap"));
    size_t nmoduli = pt.coeff_count() / degree;
    if (nmoduli != precomp->nmoduli) {
      throw std::length_error("ConventionalForm: invalid plaintext nmoduli");
    }

    mpz_t tmp;
    mpz_init2(tmp,
              precomp->nshift_modulus_shoup - 1 + precomp->nbits_modulus_shoup);
    mpz_t Qhalf;
    mpz_init_set(Qhalf, precomp->moduli_product.get_mpz_t());
    mpz_cdiv_q_ui(Qhalf, Qhalf, 2);

    out->resize(degree / gap);
    for (auto& coeff : *out) mpz_set_ui(coeff.get_mpz_t(), 0);

    std::vector<U64> non_ntt;
    for (size_t j = 0; j < nmoduli; ++j) {
      const U64* raw_data;
      if (pt.is_ntt_form()) {
        non_ntt.resize(degree);
        std::memcpy(non_ntt.data(), pt.data() + j * degree,
                    sizeof(U64) * degree);
        seal::util::inverse_ntt_negacyclic_harvey(non_ntt.data(),
                                                  ntt_tables[j]);
        raw_data = non_ntt.data();
      } else {
        raw_data = pt.data() + j * degree;
      }

      const auto& lift = precomp->lifting_integers[j].get_mpz_t();
      for (auto& coeff : *out) {
        if (*raw_data != 0) {
          mpz_addmul_ui(coeff.get_mpz_t(), lift, *raw_data);
        }
        raw_data += gap;
      }
    }

    for (auto& coeff : *out) {
      mpz_mul(tmp, coeff.get_mpz_t(), precomp->modulus_shoup.get_mpz_t());
      mpz_tdiv_q_2exp(tmp, tmp, precomp->nshift_modulus_shoup);
      mpz_submul(coeff.get_mpz_t(), tmp, precomp->moduli_product.get_mpz_t());
      if (cmp(coeff, precomp->moduli_product) >= 0) {
        coeff -= precomp->moduli_product;
      }

      if (neg && (mpz_cmp(coeff.get_mpz_t(), Qhalf) >= 0)) {
        coeff -= precomp->moduli_product;
      }
    }

    mpz_clears(tmp, Qhalf, NULL);
    return Status::Ok();
  }

  sf::contention_free_shared_mutex<> precompGuard_;
  std::unordered_map<size_t, ConventionalFormPrecomp*> convFormPrecomp_;
#endif  // GEMINI_ENABLE_CONVENTIONAL_FORM

  U64 GetModulusPrime(size_t cm) const {
    auto pid = context_->first_parms_id();
    return context_->get_context_data(pid)
        ->parms()
        .coeff_modulus()
        .at(cm)
        .value();
  }

  template <class Obj>
  Status saveObject(Obj const& obj, std::ostream& out) const {
    try {
      obj.save(out);
    } catch (std::exception e) {
      return Status::FileIOError(std::string(e.what()));
    }
    return Status::Ok();
  }

  template <class Obj>
  Status loadObject(Obj& obj, std::istream& in) const {
    try {
      obj.load(context_, in);
    } catch (std::exception e) {
      return Status::FileIOError(std::string(e.what()));
    }
    return Status::Ok();
  }
};

RunTime::RunTime(int log2PolyDegree, std::vector<int> const& moduliBits,
                 int nSpecialPrimes, std::array<uint64_t, 8> const& seed,
                 bool decryptionOnly, bool isCompresskKey) {
  impl_ =
      std::unique_ptr<Impl>(new Impl(log2PolyDegree, moduliBits, nSpecialPrimes,
                                     seed, decryptionOnly, isCompresskKey));
}

RunTime::RunTime(int log2PolyDegree, std::vector<int> const& moduliBits,
                 int nSpecialPrimes) {
  impl_ = std::unique_ptr<Impl>(
      new Impl(log2PolyDegree, moduliBits, nSpecialPrimes));
}

RunTime::~RunTime() {}

std::shared_ptr<RunTime> RunTime::Create(std::string const& json) {
  using namespace rapidjson;
  Value v;
  Document document;

  if (!document.Parse(json.c_str(), json.length()).HasParseError()) {
    int log2PolyDegree = -1;

    if (document.HasMember(JSONField::Log2PolyDegree().c_str())) {
      v = document[JSONField::Log2PolyDegree().c_str()];
      if (v.IsInt()) {
        log2PolyDegree = v.GetInt();
      }
    }

    std::vector<int> moduliBits;
    if (document.HasMember(JSONField::ModuliArray().c_str())) {
      v = document[JSONField::ModuliArray().c_str()];
      if (v.IsArray() && v.Size() >= 1) {
        moduliBits.resize(v.Size());
        for (size_t i = 0; i < v.Size(); ++i) {
          moduliBits[i] = v[i].GetInt();
        }
      }
    }

    bool seeded = true;
    std::array<uint64_t, 8> seed;
    std::fill(seed.begin(), seed.end(), 0);
    if (document.HasMember(JSONField::Seed().c_str())) {
      v = document[JSONField::Seed().c_str()];
      if (v.IsArray() && v.Size() == 8) {
        for (size_t i = 0; i < v.Size(); ++i) {
          seed[i] = v[i].GetInt();
        }
      }
    } else {
      seeded = false;
    }

    int nSpecialPrimes = 0;
    if (document.HasMember(JSONField::NSpecialPrimes().c_str())) {
      v = document[JSONField::NSpecialPrimes().c_str()];
      if (v.IsInt()) {
        nSpecialPrimes = v.GetInt();
      }
    }

    bool decryptionOnly = false;
    if (document.HasMember(JSONField::DecryptionOnly().c_str())) {
      v = document[JSONField::DecryptionOnly().c_str()];
      if (v.IsBool()) {
        decryptionOnly = v.GetBool();
      }
    }

    bool isCompresskKey = false;
    if (document.HasMember(JSONField::CompressKey().c_str())) {
      v = document[JSONField::CompressKey().c_str()];
      if (v.IsBool()) {
        isCompresskKey = v.GetBool();
      }
    }

    if (log2PolyDegree <= 0 || nSpecialPrimes >= (int)moduliBits.size()) {
      throw std::runtime_error(
          "RunTime::Create: Missing required fields: log2PolyDegree, "
          "moduliArray");
    }

    if (seeded) {
      return std::shared_ptr<RunTime>(
          new RunTime(log2PolyDegree, moduliBits, nSpecialPrimes, seed,
                      decryptionOnly, isCompresskKey));
    } else {
      return std::shared_ptr<RunTime>(
          new RunTime(log2PolyDegree, moduliBits, nSpecialPrimes));
    }
  } else {
    throw std::runtime_error("RunTime::Create: Can not parse this JSON");
  }
}

void RunTime::ShowContext(std::ostream& ss) const {
  return impl_->ShowContext(ss);
}

Status RunTime::Encrypt(Ptx const& pt, Ctx* out) const {
  return impl_->Encrypt(pt, out);
}

Status RunTime::SymEncrypt(Ptx const& pt, Ctx* out) const {
  return impl_->SymEncrypt(pt, out);
}

Status RunTime::EncryptZero(Ctx* out) const { return impl_->EncryptZero(out); }

Status RunTime::EncryptMonicPoly(std::tuple<F64, U64> const& tuple,
                                 Ctx* out) const {
  return impl_->EncryptMonicPoly(tuple, out);
}

Status RunTime::Decrypt(Ctx const& ct, Ptx* out) const {
  return impl_->Decrypt(ct, out);
}

Status RunTime::Encode(C64Vec const& vec, const F64 scale, Ptx* out) const {
  return impl_->Encode(vec, scale, out);
}

Status RunTime::Encode(C64 const scalar, const F64 scale, Ptx* out) const {
  return impl_->Encode(scalar, scale, out);
}

Status RunTime::Encode(F64Vec const& vec, const F64 scale, Ptx* out) const {
  return impl_->Encode(vec, scale, out);
}

Status RunTime::Encode(F64 const scalar, const F64 scale, Ptx* out) const {
  return impl_->Encode(scalar, scale, out);
}

Status RunTime::Decode(Ptx const& pt, size_t nslots, C64Vec* out) const {
  return impl_->Decode(pt, nslots, out);
}

Status RunTime::Decode(Ptx const& pt, size_t nslots, F64Vec* out) const {
  return impl_->Decode(pt, nslots, out);
}

Status RunTime::Add(Ctx* lhs, Ctx const& rhs) const {
  return impl_->Add(lhs, rhs);
}

Status RunTime::AddPlain(Ctx* lhs, Ptx const& rhs) const {
  return impl_->AddPlain(lhs, rhs);
}

Status RunTime::AddScalar(Ctx* lhs, F64 const rhs) const {
  return impl_->AddScalar(lhs, rhs);
}

Status RunTime::Sub(Ctx* lhs, Ctx const& rhs) const {
  return impl_->Sub(lhs, rhs);
}

Status RunTime::SubPlain(Ctx* lhs, Ptx const& rhs) const {
  return impl_->SubPlain(lhs, rhs);
}

Status RunTime::SubScalar(Ctx* lhs, F64 const rhs) const {
  return impl_->SubScalar(lhs, rhs);
}

Status RunTime::AddSub(Ctx* lhs, Ctx* rhs) const {
  return impl_->AddSub(lhs, rhs);
}

Status RunTime::Negate(Ctx* op) const { return impl_->Negate(op); }

Status RunTime::Mul(Ctx* lhs, Ctx const& rhs) const {
  return impl_->Mul(lhs, rhs);
}

Status RunTime::MulPlain(Ctx* lhs, Ptx const& rhs) const {
  return impl_->MulPlain(lhs, rhs);
}

Status RunTime::MulScalar(Ctx* lhs, F64 const rhs, F64 const prec) const {
  return impl_->MulScalar(lhs, rhs, prec);
}

Status RunTime::MulImageUnit(Ctx* op) const { return impl_->MulImageUnit(op); }

Status RunTime::FMA(Ctx* accum, Ctx const& lhs, Ptx const& rhs) const {
  return impl_->FMA(accum, lhs, rhs);
}

Status RunTime::FMAMontgomery(Ctx* accum, Ctx const& lhs,
                              Ptx const& rhs_mont) const {
  return impl_->FMAMontgomery(accum, lhs, rhs_mont);
}

Status RunTime::Relin(Ctx* ct) const { return impl_->Relin(ct); }

Status RunTime::MulRelinRescale(Ctx* lhs, const Ctx& rhs) const {
  return impl_->MulRelinRescale(lhs, rhs);
}

Status RunTime::MulRelin(Ctx* out, const Ctx& lhs, const Ctx& rhs) const {
  return impl_->MulRelin(out, lhs, rhs);
}
Status RunTime::RescaleNext(Ctx* ct) const { return impl_->RescaleNext(ct); }

Status RunTime::RescaleByBits(Ctx* ct, int nbits) const {
  return impl_->RescaleByBits(ct, nbits);
}

Status RunTime::RescaleByFactor(Ctx* ct, F64 factor) const {
  return impl_->RescaleByFactor(ct, factor);
}

Status RunTime::AlignModuli(Ctx* ct0, Ctx* ct1) const {
  return impl_->AlignModuli(ct0, ct1);
}

Status RunTime::AlignModuli(Ctx* ct0, const Ctx& ct1) const {
  return impl_->AlignModuli(ct0, ct1);
}

Status RunTime::AlignModuliCopy(Ctx* dst, const Ctx& src,
                                const int nModuli) const {
  return impl_->AlignModuliCopy(dst, src, nModuli);
}

Status RunTime::DropModuli(Ctx* ct0, int n) const {
  return impl_->DropModuli(ct0, n);
}

Status RunTime::KeepLastModuli(Ctx* ct) const {
  return impl_->KeepLastModuli(ct);
}

Status RunTime::RotateLeft(Ctx* vec, size_t offset) const {
  return impl_->RotateLeft(vec, offset);
}

Status RunTime::RotateRight(Ctx* vec, size_t offset) const {
  return impl_->RotateRight(vec, offset);
}

Status RunTime::Conjugate(Ctx* compx) const { return impl_->Conjugate(compx); }

Status RunTime::Montgomerize(Ptx* pt) const { return impl_->Montgomerize(pt); }

Status RunTime::MulPlainMontgomery(Ctx* lhs, Ptx const& rhs_mont) const {
  return impl_->MulPlainMontgomery(lhs, rhs_mont);
}

Status RunTime::ModulusUp(Ctx* ct) const { return impl_->ModulusUp(ct); }

Status RunTime::Assign(Ctx* out, seal::parms_id_type pid, Ctx const& in) const {
  return impl_->Assign(out, pid, in);
}

#if GEMINI_ENABLE_CONVENTIONAL_FORM
Status RunTime::ConventionalForm(Ptx const& pt, std::vector<mpz_class>* coeff,
                                 bool neg) {
  return impl_->ConventionalForm(pt, coeff, neg);
}

Status RunTime::ConventionalForm(Ptx const& pt, std::vector<F64>* coeff,
                                 bool neg) {
  return impl_->ConventionalForm(pt, coeff, neg);
}
#endif  // GEMINI_ENABLE_CONVENTIONAL_FORM

U64 RunTime::GetModulusPrime(size_t cm) const {
  return impl_->GetModulusPrime(cm);
}

size_t RunTime::MaximumNSlots() const { return impl_->MaximumNSlots(); }

size_t RunTime::MaximumNModuli() const { return impl_->MaximumNModuli(); }

bool RunTime::IsSupported(Supports f) const { return impl_->IsSupported(f); }

Status RunTime::SaveKey(Key obj, std::ostream& os) const {
  return impl_->SaveKey(obj, os);
}

Status RunTime::LoadKey(Key obj, std::istream& is) {
  return impl_->LoadKey(obj, is);
}

Status RunTime::LoadCtx(Ctx* ct, std::istream& is) const {
  return impl_->LoadCtx(ct, is);
}

Status RunTime::SaveCtx(Ctx const& ct, std::ostream& os) const {
  return impl_->SaveCtx(ct, os);
}

Status RunTime::LoadOneModuli(Ctx* ct, size_t modIdx, std::istream& is) const {
  return impl_->LoadOneModuli(ct, modIdx, is);
}

Status RunTime::SaveOneModuli(Ctx const& ct, size_t modIdx,
                              std::ostream& os) const {
  return impl_->SaveOneModuli(ct, modIdx, os);
}

std::vector<size_t> RunTime::GaloisKeyIndices() const {
  return impl_->GaloisKeyIndices();
}

Status RunTime::SaveGaloisKey(size_t keyIndex, std::ostream& os) const {
  return impl_->SaveGaloisKey(keyIndex, os);
}

Status RunTime::LoadGaloisKey(size_t keyIndex, std::istream& is) {
  return impl_->LoadGaloisKey(keyIndex, is);
}

std::shared_ptr<seal::SEALContext> RunTime::SEALRunTime() {
  return impl_->context_;
}
const seal::SecretKey& RunTime::SEALSecretKey() const { return *impl_->sk_; }

size_t GetNModuli(Ctx ct) { return ct.coeff_modulus_size(); }

}  // namespace gemini

