// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "seal/util/ntt.h"
#include "seal/util/uintarith.h"
#include "seal/util/uintarithsmallmod.h"
#include <algorithm>

using namespace std;

namespace seal
{
    namespace util
    {
        NTTTables::NTTTables(int coeff_count_power, const Modulus &modulus, MemoryPoolHandle pool) : pool_(move(pool))
        {
#ifdef SEAL_DEBUG
            if (!pool_)
            {
                throw invalid_argument("pool is uninitialized");
            }
#endif
            initialize(coeff_count_power, modulus);
        }

        void NTTTables::initialize(int coeff_count_power, const Modulus &modulus)
        {
#ifdef SEAL_DEBUG
            if ((coeff_count_power < get_power_of_two(SEAL_POLY_MOD_DEGREE_MIN)) ||
                coeff_count_power > get_power_of_two(SEAL_POLY_MOD_DEGREE_MAX))
            {
                throw invalid_argument("coeff_count_power out of range");
            }
#endif
            coeff_count_power_ = coeff_count_power;
            coeff_count_ = size_t(1) << coeff_count_power_;

            // Allocate memory for the tables
            root_powers_ = allocate<MultiplyUIntModOperand>(coeff_count_, pool_);
            inv_root_powers_ = allocate<MultiplyUIntModOperand>(coeff_count_, pool_);
            modulus_ = modulus;

            // We defer parameter checking to try_minimal_primitive_root(...)
            if (!try_minimal_primitive_root(2 * coeff_count_, modulus_, root_))
            {
                throw invalid_argument("invalid modulus");
            }

            uint64_t inverse_root;
            if (!try_invert_uint_mod(root_, modulus_, inverse_root))
            {
                throw invalid_argument("invalid modulus");
            }

            // Populate the tables storing (scaled version of) powers of root
            // mod q in bit-scrambled order.
            ntt_powers_of_primitive_root(root_, root_powers_.get());

            // Populate the tables storing (scaled version of) powers of
            // (root)^{-1} mod q in bit-scrambled order.
            ntt_powers_of_primitive_root(inverse_root, inv_root_powers_.get());

            // Reordering inv_root_powers_ so that the access pattern in inverse NTT is sequential.
            auto temp = allocate<MultiplyUIntModOperand>(coeff_count_, pool_);
            MultiplyUIntModOperand *temp_ptr = temp.get() + 1;
            for (size_t m = (coeff_count_ >> 1); m > 0; m >>= 1)
            {
                for (size_t i = 0; i < m; i++)
                {
                    *temp_ptr++ = inv_root_powers_[m + i];
                }
            }
            copy_n(temp.get() + 1, coeff_count_ - 1, inv_root_powers_.get() + 1);

            // Last compute n^(-1) modulo q.
            uint64_t degree_uint = static_cast<uint64_t>(coeff_count_);
            if (!try_invert_uint_mod(degree_uint, modulus_, inv_degree_modulo_.operand))
            {
                throw invalid_argument("invalid modulus");
            }
            inv_degree_modulo_.set_quotient(modulus_);
#if GEMINI_SEAL
            reduce_precomp_.set(1, modulus_); // 2^64/p
            // merge the last inv_root_powers with n^{-1}
            auto inv_n_w = multiply_uint_mod(inv_root_powers_[coeff_count_ - 1].operand, inv_degree_modulo_, modulus_);
            inv_root_powers_[coeff_count_ - 1].set(inv_n_w, modulus_);
#endif
            return;
        }

        void NTTTables::ntt_powers_of_primitive_root(uint64_t root, MultiplyUIntModOperand *destination) const
        {
            MultiplyUIntModOperand *destination_start = destination;
            destination_start->set(1, modulus_);
            for (size_t i = 1; i < coeff_count_; i++)
            {
                MultiplyUIntModOperand *next_destination = destination_start + reverse_bits(i, coeff_count_power_);
                next_destination->set(multiply_uint_mod(destination->operand, root, modulus_), modulus_);
                destination = next_destination;
            }
        }

        class NTTTablesCreateIter
        {
        public:
            using value_type = NTTTables;
            using pointer = void;
            using reference = value_type;
            using difference_type = ptrdiff_t;

            // LegacyInputIterator allows reference to be equal to value_type so we can construct
            // the return objects on the fly and return by value.
            using iterator_category = input_iterator_tag;

            // Require default constructor
            NTTTablesCreateIter()
            {}

            // Other constructors
            NTTTablesCreateIter(int coeff_count_power, vector<Modulus> modulus, MemoryPoolHandle pool)
                : coeff_count_power_(coeff_count_power), modulus_(modulus), pool_(pool)
            {}

            // Require copy and move constructors and assignments
            NTTTablesCreateIter(const NTTTablesCreateIter &copy) = default;

            NTTTablesCreateIter(NTTTablesCreateIter &&source) = default;

            NTTTablesCreateIter &operator=(const NTTTablesCreateIter &assign) = default;

            NTTTablesCreateIter &operator=(NTTTablesCreateIter &&assign) = default;

            // Dereferencing creates NTTTables and returns by value
            inline value_type operator*() const
            {
                return { coeff_count_power_, modulus_[index_], pool_ };
            }

            // Pre-increment
            inline NTTTablesCreateIter &operator++() noexcept
            {
                index_++;
                return *this;
            }

            // Post-increment
            inline NTTTablesCreateIter operator++(int) noexcept
            {
                NTTTablesCreateIter result(*this);
                index_++;
                return result;
            }

            // Must be EqualityComparable
            inline bool operator==(const NTTTablesCreateIter &compare) const noexcept
            {
                return (compare.index_ == index_) && (coeff_count_power_ == compare.coeff_count_power_);
            }

            inline bool operator!=(const NTTTablesCreateIter &compare) const noexcept
            {
                return !operator==(compare);
            }

            // Arrow operator must be defined
            value_type operator->() const
            {
                return **this;
            }

        private:
            size_t index_ = 0;
            int coeff_count_power_ = 0;
            vector<Modulus> modulus_;
            MemoryPoolHandle pool_;
        };

        void CreateNTTTables(
            int coeff_count_power, const vector<Modulus> &modulus, Pointer<NTTTables> &tables, MemoryPoolHandle pool)
        {
            if (!pool)
            {
                throw invalid_argument("pool is uninitialized");
            }
            if (!modulus.size())
            {
                throw invalid_argument("invalid modulus");
            }
            // coeff_count_power and modulus will be validated by "allocate"

            NTTTablesCreateIter iter(coeff_count_power, modulus, pool);
            tables = allocate(iter, modulus.size(), pool);
        }

#if !GEMINI_SEAL
        /**
        This function computes in-place the negacyclic NTT. The input is
        a polynomial a of degree n in R_q, where n is assumed to be a power of
        2 and q is a prime such that q = 1 (mod 2n).

        The output is a vector A such that the following hold:
        A[j] =  a(psi**(2*bit_reverse(j) + 1)), 0 <= j < n.

        For details, see Michael Naehrig and Patrick Longa.
        */
        void ntt_negacyclic_harvey_lazy(CoeffIter operand, const NTTTables &tables)
        {
#ifdef SEAL_DEBUG
            if (!operand)
            {
                throw invalid_argument("operand");
            }
#endif
            Modulus modulus = tables.modulus();
            uint64_t two_times_modulus = modulus.value() << 1;

            // Return the NTT in scrambled order
            size_t n = size_t(1) << tables.coeff_count_power();
            size_t t = n >> 1;
            for (size_t m = 1; m < n; m <<= 1)
            {
                size_t j1 = 0;
                if (t >= 4)
                {
                    for (size_t i = 0; i < m; i++)
                    {
                        size_t j2 = j1 + t;
                        const MultiplyUIntModOperand W = tables.get_from_root_powers(m + i);

                        uint64_t *X = operand + j1;
                        uint64_t *Y = X + t;
                        uint64_t tx;
                        uint64_t Q;
                        for (size_t j = j1; j < j2; j += 4)
                        {
                            tx = *X - (two_times_modulus &
                                       static_cast<uint64_t>(-static_cast<int64_t>(*X >= two_times_modulus)));
                            Q = multiply_uint_mod_lazy(*Y, W, modulus);
                            *X++ = tx + Q;
                            *Y++ = tx + two_times_modulus - Q;

                            tx = *X - (two_times_modulus &
                                       static_cast<uint64_t>(-static_cast<int64_t>(*X >= two_times_modulus)));
                            Q = multiply_uint_mod_lazy(*Y, W, modulus);
                            *X++ = tx + Q;
                            *Y++ = tx + two_times_modulus - Q;

                            tx = *X - (two_times_modulus &
                                       static_cast<uint64_t>(-static_cast<int64_t>(*X >= two_times_modulus)));
                            Q = multiply_uint_mod_lazy(*Y, W, modulus);
                            *X++ = tx + Q;
                            *Y++ = tx + two_times_modulus - Q;

                            tx = *X - (two_times_modulus &
                                       static_cast<uint64_t>(-static_cast<int64_t>(*X >= two_times_modulus)));
                            Q = multiply_uint_mod_lazy(*Y, W, modulus);
                            *X++ = tx + Q;
                            *Y++ = tx + two_times_modulus - Q;
                        }
                        j1 += (t << 1);
                    }
                }
                else
                {
                    for (size_t i = 0; i < m; i++)
                    {
                        size_t j2 = j1 + t;
                        const MultiplyUIntModOperand W = tables.get_from_root_powers(m + i);

                        uint64_t *X = operand + j1;
                        uint64_t *Y = X + t;
                        uint64_t tx;
                        uint64_t Q;
                        for (size_t j = j1; j < j2; j++)
                        {
                            // The Harvey butterfly: assume X, Y in [0, 2p), and return X', Y' in [0, 4p).
                            // X', Y' = X + WY, X - WY (mod p).
                            tx = *X - (two_times_modulus &
                                       static_cast<uint64_t>(-static_cast<int64_t>(*X >= two_times_modulus)));
                            Q = multiply_uint_mod_lazy(*Y, W, modulus);
                            *X++ = tx + Q;
                            *Y++ = tx + two_times_modulus - Q;
                        }
                        j1 += (t << 1);
                    }
                }
                t >>= 1;
            }
        }

        // Inverse negacyclic NTT using Harvey's butterfly. (See Patrick Longa and Michael Naehrig).
        void inverse_ntt_negacyclic_harvey_lazy(CoeffIter operand, const NTTTables &tables)
        {
#ifdef SEAL_DEBUG
            if (!operand)
            {
                throw invalid_argument("operand");
            }
#endif
            Modulus modulus = tables.modulus();
            uint64_t two_times_modulus = modulus.value() << 1;

            // return the bit-reversed order of NTT.
            size_t n = size_t(1) << tables.coeff_count_power();
            size_t t = 1;
            size_t root_index = 1;
            for (size_t m = (n >> 1); m > 1; m >>= 1)
            {
                size_t j1 = 0;
                if (t >= 4)
                {
                    for (size_t i = 0; i < m; i++, root_index++)
                    {
                        size_t j2 = j1 + t;
                        const MultiplyUIntModOperand W = tables.get_from_inv_root_powers(root_index);

                        uint64_t *X = operand + j1;
                        uint64_t *Y = X + t;
                        uint64_t tx;
                        uint64_t ty;
                        for (size_t j = j1; j < j2; j += 4)
                        {
                            tx = *X + *Y;
                            ty = *X + two_times_modulus - *Y;
                            *X++ = tx - (two_times_modulus &
                                         static_cast<uint64_t>(-static_cast<int64_t>(tx >= two_times_modulus)));
                            *Y++ = multiply_uint_mod_lazy(ty, W, modulus);

                            tx = *X + *Y;
                            ty = *X + two_times_modulus - *Y;
                            *X++ = tx - (two_times_modulus &
                                         static_cast<uint64_t>(-static_cast<int64_t>(tx >= two_times_modulus)));
                            *Y++ = multiply_uint_mod_lazy(ty, W, modulus);

                            tx = *X + *Y;
                            ty = *X + two_times_modulus - *Y;
                            *X++ = tx - (two_times_modulus &
                                         static_cast<uint64_t>(-static_cast<int64_t>(tx >= two_times_modulus)));
                            *Y++ = multiply_uint_mod_lazy(ty, W, modulus);

                            tx = *X + *Y;
                            ty = *X + two_times_modulus - *Y;
                            *X++ = tx - (two_times_modulus &
                                         static_cast<uint64_t>(-static_cast<int64_t>(tx >= two_times_modulus)));
                            *Y++ = multiply_uint_mod_lazy(ty, W, modulus);
                        }
                        j1 += (t << 1);
                    }
                }
                else
                {
                    for (size_t i = 0; i < m; i++, root_index++)
                    {
                        size_t j2 = j1 + t;
                        const MultiplyUIntModOperand W = tables.get_from_inv_root_powers(root_index);

                        uint64_t *X = operand + j1;
                        uint64_t *Y = X + t;
                        uint64_t tx;
                        uint64_t ty;
                        for (size_t j = j1; j < j2; j++)
                        {
                            tx = *X + *Y;
                            ty = *X + two_times_modulus - *Y;
                            *X++ = tx - (two_times_modulus &
                                         static_cast<uint64_t>(-static_cast<int64_t>(tx >= two_times_modulus)));
                            *Y++ = multiply_uint_mod_lazy(ty, W, modulus);
                        }
                        j1 += (t << 1);
                    }
                }
                t <<= 1;
            }

            MultiplyUIntModOperand inv_N = tables.inv_degree_modulo();
            MultiplyUIntModOperand W = tables.get_from_inv_root_powers(root_index);
            MultiplyUIntModOperand inv_N_W;
            inv_N_W.set(multiply_uint_mod(inv_N.operand, W, modulus), modulus);

            uint64_t *X = operand;
            uint64_t *Y = X + (n >> 1);
            uint64_t tx;
            uint64_t ty;
            for (size_t j = (n >> 1); j < n; j++)
            {
                tx = *X + *Y;
                tx -= two_times_modulus & static_cast<uint64_t>(-static_cast<int64_t>(tx >= two_times_modulus));
                ty = *X + two_times_modulus - *Y;
                *X++ = multiply_uint_mod_lazy(tx, inv_N, modulus);
                *Y++ = multiply_uint_mod_lazy(ty, inv_N_W, modulus);
            }
        }
#else
        struct NTTBase
        {
            const Modulus &modulus;
            uint64_t Lp; // for now, Lp = 2*p
            using MUMO = MultiplyUIntModOperand;
            MUMO reducer;
            explicit NTTBase(const Modulus &modulus, uint64_t Lp, const MUMO &reducer)
                : modulus(modulus), Lp(Lp), reducer(reducer)
            {}

            ~NTTBase()
            {}

            // return 0 if cond = true, else return b if cond = false
            inline uint64_t select(uint64_t b, bool cond) const
            {
                return (b & -(uint64_t)cond) ^ b;
            }
        };

        struct NormalNTT : public NTTBase
        {
            using MUMO = NTTBase::MUMO;
            explicit NormalNTT(const Modulus &modulus, uint64_t Lp, const MUMO &reducer) : NTTBase(modulus, Lp, reducer)
            {}

            // x0' <- x0 + w * x1 mod p
            // x1' <- x0 - w * x1 mod p
            void Forward(uint64_t *x0, uint64_t *x1, const MUMO &w) const
            {
                uint64_t u, v;
                u = *x0;
                u -= select(Lp, u < Lp);
                v = multiply_uint_mod_lazy(*x1, w, modulus);
                *x0 = u + v;
                *x1 = u - v + Lp;
            }

            void ForwardLast(uint64_t *x0, uint64_t *x1, const MUMO &w) const
            {
                Forward(x0, x1, w);
            }

            // x0' <- x0 + x1 mod p
            // x1' <- x0 - w * x1 mod p
            inline void Backward(uint64_t *x0, uint64_t *x1, const MUMO &w) const
            {
                uint64_t u = *x0;
                uint64_t v = *x1;
                uint64_t t = u + v;
                t -= select(Lp, t < Lp);
                *x0 = t;
                *x1 = multiply_uint_mod_lazy(u - v + Lp, w, modulus);
            }

            inline void BackwardLast(uint64_t *x0, uint64_t *x1, const MUMO &inv_n, const MUMO &inv_n_w) const
            {
                uint64_t u = *x0;
                uint64_t v = *x1;
                uint64_t t = u + v;
                t -= select(Lp, t < Lp);
                *x0 = multiply_uint_mod_lazy(t, inv_n, modulus);
                *x1 = multiply_uint_mod_lazy(u - v + Lp, inv_n_w, modulus);
            }
        };

        struct SlothfulNTT : public NTTBase
        {
            using MUMO = NTTBase::MUMO;
            explicit SlothfulNTT(const Modulus &modulus, uint64_t Lp, const MUMO &reducer)
                : NTTBase(modulus, Lp, reducer)
            {}

            // x0' <- x0 + w * x1 mod p
            // x1' <- x0 - w * x1 mod p
            inline void Forward(uint64_t *x0, uint64_t *x1, const MUMO &w) const
            {
                uint64_t u, v;
                u = *x0;
                v = multiply_uint_mod_lazy(*x1, w, modulus);
                *x0 = u + v;
                *x1 = u - v + Lp;
            }

            inline void ForwardLast(uint64_t *x0, uint64_t *x1, const MUMO &w) const
            {
                uint64_t u, v;
                u = multiply_uint_mod_lazy(*x0, reducer, modulus);
                v = multiply_uint_mod_lazy(*x1, w, modulus);

                *x0 = u + v;
                *x1 = u - v + Lp;
            }
        };

        template <class NTTDoer>
        inline void do_ntt_negacyclic_lazy(CoeffIter operand, const NTTTables &tables, const NTTDoer &base)
        {
            const uint64_t p = tables.modulus().value();
            const size_t n = size_t(1) << tables.coeff_count_power();
            const MultiplyUIntModOperand *w = tables.root_powers() + 1;

            // main loop: for h >= 4
            size_t m = 1;
            size_t h = n >> 1;
            for (; h > 2; m <<= 1, h >>= 1)
            {
                // invariant: h * m = degree / 2
                // different buttefly groups
                uint64_t *x0 = operand;
                uint64_t *x1 = x0 + h; // invariant: x1 = x0 + h during the iteration
                for (size_t r = 0; r < m; ++r, ++w)
                {
                    for (size_t i = 0; i < h; i += 4)
                    { // unrolling
                        base.Forward(x0++, x1++, *w);
                        base.Forward(x0++, x1++, *w);
                        base.Forward(x0++, x1++, *w);
                        base.Forward(x0++, x1++, *w);
                    }
                    x0 += h;
                    x1 += h;
                }
            }

            // m = degree / 4, h = 2
            m = n >> 2;
            uint64_t *x0 = operand;
            uint64_t *x1 = x0 + 2;
            for (size_t r = 0; r < m; ++r, ++w)
            { // unrolling
                base.Forward(x0++, x1++, *w);
                base.Forward(x0, x1, *w); // combine the incr to following steps
                x0 += 3;
                x1 += 3;
            }

            // m = degree / 2, h = 1
            m = n >> 1;
            x0 = operand;
            x1 = x0 + 1;
            for (size_t r = 0; r < m; ++r, ++w)
            {
                base.ForwardLast(x0, x1, *w);
                x0 += 2;
                x1 += 2;
            }
            // At the end operand[0 .. n) stay in [0, 4p).
        }

        template <class NTTDoer>
        inline void do_inverse_ntt_negacyclic_lazy(CoeffIter operand, const NTTTables &tables, const NTTDoer &base)
        {
            const uint64_t p = tables.modulus().value();
            const size_t n = 1L << tables.coeff_count_power();
            const MultiplyUIntModOperand *w = tables.inv_root_powers() + 1;
            // first loop: m = degree / 2, h = 1
            // m > 1 to skip the last layer
            size_t m = n >> 1;
            auto x0 = operand;
            auto x1 = x0 + 1; // invariant: x1 = x0 + h during the iteration
            for (size_t r = 0; m > 1 && r < m; ++r, ++w)
            {
                base.Backward(x0, x1, *w);
                x0 += 2;
                x1 += 2;
            }

            // second loop: m = degree / 4, h = 2
            // m > 1 to skip the last layer
            m = n >> 2;
            x0 = operand;
            x1 = x0 + 2;
            for (size_t r = 0; m > 1 && r < m; ++r, ++w)
            {
                base.Backward(x0++, x1++, *w);
                base.Backward(x0, x1, *w);
                x0 += 3;
                x1 += 3;
            }
            // main loop: for h >= 4
            m = n >> 3;
            size_t h = 4;
            // m > 1 to skip the last layer
            for (; m > 1; m >>= 1, h <<= 1)
            {
                x0 = operand;
                x1 = x0 + h;
                for (size_t r = 0; r < m; ++r, ++w)
                {
                    for (size_t i = 0; i < h; i += 4)
                    { // unrolling
                        base.Backward(x0++, x1++, *w);
                        base.Backward(x0++, x1++, *w);
                        base.Backward(x0++, x1++, *w);
                        base.Backward(x0++, x1++, *w);
                    }
                    x0 += h;
                    x1 += h;
                }
            }

            x0 = operand;
            x1 = x0 + (n >> 1);
            const MultiplyUIntModOperand &inv_n = tables.inv_degree_modulo();

            for (size_t r = n >> 1; r < n; ++r)
            {
                base.BackwardLast(x0++, x1++, inv_n, *w);
            }
            // At the end operand[0 .. n) lies in [0, 2p)
        }

        void ntt_negacyclic_harvey_lazy(CoeffIter operand, const NTTTables &tables)
        {
#ifdef SEAL_DEBUG
            if (!operand)
            {
                throw invalid_argument("ntt_negacyclic_harvey_lazy: operand");
            }
#endif
            const uint64_t p = tables.modulus().value();
            const uint64_t logn = tables.modulus().bit_count();
            const int max_accum_bits = tables.modulus().bit_count() + 1 + (int)std::ceil(std::log2(1. * logn));
            if (max_accum_bits < 64)
            {
                SlothfulNTT doer(p, p << 1u, tables.reduce_precomp());
                do_ntt_negacyclic_lazy(operand, tables, doer);
            }
            else
            {
                NormalNTT doer(p, p << 1u, tables.reduce_precomp());
                do_ntt_negacyclic_lazy(operand, tables, doer);
            }
        }

        void inverse_ntt_negacyclic_harvey_lazy(CoeffIter operand, const NTTTables &tables)
        {
#ifdef SEAL_DEBUG
            if (!operand)
            {
                throw invalid_argument("inverse_ntt_negacyclic_harvey: operand");
            }
#endif
            const uint64_t p = tables.modulus().value();
            NormalNTT doer(p, p << 1u, tables.reduce_precomp());
            do_inverse_ntt_negacyclic_lazy(operand, tables, doer);
        }
#endif
    } // namespace util
} // namespace seal
