#include "pegasus/pegasus_runtime.h"
#include "pegasus/timer.h"

int repacking_demo(int nslots) {
  using namespace gemini;
  PegasusRunTime::Parms pp;

  pp.lvl0_lattice_dim = lwe::params::n();
  pp.lvl1_lattice_dim = 4096;
  pp.lvl2_lattice_dim = 1 << 14;
  pp.nlevels = 4;  // CKKS levels
  pp.scale = std::pow(2., 40);
  pp.nslots = nslots;
  pp.enable_repacking = true;

  PegasusRunTime pg_rt(pp, /*num_threads*/ 4);

  F64Vec sqrt_slots(pp.nslots);
  F64Vec sigmoid_slots(pp.nslots);
  {
    std::random_device rdv;
    std::uniform_real_distribution<double> uniform(-8., 8.);
    std::generate_n(sqrt_slots.begin(), pp.nslots,
                    [&]() { return uniform(rdv); });
    std::generate_n(sigmoid_slots.begin(), pp.nslots,
                    [&]() { return uniform(rdv); });
  }

  Ctx ckks_ct[2];
  CHECK_AND_ABORT(pg_rt.EncodeThenEncrypt(sqrt_slots, ckks_ct[0]));
  CHECK_AND_ABORT(pg_rt.EncodeThenEncrypt(sigmoid_slots, ckks_ct[1]));

  // Step 1, S2C and KS

  F64 s2c_time{0.};
  {
    AutoTimer timer(&s2c_time);
    CHECK_AND_ABORT(pg_rt.SlotsToCoeffs(ckks_ct[0]));
    CHECK_AND_ABORT(pg_rt.SlotsToCoeffs(ckks_ct[1]));
  }

  std::vector<lwe::Ctx_st> lwe_sqrt;
  std::vector<lwe::Ctx_st> lwe_sigmoid;

  F64 extract_time{0.};
  {
    AutoTimer timer(&extract_time);
    CHECK_AND_ABORT(pg_rt.ExtraAllCoefficients(ckks_ct[0], lwe_sqrt));
    CHECK_AND_ABORT(pg_rt.ExtraAllCoefficients(ckks_ct[1], lwe_sigmoid));
  }

  F64 lut_time{0.};
  {
    AutoTimer timer(&lut_time);
    // compute sqrt(|x|) on LWE ciphertexts
    pg_rt.AbsSqrt(lwe_sqrt.data(), lwe_sqrt.size());
    // compute sigmoid(x) on LWE ciphertexts
    pg_rt.Sigmoid(lwe_sigmoid.data(), lwe_sigmoid.size());
  }

  // rearange before repacking
  std::vector<lwe::Ctx_st> lwe_inputs(lwe_sigmoid.size() + lwe_sqrt.size());
  for (int i = 0; i < lwe_inputs.size(); i += 2) {
    lwe_inputs[i] = lwe_sqrt[i / 2];
    lwe_inputs[i + 1] = lwe_sigmoid[i / 2];
  }

  // repacking to CKKS
  F64 repacking_time{0.};
  Ctx repacked;
  {
    AutoTimer timer(&repacking_time);
    CHECK_AND_ABORT(pg_rt.Repack(repacked, lwe_inputs));
  }

  printf("S2C (nslots = %d), took %f seconds\n", pp.nslots, s2c_time / 1000. / 2);
  printf("Extract and KS (nslots = %d), took %f seconds\n", pp.nslots, extract_time / 1000. / 2);
  printf("LUT (nslots = %d), took %f seconds\n", pp.nslots, lut_time / 1000. / 2);
  printf("Repack (nslots = %d), took %f seconds\n", 2 * pp.nslots, repacking_time / 1000.);

  F64Vec computed;
  pg_rt.DecryptThenDecode(repacked, pp.nslots * 2, computed);

  double err0 = 0.;
  double err1 = 0.;
  for (size_t i = 0; i < pp.nslots; ++i) {
    double s0 = sqrt_slots[i];
    double s1 = sigmoid_slots[i];
    double gnd0 = std::sqrt(std::abs(s0));
    double gnd1 = 1. / (1. + std::exp(-s1));
    double c0 = computed[2 * i];
    double c1 = computed[2 * i + 1];
    if (i < 4) {
      printf("sqrt(|%f|) = %f, LUT gives %f\n", s0, gnd0, c0);
      printf("sigmoid(%f) = %f, LUT gives %f\n", s1, gnd1, c1);
    }
    err0 += std::abs(gnd0 - c0);
    err1 += std::abs(gnd1 - c1);
  }

  printf("sqrt: average error = %f ~ 2^%.1f\n", err0 / pp.nslots,
         std::log2(err0 / pp.nslots));
  printf("sigmoid: average error = %f ~ 2^%.1f\n", err1 / pp.nslots,
         std::log2(err1 / pp.nslots));
  return 0;
}

int main() {
  repacking_demo(/*nslots*/ 256);
  return 0;
}
