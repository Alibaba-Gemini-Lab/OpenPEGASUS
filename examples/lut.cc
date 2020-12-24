#include "pegasus/pegasus_runtime.h"
#include "pegasus/timer.h"

int main() {
  using namespace gemini;
  PegasusRunTime::Parms pp;

  pp.lvl0_lattice_dim = lwe::params::n();
  pp.lvl1_lattice_dim = 1 << 12;
  pp.lvl2_lattice_dim = 1 << 16;
  pp.nlevels = 4; // CKKS levels
  pp.scale = std::pow(2., 40);
  pp.nslots = 1 << 8;
  pp.s2c_multiplier = 1.;

  PegasusRunTime pg_rt(pp, /*num_threads*/ 4);

  for (int f = 0; f < 7; ++f) {
    F64Vec slots(pp.nslots);

    {
      std::random_device rdv;
      std::uniform_real_distribution<double> uniform(-8., 8.);
      std::generate_n(slots.begin(), pp.nslots, [&]() { return uniform(rdv); });
    }

    Ctx ckks_ct;
    CHECK_AND_ABORT(pg_rt.EncodeThenEncrypt(slots, ckks_ct));
    CHECK_AND_ABORT(pg_rt.SlotsToCoeffs(ckks_ct));

    std::vector<lwe::Ctx_st> lwe_ct;
    double s2c_ks_time{0.};
    {
      AutoTimer timer(&s2c_ks_time);
      CHECK_AND_ABORT(pg_rt.ExtraAllCoefficients(ckks_ct, lwe_ct));
    }

    double accum_s2c_err{0.};
    for (int i = 0; i < pp.nslots; ++i) {
      accum_s2c_err += std::abs(slots[i] - pg_rt.DecryptLWE(lwe_ct[i]));
    }
    printf("\nS2C.error 2^%f\t%f sec\n", std::log2(accum_s2c_err / pp.nslots), s2c_ks_time / 1000.);

    std::string tag;
    std::function<double(double)> target_func;
    double lut_time{0.};

    AutoTimer timer(&lut_time);
    switch (f) {
      case 0: {
        pg_rt.LeakyReLU(lwe_ct.data(), lwe_ct.size());
        tag = "LeakyReLU";
        target_func = [](double e) { return std::signbit(e) ? 0.01 * e : e; };
      } break;
      case 1: {
        pg_rt.AbsLog(lwe_ct.data(), lwe_ct.size());
        tag = "AbsLog";
        target_func = [](double e) -> double { return std::log(std::abs(e)); };
      } break;
      case 2: {
        pg_rt.ReLU(lwe_ct.data(), lwe_ct.size());
        tag = "ReLU";
        target_func = [](double e) -> double { return std::max(0., e); };
      } break;
      case 3: {
        pg_rt.AbsSqrt(lwe_ct.data(), lwe_ct.size());
        tag = "AbsSqrt";
        target_func = [](double e) -> double { return std::sqrt(std::abs(e)); };
      } break;
      case 4: {
        pg_rt.Tanh(lwe_ct.data(), lwe_ct.size());
        tag = "Tanh";
        target_func = [](double e) -> double { return std::tanh(e); };
      } break;
      case 5: {
        pg_rt.Sign(lwe_ct.data(), lwe_ct.size());
        tag = "Sign";
        target_func = [](double e) -> double { return std::signbit(e); };
      } break;
      case 6: {
        pg_rt.Sigmoid(lwe_ct.data(), lwe_ct.size());
        tag = "Sigmoid";
        target_func = [](double e) { return 1. / (1. + std::exp(-e)); };
      } break;
    }
    timer.stop();

    double accum_lut_err{0.};
    for (int i = 0; i < pp.nslots; ++i) {
      double gnd = target_func(slots[i]);
      double cmp = pg_rt.DecryptLWE(lwe_ct[i]);
      accum_lut_err += std::abs(gnd - cmp);
    }

    printf("LUT (%s).error 2^%f\t%f sec\n", tag.c_str(), std::log2(accum_lut_err / pp.nslots), lut_time / 1000.);
  }
  return 0;
}
