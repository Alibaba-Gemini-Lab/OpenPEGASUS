#pragma once
#include <chrono>
#include <string>

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
