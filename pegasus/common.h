#pragma once
#include <cmath>
#include <string>

#include "pegasus/status.h"
#include "pegasus/types.h"

namespace gemini {
constexpr F64 MM_PI = 3.1415926535897932384626433;

// floor(sqrt(n))
template <typename T>
static inline T FloorSqrt(T n) {
  return static_cast<T>(std::floor(std::sqrt(1. * n)));
}

// ceil(sqrt(n))
template <typename T>
static inline T CeilSqrt(T n) {
  return static_cast<T>(std::ceil(std::sqrt(1. * n)));
}

// ceil(a / b)
template <typename T>
static inline T CeilDiv(T a, T b) {
  return (a + b - 1) / b;
}

template <typename T>
static inline bool IsTwoPower(T v) {
  return v && !(v & (v - 1));
}

inline constexpr U64 Log2(U64 x) { return x == 1 ? 0 : 1 + Log2(x >> 1); }
}  // namespace gemini

