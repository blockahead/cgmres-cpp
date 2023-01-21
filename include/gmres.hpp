#pragma once

#include <stdint.h>

#include <functional>

using std::function;

class Gmres {
 public:
  Gmres(const uint16_t len, const uint16_t k_max, const double tol)
      : len(len), k_max(k_max), tol(tol) {}

  ~Gmres() {}

  void gmres(double *x, const double *b, function<void(double *Ax, const double *x)> Ax_func);

 private:
  // --------------------
  // Parameters
  // --------------------
  static constexpr uint16_t g_len = 3;
  const uint16_t len;
  const uint16_t k_max;
  const double tol;

  // --------------------
  // Copy constructor
  // --------------------
  Gmres(const Gmres &);
  Gmres &operator=(const Gmres &);
};
