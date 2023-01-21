#include "gmres.hpp"

#include <float.h>
#include <math.h>
#include <stdint.h>

#include <memory>

#include "matrix.hpp"

void Gmres::gmres(double *x, const double *b, function<void(double *Ax, const double *x)> Ax_func) {
  uint16_t k, idx_v1, idx_v2, idx_h, idx_g;
  auto V = std::make_unique<double[]>((len) * (k_max + 1));
  auto H = std::make_unique<double[]>((k_max + 1) * (k_max + 1));
  auto rho = std::make_unique<double[]>(k_max + 1);
  auto g = std::make_unique<double[]>((g_len) * (k_max));
  auto U_buf = std::make_unique<double[]>(len);
  double buf;

  // r0 = b - Ax(x0)
  Ax_func(&V[0], x);
  sub(&V[0], b, &V[0], len);

  // rho = sqrt(r0' * r0)
  rho[0] = norm(&V[0], len);

  if (rho[0] < tol) {
    return;
  }

  // V(0) = r0 / rho
  div(&V[0], &V[0], rho[0], len);

  for (k = 0; k < k_max; k++) {
    // V(k + 1) = Ax(v(k))
    Ax_func(&V[len * (k + 1)], &V[len * k]);

    idx_v1 = len * (k + 1);
    // Modified Gram-Schmidt
    for (uint16_t i = 0; i < k + 1; i++) {
      idx_v2 = len * i;
      idx_h = (k_max + 1) * k + i;
      H[idx_h] = dot(&V[idx_v2], &V[idx_v1], len);
      mul(U_buf.get(), &V[idx_v2], H[idx_h], len);
      sub(&V[idx_v1], &V[idx_v1], U_buf.get(), len);
    }
    idx_h = (k_max + 1) * k + (k + 1);
    H[idx_h] = norm(&V[idx_v1], len);

    // Check breakdown
    if (fabs(H[idx_h]) < DBL_EPSILON) {
      printf("Breakdown\n");
      return;
    } else {
      div(&V[idx_v1], &V[idx_v1], H[idx_h], len);
    }

    // Transformation H to upper triangular matrix by Householder transformation
    for (uint16_t i = 0; i < k; i++) {
      idx_h = (k_max + 1) * k + i;
      idx_g = g_len * i;
      buf = (g[idx_g + 0] * H[idx_h + 0] + g[idx_g + 1] * H[idx_h + 1]) * g[idx_g + 2];
      H[idx_h + 0] = H[idx_h + 0] - buf * g[idx_g + 0];
      H[idx_h + 1] = H[idx_h + 1] - buf * g[idx_g + 1];
    }
    idx_h = (k_max + 1) * k + k;
    idx_g = g_len * k;
    buf = -sign(H[idx_h]) * norm(&H[idx_h], 2);  // Vector length
    g[idx_g + 0] = H[idx_h + 0] - buf;
    g[idx_g + 1] = H[idx_h + 1];
    g[idx_g + 2] = 2.0 / dot(&g[idx_g], &g[idx_g], 2);
    H[idx_h + 0] = buf;
    H[idx_h + 1] = 0.0;

    // Update residual
    buf = g[idx_g + 0] * rho[k + 0] * g[idx_g + 2];
    rho[k + 0] = rho[k + 0] - buf * g[idx_g + 0];
    rho[k + 1] = -buf * g[idx_g + 1];

    // Check convergence
    if (fabs(rho[k + 1]) < tol) {
      break;
    }
  }

  // Solve H * y = rho
  // H is upper triangle matrix
  for (int16_t i = k - 1; i >= 0; i--) {
    for (int16_t j = k - 1; j > i; j--) {
      idx_h = (k_max + 1) * j + i;
      rho[i] -= H[idx_h] * rho[j];
    }
    idx_h = (k_max + 1) * i + i;
    rho[i] /= H[idx_h];
  }

  // x = x + V * y
  mul(&V[len * k_max], V.get(), rho.get(), len, k);
  add(x, x, &V[len * k_max], len);
}