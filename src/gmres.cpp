#include "gmres.hpp"

#include <cblas.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#ifdef DEBUG
#include <stdio.h>
#endif

#include <memory>

#include "matrix.hpp"

void Gmres::gmres(double *x, const double *b, function<void(double *Ax, const double *x)> Ax_func) {
  uint16_t k, idx_h, idx_v, idx_w, idx_g;
  auto H = std::make_unique<double[]>((k_max + 1) * (k_max + 1));
  auto V = std::make_unique<double[]>((len) * (k_max + 1));
  auto rho = std::make_unique<double[]>(k_max + 1);
  auto g = std::make_unique<double[]>((g_len) * (k_max));
  auto U_buf = std::make_unique<double[]>(len);
  double buf;

  // r0 = b - Ax(x0)
  Ax_func(&V[0], x);
  cblas_daxpby(len, 1.0, b, cblas_inc, -1.0, &V[0], cblas_inc);

  // rho = ||r0||_2
  rho[0] = cblas_dnrm2(len, &V[0], cblas_inc);

  if (rho[0] < tol) {
    return;
  }

  // V(0) = r0 / ||r0||_2
  cblas_dscal(len, 1.0 / rho[0], &V[0], cblas_inc);

  for (k = 0; k < k_max; k++) {
    // Krylov subspace
    // V = [v0, ..., vk, Ax(vk)]
    Ax_func(&V[len * (k + 1)], &V[len * k]);

    // Modified Gram-Schmidt
    // w = V(k+1)
    // h = H(k)
    idx_w = len * (k + 1);
    idx_h = (k_max + 1) * k;

    for (uint16_t i = 0; i <= k; i++) {
      // v = V(i)
      idx_v = len * i;

      // h(i) = (w, v)
      // w = w - h(i) * v
      H[idx_h + i] = cblas_ddot(len, &V[idx_w], cblas_inc, &V[idx_v], cblas_inc);
      cblas_daxpy(len, -H[idx_h + i], &V[idx_v], cblas_inc, &V[idx_w], cblas_inc);
    }

    // h(k+1) = ||w||_2
    H[idx_h + (k + 1)] = cblas_dnrm2(len, &V[idx_w], cblas_inc);

    // Check breakdown
    // Stop if ||w||_2 == 0
    // TODO: should be positive?
    if (fabs(H[idx_h + (k + 1)]) < DBL_EPSILON) {
#ifdef DEBUG
      printf("Breakdown\n");
#endif
      return;
    } else {
      // V(k) = w / ||w||_2
      cblas_dscal(len, 1.0 / H[idx_h + (k + 1)], &V[idx_w], cblas_inc);
    }
    // TODO: ここまで完了
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
