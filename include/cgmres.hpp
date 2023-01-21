#pragma once

#include <math.h>
#include <stdint.h>

#include <functional>
#include <memory>

#include "gmres.hpp"
#include "matrix.hpp"

using std::function;
using std::unique_ptr;
using std::placeholders::_1;
using std::placeholders::_2;
using std::placeholders::_3;
using std::placeholders::_4;

class Cgmres {
 public:
  Cgmres() {}

  ~Cgmres() {}

  template <class Model>
  void register_model(void) {
    dim_x = Model::dim_x;
    dim_u = Model::dim_u;
    dim_p = Model::dim_p;

    dxdt = Model::dxdt;
    dPhidx = Model::dPhidx;
    dHdx = Model::dHdx;
    dHdu = Model::dHdu;
    ddHduu = Model::ddHduu;

    dt = Model::dt;
    Tf = Model::Tf;
    dv = Model::dv;
    warm_start = Model::warm_start;

    alpha = Model::alpha;
    h = Model::h;
    zeta = Model::zeta;

    len = (dim_u * dv);

    gmres = std::make_unique<Gmres>(len, Model::k_max, Model::tol);

    t_ = 0.0;
    U_ = std::make_unique<double[]>(dim_u * dv);
    dUdt_ = std::make_unique<double[]>(dim_u * dv);
    ptau_ = std::make_unique<double[]>(dim_p * (dv + 1));

    isvalid = true;
  }

  inline uint16_t get_dim_x(void) const {
    return dim_x;
  }

  inline uint16_t get_dim_u(void) const {
    return dim_u;
  }

  inline uint16_t get_dim_p(void) const {
    return dim_p;
  }

  inline uint16_t get_dv(void) const {
    return dv;
  }

  // Get the controller time
  inline double get_time(void) const {
    return t_;
  }

  // Get the time increment in the prediction horizon at time t
  inline double get_dtau(const double t) const {
    return Tf * (1 - exp(-alpha * t)) / (double)dv;
  }

  // ptau = [p(t), p(t + dtau), ..., p(t + dv * dtau)]
  void set_ptau(const double *ptau_buf) const {
    mov(ptau_.get(), ptau_buf, dim_p * (dv + 1));
  }

  // U(i) = u0
  void init_U(const double *u0);

  void init_U_newton(const double *u0, const double *x0, const double *p0, const uint16_t n_loop);

  // Note: This function should be placed ``before'' the hardware output functions.
  void control(double *u) const;

  // Note: This function should be placed ``after'' the hardware output functions.
  void update(const double *x);

 private:
  void F_func(double *ret, const double *U, const double *x, const double t) const;
  void Ax_func(double *Ax, const double *dUdt, const double *U, const double t, const double *x_dxh, const double *F_dxh_h) const;

 private:
  bool isvalid;

  // --------------------
  // Parameters
  // --------------------
  // Dimensions
  uint16_t dim_x, dim_u, dim_p;

  // Internal functions
  function<void(double *ret, const double *x, const double *u, const double *p)> dxdt;
  function<void(double *ret, const double *x, const double *p)> dPhidx;
  function<void(double *ret, const double *x, const double *u, const double *p, const double *lmd)> dHdx;
  function<void(double *ret, const double *x, const double *u, const double *p, const double *lmd)> dHdu;
  function<void(double *ret, const double *x, const double *u, const double *p, const double *lmd)> ddHduu;

  // Sampling period (s)
  double dt;
  // Prediction horizon (s)
  double Tf;
  // Number of divisions of the prediction horizon
  uint16_t dv;
  //
  double alpha;
  //
  double h;
  //
  double zeta;
  // Set true to enable warm start
  bool warm_start;

  // Internal parameter (dim_x * dim_u)
  uint16_t len;

  // GMRES instance
  unique_ptr<Gmres> gmres;

  // --------------------
  // Variables
  // --------------------
  // Controller time (s)
  double t_;

  // Control input vector
  unique_ptr<double[]> U_;
  // Derivative of control input vector
  unique_ptr<double[]> dUdt_;
  // Time variant parameter vector
  unique_ptr<double[]> ptau_;

  // --------------------
  // Copy constructor
  // --------------------
  Cgmres(const Cgmres &);
  Cgmres &operator=(const Cgmres &);
};
