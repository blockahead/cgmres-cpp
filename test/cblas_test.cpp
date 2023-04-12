#include <cblas.h>
#include <gtest/gtest.h>

#define ROW (3)
#define COL (3)
#define LEN (ROW * COL)
#define CblasInc (1)

TEST(cblas, mov_vec) {
  double x[LEN] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
  double y[LEN] = {0};

  cblas_dcopy(LEN, x, CblasInc, y, CblasInc);

  for (int i = 0; i < LEN; i++) {
    EXPECT_FLOAT_EQ(1.0, y[i]);
  }
}

TEST(cblas, mov_mat) {
  double A[ROW * COL] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
  double B[ROW * COL] = {0};

  cblas_dcopy(ROW * COL, A, CblasInc, B, CblasInc);

  for (int i = 0; i < ROW * COL; i++) {
    EXPECT_FLOAT_EQ(1.0, B[i]);
  }
}

TEST(cblas, add_vec) {
  double x[LEN] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
  double y[LEN] = {2, 2, 2, 2, 2, 2, 2, 2, 2};

  cblas_daxpy(LEN, 1.0, x, CblasInc, y, CblasInc);

  for (int i = 0; i < LEN; i++) {
    EXPECT_FLOAT_EQ(3.0, y[i]);
  }
}

TEST(cblas, add_mat) {
  double A[ROW * COL] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
  double B[ROW * COL] = {2, 2, 2, 2, 2, 2, 2, 2, 2};

  cblas_daxpy(ROW * COL, 1.0, A, CblasInc, B, CblasInc);

  for (int i = 0; i < ROW * COL; i++) {
    EXPECT_FLOAT_EQ(3.0, B[i]);
  }
}

TEST(cblas, sub_vec) {
  double x[LEN] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
  double y[LEN] = {2, 2, 2, 2, 2, 2, 2, 2, 2};

  cblas_daxpby(LEN, 1.0, x, CblasInc, -1.0, y, CblasInc);

  for (int i = 0; i < LEN; i++) {
    EXPECT_FLOAT_EQ(-1.0, y[i]);
  }
}

TEST(cblas, sub_mat) {
  double A[ROW * COL] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
  double B[ROW * COL] = {2, 2, 2, 2, 2, 2, 2, 2, 2};

  cblas_daxpby(ROW * COL, 1.0, A, CblasInc, -1.0, B, CblasInc);

  for (int i = 0; i < ROW * COL; i++) {
    EXPECT_FLOAT_EQ(-1.0, B[i]);
  }
}

TEST(cblas, mul_vec_scalar) {
  double x[LEN] = {1, 1, 1, 1, 1, 1, 1, 1, 1};

  cblas_dscal(LEN, 5.0, x, CblasInc);

  for (int i = 0; i < LEN; i++) {
    EXPECT_FLOAT_EQ(5.0, x[i]);
  }
}

TEST(cblas, mul_mat_scalar) {
  double x[ROW * COL] = {1, 1, 1, 1, 1, 1, 1, 1, 1};

  cblas_dscal(ROW * COL, 5.0, x, CblasInc);

  for (int i = 0; i < ROW * COL; i++) {
    EXPECT_FLOAT_EQ(5.0, x[i]);
  }
}

TEST(cblas, mul_mat_vec) {
  // Column major
  // A =
  // 1, 2, 3;
  // 4, 5, 6;
  // 7, 8, 9;
  double A[ROW * COL] = {1, 4, 7,
                         2, 5, 8,
                         3, 6, 9};
  double x[ROW] = {1, 1, 1};
  double y[ROW] = {0};

  // LDA is set to row of matrix
  cblas_dgemv(CblasColMajor, CblasNoTrans, ROW, COL, 1.0, A, ROW, x, CblasInc, 0.0, y, CblasInc);

  EXPECT_FLOAT_EQ(6.0, y[0]);
  EXPECT_FLOAT_EQ(15.0, y[1]);
  EXPECT_FLOAT_EQ(24.0, y[2]);
}

TEST(cblas, mul_mat2x3_vec3) {
  // Column major
  // A =
  // 1, 2, 3;
  // 4, 5, 6;
  double A[2 * 3] = {1, 4,
                     2, 5,
                     3, 6};
  double x[3] = {1, 1, 1};
  double y[2] = {0};

  // LDA is set to row of matrix
  cblas_dgemv(CblasColMajor, CblasNoTrans, 2, 3, 1.0, A, 2, x, CblasInc, 0.0, y, CblasInc);

  EXPECT_FLOAT_EQ(6.0, y[0]);
  EXPECT_FLOAT_EQ(15.0, y[1]);
}

TEST(cblas, mul_mat3x2_vec2) {
  // Column major
  // A =
  // 1, 2;
  // 3, 4;
  // 5, 6;
  double A[3 * 2] = {1, 3, 5,
                     2, 4, 6};
  double x[2] = {1, 1};
  double y[3] = {0};

  // LDA is set to row of matrix
  cblas_dgemv(CblasColMajor, CblasNoTrans, 3, 2, 1.0, A, 3, x, CblasInc, 0.0, y, CblasInc);

  EXPECT_FLOAT_EQ(3.0, y[0]);
  EXPECT_FLOAT_EQ(7.0, y[1]);
  EXPECT_FLOAT_EQ(11.0, y[2]);
}

TEST(cblas, div_vec_scalar) {
  double x[LEN] = {1, 1, 1, 1, 1, 1, 1, 1, 1};

  cblas_dscal(LEN, 1 / 5.0, x, CblasInc);

  for (int i = 0; i < LEN; i++) {
    EXPECT_FLOAT_EQ(0.2, x[i]);
  }
}

TEST(cblas, div_mat_scalar) {
  double A[ROW * COL] = {1, 1, 1, 1, 1, 1, 1, 1, 1};

  cblas_dscal(ROW * COL, 1 / 5.0, A, CblasInc);

  for (int i = 0; i < ROW * COL; i++) {
    EXPECT_FLOAT_EQ(0.2, A[i]);
  }
}

TEST(cblas, norm_vec) {
  double x[3] = {4, 28, 35};

  EXPECT_FLOAT_EQ(45.0, cblas_dnrm2(3, x, CblasInc));
}

TEST(cblas, dot_vec) {
  double x[LEN] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  double y[LEN] = {9, 8, 7, 6, 5, 4, 3, 2, 1};

  EXPECT_FLOAT_EQ(165.0, cblas_ddot(LEN, x, CblasInc, y, CblasInc));
}
