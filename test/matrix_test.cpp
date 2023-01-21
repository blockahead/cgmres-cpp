#include "matrix.hpp"

#include <gtest/gtest.h>

#define ROW (3)
#define COL (3)
#define LEN (ROW * COL)

TEST(matrix, clear) {
  double a[LEN] = {1, 1, 1, 1, 1, 1, 1, 1, 1};

  clear(a, LEN);

  for (int i = 0; i < LEN; i++) {
    EXPECT_FLOAT_EQ(0.0, a[i]);
  }
}

TEST(matrix, mov_vec) {
  double a[LEN] = {0};
  double b[LEN] = {1, 1, 1, 1, 1, 1, 1, 1, 1};

  mov(a, b, LEN);

  for (int i = 0; i < LEN; i++) {
    EXPECT_FLOAT_EQ(1.0, a[i]);
  }
}

TEST(matrix, mov_mat) {
  double a[ROW * COL] = {0};
  double b[ROW * COL] = {1, 1, 1, 1, 1, 1, 1, 1, 1};

  mov(a, b, ROW, COL);

  for (int i = 0; i < ROW * COL; i++) {
    EXPECT_FLOAT_EQ(1.0, a[i]);
  }
}

TEST(matrix, add_vec) {
  double a[LEN] = {0};
  double b[LEN] = {2, 2, 2, 2, 2, 2, 2, 2, 2};
  double c[LEN] = {1, 1, 1, 1, 1, 1, 1, 1, 1};

  add(a, b, c, LEN);

  for (int i = 0; i < LEN; i++) {
    EXPECT_FLOAT_EQ(3.0, a[i]);
  }
}

TEST(matrix, add_mat) {
  double a[ROW * COL] = {0};
  double b[ROW * COL] = {2, 2, 2, 2, 2, 2, 2, 2, 2};
  double c[ROW * COL] = {1, 1, 1, 1, 1, 1, 1, 1, 1};

  add(a, b, c, ROW, COL);

  for (int i = 0; i < ROW * COL; i++) {
    EXPECT_FLOAT_EQ(3.0, a[i]);
  }
}

TEST(matrix, sub_vec) {
  double a[LEN] = {0};
  double b[LEN] = {2, 2, 2, 2, 2, 2, 2, 2, 2};
  double c[LEN] = {1, 1, 1, 1, 1, 1, 1, 1, 1};

  sub(a, b, c, LEN);

  for (int i = 0; i < LEN; i++) {
    EXPECT_FLOAT_EQ(1.0, a[i]);
  }
}

TEST(matrix, sub_mat) {
  double a[ROW * COL] = {0};
  double b[ROW * COL] = {2, 2, 2, 2, 2, 2, 2, 2, 2};
  double c[ROW * COL] = {1, 1, 1, 1, 1, 1, 1, 1, 1};

  sub(a, b, c, ROW, COL);

  for (int i = 0; i < ROW * COL; i++) {
    EXPECT_FLOAT_EQ(1.0, a[i]);
  }
}

TEST(matrix, mul_vec_scalar) {
  double a[LEN] = {0};
  double b[LEN] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
  double c = 5;

  mul(a, b, c, LEN);

  for (int i = 0; i < LEN; i++) {
    EXPECT_FLOAT_EQ(5.0, a[i]);
  }
}

TEST(matrix, mul_mat_scalar) {
  double a[ROW * COL] = {0};
  double b[ROW * COL] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
  double c = 5;

  mul(a, b, c, ROW, COL);

  for (int i = 0; i < ROW * COL; i++) {
    EXPECT_FLOAT_EQ(5.0, a[i]);
  }
}

TEST(matrix, mul_mat_vec) {
  double a[ROW] = {0};
  // Column major
  double b[ROW * COL] = {1, 4, 7,
                         2, 5, 8,
                         3, 6, 9};
  double c[COL] = {1, 1, 1};

  mul(a, b, c, ROW, COL);

  EXPECT_FLOAT_EQ(6.0, a[0]);
  EXPECT_FLOAT_EQ(15.0, a[1]);
  EXPECT_FLOAT_EQ(24.0, a[2]);
}

TEST(matrix, mul_mat2x3_vec3) {
  double a[2] = {0};
  // Column major
  double b[2 * 3] = {1, 4,
                     2, 5,
                     3, 6};
  double c[3] = {1, 1, 1};

  mul(a, b, c, 2, 3);

  EXPECT_FLOAT_EQ(6.0, a[0]);
  EXPECT_FLOAT_EQ(15.0, a[1]);
}

TEST(matrix, mul_mat3x2_vec2) {
  double a[3] = {0};
  // Column major
  double b[3 * 2] = {1, 3, 5,
                     2, 4, 6};
  double c[2] = {1, 1};

  mul(a, b, c, 3, 2);

  EXPECT_FLOAT_EQ(3.0, a[0]);
  EXPECT_FLOAT_EQ(7.0, a[1]);
  EXPECT_FLOAT_EQ(11.0, a[2]);
}

TEST(matrix, div_vec_scalar) {
  double a[LEN] = {0};
  double b[LEN] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
  double c = 5;

  div(a, b, c, LEN);

  for (int i = 0; i < LEN; i++) {
    EXPECT_FLOAT_EQ(0.2, a[i]);
  }
}

TEST(matrix, div_mat_scalar) {
  double a[ROW * COL] = {0};
  double b[ROW * COL] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
  double c = 5;

  div(a, b, c, ROW, COL);

  for (int i = 0; i < ROW * COL; i++) {
    EXPECT_FLOAT_EQ(0.2, a[i]);
  }
}

TEST(matrix, norm) {
  double a[LEN] = {1, 2, 3, 4, 5, 6, 7, 8, 9};

  EXPECT_NEAR(16.88, norm(a, LEN), 1e-2);
}

TEST(matrix, dot) {
  double a[LEN] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  double b[LEN] = {9, 8, 7, 6, 5, 4, 3, 2, 1};

  EXPECT_FLOAT_EQ(165.0, dot(a, b, LEN));
}

TEST(matrix, sign) {
  EXPECT_FLOAT_EQ(-1.0, sign(-100));
  EXPECT_FLOAT_EQ(1.0, sign(100));
  EXPECT_FLOAT_EQ(1.0, sign(0));
}

TEST(matrix, linsolve) {
  // Column major
  double a[4 * 4] = {1, 1, 1, 1,
                     1, 1, 1, -1,
                     1, 1, -1, 1,
                     1, -1, 1, 1};
  double b[4] = {0, 4, -4, 2};

  linsolve(b, a, 4);

  EXPECT_FLOAT_EQ(1, b[0]);
  EXPECT_FLOAT_EQ(-1, b[1]);
  EXPECT_FLOAT_EQ(2, b[2]);
  EXPECT_FLOAT_EQ(-2, b[3]);
}
