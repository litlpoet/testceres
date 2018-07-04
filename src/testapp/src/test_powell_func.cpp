#include <gtest/gtest.h>

#include <ceres/ceres.h>
#include <glog/logging.h>

#include <iostream>

struct F1
{
  template<typename T>
  bool
  operator()(T const* const x1, T const* const x2, T* residual) const
  {
    residual[0] = x1[0] + T(10.0) * x2[0];
    return true;
  }
};

struct F2
{
  template<typename T>
  bool
  operator()(T const* const x3, T const* const x4, T* residual) const
  {
    residual[0] = T(sqrt(5.0)) * (x3[0] - x4[0]);
    return true;
  }
};

struct F3
{
  template<typename T>
  bool
  operator()(T const* const x2, T const* const x3, T* residual) const
  {
    residual[0] = (x2[0] - T(2.0) * x3[0]) * (x2[0] - T(2.0) * x3[0]);
    return true;
  }
};

struct F4
{
  template<typename T>
  bool
  operator()(T const* const x1, T const* const x4, T* residual) const
  {
    residual[0] = T(sqrt(10.0)) * (x1[0] - x4[0]) * (x1[0] - x4[0]);
    return true;
  }
};

TEST(TestPowell, PowellFunc)
{
  double x[] = {3.0, -1.0, 0.0, 1.0};

  ceres::Problem problem;

  problem.AddResidualBlock(
      new ceres::AutoDiffCostFunction<F1, 1, 1, 1>(new F1), nullptr, &x[0], &x[1]);
  problem.AddResidualBlock(
      new ceres::AutoDiffCostFunction<F2, 1, 1, 1>(new F2), nullptr, &x[2], &x[3]);
  problem.AddResidualBlock(
      new ceres::AutoDiffCostFunction<F3, 1, 1, 1>(new F3), nullptr, &x[1], &x[2]);
  problem.AddResidualBlock(
      new ceres::AutoDiffCostFunction<F4, 1, 1, 1>(new F4), nullptr, &x[0], &x[3]);

  ceres::Solver::Summary summary;
  ceres::Solver::Options options;
  options.minimizer_progress_to_stdout = true;
  ceres::Solve(options, &problem, &summary);
  std::cout << summary.FullReport() << std::endl;
}
