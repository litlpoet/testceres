#include <gtest/gtest.h>

#include <ceres/ceres.h>
#include <glog/logging.h>

#include <iostream>

TEST(TestHello, InitialTest)
{
  std::cout << "hello test ceres" << std::endl;
}

struct CostFunctor
{
  template<typename T>
  bool
  operator()(T const* const x, T* residual) const
  {
    residual[0] = 10.0 - x[0];
    return true;
  }
};

TEST(TestHello, HelloCeres)
{
  google::InitGoogleLogging("hello ceres");

  auto       x         = 0.5;
  auto const initial_x = x;

  ceres::CostFunction* cost_function =
      new ceres::AutoDiffCostFunction<CostFunctor, 1, 1>(new CostFunctor);

  ceres::Problem problem;
  problem.AddResidualBlock(cost_function, nullptr, &x);

  ceres::Solver::Summary summary;
  ceres::Solver::Options options;
  options.minimizer_progress_to_stdout = true;
  ceres::Solve(options, &problem, &summary);

  std::cout << summary.BriefReport() << std::endl;
  std::cout << "x: " << initial_x << " -> " << x << std::endl;
  EXPECT_NE(initial_x, x);
}
