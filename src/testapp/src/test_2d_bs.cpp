#include <gtest/gtest.h>

#include <ceres/ceres.h>
#include <glog/logging.h>

#include <iostream>

double const b0[] = {
    0.0,
    0.0,  // v1
    0.0,
    1.0,  // v2
    0.0,
    2.0  // v3
};

double const b1_ws[] = {
    0.0,
    0.0,  // v1
    -0.5,
    0.8,  // v2
    -1.7,
    1.0  // v3
};

double const b2_ws[] = {
    0.0,
    0.0,  // v1
    0.5,
    0.6,  // v2
    1.8,
    1.2  // v3
};

double const target1[] = {
    0.0,
    0.0,  // v1
    0.5,
    0.6,  // v2
    1.8,
    1.2  // v3
};

double const target2[] = {
    0.0,
    0.0,  // v1
    1.0,
    1.0,  // v2
    2.0,
    2.0  // v3
};

struct BSResidual
{
  BSResidual(int const n, double const* const x)
      : _n(n)
  {
    for (int i = 0; i < n; ++i)
      _x.push_back(x[i]);
  }

  template<typename T>
  bool
  operator()(T const* const w, T* residual) const
  {
    for (int i = 0; i < _n; ++i)
    {
      residual[i] = T(_x[i]) - T(b0[i]) - w[0] * T(b1_ws[i] - b0[i]) - w[1] * T(b2_ws[i] - b0[i]);
    }
    return true;
  }

 private:
  int const           _n;
  std::vector<double> _x;
};

TEST(TestBSWeightFit, WeightFit)
{
  double w[] = {0.0, 0.0};

  ceres::Problem problem;
  auto cost_f = new ceres::AutoDiffCostFunction<BSResidual, 6, 2>(new BSResidual(6, target2));
  problem.AddResidualBlock(cost_f, nullptr, w);

  ceres::Solver::Options options;
  options.max_num_iterations           = 25;
  options.linear_solver_type           = ceres::DENSE_QR;
  options.minimizer_progress_to_stdout = true;

  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  std::cout << summary.FullReport() << std::endl;
  std::cout << "w0: " << w[0] << std::endl;
  std::cout << "w1: " << w[1] << std::endl;
  for (int i = 0; i < 6; ++i)
  {
    auto result = b0[i] + w[0] * (b1_ws[i] - b0[i]) + w[1] * (b2_ws[i] - b0[i]);
    std::cout << result << std::endl;
  }
}
