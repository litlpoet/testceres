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

class BSShapeCost : public ceres::CostFunction
{
  size_t              _n_bs;
  size_t              _n_v;
  std::vector<double> _t;
  std::vector<double> _w;

 public:
  BSShapeCost(size_t const& n_bs, size_t const& n_v, double const* const t, double const* const w)
      : _n_bs(n_bs)
      , _n_v(n_v)
  {
    set_num_residuals(n_v);
    for (size_t i = 0; i < n_bs; ++i)
      mutable_parameter_block_sizes()->push_back(n_v);
    for (size_t i = 0; i < n_bs; ++i)
      _w.push_back(w[i]);
    for (size_t i = 0; i < n_v; ++i)
      _t.push_back(t[i]);
  }

  virtual ~BSShapeCost();

  BSShapeCost(BSShapeCost const&) = delete;

  BSShapeCost&
  operator=(BSShapeCost const&) = delete;

  bool
  Evaluate(double const* const* x, double* residuals, double** jacobians) const final
  {
    for (size_t i = 0; i < _n_v; ++i)
    {
      residuals[i] = _t[i];
      for (size_t j = 0; j < _n_bs; ++j)
      {
        residuals[i] -= (b0[i] + _w[j] * (b1_ws[i] - b0[i] + x[j][i]));
      }
    }

    if (!jacobians)
      return true;

    for (size_t i = 0; i < _n_bs; ++i)
    {
      if (jacobians[i])
      {
        for (size_t j = 0; j < _n_v; ++j)
        {
          for (size_t k = 0; k < _n_v; ++k)
          {
            jacobians[i][j * parameter_block_sizes()[i] + k] = -_w[i];
          }
        }
      }
    }

    return true;
  }
};

struct BSResidual
{
  int const           _n;
  std::vector<double> _x;

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
};

struct BSResidualDyn
{
  int const           _n;
  std::vector<double> _x;

  BSResidualDyn(int const n, double const* const x)
      : _n(n)
  {
    for (int i = 0; i < n; ++i)
      _x.push_back(x[i]);
  }

  template<typename T>
  bool
  operator()(T const* const* w, T* residual) const
  {
    for (int i = 0; i < _n; ++i)
    {
      residual[i] =
          T(_x[i]) - T(b0[i]) - w[0][0] * T(b1_ws[i] - b0[i]) - w[0][1] * T(b2_ws[i] - b0[i]);
    }
    return true;
  }
};

struct BSResidualSingle
{
  int const           _v_id;
  std::vector<double> _x;

  BSResidualSingle(int v_id, double const* const x)
      : _v_id(v_id)
  {
    for (int i = 0; i < 2; ++i)
      _x.push_back(x[i]);
  }

  template<typename T>
  bool
  operator()(T const* const w, T* residual) const
  {
    for (int i = 0; i < 2; ++i)
    {
      auto v_id   = 2 * _v_id + i;
      residual[i] = T(_x[i]) - T(b0[v_id]) - w[0] * T(b1_ws[v_id] - b0[v_id]) -
                    w[1] * T(b2_ws[v_id] - b0[v_id]);
    }
    return true;
  }
};

TEST(TestBSWeightFit, WeightFit)
{
  double w_f1[] = {0.0, 0.0};
  double w_f2[] = {0.0, 0.0};

  ceres::Problem problem;
  auto cost_f1 = new ceres::AutoDiffCostFunction<BSResidual, 6, 2>(new BSResidual(6, target2));
  auto cost_f2 = new ceres::AutoDiffCostFunction<BSResidual, 6, 2>(new BSResidual(6, target1));
  problem.AddResidualBlock(cost_f1, nullptr, w_f1);
  problem.SetParameterLowerBound(w_f1, 0, 0.0);
  problem.SetParameterUpperBound(w_f1, 0, 1.0);
  problem.SetParameterLowerBound(w_f1, 1, 0.0);
  problem.SetParameterUpperBound(w_f1, 1, 1.0);
  problem.AddResidualBlock(cost_f2, nullptr, w_f2);
  problem.SetParameterLowerBound(w_f2, 0, 0.0);
  problem.SetParameterUpperBound(w_f2, 0, 1.0);
  problem.SetParameterLowerBound(w_f2, 1, 0.0);
  problem.SetParameterUpperBound(w_f2, 1, 1.0);

  ceres::Solver::Options options;
  options.max_num_iterations           = 100;
  options.linear_solver_type           = ceres::DENSE_QR;
  options.minimizer_progress_to_stdout = true;

  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  std::cout << "PER MESH COST!!!!!!!!" << std::endl;
  std::cout << summary.FullReport() << std::endl;
  std::cout << "f1_w0: " << w_f1[0] << std::endl;
  std::cout << "f1_w1: " << w_f1[1] << std::endl;
  std::cout << "f2_w0: " << w_f2[0] << std::endl;
  std::cout << "f2_w1: " << w_f2[1] << std::endl;
  std::cout << "result1" << std::endl;
  for (int i = 0; i < 6; ++i)
  {
    auto result = b0[i] + w_f1[0] * (b1_ws[i] - b0[i]) + w_f1[1] * (b2_ws[i] - b0[i]);
    std::cout << result << std::endl;
  }
  std::cout << "result2" << std::endl;
  for (int i = 0; i < 6; ++i)
  {
    auto result = b0[i] + w_f2[0] * (b1_ws[i] - b0[i]) + w_f2[1] * (b2_ws[i] - b0[i]);
    std::cout << result << std::endl;
  }
}

TEST(TestBSWeightFit, WeightFit2)
{
  double w_f1[] = {0.0, 0.0};
  double w_f2[] = {0.0, 0.0};

  ceres::Problem      problem;
  double const* const x = target2;
  for (auto i = 0, n = 3; i < n; ++i)
  {
    problem.AddResidualBlock(
        new ceres::AutoDiffCostFunction<BSResidualSingle, 2, 2>(new BSResidualSingle(i, x + 2 * i)),
        nullptr,
        w_f1);
  }
  problem.SetParameterLowerBound(w_f1, 0, 0.0);
  problem.SetParameterUpperBound(w_f1, 0, 1.0);
  problem.SetParameterLowerBound(w_f1, 1, 0.0);
  problem.SetParameterUpperBound(w_f1, 1, 1.0);

  double const* const x2 = target1;
  for (auto i = 0, n = 3; i < n; ++i)
  {
    problem.AddResidualBlock(new ceres::AutoDiffCostFunction<BSResidualSingle, 2, 2>(
                                 new BSResidualSingle(i, x2 + 2 * i)),
                             nullptr,
                             w_f2);
  }
  problem.SetParameterLowerBound(w_f2, 0, 0.0);
  problem.SetParameterUpperBound(w_f2, 0, 1.0);
  problem.SetParameterLowerBound(w_f2, 1, 0.0);
  problem.SetParameterUpperBound(w_f2, 1, 1.0);

  ceres::Solver::Options options;
  options.max_num_iterations           = 100;
  options.linear_solver_type           = ceres::DENSE_QR;
  options.minimizer_progress_to_stdout = true;

  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  std::cout << "PER VERTEX COST!!!!!!!!" << std::endl;
  std::cout << summary.FullReport() << std::endl;
  std::cout << "f1_w0: " << w_f1[0] << std::endl;
  std::cout << "f1_w1: " << w_f1[1] << std::endl;
  std::cout << "f2_w0: " << w_f2[0] << std::endl;
  std::cout << "f2_w1: " << w_f2[1] << std::endl;
  std::cout << "result1" << std::endl;
  for (int i = 0; i < 6; ++i)
  {
    auto result = b0[i] + w_f1[0] * (b1_ws[i] - b0[i]) + w_f1[1] * (b2_ws[i] - b0[i]);
    std::cout << result << std::endl;
  }
  std::cout << "result2" << std::endl;
  for (int i = 0; i < 6; ++i)
  {
    auto result = b0[i] + w_f2[0] * (b1_ws[i] - b0[i]) + w_f2[1] * (b2_ws[i] - b0[i]);
    std::cout << result << std::endl;
  }
}

TEST(TestBSWeightFit, WeightFit3)
{
  double w_f1[] = {0.0, 0.0};
  double w_f2[] = {0.0, 0.0};

  ceres::Problem problem;
  auto cost_f1 = new ceres::AutoDiffCostFunction<BSResidual, 6, 2>(new BSResidual(6, target2));
  auto cost_f2 = new ceres::AutoDiffCostFunction<BSResidual, 6, 2>(new BSResidual(6, target1));

  auto cost_dyn_f1 =
      new ceres::DynamicAutoDiffCostFunction<BSResidualDyn, 4>(new BSResidualDyn(6, target2));
  cost_dyn_f1->AddParameterBlock(2);
  cost_dyn_f1->SetNumResiduals(6);
  problem.AddResidualBlock(cost_dyn_f1, nullptr, w_f1);
  problem.SetParameterLowerBound(w_f1, 0, 0.0);
  problem.SetParameterUpperBound(w_f1, 0, 1.0);
  problem.SetParameterLowerBound(w_f1, 1, 0.0);
  problem.SetParameterUpperBound(w_f1, 1, 1.0);

  auto cost_dyn_f2 =
      new ceres::DynamicAutoDiffCostFunction<BSResidualDyn, 4>(new BSResidualDyn(6, target1));
  cost_dyn_f2->AddParameterBlock(2);
  cost_dyn_f2->SetNumResiduals(6);
  problem.AddResidualBlock(cost_dyn_f2, nullptr, w_f2);
  problem.SetParameterLowerBound(w_f2, 0, 0.0);
  problem.SetParameterUpperBound(w_f2, 0, 1.0);
  problem.SetParameterLowerBound(w_f2, 1, 0.0);
  problem.SetParameterUpperBound(w_f2, 1, 1.0);

  ceres::Solver::Options options;
  options.max_num_iterations           = 100;
  options.linear_solver_type           = ceres::DENSE_QR;
  options.minimizer_progress_to_stdout = true;

  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  std::cout << "DYNAMIC PER MESH COST!!!!!!!!" << std::endl;
  std::cout << summary.FullReport() << std::endl;
  std::cout << "f1_w0: " << w_f1[0] << std::endl;
  std::cout << "f1_w1: " << w_f1[1] << std::endl;
  std::cout << "f2_w0: " << w_f2[0] << std::endl;
  std::cout << "f2_w1: " << w_f2[1] << std::endl;
  std::cout << "result1" << std::endl;
  for (int i = 0; i < 6; ++i)
  {
    auto result = b0[i] + w_f1[0] * (b1_ws[i] - b0[i]) + w_f1[1] * (b2_ws[i] - b0[i]);
    std::cout << result << std::endl;
  }
  std::cout << "result2" << std::endl;
  for (int i = 0; i < 6; ++i)
  {
    auto result = b0[i] + w_f2[0] * (b1_ws[i] - b0[i]) + w_f2[1] * (b2_ws[i] - b0[i]);
    std::cout << result << std::endl;
  }
}
