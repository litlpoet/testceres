#ifndef SRC_TESTAPP_INC_WEIGHTOPTIMIZINGCOSTANALYTIC_HPP_
#define SRC_TESTAPP_INC_WEIGHTOPTIMIZINGCOSTANALYTIC_HPP_

#include <ceres/ceres.h>

#include <cstring>

#include <eigentypes.hpp>

namespace facemodel
{
class BlendShapeModel;
}  // namespace facemodel

class WeightOptimizingCostAnalytic : public ceres::CostFunction
{
  core::SpMatRM const* _bsmat;
  size_t               _n_full_wts;
  size_t               _n_ctrl_wts;
  size_t               _v_dim;
  core::Idx            _v3;

 public:
  WeightOptimizingCostAnalytic() {}

  virtual ~WeightOptimizingCostAnalytic() = default;

  // bool
  // Evaluate(double const* const* parameters, double* residuals, double** jacobians) const final
  // {
  //   double const* ctrl_w = parameters[0];
  //   core::RVecX   full_w(_n_full_wts);
  //   for (size_t i = 0; i < _n_ctrl_wts; ++i)
  //   {
  //     full_w(i) = ctrl_w[i];
  //   }
  //   std::vector<std::pair<core::Idx, core::Idx>> activated_pn;
  //   for (auto const& pn : (*_pos_negatives))
  //   {
  //     auto& pos_w = full_w(pn.sourceWeightId());
  //     if (pos_w < 0.0)
  //     {
  //       full_w(pn.targetWeightId()) = -pos_w;
  //       pos_w                       = 0.0;
  //       activated_pn.emplace_back(std::make_pair(pn.sourceWeightId(), pn.targetWeightId()));
  //     }
  //   }

  //   for (size_t i = 0; i < _n_v_dim; ++i)
  //   {
  //     residuals[i] = _target_shape[i] - _home_shape[i];
  //   }
  //   Eigen::Map<core::RVecX> r_vec(residuals, _v_dim);
  //   r_vec -= full_w * (*_bsmat);

  //   if (jacobians == nullptr || jacobians[0] == nullptr)
  //     return true;

  //   double* jac = jacobians[0];
  //   for (size_t r = 0; r < _v_dim; ++r)
  //   {
  //     auto const r_stride = r * _n_ctrl_wts;
  //     for (size_t c = 0; c < _n_ctrl_wts; ++c)
  //     {
  //       jac[r_stride + c] = -(*_bsmat_dense)(c, r);
  //     }
  //     for (auto const& c : activated_pn)
  //     {
  //       jac[r_stride + c.first] = -(*_bsmat_dense)(c.second, r);
  //     }
  //   }

  //   return true;
  // }

  bool
  Evaluate(double const* const* parameters, double* residuals, double** jacobians) const final
  {
    double const* ctrl_w = parameters[0];
    core::RVecX   full_w = core::RVecX::Zero(_n_full_wts);
    for (size_t i = 0; i < _n_ctrl_wts; ++i)
    {
      full_w(i) = ctrl_w[i];
    }

    std::vector<std::pair<core::Idx, core::Idx>> activated_pn;
    for (auto const& pn : (*_pos_negatives))
    {
      auto& pos_w = full_w(pn.sourceWeightId());
      if (pos_w < 0.0)
      {
        full_w(pn.targetWeightId()) = -pos_w;
        pos_w                       = 0.0;
        activated_pn.emplace_back(std::make_pair(pn.sourceWeightId(), pn.targetWeightId()));
      }
    }

    for (size_t i = 0; i < 3; ++i)
    {
      residuals[i] = _target_residuals[_v3 + i];
    }
    Eigen::Map<core::RVec3> r_vec(residuals, _v_dim);
    r_vec -= full_w * _bsmat->block(0, _v3, _n_full_wts, 3);

    if (jacobians == nullptr || jacobians[0] == nullptr)
      return true;

    Eigen::Map<core::MatX> jac_map(jacobians[0], _n_ctrl_wts, _v_dim);
    jac_map = -_bsmat->block(0, _v3, _n_ctrl_wts, 3);
    for (auto const& pn : activated_pn)
    {
      jac_map.block(pn.first, _v3, 1, 3) = -_bsmat->block(pn.second, _v3, 1, 3);
    }
    return true;
  }
};
#endif  // SRC_TESTAPP_INC_WEIGHTOPTIMIZINGCOSTANALYTIC_HPP_
