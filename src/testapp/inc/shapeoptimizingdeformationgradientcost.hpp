#ifndef SRC_TESTAPP_INC_SHAPEOPTIMIZINGDEFORMATIONGRADIENTCOST_HPP_
#define SRC_TESTAPP_INC_SHAPEOPTIMIZINGDEFORMATIONGRADIENTCOST_HPP_

#include <mathtypes.hpp>

namespace faceopt
{

namespace opt
{

struct ShapeOptimizingDeformationGradientCost
{
  core::Real const*        _target_local_frame;  // should be precomputed \in R^6
  core::Real const* const* _base_tri_per_bs;     // should be precomputed \in #(full weights) * R^9
  core::Real const*        _opt_wts_per_bs;      // should be precomputed \in R^#(full weights)
  core::Real               _tri_wt;              // global optimization weights for triangle
  int                      _n_opt_wts;           // #full weights (in reduced model)

  ShapeOptimizingDeformationGradientCost() {}

  template<typename T>
  bool
  operator()(T const* const* tri_delta_per_bs,  // #opt_bs * 9 (3 verts to compute local frame)
             T*              residual           // residual = 6
             ) const
  {
    // (1) set initial (precompute) target local frame
    for (int i = 0; i < 6; ++i)
    {
      residual[i] = T(_target_local_frame[i]);
    }

    // (2) compute local frame difference (approx. deformation gradient)
    for (int w = 0; w < _n_opt_wts; ++w)
    {
      auto const* tri      = tri_delta_per_bs[w];
      auto const* base_tri = _base_tri_per_bs[w];
      for (int v = 0; v < 9; ++v)
      {
        tri[v] += base_tri[v];
      }
      auto const* v1 = tri[0];
      auto const* v2 = tri[3];
      auto const* v3 = tri[6];
      // edge 1
      residual[0] -= (v1[0] - v2[0]);
      residual[1] -= (v1[1] - v2[1]);
      residual[2] -= (v1[2] - v2[2]);
      // edge 2
      residual[3] -= (v3[0] - v2[0]);
      residual[4] -= (v3[1] - v2[1]);
      residual[5] -= (v3[2] - v2[2]);
    }

    // (3) apply optimization weights
    for (int i = 0; i < 6; ++i)
    {
      residual[i] *= _tri_wt;
    }
  }
};

}  // namespace opt

}  // namespace faceopt

#endif  // SRC_TESTAPP_INC_SHAPEOPTIMIZINGDEFORMATIONGRADIENTCOST_HPP_
