#ifndef SRC_TESTAPP_INC_EIGENTYPES_HPP_
#define SRC_TESTAPP_INC_EIGENTYPES_HPP_

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <mathtypes.hpp>

namespace core
{

using RVec3 = Eigen::Matrix<core::Real, 1, 3, Eigen::RowMajor>;
using RVecX = Eigen::Matrix<core::Real, 1, Eigen::Dynamic, Eigen::RowMajor>;

using MatX = Eigen::Matrix<core::Real, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

using SpMatRM = Eigen::SparseMatrix<core::Real, Eigen::RowMajor>;

}  // namespace core

#endif  // SRC_TESTAPP_INC_EIGENTYPES_HPP_
