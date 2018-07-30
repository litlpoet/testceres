#ifndef SRC_TESTAPP_INC_EIGENTYPES_HPP_
#define SRC_TESTAPP_INC_EIGENTYPES_HPP_

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <mathtypes.hpp>

namespace core
{

using RVecX = Eigen::Matrix<core::Real, 1, Eigen::Dynamic, Eigen::RowMajor>;

using SpMatRM = Eigen::SparseMatrix<core::Real, Eigen::RowMajor>;

}  // namespace core

#endif  // SRC_TESTAPP_INC_EIGENTYPES_HPP_
