# testapp
project(ceres_tests VERSION 0.0.1 LANGUAGES CXX)

add_executable(
  ${PROJECT_NAME}
  src/test_2d_bs.cpp
  src/test_hello.cpp
  src/test_curvefit.cpp
  src/test_powell_func.cpp
  )

target_include_directories(
  ${PROJECT_NAME}
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
  ${CERES_INCLUDE_DIRS}
  PRIVATE
  src
  )

target_link_libraries(
  ${PROJECT_NAME}
  Eigen3::Eigen
  GTest::GTest
  GTest::Main
  ${CERES_LIBRARIES}
  )

add_test(
  ${PROJECT_NAME}
  ${PROJECT_NAME}
  )
