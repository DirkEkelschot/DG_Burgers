cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(custom_ops)


include(platforms_cmake/work_laptop.cmake)

#find_package(LAPACK)
#find_package(Boost COMPONENTS system filesystem REQUIRED)


list(APPEND CMAKE_PREFIX_PATH ${DEFAULT_LIBTORCH_ROOT})
find_package(Torch REQUIRED)

list (APPEND CMAKE_PREFIX_PATH ${DEFAULT_BLAS_ROOT})
find_package(BLAS REQUIRED)

file(GLOB SRC CONFIGURE_DEPENDS "src/*.cpp" "1D-Euler-app.cpp")
set(CMAKE_CXX_FLAGS "-Wl,-no_compact_unwind")
add_executable(app ${SRC})



include_directories("${DEFAULT_BLAS_ROOT}/include")
target_link_libraries(app ${BLAS_LIBRARIES})


target_link_libraries(app "${TORCH_LIBRARIES}")
set_property(TARGET app PROPERTY CXX_STANDARD 17)
