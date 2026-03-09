cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(custom_ops LANGUAGES CXX)

if(EXISTS "${CMAKE_SOURCE_DIR}/platforms_cmake/work_laptop.cmake")
    include(platforms_cmake/work_laptop.cmake)
endif()

# 1D Solver (optional, requires LibTorch)
find_package(Torch QUIET)
if(Torch_FOUND)
    list(APPEND CMAKE_PREFIX_PATH ${DEFAULT_BLAS_ROOT})
    find_package(BLAS REQUIRED)
    file(GLOB SRC CONFIGURE_DEPENDS "src/*.cpp" "1D-Euler-app.cpp")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wl,-no_compact_unwind")
    add_executable(app ${SRC})
    if(DEFAULT_BLAS_ROOT)
        include_directories("${DEFAULT_BLAS_ROOT}/include")
    endif()
    target_link_libraries(app ${BLAS_LIBRARIES} "${TORCH_LIBRARIES}")
    set_property(TARGET app PROPERTY CXX_STANDARD 17)
endif()

# 2D Euler DG Solver with CUDA acceleration
enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

add_executable(app2d
    src/Polylib.cpp
    src/basis.cpp
    src/basis_poly.cpp
    src/basis_functions.cpp
    src/io.cpp
    src/tinyxml.cpp
    src/tinyxmlerror.cpp
    src/tinyxmlparser.cpp
    src/tinystr.cpp
    src/mesh2d.cpp
    src/geom2d.cpp
    src/euler2d.cpp
    src/euler2d_gpu.cu
    src/adjoint2d_gpu.cu
    src/implicit_solver.cu
    src/p_adapt.cpp
    2D-Euler-app.cpp
)
if(DEFAULT_BLAS_ROOT AND EXISTS "${DEFAULT_BLAS_ROOT}/include")
    target_include_directories(app2d PRIVATE "${DEFAULT_BLAS_ROOT}/include")
endif()
target_link_libraries(app2d ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES} CUDA::cublas)
set_property(TARGET app2d PROPERTY CXX_STANDARD 17)
set_property(TARGET app2d PROPERTY CUDA_STANDARD 17)
set_target_properties(app2d PROPERTIES CUDA_ARCHITECTURES "90")

# 2D Euler Adjoint Solver
add_executable(app2d_adjoint
    src/Polylib.cpp
    src/basis.cpp
    src/basis_poly.cpp
    src/basis_functions.cpp
    src/io.cpp
    src/tinyxml.cpp
    src/tinyxmlerror.cpp
    src/tinyxmlparser.cpp
    src/tinystr.cpp
    src/mesh2d.cpp
    src/geom2d.cpp
    src/euler2d.cpp
    src/euler2d_gpu.cu
    src/adjoint2d_gpu.cu
    src/p_adapt.cpp
    2D-Euler-adjoint-app.cpp
)
if(DEFAULT_BLAS_ROOT AND EXISTS "${DEFAULT_BLAS_ROOT}/include")
    target_include_directories(app2d_adjoint PRIVATE "${DEFAULT_BLAS_ROOT}/include")
endif()
target_link_libraries(app2d_adjoint ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
set_property(TARGET app2d_adjoint PROPERTY CXX_STANDARD 17)
set_property(TARGET app2d_adjoint PROPERTY CUDA_STANDARD 17)
set_target_properties(app2d_adjoint PROPERTIES CUDA_ARCHITECTURES "90")

# 2D Euler Discrete Adjoint Solver
add_executable(app2d_discrete_adjoint
    src/Polylib.cpp
    src/basis.cpp
    src/basis_poly.cpp
    src/basis_functions.cpp
    src/io.cpp
    src/tinyxml.cpp
    src/tinyxmlerror.cpp
    src/tinyxmlparser.cpp
    src/tinystr.cpp
    src/mesh2d.cpp
    src/geom2d.cpp
    src/euler2d.cpp
    src/euler2d_gpu.cu
    src/discrete_adjoint2d_gpu.cu
    src/p_adapt.cpp
    2D-Euler-discrete-adjoint-app.cpp
)
if(DEFAULT_BLAS_ROOT AND EXISTS "${DEFAULT_BLAS_ROOT}/include")
    target_include_directories(app2d_discrete_adjoint PRIVATE "${DEFAULT_BLAS_ROOT}/include")
endif()
target_link_libraries(app2d_discrete_adjoint ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
set_property(TARGET app2d_discrete_adjoint PROPERTY CXX_STANDARD 17)
set_property(TARGET app2d_discrete_adjoint PROPERTY CUDA_STANDARD 17)
set_target_properties(app2d_discrete_adjoint PROPERTIES CUDA_ARCHITECTURES "90")

# Mesh deformation tool (CPU only, no CUDA)
add_executable(deform_mesh
    src/Polylib.cpp
    src/basis.cpp
    src/basis_poly.cpp
    src/basis_functions.cpp
    src/io.cpp
    src/tinyxml.cpp
    src/tinyxmlerror.cpp
    src/tinyxmlparser.cpp
    src/tinystr.cpp
    src/mesh2d.cpp
    src/geom2d.cpp
    src/hicks_henne.cpp
    src/mesh_deform.cpp
    deform-mesh-app.cpp
)
if(DEFAULT_BLAS_ROOT AND EXISTS "${DEFAULT_BLAS_ROOT}/include")
    target_include_directories(deform_mesh PRIVATE "${DEFAULT_BLAS_ROOT}/include")
endif()
target_link_libraries(deform_mesh ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
set_property(TARGET deform_mesh PROPERTY CXX_STANDARD 17)

# Shape gradient computation (CUDA)
add_executable(shape_grad
    src/Polylib.cpp
    src/basis.cpp
    src/basis_poly.cpp
    src/basis_functions.cpp
    src/io.cpp
    src/tinyxml.cpp
    src/tinyxmlerror.cpp
    src/tinyxmlparser.cpp
    src/tinystr.cpp
    src/mesh2d.cpp
    src/geom2d.cpp
    src/euler2d.cpp
    src/euler2d_gpu.cu
    src/discrete_adjoint2d_gpu.cu
    src/hicks_henne.cpp
    src/mesh_deform.cpp
    src/shape_sensitivity.cu
    shape-gradient-app.cpp
)
if(DEFAULT_BLAS_ROOT AND EXISTS "${DEFAULT_BLAS_ROOT}/include")
    target_include_directories(shape_grad PRIVATE "${DEFAULT_BLAS_ROOT}/include")
endif()
target_link_libraries(shape_grad ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
set_property(TARGET shape_grad PROPERTY CXX_STANDARD 17)
set_property(TARGET shape_grad PROPERTY CUDA_STANDARD 17)
set_target_properties(shape_grad PROPERTIES CUDA_ARCHITECTURES "90")

# Truncation error utility (CPU only, no CUDA)
add_executable(truncerr
    src/Polylib.cpp
    src/basis.cpp
    src/basis_poly.cpp
    src/basis_functions.cpp
    src/io.cpp
    src/tinyxml.cpp
    src/tinyxmlerror.cpp
    src/tinyxmlparser.cpp
    src/tinystr.cpp
    src/mesh2d.cpp
    src/geom2d.cpp
    src/euler2d.cpp
    truncation-error-app.cpp
)
if(DEFAULT_BLAS_ROOT AND EXISTS "${DEFAULT_BLAS_ROOT}/include")
    target_include_directories(truncerr PRIVATE "${DEFAULT_BLAS_ROOT}/include")
endif()
target_link_libraries(truncerr ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
set_property(TARGET truncerr PROPERTY CXX_STANDARD 17)
