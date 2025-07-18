cmake_minimum_required(VERSION 3.10)
project(LDDMM3D LANGUAGES C CXX)

# Use C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ITK installation directory
set(ITK_DIR "$ENV{HOME}/ITK-install/lib/cmake/ITK-6.0")

# Find the required ITK modules
find_package(ITK REQUIRED
  HINTS    "${ITK_DIR}"
  NO_DEFAULT_PATH
  COMPONENTS
    ITKDisplacementField   # warping filter
    ITKVTK                # ITK–VTK glue
    ITKIOGDCM             # DICOM I/O
    ITKIOMeta             # MetaImage I/O
    ITKRegistrationCommon
    ITKRegistrationDemons  # diffeomorphic demons (LDDMM-like)
)
include(${ITK_USE_FILE})

# MPI for VTK's MPI support
find_package(MPI REQUIRED)

# Required VTK components
find_package(VTK CONFIG REQUIRED
  COMPONENTS
    CommonCore
    CommonDataModel
    IOImage
    FiltersCore
    FiltersSources
    InteractionStyle
    RenderingCore
    RenderingOpenGL2
)

# Build the executable
add_executable(LDDMM3D FLux3D_LDDMM.cpp)

target_link_libraries(LDDMM3D PRIVATE
  ${ITK_LIBRARIES}
  MPI::MPI_C
  MPI::MPI_CXX
  VTK::CommonCore
  VTK::CommonDataModel
  VTK::IOImage
  VTK::FiltersCore
  VTK::FiltersSources
  VTK::InteractionStyle
  VTK::RenderingCore
  VTK::RenderingOpenGL2
)
