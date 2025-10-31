cmake_minimum_required(VERSION 3.8)

set(TARGET_NAME laszip_playground)
project(${TARGET_NAME} LANGUAGES CXX CUDA)  # <-- enable CUDA language

set(CMAKE_CXX_STANDARD 23)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# --------------------------------------------------------
# Source files (C++ and CUDA)
# --------------------------------------------------------
file(GLOB PROJECT_SOURCES CONFIGURE_DEPENDS 
    src/*.[ch]pp
    src/*.cu      # <-- include CUDA files
)

# --------------------------------------------------------
# Dependencies
# --------------------------------------------------------
add_subdirectory(libs/laszip)

# --------------------------------------------------------
# Target setup
# --------------------------------------------------------
add_executable(${TARGET_NAME} ${PROJECT_SOURCES})

target_link_libraries(${TARGET_NAME}
    PRIVATE
    laszip
)

target_include_directories(${TARGET_NAME} PRIVATE 
    ./src
    ./libs
)

# --------------------------------------------------------
# CUDA configuration
# --------------------------------------------------------
# Optional: choose GPU architectures to compile for
set(CMAKE_CUDA_ARCHITECTURES 75 86 89 90)  # adapt to your GPU(s)
# You can check your compute capability here: https://developer.nvidia.com/cuda-gpus

# Treat all CUDA warnings as errors (optional)
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Werror all-warnings")

# Enable separable compilation (if your .cu files use device functions across TUs)
set_target_properties(${TARGET_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# --------------------------------------------------------
# Windows-specific options
# --------------------------------------------------------
if (WIN32)
    # set Visual Studio working directory
    set_property(TARGET ${TARGET_NAME} PROPERTY 
        VS_DEBUGGER_WORKING_DIRECTORY "$(SolutionDir).."
    )
endif()

# --------------------------------------------------------
# Set startup project (Visual Studio)
# --------------------------------------------------------
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${TARGET_NAME})
