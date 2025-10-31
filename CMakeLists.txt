cmake_minimum_required(VERSION 3.8)

set(TARGET_NAME laszip_playground)
project(${TARGET_NAME} LANGUAGES CXX CUDA)  # CUDA enabled

set(CMAKE_CXX_STANDARD 23)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# --------------------------------------------------------
# Source files (C++ and CUDA)
# --------------------------------------------------------
file(GLOB PROJECT_SOURCES CONFIGURE_DEPENDS 
    src/*.[ch]pp
    src/*.cu
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
# Choose GPU architectures (adjust for your machine)
set(CMAKE_CUDA_ARCHITECTURES 75 86 89 90)

# Enable separable compilation (for multi-file CUDA builds)
set_target_properties(${TARGET_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Treat warnings as errors (optional)
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Werror all-warnings")

# --------------------------------------------------------
# Windows-specific options
# --------------------------------------------------------
if (WIN32)
    set_property(TARGET ${TARGET_NAME} PROPERTY 
        VS_DEBUGGER_WORKING_DIRECTORY "$(SolutionDir).."
    )
endif()

# --------------------------------------------------------
# Set startup project (Visual Studio)
# --------------------------------------------------------
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${TARGET_NAME})
