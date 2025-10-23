cmake_minimum_required(VERSION 3.8)
#export CMAKE_POLICY_VERSION_MINIMUM=3.1

set(TARGET_NAME laszip_playground) 
project(${TARGET_NAME})
set(CMAKE_CXX_STANDARD 23)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)


file(GLOB PROJECT_SOURCES CONFIGURE_DEPENDS 
	src/*.*
)

add_subdirectory(libs/laszip)
add_executable(${TARGET_NAME} ${PROJECT_SOURCES})

target_link_libraries(${PROJECT_NAME}
	laszip
)

if (WIN32)
	# set workdir to ../$(SolutionDir), so that .cu files can be monitored/reloaded/compiled
	set_property(TARGET ${TARGET_NAME} PROPERTY VS_DEBUGGER_WORKING_DIRECTORY "$(SolutionDir)..")
endif (WIN32)

target_include_directories(${TARGET_NAME} PRIVATE 
	./src
	./libs
)

# set startup project
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${TARGET_NAME})