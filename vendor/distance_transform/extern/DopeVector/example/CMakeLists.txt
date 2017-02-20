cmake_minimum_required(VERSION 3.3)

project(DopeVector_example)


#option(WITH_RTTI "Build DopeVector using RTTI or not" ON)
option(WITH_EIGEN "Build DopeVector with Index<Dimension> as Eigen Matrix (if present) or not." ON)
option(ATTACH_SOURCES "When generating an IDE project, add DopeVector header files to project sources." ON)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/.. ${CMAKE_CURRENT_BINARY_DIR}/lib_dope_vector)
include(${CMAKE_CURRENT_SOURCE_DIR}/../sources_properties.cmake)

set(src
	${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
)
add_executable(${PROJECT_NAME} ${src})

target_link_libraries(${PROJECT_NAME} DopeVector)
set_dope_vector_source_files_properties()
