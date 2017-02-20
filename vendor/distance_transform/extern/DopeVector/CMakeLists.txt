cmake_minimum_required(VERSION 3.3)

project(DopeVector)



option(WITH_RTTI "Build DopeVector using RTTI or not" OFF)
option(WITH_EIGEN "Build DopeVector with Index<Dimension> as Eigen Matrix (if present) or not." OFF)
option(ATTACH_SOURCES "When generating an IDE project, add DopeVector header files to project sources." OFF)



set(hdr_dir ${CMAKE_CURRENT_SOURCE_DIR}/include)

set(hdr_internal_files
	${hdr_dir}/DopeVector/internal/Common.hpp
	${hdr_dir}/DopeVector/internal/Expression.hpp
	${hdr_dir}/DopeVector/internal/eigen_support/EigenExpression.hpp
	${hdr_dir}/DopeVector/internal/Iterator.hpp
)
source_group("DopeVector\\internal" FILES ${hdr_internal_files})

set(hdr_internal_inline_files
	${hdr_dir}/DopeVector/internal/inlines/Index.inl
	${hdr_dir}/DopeVector/internal/inlines/Expression.inl
	${hdr_dir}/DopeVector/internal/inlines/eigen_support/EigenExpression.inl
	${hdr_dir}/DopeVector/internal/inlines/Iterator.inl
	${hdr_dir}/DopeVector/internal/inlines/DopeVector.inl
	${hdr_dir}/DopeVector/internal/inlines/Grid.inl
)
set_source_files_properties(${hdr_internal_inline_files} PROPERTIES XCODE_EXPLICIT_FILE_TYPE "sourcecode.cpp.h")
source_group("DopeVector\\internal\\inlines" FILES ${hdr_internal_inline_files})

set(hdr_main_files
	${hdr_dir}/DopeVector/DopeVector.hpp
	${hdr_dir}/DopeVector/Grid.hpp
	${hdr_dir}/DopeVector/Index.hpp
)
source_group("DopeVector" FILES ${hdr_main_files})

set(all_hdr ${hdr_internal_files} ${hdr_internal_inline_files} ${hdr_main_files})


add_library(${PROJECT_NAME} INTERFACE)

set(required_cxx_features
	cxx_alias_templates
	cxx_auto_type
	cxx_constexpr
	cxx_decltype
	cxx_default_function_template_args
	cxx_defaulted_functions
	cxx_defaulted_move_initializers
	cxx_deleted_functions
	cxx_explicit_conversions
	cxx_generalized_initializers
	cxx_lambdas
	cxx_noexcept
	cxx_nonstatic_member_init
	cxx_nullptr
	cxx_override
	cxx_range_for
	cxx_right_angle_brackets
	cxx_rvalue_references
	cxx_static_assert
	cxx_variadic_templates
)

target_compile_features(${PROJECT_NAME} INTERFACE ${required_cxx_features})
target_include_directories(${PROJECT_NAME} INTERFACE ${hdr_dir})
if(ATTACH_SOURCES)
	target_sources(${PROJECT_NAME} INTERFACE ${all_hdr})
endif()

if(WITH_EIGEN)
	find_package(Eigen3 3.3.1 QUIET NO_MODULE) # Eigen 3.3 uses modern cmake
	if(EIGEN3_FOUND)
		message(STATUS "Found Eigen ${Eigen3_VERSION} for use in DopeVector")
        target_link_libraries(${PROJECT_NAME} INTERFACE Eigen3::Eigen)
		target_compile_definitions(${PROJECT_NAME} INTERFACE DOPE_USE_EIGEN)
	else()
        set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules ${CMAKE_MODULE_PATH})
        find_package(Eigen3 QUIET)
    	if(EIGEN3_FOUND)
    		message(STATUS "Found Eigen ${Eigen3_VERSION} for use in DopeVector")
    		target_include_directories(${PROJECT_NAME} INTERFACE ${EIGEN3_INCLUDE_DIR})
    		target_compile_definitions(${PROJECT_NAME} INTERFACE DOPE_USE_EIGEN)
    	else()
    		message(STATUS "Eigen was not found")
    	endif()
	endif()
endif(WITH_EIGEN)

if(WITH_RTTI)
	message(STATUS "RTTI is on")
	target_compile_definitions(${PROJECT_NAME} INTERFACE DOPE_USE_RTTI)
else()
	message(STATUS "RTTI is off")
endif()