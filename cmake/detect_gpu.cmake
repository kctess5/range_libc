find_package(CUDA REQUIRED)

set(OUTPUT_DIR ./)

message(${CUDA_NVCC_EXECUTABLE})

# ################################################################################################
# # A function for automatic detection of GPUs installed  (if autodetection is enabled)
# # Usage:
# #   detect_installed_gpus(out_variable)
# function(detect_installed_gpus out_variable)
#   if(NOT CUDA_gpu_detect_output)
#     set(__cufile ${OUTPUT_DIR}/detect_cuda_archs.cu)

#     file(WRITE ${__cufile} ""
#       "#include <cstdio>\n"
#       "int main()\n"
#       "{\n"
#       "  int count = 0;\n"
#       "  if (cudaSuccess != cudaGetDeviceCount(&count)) return -1;\n"
#       "  if (count == 0) return -1;\n"
#       "  for (int device = 0; device < count; ++device)\n"
#       "  {\n"
#       "    cudaDeviceProp prop;\n"
#       "    if (cudaSuccess == cudaGetDeviceProperties(&prop, device))\n"
#       "      std::printf(\"%d.%d \", prop.major, prop.minor);\n"
#       "  }\n"
#       "  return 0;\n"
#       "}\n")

#     execute_process(COMMAND "${CUDA_NVCC_EXECUTABLE}" "--run" "${__cufile}"
#                     WORKING_DIRECTORY "${OUTPUT_DIR}/CMakeFiles/"
#                     RESULT_VARIABLE __nvcc_res OUTPUT_VARIABLE __nvcc_out
#                     ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)

#     if(__nvcc_res EQUAL 0)
#       string(REPLACE "2.1" "2.1(2.0)" __nvcc_out "${__nvcc_out}")
#       set(CUDA_gpu_detect_output ${__nvcc_out} CACHE INTERNAL "Returned GPU architetures from detect_gpus tool" FORCE)
#     endif()
#   endif()

#   if(NOT CUDA_gpu_detect_output)
#     message(STATUS "Automatic GPU detection failed. Building for all known architectures.")
#     set(${out_variable} ${Caffe_known_gpu_archs} PARENT_SCOPE)
#   else()
#     set(${out_variable} ${CUDA_gpu_detect_output} PARENT_SCOPE)
#   endif()
# endfunction()


# ################################################################################################
# # Function for selecting GPU arch flags for nvcc based on CUDA_ARCH_NAME
# # Usage:
# #   select_nvcc_arch_flags(out_variable)
# function(select_nvcc_arch_flags out_variable)
#   # List of arch names
#   set(__archs_names "Fermi" "Kepler" "Maxwell" "All" "Manual")
#   set(__archs_name_default "All")
#   if(NOT CMAKE_CROSSCOMPILING)
#     list(APPEND __archs_names "Auto")
#     set(__archs_name_default "Auto")
#   endif()

#   # set CUDA_ARCH_NAME strings (so it will be seen as dropbox in CMake-Gui)
#   set(CUDA_ARCH_NAME ${__archs_name_default} CACHE STRING "Select target NVIDIA GPU achitecture.")
#   set_property( CACHE CUDA_ARCH_NAME PROPERTY STRINGS "" ${__archs_names} )
#   mark_as_advanced(CUDA_ARCH_NAME)

#   # verify CUDA_ARCH_NAME value
#   if(NOT ";${__archs_names};" MATCHES ";${CUDA_ARCH_NAME};")
#     string(REPLACE ";" ", " __archs_names "${__archs_names}")
#     message(FATAL_ERROR "Only ${__archs_names} architeture names are supported.")
#   endif()

#   if(${CUDA_ARCH_NAME} STREQUAL "Manual")
#     set(CUDA_ARCH_BIN ${Caffe_known_gpu_archs} CACHE STRING "Specify 'real' GPU architectures to build binaries for, BIN(PTX) format is supported")
#     set(CUDA_ARCH_PTX "50"                     CACHE STRING "Specify 'virtual' PTX architectures to build PTX intermediate code for")
#     mark_as_advanced(CUDA_ARCH_BIN CUDA_ARCH_PTX)
#   else()
#     unset(CUDA_ARCH_BIN CACHE)
#     unset(CUDA_ARCH_PTX CACHE)
#   endif()

#   if(${CUDA_ARCH_NAME} STREQUAL "Fermi")
#     set(__cuda_arch_bin "20 21(20)")
#   elseif(${CUDA_ARCH_NAME} STREQUAL "Kepler")
#     set(__cuda_arch_bin "30 35")
#   elseif(${CUDA_ARCH_NAME} STREQUAL "Maxwell")
#     set(__cuda_arch_bin "50")
#   elseif(${CUDA_ARCH_NAME} STREQUAL "All")
#     set(__cuda_arch_bin ${Caffe_known_gpu_archs})
#   elseif(${CUDA_ARCH_NAME} STREQUAL "Auto")
#     detect_installed_gpus(__cuda_arch_bin)
#   else()  # (${CUDA_ARCH_NAME} STREQUAL "Manual")
#     set(__cuda_arch_bin ${CUDA_ARCH_BIN})
#   endif()

#   # remove dots and convert to lists
#   string(REGEX REPLACE "\\." "" __cuda_arch_bin "${__cuda_arch_bin}")
#   string(REGEX REPLACE "\\." "" __cuda_arch_ptx "${CUDA_ARCH_PTX}")
#   string(REGEX MATCHALL "[0-9()]+" __cuda_arch_bin "${__cuda_arch_bin}")
#   string(REGEX MATCHALL "[0-9]+"   __cuda_arch_ptx "${__cuda_arch_ptx}")
#   list_unique(__cuda_arch_bin __cuda_arch_ptx)

#   set(__nvcc_flags "")
#   set(__nvcc_archs_readable "")

#   # Tell NVCC to add binaries for the specified GPUs
#   foreach(__arch ${__cuda_arch_bin})
#     if(__arch MATCHES "([0-9]+)\\(([0-9]+)\\)")
#       # User explicitly specified PTX for the concrete BIN
#       list(APPEND __nvcc_flags -gencode arch=compute_${CMAKE_MATCH_2},code=sm_${CMAKE_MATCH_1})
#       list(APPEND __nvcc_archs_readable sm_${CMAKE_MATCH_1})
#     else()
#       # User didn't explicitly specify PTX for the concrete BIN, we assume PTX=BIN
#       list(APPEND __nvcc_flags -gencode arch=compute_${__arch},code=sm_${__arch})
#       list(APPEND __nvcc_archs_readable sm_${__arch})
#     endif()
#   endforeach()

#   # Tell NVCC to add PTX intermediate code for the specified architectures
#   foreach(__arch ${__cuda_arch_ptx})
#     list(APPEND __nvcc_flags -gencode arch=compute_${__arch},code=compute_${__arch})
#     list(APPEND __nvcc_archs_readable compute_${__arch})
#   endforeach()

#   string(REPLACE ";" " " __nvcc_archs_readable "${__nvcc_archs_readable}")
#   set(${out_variable}          ${__nvcc_flags}          PARENT_SCOPE)
#   set(${out_variable}_readable ${__nvcc_archs_readable} PARENT_SCOPE)
# endfunction()

# detect_installed_gpus(CUDA_COMPUTE_NUMBER)
# # string(REGEX REPLACE "\\." "" CUDA_COMPUTE_NUMBER "${CUDA_COMPUTE_NUMBER}")
# message("TEST")
# message(${CUDA_COMPUTE_NUMBER})