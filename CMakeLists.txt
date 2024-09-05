cmake_minimum_required(VERSION 3.14)

project(CudaSift)
set(PROJECT_VERSION_MAJOR 3)
set(PROJECT_VERSION_MINOR 0)
set(PROJECT_VERSION_PATCH 0)
SET(PROJECT_VERSION ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH})
SET(PROJECT_SOVERSION "${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}")

include(GenerateExportHeader)
include(GNUInstallDirs)

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

OPTION( BUILD_SHARED_LIBS "Set to OFF to build static libraries" ON )
OPTION( VERBOSE "Print internal processing time" ON )
if(VERBOSE)
  add_definitions(-DVERBOSE)
endif()

if (WIN32)
  set(EXTRA_CXX_FLAGS "/DVERBOSE /D_CRT_SECURE_NO_WARNINGS ")
  list(APPEND CUDA_NVCC_FLAGS "--compiler-options;-O2;-DVERBOSE") 
endif()
if (UNIX)
  if (APPLE)
    set(EXTRA_CXX_FLAGS "-DVERBOSE -msse2")
    list(APPEND CUDA_NVCC_FLAGS "--compiler-options;-O2;-DVERBOSE") 
  else()
     if(NOT ${CMAKE_SYSTEM_PROCESSOR} MATCHES "^arm" AND NOT ${CMAKE_SYSTEM_PROCESSOR} MATCHES "^aarch")
       set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -msse2 ")
    else() # ARM platforms do not have SSE
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 ")
    endif()
    list(APPEND CUDA_NVCC_FLAGS "-lineinfo;-ccbin;/usr/bin/gcc;--compiler-options;-O2;-D_FORCE_INLINES;-DVERBOSE_NOT") 
  endif()
endif()

IF(${CMAKE_GENERATOR} MATCHES ".*Makefiles")
    IF("${CMAKE_BUILD_TYPE}" STREQUAL "")
        set(CMAKE_BUILD_TYPE Release)
    ENDIF("${CMAKE_BUILD_TYPE}" STREQUAL "")
ENDIF(${CMAKE_GENERATOR} MATCHES ".*Makefiles")

message(STATUS "BUILD_SHARED_LIBS=${BUILD_SHARED_LIBS}")
message(STATUS "CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}")
message(STATUS "VERBOSE=${VERBOSE}")

set(INSTALL_INCLUDE_DIR include/cudasift-${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR})
if(WIN32 AND NOT CYGWIN)
  set(DEF_INSTALL_CMAKE_DIR CMake)
else()
  set(DEF_INSTALL_CMAKE_DIR ${CMAKE_INSTALL_LIBDIR}/cudasift-${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR})
endif()
set(INSTALL_CMAKE_DIR ${DEF_INSTALL_CMAKE_DIR})

set(cuda_sources
  src/cudaImage.cu
  src/cudaSiftH.cu
  src/matching.cu
  include/cudasift/cudaImage.h  
  include/cudasift/cudaSiftH.h  
  include/cudasift/cudaSiftD.h  
  include/cudasift/cudaSift.h  
  include/cudasift/cudautils.h
)  

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/include 
)

# library
cuda_add_library(cudasift ${cuda_sources})
set_target_properties(cudasift PROPERTIES
  COMPILE_FLAGS "${EXTRA_CXX_FLAGS}"			   
)
target_link_libraries(cudasift ${CUDA_cudadevrt_LIBRARY})
generate_export_header(cudasift)
target_include_directories(cudasift PUBLIC 
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include;${CMAKE_CURRENT_BINARY_DIR}/include>"
  "$<INSTALL_INTERFACE:${INSTALL_INCLUDE_DIR}>")

SET_TARGET_PROPERTIES(
  cudasift
   PROPERTIES
     VERSION ${PROJECT_VERSION}
     SOVERSION ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}
)

# executable
set(sources
  example/geomFuncs.cpp
  example/mainSift.cpp
)
cuda_add_executable(cudasift_example ${sources})
TARGET_LINK_LIBRARIES(cudasift_example cudasift ${OpenCV_LIBS})

#######################
# Setup CudaSiftConfig.cmake
#######################
IF (NOT TARGET uninstall)
CONFIGURE_FILE(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake_uninstall.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
  IMMEDIATE @ONLY)

ADD_CUSTOM_TARGET(uninstall
  "${CMAKE_COMMAND}" -P "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake")
ENDIF()

install(
  DIRECTORY
    ${CMAKE_CURRENT_SOURCE_DIR}/include/cudasift
    ${CMAKE_CURRENT_BINARY_DIR}/include/cudasift
  DESTINATION
    ${INSTALL_INCLUDE_DIR}
  COMPONENT
    devel
  FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)

INSTALL(TARGETS cudasift EXPORT cudasiftTargets
        RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}" COMPONENT runtime
        LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}" COMPONENT devel
        ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}" COMPONENT devel)

configure_file(
        ${CMAKE_CURRENT_BINARY_DIR}/cudasift_export.h
        ${CMAKE_CURRENT_BINARY_DIR}/include/cudasift/cudasift_export.h
        COPYONLY)

export(EXPORT cudasiftTargets
  FILE "${CMAKE_CURRENT_BINARY_DIR}/../../${PROJECT_NAME}_coreTargets.cmake"
)

install(EXPORT cudasiftTargets
  FILE
    ${PROJECT_NAME}Targets.cmake
  DESTINATION
    ${INSTALL_CMAKE_DIR}
  COMPONENT
    devel
)

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY AnyNewerVersion
)

# Build tree:
SET(CONF_MODULES_DIR "../cmake_modules")
configure_file(
  ${PROJECT_NAME}Config.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
  @ONLY
)

# Install tree:
configure_file(
  ${PROJECT_NAME}Config.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/${PROJECT_NAME}Config.cmake"
  @ONLY
)
install(
  FILES
    "${CMAKE_CURRENT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/${PROJECT_NAME}Config.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
  DESTINATION
    ${INSTALL_CMAKE_DIR}
  COMPONENT
    devel
)
