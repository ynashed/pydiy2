CMAKE_MINIMUM_REQUIRED (VERSION 2.8.12)

PROJECT (pydiy2)
SET (pydiy2_VERSION_MAJOR 0)
SET (pydiy2_VERSION_MINOR 1)
SET(CMAKE_BUILD_TYPE Release)

SET(CMAKE_CXX_STANDARD 11)
INCLUDE(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
IF(COMPILER_SUPPORTS_CXX11)
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
ELSEIF(COMPILER_SUPPORTS_CXX0X)
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
ELSE()
	MESSAGE(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
ENDIF()

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})

OPTION (debug "Build pydiy2 with debugging on" OFF)
OPTION (optimize "Build pydiy2 with optimization" OFF)

IF (debug)
	IF (optimize)
		SET(CMAKE_BUILD_TYPE RelWithDebInfo)
    ELSE (optimize)
		SET(CMAKE_BUILD_TYPE Debug)
    ENDIF (optimize)
    ADD_DEFINITIONS(-DDEBUG)
ELSE (debug)
	SET(CMAKE_BUILD_TYPE Release)
ENDIF (debug)

INCLUDE(ExternalProject)
################################## Dependencies #####################################
#DIY2
SET(DIY2_PATH "" CACHE PATH "path where DIY2 resides")
IF ("${DIY2_PATH}" STREQUAL "")
	ExternalProject_Add(diy2
   						SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/diy2
   						GIT_REPOSITORY https://github.com/diatomic/diy2.git
   						CONFIGURE_COMMAND ""
   						BUILD_COMMAND ""
   						INSTALL_COMMAND ""
       					)
	ExternalProject_Get_Property(diy2 SOURCE_DIR)
	SET(DIY2_PATH ${SOURCE_DIR})
ENDIF()
INCLUDE_DIRECTORIES(${DIY2_PATH}/include)

#MPI
FIND_PACKAGE(MPI REQUIRED)

SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${MPI_COMPILE_FLAGS}")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${MPI_COMPILE_FLAGS}")
INCLUDE_DIRECTORIES(${MPI_INCLUDE_PATH})

#Pybind11
SET(PYBIND11_PATH "" CACHE PATH "path where pybind11 is installed")
IF ("${PYBIND11_PATH}" STREQUAL "")
	MESSAGE(FATAL_ERROR "PYBIND11_PATH not set: Instructions to download Pybind11 can be found at https://github.com/pybind/pybind11")
ENDIF()
SET(PYTHON_EXECUTABLE "/usr/bin/python" CACHE PATH "path for the python executable/version")
INCLUDE_DIRECTORIES(${PYBIND11_PATH}/include)
ADD_SUBDIRECTORY(${PYBIND11_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/pybind11)

######################################################################################
FILE(GLOB pydiy2_sources "${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/*.cpp")

PYBIND11_ADD_MODULE (pydiy2 MODULE ${pydiy2_sources})
TARGET_LINK_LIBRARIES(pydiy2 ${MPI_LIBRARIES})

ADD_DEPENDENCIES(pydiy2 diy2)
ADD_DEPENDENCIES(pydiy2 pybind11)

INSTALL(TARGETS pydiy2 LIBRARY DESTINATION "${CMAKE_CURRENT_SOURCE_DIR}/tests" COMPONENT modules)
