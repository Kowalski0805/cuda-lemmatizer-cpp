#=============================================================================
# Copyright (c) 2025, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#=============================================================================

#[=======================================================================[.rst:
FindDLPACK
--------

Find DLPACK

Imported targets
^^^^^^^^^^^^^^^^

This module defines the following :prop_tgt:`IMPORTED` target(s):

``DLPACK::DLPACK``
  The DLPACK library, if found.

Result variables
^^^^^^^^^^^^^^^^

This module will set the following variables in your project:

``DLPACK_FOUND``
  True if DLPACK is found.
``DLPACK_INCLUDE_DIRS``
  The include directories needed to use DLPACK.
``DLPACK_LIBRARIES``
  The libraries needed to useDLPACK.
``DLPACK_VERSION_STRING``
  The version of the DLPACK library found. [OPTIONAL]

#]=======================================================================]

# Prefer using a Config module if it exists for this project



set(DLPACK_NO_CONFIG FALSE)
if(NOT DLPACK_NO_CONFIG)
  find_package(DLPACK CONFIG QUIET)
  if(DLPACK_FOUND)
    find_package_handle_standard_args(DLPACK DEFAULT_MSG DLPACK_CONFIG)
    return()
  endif()
endif()

find_path(DLPACK_INCLUDE_DIR NAMES dlpack.h )

set(DLPACK_IS_HEADER_ONLY TRUE)
if(NOT DLPACK_LIBRARY AND NOT DLPACK_IS_HEADER_ONLY)
  find_library(DLPACK_LIBRARY_RELEASE NAMES  NAMES_PER_DIR )
  find_library(DLPACK_LIBRARY_DEBUG   NAMES    NAMES_PER_DIR )

  include(${CMAKE_ROOT}/Modules/SelectLibraryConfigurations.cmake)
  select_library_configurations(DLPACK)
  unset(DLPACK_FOUND) #incorrectly set by select_library_configurations
endif()

include(${CMAKE_ROOT}/Modules/FindPackageHandleStandardArgs.cmake)

if(DLPACK_IS_HEADER_ONLY)
  find_package_handle_standard_args(DLPACK
                                    FOUND_VAR DLPACK_FOUND
                                    REQUIRED_VARS DLPACK_INCLUDE_DIR
                                    VERSION_VAR )
else()
  find_package_handle_standard_args(DLPACK
                                    FOUND_VAR DLPACK_FOUND
                                    REQUIRED_VARS DLPACK_LIBRARY DLPACK_INCLUDE_DIR
                                    VERSION_VAR )
endif()

if(DLPACK_FOUND)
  set(DLPACK_INCLUDE_DIRS ${DLPACK_INCLUDE_DIR})

  if(NOT DLPACK_LIBRARIES)
    set(DLPACK_LIBRARIES ${DLPACK_LIBRARY})
  endif()

  if(NOT TARGET DLPACK::DLPACK)
    add_library(DLPACK::DLPACK UNKNOWN IMPORTED GLOBAL)
    set_target_properties(DLPACK::DLPACK PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${DLPACK_INCLUDE_DIRS}")

    if(DLPACK_LIBRARY_RELEASE)
      set_property(TARGET DLPACK::DLPACK APPEND PROPERTY
        IMPORTED_CONFIGURATIONS RELEASE)
      set_target_properties(DLPACK::DLPACK PROPERTIES
        IMPORTED_LOCATION_RELEASE "${DLPACK_LIBRARY_RELEASE}")
    endif()

    if(DLPACK_LIBRARY_DEBUG)
      set_property(TARGET DLPACK::DLPACK APPEND PROPERTY
        IMPORTED_CONFIGURATIONS DEBUG)
      set_target_properties(DLPACK::DLPACK PROPERTIES
        IMPORTED_LOCATION_DEBUG "${DLPACK_LIBRARY_DEBUG}")
    endif()

    if(NOT DLPACK_LIBRARY_RELEASE AND NOT DLPACK_LIBRARY_DEBUG)
      set_property(TARGET DLPACK::DLPACK APPEND PROPERTY
        IMPORTED_LOCATION "${DLPACK_LIBRARY}")
    endif()
  endif()
endif()



unset(DLPACK_NO_CONFIG)
unset(DLPACK_IS_HEADER_ONLY)
