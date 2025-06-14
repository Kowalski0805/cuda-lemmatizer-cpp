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


# CPM Search for nvcomp
#
# Make sure we search for a build-dir config module for the CPM project
set(possible_package_dir "/home/kowalski0805/UANLP/cuda_exe/_deps/nvcomp_proprietary_binary-src/lib/cmake/nvcomp")
if(possible_package_dir AND NOT DEFINED nvcomp_DIR)
  set(nvcomp_DIR "${possible_package_dir}")
endif()

CPMFindPackage(
  "NAME;nvcomp;VERSION;4.2.0.11;GIT_REPOSITORY;https://github.com/NVIDIA/nvcomp.git;GIT_TAG;v2.2.0;GIT_SHALLOW;ON;EXCLUDE_FROM_ALL;OFF;OPTIONS;BUILD_STATIC ON;BUILD_TESTS OFF;BUILD_BENCHMARKS OFF;BUILD_EXAMPLES OFF"
  )

if(possible_package_dir)
  unset(possible_package_dir)
endif()
#=============================================================================
