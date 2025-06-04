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


# CPM Search for rapids_logger
#
# Make sure we search for a build-dir config module for the CPM project
set(possible_package_dir "/home/kowalski0805/UANLP/cuda_exe/_deps/rapids_logger-build")
if(possible_package_dir AND NOT DEFINED rapids_logger_DIR)
  set(rapids_logger_DIR "${possible_package_dir}")
endif()

CPMFindPackage(
  "NAME;rapids_logger;VERSION;0.1.0;GIT_REPOSITORY;https://github.com/rapidsai/rapids-logger.git;GIT_TAG;46070bb255482f0782ca840ae45de9354380e298;GIT_SHALLOW;OFF;OPTIONS;BUILD_TESTS OFF"
  )

if(possible_package_dir)
  unset(possible_package_dir)
endif()
#=============================================================================
