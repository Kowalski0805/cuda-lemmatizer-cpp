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


# CPM Search for bs_thread_pool
#
# Make sure we search for a build-dir config module for the CPM project
set(possible_package_dir "/home/kowalski0805/UANLP/cuda_exe/_deps/bs_thread_pool-build")
if(possible_package_dir AND NOT DEFINED bs_thread_pool_DIR)
  set(bs_thread_pool_DIR "${possible_package_dir}")
endif()

CPMFindPackage(
  "NAME;bs_thread_pool;VERSION;4.1.0;GIT_REPOSITORY;https://github.com/bshoshany/thread-pool.git;GIT_TAG;097aa718f25d44315cadb80b407144ad455ee4f9;GIT_SHALLOW;ON;EXCLUDE_FROM_ALL;OFF;DOWNLOAD_ONLY;ON"
  )

if(possible_package_dir)
  unset(possible_package_dir)
endif()
#=============================================================================
