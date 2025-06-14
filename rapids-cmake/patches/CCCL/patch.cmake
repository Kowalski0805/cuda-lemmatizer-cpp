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

set(msg_state)
set(error_state)
function(rapids_cpm_run_git_patch file issue require)
  set(git_command /usr/bin/git)
  cmake_path(GET file FILENAME file_name)
  cmake_path(GET file_name EXTENSION LAST_ONLY ext)
  string(SUBSTRING "${ext}" 1 -1 ext)

  if(NOT (ext STREQUAL "diff" OR ext STREQUAL "patch") )
    list(APPEND msg_state "rapids-cmake: Unable to apply ${file} as ${ext} is unsupported. Only .diff and .patch are supported")
    set(msg_state ${msg_state} PARENT_SCOPE)
    return()
  endif()

  set(command apply)
  set(args)
  if(ext STREQUAL "patch")
    set(command am)
    set(args -3)
  endif()

  set(result 1)
  if(ext STREQUAL "diff")
    execute_process(
      COMMAND ${git_command} apply --whitespace=fix ${file}
      RESULT_VARIABLE result
      ERROR_VARIABLE repo_error_info
    )
    if(NOT result EQUAL 0)
      # See if the diff was previously applied
      execute_process(
        COMMAND ${git_command} apply --reverse --check ${file}
        RESULT_VARIABLE result
      )
    endif()
  elseif(ext STREQUAL "patch")
    # Setup a fake committer name/email so we execute everywhere
    set(ENV{GIT_COMMITTER_NAME} "rapids-cmake" )
    set(ENV{GIT_COMMITTER_EMAIL} "rapids.cmake@rapids.ai" )
    # no need to check if the git patch was already applied
    # `am` does that and returns a success error code for those cases
    execute_process(
      COMMAND ${git_command} am -3 ${file}
      RESULT_VARIABLE result
      ERROR_VARIABLE repo_error_info
    )
  endif()

  # Setup where we log error message too
  set(error_msg_var msg_state)
  if(require)
    set(error_msg_var error_state)
  endif()

  if(result EQUAL 0)
    list(APPEND msg_state "rapids-cmake [CCCL]: applied ${ext} ${file_name} to fix issue: '${issue}'\n")
  else()
    list(APPEND ${error_msg_var} "rapids-cmake [CCCL]: failed to apply ${ext} ${file_name}\n")
    list(APPEND ${error_msg_var} "rapids-cmake [CCCL]: git ${ext} output: ${repo_error_info}\n")
  endif()
  set(msg_state ${msg_state} PARENT_SCOPE)
  set(error_state ${error_state} PARENT_SCOPE)
endfunction()

# We want to ensure that any patched files have a timestamp
# that is at least 1 second newer compared to the git checkout
# This ensures that all of CMake up-to-date install logic
# considers these files as modified.
#
# This ensures that if our patch contains additional install rules
# they will execute even when an existing install rule exists
# with the same destination ( and our patch is listed last ).
execute_process(COMMAND ${CMAKE_COMMAND} -E sleep 1)

set(files "/home/kowalski0805/UANLP/cuda_exe/_deps/cudf-src/cpp/cmake/thirdparty/patches/thrust_faster_sort_compile_times.diff;/home/kowalski0805/UANLP/cuda_exe/_deps/cudf-src/cpp/cmake/thirdparty/patches/thrust_faster_scan_compile_times.diff")
set(issues "Improve Thrust sort compile times by not unrolling loops for inlined comparators [https://github.com/rapidsai/cudf/pull/10577];Improve Thrust scan compile times by reducing the number of kernels generated [https://github.com/rapidsai/cudf/pull/8183]")
set(required "FALSE;FALSE")
set(output_file "/home/kowalski0805/UANLP/cuda_exe/rapids-cmake/patches/CCCL/log")
set(error_file "/home/kowalski0805/UANLP/cuda_exe/rapids-cmake/patches/CCCL/err")
foreach(file issue require IN ZIP_LISTS files issues required)
  rapids_cpm_run_git_patch(${file} ${issue} ${require})
endforeach()
if(msg_state)
  file(WRITE "${output_file}" ${msg_state})
endif()
if(error_state)
  file(WRITE "${error_file}" ${error_state})
endif()
