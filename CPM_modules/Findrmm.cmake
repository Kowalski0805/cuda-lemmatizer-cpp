include("/home/kowalski0805/UANLP/cuda_exe/cmake/CPM_0.40.0.cmake")
CPMAddPackage("NAME;rmm;VERSION;25.04;GIT_REPOSITORY;https://github.com/rapidsai/rmm.git;GIT_TAG;branch-25.04;GIT_SHALLOW;ON;EXCLUDE_FROM_ALL;OFF;OPTIONS;BUILD_TESTS OFF;BUILD_BENCHMARKS OFF")
set(rmm_FOUND TRUE)