include("/home/kowalski0805/UANLP/cuda_exe/cmake/CPM_0.40.0.cmake")
CPMAddPackage("NAME;nvtx3;VERSION;3.1.0;GIT_REPOSITORY;https://github.com/NVIDIA/NVTX.git;GIT_TAG;v3.1.0;GIT_SHALLOW;ON;SOURCE_SUBDIR;c;EXCLUDE_FROM_ALL;OFF")
set(nvtx3_FOUND TRUE)