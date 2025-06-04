include("/home/kowalski0805/UANLP/cuda_exe/cmake/CPM_0.40.0.cmake")
CPMAddPackage("NAME;dlpack;VERSION;0.8;GIT_REPOSITORY;https://github.com/dmlc/dlpack.git;GIT_TAG;v0.8;GIT_SHALLOW;TRUE;DOWNLOAD_ONLY;TRUE;OPTIONS;BUILD_MOCK OFF")
set(dlpack_FOUND TRUE)