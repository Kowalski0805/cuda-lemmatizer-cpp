include("/home/kowalski0805/UANLP/cuda_exe/cmake/CPM_0.40.0.cmake")
CPMAddPackage("NAME;kvikio;VERSION;25.04;GIT_REPOSITORY;https://github.com/rapidsai/kvikio.git;GIT_TAG;branch-25.04;GIT_SHALLOW;TRUE;SOURCE_SUBDIR;cpp;OPTIONS;KvikIO_BUILD_EXAMPLES OFF;KvikIO_REMOTE_SUPPORT OFF")
set(kvikio_FOUND TRUE)