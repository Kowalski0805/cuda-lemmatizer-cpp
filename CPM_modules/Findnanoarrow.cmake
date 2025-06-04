include("/home/kowalski0805/UANLP/cuda_exe/cmake/CPM_0.40.0.cmake")
CPMAddPackage("NAME;nanoarrow;VERSION;0.7.0.dev;GIT_REPOSITORY;https://github.com/apache/arrow-nanoarrow.git;GIT_TAG;4bf5a9322626e95e3717e43de7616c0a256179eb;GIT_SHALLOW;FALSE;OPTIONS;BUILD_SHARED_LIBS OFF;NANOARROW_NAMESPACE cudf;EXCLUDE_FROM_ALL;TRUE")
set(nanoarrow_FOUND TRUE)