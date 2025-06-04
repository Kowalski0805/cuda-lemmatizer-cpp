include("/home/kowalski0805/UANLP/cuda_exe/cmake/CPM_0.40.0.cmake")
CPMAddPackage("NAME;jitify;VERSION;2.0.0;GIT_REPOSITORY;https://github.com/rapidsai/jitify.git;GIT_TAG;jitify2;GIT_SHALLOW;TRUE;DOWNLOAD_ONLY;TRUE")
set(jitify_FOUND TRUE)