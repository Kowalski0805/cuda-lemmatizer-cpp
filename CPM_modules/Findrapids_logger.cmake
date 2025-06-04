include("/home/kowalski0805/UANLP/cuda_exe/cmake/CPM_0.40.0.cmake")
CPMAddPackage("NAME;rapids_logger;VERSION;0.1.0;GIT_REPOSITORY;https://github.com/rapidsai/rapids-logger.git;GIT_TAG;46070bb255482f0782ca840ae45de9354380e298;GIT_SHALLOW;OFF;OPTIONS;BUILD_TESTS OFF")
set(rapids_logger_FOUND TRUE)