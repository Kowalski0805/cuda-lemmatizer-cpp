include("/home/kowalski0805/UANLP/cuda_exe/cmake/CPM_0.40.0.cmake")
CPMAddPackage("NAME;bs_thread_pool;VERSION;4.1.0;GIT_REPOSITORY;https://github.com/bshoshany/thread-pool.git;GIT_TAG;097aa718f25d44315cadb80b407144ad455ee4f9;GIT_SHALLOW;ON;EXCLUDE_FROM_ALL;OFF;DOWNLOAD_ONLY;ON")
set(bs_thread_pool_FOUND TRUE)