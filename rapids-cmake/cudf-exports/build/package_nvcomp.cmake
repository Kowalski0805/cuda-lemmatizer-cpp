#=============================================================================
# find_dependency Search for nvcomp
#
# Make sure we search for a build-dir config module for the project
set(possible_package_dir "/home/kowalski0805/UANLP/cuda_exe/_deps/nvcomp_proprietary_binary-src/lib/cmake/nvcomp")
if(possible_package_dir AND NOT DEFINED nvcomp_DIR)
  set(nvcomp_DIR "${possible_package_dir}")
endif()

find_package(nvcomp 4.2.0.11 QUIET)
find_dependency(nvcomp)

if(possible_package_dir)
  unset(possible_package_dir)
endif()
#=============================================================================
