# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The program to use to edit the cache.
CMAKE_EDIT_COMMAND = /usr/bin/cmake-gui

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/jpl/ORB_SLAM2/Thirdparty/g2o

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/jpl/ORB_SLAM2/Thirdparty/g2o/build

# Include any dependencies generated for this target.
include g2o/solvers/eigen/CMakeFiles/solver_eigen.dir/depend.make

# Include the progress variables for this target.
include g2o/solvers/eigen/CMakeFiles/solver_eigen.dir/progress.make

# Include the compile flags for this target's objects.
include g2o/solvers/eigen/CMakeFiles/solver_eigen.dir/flags.make

g2o/solvers/eigen/CMakeFiles/solver_eigen.dir/solver_eigen.cpp.o: g2o/solvers/eigen/CMakeFiles/solver_eigen.dir/flags.make
g2o/solvers/eigen/CMakeFiles/solver_eigen.dir/solver_eigen.cpp.o: ../g2o/solvers/eigen/solver_eigen.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/jpl/ORB_SLAM2/Thirdparty/g2o/build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object g2o/solvers/eigen/CMakeFiles/solver_eigen.dir/solver_eigen.cpp.o"
	cd /home/jpl/ORB_SLAM2/Thirdparty/g2o/build/g2o/solvers/eigen && /usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/solver_eigen.dir/solver_eigen.cpp.o -c /home/jpl/ORB_SLAM2/Thirdparty/g2o/g2o/solvers/eigen/solver_eigen.cpp

g2o/solvers/eigen/CMakeFiles/solver_eigen.dir/solver_eigen.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/solver_eigen.dir/solver_eigen.cpp.i"
	cd /home/jpl/ORB_SLAM2/Thirdparty/g2o/build/g2o/solvers/eigen && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/jpl/ORB_SLAM2/Thirdparty/g2o/g2o/solvers/eigen/solver_eigen.cpp > CMakeFiles/solver_eigen.dir/solver_eigen.cpp.i

g2o/solvers/eigen/CMakeFiles/solver_eigen.dir/solver_eigen.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/solver_eigen.dir/solver_eigen.cpp.s"
	cd /home/jpl/ORB_SLAM2/Thirdparty/g2o/build/g2o/solvers/eigen && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/jpl/ORB_SLAM2/Thirdparty/g2o/g2o/solvers/eigen/solver_eigen.cpp -o CMakeFiles/solver_eigen.dir/solver_eigen.cpp.s

g2o/solvers/eigen/CMakeFiles/solver_eigen.dir/solver_eigen.cpp.o.requires:
.PHONY : g2o/solvers/eigen/CMakeFiles/solver_eigen.dir/solver_eigen.cpp.o.requires

g2o/solvers/eigen/CMakeFiles/solver_eigen.dir/solver_eigen.cpp.o.provides: g2o/solvers/eigen/CMakeFiles/solver_eigen.dir/solver_eigen.cpp.o.requires
	$(MAKE) -f g2o/solvers/eigen/CMakeFiles/solver_eigen.dir/build.make g2o/solvers/eigen/CMakeFiles/solver_eigen.dir/solver_eigen.cpp.o.provides.build
.PHONY : g2o/solvers/eigen/CMakeFiles/solver_eigen.dir/solver_eigen.cpp.o.provides

g2o/solvers/eigen/CMakeFiles/solver_eigen.dir/solver_eigen.cpp.o.provides.build: g2o/solvers/eigen/CMakeFiles/solver_eigen.dir/solver_eigen.cpp.o

# Object files for target solver_eigen
solver_eigen_OBJECTS = \
"CMakeFiles/solver_eigen.dir/solver_eigen.cpp.o"

# External object files for target solver_eigen
solver_eigen_EXTERNAL_OBJECTS =

../lib/libg2o_solver_eigen.so: g2o/solvers/eigen/CMakeFiles/solver_eigen.dir/solver_eigen.cpp.o
../lib/libg2o_solver_eigen.so: g2o/solvers/eigen/CMakeFiles/solver_eigen.dir/build.make
../lib/libg2o_solver_eigen.so: ../lib/libg2o_core.so
../lib/libg2o_solver_eigen.so: ../lib/libg2o_stuff.so
../lib/libg2o_solver_eigen.so: g2o/solvers/eigen/CMakeFiles/solver_eigen.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX shared library ../../../../lib/libg2o_solver_eigen.so"
	cd /home/jpl/ORB_SLAM2/Thirdparty/g2o/build/g2o/solvers/eigen && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/solver_eigen.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
g2o/solvers/eigen/CMakeFiles/solver_eigen.dir/build: ../lib/libg2o_solver_eigen.so
.PHONY : g2o/solvers/eigen/CMakeFiles/solver_eigen.dir/build

g2o/solvers/eigen/CMakeFiles/solver_eigen.dir/requires: g2o/solvers/eigen/CMakeFiles/solver_eigen.dir/solver_eigen.cpp.o.requires
.PHONY : g2o/solvers/eigen/CMakeFiles/solver_eigen.dir/requires

g2o/solvers/eigen/CMakeFiles/solver_eigen.dir/clean:
	cd /home/jpl/ORB_SLAM2/Thirdparty/g2o/build/g2o/solvers/eigen && $(CMAKE_COMMAND) -P CMakeFiles/solver_eigen.dir/cmake_clean.cmake
.PHONY : g2o/solvers/eigen/CMakeFiles/solver_eigen.dir/clean

g2o/solvers/eigen/CMakeFiles/solver_eigen.dir/depend:
	cd /home/jpl/ORB_SLAM2/Thirdparty/g2o/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/jpl/ORB_SLAM2/Thirdparty/g2o /home/jpl/ORB_SLAM2/Thirdparty/g2o/g2o/solvers/eigen /home/jpl/ORB_SLAM2/Thirdparty/g2o/build /home/jpl/ORB_SLAM2/Thirdparty/g2o/build/g2o/solvers/eigen /home/jpl/ORB_SLAM2/Thirdparty/g2o/build/g2o/solvers/eigen/CMakeFiles/solver_eigen.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : g2o/solvers/eigen/CMakeFiles/solver_eigen.dir/depend

