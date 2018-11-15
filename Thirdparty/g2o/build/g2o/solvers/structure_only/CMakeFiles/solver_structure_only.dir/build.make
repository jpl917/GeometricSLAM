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
include g2o/solvers/structure_only/CMakeFiles/solver_structure_only.dir/depend.make

# Include the progress variables for this target.
include g2o/solvers/structure_only/CMakeFiles/solver_structure_only.dir/progress.make

# Include the compile flags for this target's objects.
include g2o/solvers/structure_only/CMakeFiles/solver_structure_only.dir/flags.make

g2o/solvers/structure_only/CMakeFiles/solver_structure_only.dir/structure_only.cpp.o: g2o/solvers/structure_only/CMakeFiles/solver_structure_only.dir/flags.make
g2o/solvers/structure_only/CMakeFiles/solver_structure_only.dir/structure_only.cpp.o: ../g2o/solvers/structure_only/structure_only.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/jpl/ORB_SLAM2/Thirdparty/g2o/build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object g2o/solvers/structure_only/CMakeFiles/solver_structure_only.dir/structure_only.cpp.o"
	cd /home/jpl/ORB_SLAM2/Thirdparty/g2o/build/g2o/solvers/structure_only && /usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/solver_structure_only.dir/structure_only.cpp.o -c /home/jpl/ORB_SLAM2/Thirdparty/g2o/g2o/solvers/structure_only/structure_only.cpp

g2o/solvers/structure_only/CMakeFiles/solver_structure_only.dir/structure_only.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/solver_structure_only.dir/structure_only.cpp.i"
	cd /home/jpl/ORB_SLAM2/Thirdparty/g2o/build/g2o/solvers/structure_only && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/jpl/ORB_SLAM2/Thirdparty/g2o/g2o/solvers/structure_only/structure_only.cpp > CMakeFiles/solver_structure_only.dir/structure_only.cpp.i

g2o/solvers/structure_only/CMakeFiles/solver_structure_only.dir/structure_only.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/solver_structure_only.dir/structure_only.cpp.s"
	cd /home/jpl/ORB_SLAM2/Thirdparty/g2o/build/g2o/solvers/structure_only && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/jpl/ORB_SLAM2/Thirdparty/g2o/g2o/solvers/structure_only/structure_only.cpp -o CMakeFiles/solver_structure_only.dir/structure_only.cpp.s

g2o/solvers/structure_only/CMakeFiles/solver_structure_only.dir/structure_only.cpp.o.requires:
.PHONY : g2o/solvers/structure_only/CMakeFiles/solver_structure_only.dir/structure_only.cpp.o.requires

g2o/solvers/structure_only/CMakeFiles/solver_structure_only.dir/structure_only.cpp.o.provides: g2o/solvers/structure_only/CMakeFiles/solver_structure_only.dir/structure_only.cpp.o.requires
	$(MAKE) -f g2o/solvers/structure_only/CMakeFiles/solver_structure_only.dir/build.make g2o/solvers/structure_only/CMakeFiles/solver_structure_only.dir/structure_only.cpp.o.provides.build
.PHONY : g2o/solvers/structure_only/CMakeFiles/solver_structure_only.dir/structure_only.cpp.o.provides

g2o/solvers/structure_only/CMakeFiles/solver_structure_only.dir/structure_only.cpp.o.provides.build: g2o/solvers/structure_only/CMakeFiles/solver_structure_only.dir/structure_only.cpp.o

# Object files for target solver_structure_only
solver_structure_only_OBJECTS = \
"CMakeFiles/solver_structure_only.dir/structure_only.cpp.o"

# External object files for target solver_structure_only
solver_structure_only_EXTERNAL_OBJECTS =

../lib/libg2o_solver_structure_only.so: g2o/solvers/structure_only/CMakeFiles/solver_structure_only.dir/structure_only.cpp.o
../lib/libg2o_solver_structure_only.so: g2o/solvers/structure_only/CMakeFiles/solver_structure_only.dir/build.make
../lib/libg2o_solver_structure_only.so: ../lib/libg2o_core.so
../lib/libg2o_solver_structure_only.so: ../lib/libg2o_stuff.so
../lib/libg2o_solver_structure_only.so: g2o/solvers/structure_only/CMakeFiles/solver_structure_only.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX shared library ../../../../lib/libg2o_solver_structure_only.so"
	cd /home/jpl/ORB_SLAM2/Thirdparty/g2o/build/g2o/solvers/structure_only && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/solver_structure_only.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
g2o/solvers/structure_only/CMakeFiles/solver_structure_only.dir/build: ../lib/libg2o_solver_structure_only.so
.PHONY : g2o/solvers/structure_only/CMakeFiles/solver_structure_only.dir/build

g2o/solvers/structure_only/CMakeFiles/solver_structure_only.dir/requires: g2o/solvers/structure_only/CMakeFiles/solver_structure_only.dir/structure_only.cpp.o.requires
.PHONY : g2o/solvers/structure_only/CMakeFiles/solver_structure_only.dir/requires

g2o/solvers/structure_only/CMakeFiles/solver_structure_only.dir/clean:
	cd /home/jpl/ORB_SLAM2/Thirdparty/g2o/build/g2o/solvers/structure_only && $(CMAKE_COMMAND) -P CMakeFiles/solver_structure_only.dir/cmake_clean.cmake
.PHONY : g2o/solvers/structure_only/CMakeFiles/solver_structure_only.dir/clean

g2o/solvers/structure_only/CMakeFiles/solver_structure_only.dir/depend:
	cd /home/jpl/ORB_SLAM2/Thirdparty/g2o/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/jpl/ORB_SLAM2/Thirdparty/g2o /home/jpl/ORB_SLAM2/Thirdparty/g2o/g2o/solvers/structure_only /home/jpl/ORB_SLAM2/Thirdparty/g2o/build /home/jpl/ORB_SLAM2/Thirdparty/g2o/build/g2o/solvers/structure_only /home/jpl/ORB_SLAM2/Thirdparty/g2o/build/g2o/solvers/structure_only/CMakeFiles/solver_structure_only.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : g2o/solvers/structure_only/CMakeFiles/solver_structure_only.dir/depend

