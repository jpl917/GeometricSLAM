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
include g2o/examples/sba/CMakeFiles/sba_demo.dir/depend.make

# Include the progress variables for this target.
include g2o/examples/sba/CMakeFiles/sba_demo.dir/progress.make

# Include the compile flags for this target's objects.
include g2o/examples/sba/CMakeFiles/sba_demo.dir/flags.make

g2o/examples/sba/CMakeFiles/sba_demo.dir/sba_demo.cpp.o: g2o/examples/sba/CMakeFiles/sba_demo.dir/flags.make
g2o/examples/sba/CMakeFiles/sba_demo.dir/sba_demo.cpp.o: ../g2o/examples/sba/sba_demo.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/jpl/ORB_SLAM2/Thirdparty/g2o/build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object g2o/examples/sba/CMakeFiles/sba_demo.dir/sba_demo.cpp.o"
	cd /home/jpl/ORB_SLAM2/Thirdparty/g2o/build/g2o/examples/sba && /usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/sba_demo.dir/sba_demo.cpp.o -c /home/jpl/ORB_SLAM2/Thirdparty/g2o/g2o/examples/sba/sba_demo.cpp

g2o/examples/sba/CMakeFiles/sba_demo.dir/sba_demo.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/sba_demo.dir/sba_demo.cpp.i"
	cd /home/jpl/ORB_SLAM2/Thirdparty/g2o/build/g2o/examples/sba && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/jpl/ORB_SLAM2/Thirdparty/g2o/g2o/examples/sba/sba_demo.cpp > CMakeFiles/sba_demo.dir/sba_demo.cpp.i

g2o/examples/sba/CMakeFiles/sba_demo.dir/sba_demo.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/sba_demo.dir/sba_demo.cpp.s"
	cd /home/jpl/ORB_SLAM2/Thirdparty/g2o/build/g2o/examples/sba && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/jpl/ORB_SLAM2/Thirdparty/g2o/g2o/examples/sba/sba_demo.cpp -o CMakeFiles/sba_demo.dir/sba_demo.cpp.s

g2o/examples/sba/CMakeFiles/sba_demo.dir/sba_demo.cpp.o.requires:
.PHONY : g2o/examples/sba/CMakeFiles/sba_demo.dir/sba_demo.cpp.o.requires

g2o/examples/sba/CMakeFiles/sba_demo.dir/sba_demo.cpp.o.provides: g2o/examples/sba/CMakeFiles/sba_demo.dir/sba_demo.cpp.o.requires
	$(MAKE) -f g2o/examples/sba/CMakeFiles/sba_demo.dir/build.make g2o/examples/sba/CMakeFiles/sba_demo.dir/sba_demo.cpp.o.provides.build
.PHONY : g2o/examples/sba/CMakeFiles/sba_demo.dir/sba_demo.cpp.o.provides

g2o/examples/sba/CMakeFiles/sba_demo.dir/sba_demo.cpp.o.provides.build: g2o/examples/sba/CMakeFiles/sba_demo.dir/sba_demo.cpp.o

# Object files for target sba_demo
sba_demo_OBJECTS = \
"CMakeFiles/sba_demo.dir/sba_demo.cpp.o"

# External object files for target sba_demo
sba_demo_EXTERNAL_OBJECTS =

../bin/sba_demo: g2o/examples/sba/CMakeFiles/sba_demo.dir/sba_demo.cpp.o
../bin/sba_demo: g2o/examples/sba/CMakeFiles/sba_demo.dir/build.make
../bin/sba_demo: ../lib/libg2o_core.so
../bin/sba_demo: ../lib/libg2o_types_icp.so
../bin/sba_demo: ../lib/libg2o_types_sba.so
../bin/sba_demo: ../lib/libg2o_solver_cholmod.so
../bin/sba_demo: ../lib/libg2o_types_slam3d.so
../bin/sba_demo: ../lib/libg2o_opengl_helper.so
../bin/sba_demo: /usr/lib/x86_64-linux-gnu/libGLU.so
../bin/sba_demo: /usr/lib/x86_64-linux-gnu/libSM.so
../bin/sba_demo: /usr/lib/x86_64-linux-gnu/libICE.so
../bin/sba_demo: /usr/lib/x86_64-linux-gnu/libX11.so
../bin/sba_demo: /usr/lib/x86_64-linux-gnu/libXext.so
../bin/sba_demo: /usr/lib/x86_64-linux-gnu/libGL.so
../bin/sba_demo: ../lib/libg2o_core.so
../bin/sba_demo: ../lib/libg2o_stuff.so
../bin/sba_demo: /usr/lib/x86_64-linux-gnu/libcholmod.so
../bin/sba_demo: /usr/lib/x86_64-linux-gnu/libamd.so
../bin/sba_demo: /usr/lib/x86_64-linux-gnu/libcolamd.so
../bin/sba_demo: /usr/lib/x86_64-linux-gnu/libcamd.so
../bin/sba_demo: /usr/lib/x86_64-linux-gnu/libccolamd.so
../bin/sba_demo: /usr/lib/x86_64-linux-gnu/libsuitesparseconfig.a
../bin/sba_demo: /usr/lib/libcblas.so
../bin/sba_demo: /usr/lib/libf77blas.so
../bin/sba_demo: /usr/lib/libatlas.so
../bin/sba_demo: /usr/lib/liblapack.so
../bin/sba_demo: g2o/examples/sba/CMakeFiles/sba_demo.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable ../../../../bin/sba_demo"
	cd /home/jpl/ORB_SLAM2/Thirdparty/g2o/build/g2o/examples/sba && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/sba_demo.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
g2o/examples/sba/CMakeFiles/sba_demo.dir/build: ../bin/sba_demo
.PHONY : g2o/examples/sba/CMakeFiles/sba_demo.dir/build

g2o/examples/sba/CMakeFiles/sba_demo.dir/requires: g2o/examples/sba/CMakeFiles/sba_demo.dir/sba_demo.cpp.o.requires
.PHONY : g2o/examples/sba/CMakeFiles/sba_demo.dir/requires

g2o/examples/sba/CMakeFiles/sba_demo.dir/clean:
	cd /home/jpl/ORB_SLAM2/Thirdparty/g2o/build/g2o/examples/sba && $(CMAKE_COMMAND) -P CMakeFiles/sba_demo.dir/cmake_clean.cmake
.PHONY : g2o/examples/sba/CMakeFiles/sba_demo.dir/clean

g2o/examples/sba/CMakeFiles/sba_demo.dir/depend:
	cd /home/jpl/ORB_SLAM2/Thirdparty/g2o/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/jpl/ORB_SLAM2/Thirdparty/g2o /home/jpl/ORB_SLAM2/Thirdparty/g2o/g2o/examples/sba /home/jpl/ORB_SLAM2/Thirdparty/g2o/build /home/jpl/ORB_SLAM2/Thirdparty/g2o/build/g2o/examples/sba /home/jpl/ORB_SLAM2/Thirdparty/g2o/build/g2o/examples/sba/CMakeFiles/sba_demo.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : g2o/examples/sba/CMakeFiles/sba_demo.dir/depend

