# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


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

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/xiachunwei/Projects/tensor-compiler-gpu

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/xiachunwei/Projects/tensor-compiler-gpu/src

# Include any dependencies generated for this target.
include CMakeFiles/sobel.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/sobel.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/sobel.dir/flags.make

CMakeFiles/sobel.dir/sobel.cu.o: CMakeFiles/sobel.dir/flags.make
CMakeFiles/sobel.dir/sobel.cu.o: sobel.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/xiachunwei/Projects/tensor-compiler-gpu/src/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object CMakeFiles/sobel.dir/sobel.cu.o"
	/usr/local/cuda-11.0/bin/nvcc  $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -x cu -c /home/xiachunwei/Projects/tensor-compiler-gpu/src/sobel.cu -o CMakeFiles/sobel.dir/sobel.cu.o

CMakeFiles/sobel.dir/sobel.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/sobel.dir/sobel.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/sobel.dir/sobel.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/sobel.dir/sobel.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

CMakeFiles/sobel.dir/sobel.cu.o.requires:

.PHONY : CMakeFiles/sobel.dir/sobel.cu.o.requires

CMakeFiles/sobel.dir/sobel.cu.o.provides: CMakeFiles/sobel.dir/sobel.cu.o.requires
	$(MAKE) -f CMakeFiles/sobel.dir/build.make CMakeFiles/sobel.dir/sobel.cu.o.provides.build
.PHONY : CMakeFiles/sobel.dir/sobel.cu.o.provides

CMakeFiles/sobel.dir/sobel.cu.o.provides.build: CMakeFiles/sobel.dir/sobel.cu.o


# Object files for target sobel
sobel_OBJECTS = \
"CMakeFiles/sobel.dir/sobel.cu.o"

# External object files for target sobel
sobel_EXTERNAL_OBJECTS =

CMakeFiles/sobel.dir/cmake_device_link.o: CMakeFiles/sobel.dir/sobel.cu.o
CMakeFiles/sobel.dir/cmake_device_link.o: CMakeFiles/sobel.dir/build.make
CMakeFiles/sobel.dir/cmake_device_link.o: CMakeFiles/sobel.dir/dlink.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/xiachunwei/Projects/tensor-compiler-gpu/src/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CUDA device code CMakeFiles/sobel.dir/cmake_device_link.o"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/sobel.dir/dlink.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/sobel.dir/build: CMakeFiles/sobel.dir/cmake_device_link.o

.PHONY : CMakeFiles/sobel.dir/build

# Object files for target sobel
sobel_OBJECTS = \
"CMakeFiles/sobel.dir/sobel.cu.o"

# External object files for target sobel
sobel_EXTERNAL_OBJECTS =

sobel: CMakeFiles/sobel.dir/sobel.cu.o
sobel: CMakeFiles/sobel.dir/build.make
sobel: CMakeFiles/sobel.dir/cmake_device_link.o
sobel: CMakeFiles/sobel.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/xiachunwei/Projects/tensor-compiler-gpu/src/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CUDA executable sobel"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/sobel.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/sobel.dir/build: sobel

.PHONY : CMakeFiles/sobel.dir/build

CMakeFiles/sobel.dir/requires: CMakeFiles/sobel.dir/sobel.cu.o.requires

.PHONY : CMakeFiles/sobel.dir/requires

CMakeFiles/sobel.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/sobel.dir/cmake_clean.cmake
.PHONY : CMakeFiles/sobel.dir/clean

CMakeFiles/sobel.dir/depend:
	cd /home/xiachunwei/Projects/tensor-compiler-gpu/src && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/xiachunwei/Projects/tensor-compiler-gpu /home/xiachunwei/Projects/tensor-compiler-gpu /home/xiachunwei/Projects/tensor-compiler-gpu/src /home/xiachunwei/Projects/tensor-compiler-gpu/src /home/xiachunwei/Projects/tensor-compiler-gpu/src/CMakeFiles/sobel.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/sobel.dir/depend

