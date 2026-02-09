cmake_minimum_required(VERSION 3.16)

project(ml_from_scratch
    VERSION 0.1
    LANGUAGES CXX
)

# Use modern C++
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Enable warnings ()
if (MSVC)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Optional: OpenMP ()
find_package(OpenMP)
if (OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found")
endif()

# Source files
add_executable(ml
    src/main.cpp
    src/conv2d.cpp
    src/relu.cpp
    src/layer.cpp
)

# Include headers
target_include_directories(ml
    PRIVATE
        ${PROJECT_SOURCE_DIR}/src
)

# Link OpenMP if available
if (OpenMP_CXX_FOUND)
    target_link_libraries(ml PRIVATE OpenMP::OpenMP_CXX)
endif()

# Set working directory so relative dataset paths resolve correctly
set_target_properties(ml PROPERTIES
    VS_DEBUGGER_WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
)

# Copy datasets directory next to the binary for command-line runs
add_custom_command(TARGET ml POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${PROJECT_SOURCE_DIR}/datasets"
        "$<TARGET_FILE_DIR:ml>/datasets"
    COMMENT "Copying datasets to output directory"
)
