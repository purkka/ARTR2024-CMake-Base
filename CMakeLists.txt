cmake_minimum_required(VERSION 3.13)
project(ARTR2024)

if (MSVC)
    # <ranges> support requires /std:c++latest on MSVC
    set(CMAKE_CXX_STANDARD 23)
else (MSVC)
    set(CMAKE_CXX_STANDARD 20)
endif (MSVC)

include(FetchContent)

option(autoVkToolkitStarter_CreateDependencySymlinks
        "Create symbolic links instead of copying dependencies of examples, i.e. DLLs (Windows only) & assets."
        ON)

# Dependencies
## Auto-Vk-Toolkit
include(cmake/Auto-Vk-Toolkit.cmake)

# Assets
include(${Auto_Vk_Toolkit_SOURCE_DIR}/cmake/post_build_helper/add_post_build_commands.cmake)

# Assignments
include(cmake/Assignment1.cmake)

