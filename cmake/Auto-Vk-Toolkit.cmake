cmake_minimum_required(VERSION 3.14)

include(FetchContent)

set(avk_UseVMA ON)

FetchContent_Declare(
    Auto_Vk_Toolkit
    GIT_REPOSITORY      https://github.com/cg-tuwien/Auto-Vk-Toolkit.git
    GIT_TAG             artr2024_assignment1
    GIT_SUBMODULES      "auto_vk"
)

FetchContent_MakeAvailable(Auto_Vk_Toolkit)
