project(Assignment1)

add_executable(${PROJECT_NAME}
        host_code/assignment1.cpp
        host_code/utils/simple_geometry.cpp)

target_include_directories(${PROJECT_NAME} PUBLIC
        shaders/)

target_include_directories(${PROJECT_NAME} PUBLIC Auto_Vk_Toolkit)
target_link_libraries(${PROJECT_NAME} PUBLIC Auto_Vk_Toolkit)

get_target_property(autoVkToolkitStarter_BINARY_DIR ${PROJECT_NAME} BINARY_DIR)

add_post_build_commands(${PROJECT_NAME}
        ${PROJECT_SOURCE_DIR}/shaders
        ${autoVkToolkitStarter_BINARY_DIR}/shaders
        $<TARGET_FILE_DIR:${PROJECT_NAME}>/assets
        "${Auto_Vk_Toolkit_SOURCE_DIR}/assets"
        ${autoVkToolkitStarter_CreateDependencySymlinks})
