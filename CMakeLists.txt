cmake_minimum_required(VERSION 3.16)

project(RayTracingCuda CUDA CXX C)  # enable cuda language

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} "-g -G")  # enable cuda-gdb
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/$<CONFIG> CACHE STRING "")

add_definitions(-DUSE_CUDA)

add_subdirectory(externalLibs/freeglut)
include_directories(externalLibs/freeglut/include)

add_library(glew SHARED IMPORTED)
target_include_directories(glew INTERFACE externalLibs/glew-2.1.0/include)
find_library(glew_LIB NAMES glew32 HINTS ${CMAKE_CURRENT_SOURCE_DIR}/externalLibs PATH_SUFFIXES glew-2.1.0/lib/Release/x64)
set_target_properties(glew PROPERTIES IMPORTED_IMPLIB ${glew_LIB})

add_library(
        imgui STATIC 
        externalLibs/imgui/imgui.cpp
        externalLibs/imgui/imgui_demo.cpp
        externalLibs/imgui/imgui_draw.cpp
        externalLibs/imgui/imgui_tables.cpp
        externalLibs/imgui/imgui_widgets.cpp
        externalLibs/imgui/backends/imgui_impl_glut.cpp
        externalLibs/imgui/backends/imgui_impl_opengl2.cpp
)
target_include_directories(imgui PUBLIC externalLibs/imgui externalLibs/freeglut/include externalLibs/imgui/backends)

add_executable(${PROJECT_NAME} src/main.cu src/imgui_app.cpp src/texture.cpp)
target_include_directories(${PROJECT_NAME} PRIVATE src externalLibs/glm externalLibs/stb externalLibs/imgui-filebrowser)
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES OFF)
target_link_libraries(${PROJECT_NAME} PUBLIC freeglut glew imgui)

add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/data/ ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/data
)

get_target_property(freeglut_DLL_DIR freeglut RUNTIME_OUTPUT_DIRECTORY)
find_file(freeglut_DLL name freeglut.dll PATHS ${freeglut_DLL_DIR} PATH_SUFFIXES Debug Release)
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy ${freeglut_DLL} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
)
find_file(glew_DLL NAMES glew32.dll PATHS ${CMAKE_CURRENT_SOURCE_DIR}/externalLibs PATH_SUFFIXES glew-2.1.0/bin/Release/x64 NO_SYSTEM_ENVIRONMENT_PATH)
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy ${glew_DLL} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
)