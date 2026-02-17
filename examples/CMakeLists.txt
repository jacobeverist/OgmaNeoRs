cmake_minimum_required(VERSION 3.13)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(AOgmaNeo_Examples)

# AOgmaNeo source and library paths (parent project)
set(AOGMANEO_DIR "${CMAKE_CURRENT_SOURCE_DIR}/..")
set(AOGMANEO_SOURCE_DIR "${AOGMANEO_DIR}/source")

add_compile_definitions(USE_OMP)
add_compile_definitions(USE_STD_MATH)

find_package(OpenMP REQUIRED)

add_executable(cartpole cartpole.cpp)

target_include_directories(cartpole PRIVATE "${AOGMANEO_SOURCE_DIR}")
target_link_libraries(cartpole "${AOGMANEO_DIR}/build/libAOgmaNeo.a" OpenMP::OpenMP_CXX)

# On macOS, ad-hoc sign executables so they aren't killed by Gatekeeper
if(APPLE)
    add_custom_command(TARGET cartpole POST_BUILD
        COMMAND codesign -f -s - $<TARGET_FILE:cartpole>
        COMMENT "Ad-hoc signing cartpole"
    )
endif()
