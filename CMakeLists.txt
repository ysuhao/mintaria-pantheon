cmake_minimum_required(VERSION 3.22)
project(MintariaPantheon VERSION 1.0.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set_property(GLOBAL PROPERTY USE_FOLDERS YES)
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

if(MSVC)
    add_compile_options(/utf-8 /MP)
endif()

set(MINT_JUCE_DIR "C:/JUCE" CACHE PATH "Path to the JUCE 8 checkout")
set(MINT_ONLY "" CACHE STRING "Build only this plugin slug (e.g. gate). Empty builds every plugin")

add_subdirectory("${MINT_JUCE_DIR}" juce EXCLUDE_FROM_ALL)

include(cmake/MintPlugin.cmake)

add_subdirectory(tools/mint_probe)

if(MINT_ONLY)
    add_subdirectory("plugins/${MINT_ONLY}")
else()
    file(GLOB mint_plugin_lists CONFIGURE_DEPENDS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "plugins/*/CMakeLists.txt")
    foreach(mint_list IN LISTS mint_plugin_lists)
        get_filename_component(mint_dir "${mint_list}" DIRECTORY)
        add_subdirectory("${mint_dir}")
    endforeach()
endif()
