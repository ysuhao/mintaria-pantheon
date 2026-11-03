mint_add_plugin(comp
    NAME Atlas
    CODE Mcmp
    CATEGORY Dynamics
    SOURCES AtlasPlugin.cpp)

# 黑盒对比用的小工具，按子插件名从外壳型 VST3 里加载。不进插件产物，只在单独构建本插件时生成。
if(MINT_ONLY STREQUAL "comp")
    add_subdirectory(tests/shell_probe)
endif()
