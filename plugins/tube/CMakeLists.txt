mint_add_plugin(tube
    NAME Vesta
    CODE Mtub
    CATEGORY Distortion
    SOURCES TubePlugin.cpp)

# 以下两个目标只供 tests/ 使用，默认不参与构建。

# 把声音模型导出成 DLL，让 Python 能测传输曲线和未做抗混叠的基线。
add_library(mint_tube_model_bench SHARED EXCLUDE_FROM_ALL tests/model_bench.cpp)
target_include_directories(mint_tube_model_bench PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")

# mint_probe 覆盖不到的项目：单声道、运行中切换、自动化、非法输入、CPU。
juce_add_console_app(mint_tube_host_test PRODUCT_NAME "mint_tube_host_test")
target_sources(mint_tube_host_test PRIVATE tests/host_test.cpp)
target_compile_definitions(mint_tube_host_test PRIVATE
    JUCE_PLUGINHOST_VST3=1
    JUCE_MODAL_LOOPS_PERMITTED=1
    JUCE_WEB_BROWSER=0
    JUCE_USE_CURL=0
    JUCE_DISPLAY_SPLASH_SCREEN=0)
target_link_libraries(mint_tube_host_test PRIVATE
    juce::juce_audio_utils
    juce::juce_audio_processors
    juce::juce_recommended_config_flags
    juce::juce_recommended_warning_flags)
set_target_properties(mint_tube_host_test PROPERTIES EXCLUDE_FROM_ALL TRUE)
