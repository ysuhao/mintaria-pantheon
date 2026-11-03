mint_add_plugin(eq
    NAME Iris
    CODE Meqz
    CATEGORY EQ
    SOURCES IrisPlugin.cpp IrisEditor.cpp)

# 验证工具：把 VST3 当黑盒加载，做参数自动化、CPU 计时、带音频截图和模拟鼠标。
juce_add_console_app(mint_eq_bench PRODUCT_NAME "mint_eq_bench")

target_sources(mint_eq_bench PRIVATE
    tests/iris_bench.cpp
    "${CMAKE_SOURCE_DIR}/tools/mint_probe/capture_win.cpp")

target_include_directories(mint_eq_bench PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")

target_compile_definitions(mint_eq_bench PRIVATE
    JUCE_PLUGINHOST_VST3=1
    JUCE_MODAL_LOOPS_PERMITTED=1
    JUCE_WEB_BROWSER=0
    JUCE_USE_CURL=0
    JUCE_DISPLAY_SPLASH_SCREEN=0)

target_link_libraries(mint_eq_bench PRIVATE
    juce::juce_audio_utils
    juce::juce_audio_processors
    juce::juce_recommended_config_flags
    juce::juce_recommended_warning_flags)
