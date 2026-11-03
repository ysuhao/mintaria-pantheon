mint_add_plugin(reverb
    NAME Elysium
    CODE Mrev
    CATEGORY Reverb
    SOURCES ElysiumPlugin.cpp)

# 只依赖标准库的 DSP 自检程序，不进默认构建：
# cmake --build <dir> --config Release --target elysium_dsp_bench
add_executable(elysium_dsp_bench EXCLUDE_FROM_ALL tests/elysium_dsp_bench.cpp)
target_include_directories(elysium_dsp_bench PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
if(MSVC)
    target_compile_options(elysium_dsp_bench PRIVATE /O2 /W3 /utf-8)
endif()

# 查询任意 VST3 的参数文字：对标插件的归一化值要先换算成物理量，同设定对比才成立。
juce_add_console_app(elysium_param_text PRODUCT_NAME "elysium_param_text")
set_target_properties(elysium_param_text PROPERTIES EXCLUDE_FROM_ALL TRUE)
target_sources(elysium_param_text PRIVATE tests/param_text.cpp)
target_compile_definitions(elysium_param_text PRIVATE
    JUCE_PLUGINHOST_VST3=1
    JUCE_WEB_BROWSER=0
    JUCE_USE_CURL=0
    JUCE_DISPLAY_SPLASH_SCREEN=0)
target_link_libraries(elysium_param_text PRIVATE
    juce::juce_audio_utils
    juce::juce_audio_processors
    juce::juce_recommended_config_flags)
