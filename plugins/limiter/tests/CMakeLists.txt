# 浮点级验证工具：直接托管 VST3，不经过 WAV 量化。构建目标 aegis_bench。
juce_add_console_app(aegis_bench PRODUCT_NAME "aegis_bench")
set_target_properties(aegis_bench PROPERTIES EXCLUDE_FROM_ALL TRUE)

target_sources(aegis_bench PRIVATE aegis_bench.cpp)

target_compile_definitions(aegis_bench PRIVATE
    JUCE_PLUGINHOST_VST3=1
    JUCE_MODAL_LOOPS_PERMITTED=1
    JUCE_WEB_BROWSER=0
    JUCE_USE_CURL=0
    JUCE_DISPLAY_SPLASH_SCREEN=0)

target_link_libraries(aegis_bench PRIVATE
    juce::juce_audio_utils
    juce::juce_audio_processors
    juce::juce_recommended_config_flags
    juce::juce_recommended_warning_flags)
