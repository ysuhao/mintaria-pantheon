mint_add_plugin(chorus
    NAME Sirens
    CODE Mcho
    CATEGORY Modulation
    SOURCES SirensPlugin.cpp)

# 测试用的切换宿主，只在明确指定目标时才构建。
juce_add_console_app(sirens_switch_host PRODUCT_NAME "sirens_switch_host")
set_target_properties(sirens_switch_host PROPERTIES EXCLUDE_FROM_ALL TRUE)
target_sources(sirens_switch_host PRIVATE tests/switch_host.cpp)
target_compile_definitions(sirens_switch_host PRIVATE
    JUCE_PLUGINHOST_VST3=1
    JUCE_MODAL_LOOPS_PERMITTED=1
    JUCE_WEB_BROWSER=0
    JUCE_USE_CURL=0
    JUCE_DISPLAY_SPLASH_SCREEN=0)
target_link_libraries(sirens_switch_host PRIVATE
    juce::juce_audio_utils
    juce::juce_audio_processors
    juce::juce_recommended_config_flags
    juce::juce_recommended_warning_flags)
