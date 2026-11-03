juce_add_console_app(atlas_shell_probe PRODUCT_NAME "atlas_shell_probe")

target_sources(atlas_shell_probe PRIVATE main.cpp)

target_compile_definitions(atlas_shell_probe PRIVATE
    JUCE_PLUGINHOST_VST3=1
    JUCE_MODAL_LOOPS_PERMITTED=1
    JUCE_WEB_BROWSER=0
    JUCE_USE_CURL=0
    JUCE_DISPLAY_SPLASH_SCREEN=0)

target_link_libraries(atlas_shell_probe PRIVATE
    juce::juce_audio_utils
    juce::juce_audio_processors
    juce::juce_recommended_config_flags
    juce::juce_recommended_warning_flags)
