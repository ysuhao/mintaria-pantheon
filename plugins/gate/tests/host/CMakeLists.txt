juce_add_console_app(gate_host PRODUCT_NAME "gate_host")

target_sources(gate_host PRIVATE
    main.cpp
    "${CMAKE_SOURCE_DIR}/tools/mint_probe/capture_win.cpp")

target_compile_definitions(gate_host PRIVATE
    JUCE_PLUGINHOST_VST3=1
    JUCE_MODAL_LOOPS_PERMITTED=1
    JUCE_WEB_BROWSER=0
    JUCE_USE_CURL=0
    JUCE_DISPLAY_SPLASH_SCREEN=0)

target_link_libraries(gate_host PRIVATE
    juce::juce_audio_utils
    juce::juce_audio_processors
    juce::juce_recommended_config_flags
    juce::juce_recommended_warning_flags)

set_target_properties(gate_host PROPERTIES EXCLUDE_FROM_ALL TRUE)
