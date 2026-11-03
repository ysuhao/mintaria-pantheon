mint_add_plugin(delay
    NAME Echo
    CODE Mdly
    CATEGORY Delay
    SOURCES EchoPlugin.cpp EchoEngine.h)

# 测试用宿主。mint_probe 不能提供节拍信息、不能在渲染中途改参数，也选不了外壳型 VST3 里的单个插件，
# 这几项由它来做。只在单独构建本插件时加入。
if(MINT_ONLY STREQUAL "delay")
    juce_add_console_app(mint_delay_host PRODUCT_NAME "mint_delay_host")

    target_sources(mint_delay_host PRIVATE
        tests/echo_host.cpp
        tests/echo_host_factory_win.cpp
        "${CMAKE_SOURCE_DIR}/tools/mint_probe/capture_win.cpp")

    target_compile_definitions(mint_delay_host PRIVATE
        JUCE_PLUGINHOST_VST3=1
        JUCE_MODAL_LOOPS_PERMITTED=1
        JUCE_WEB_BROWSER=0
        JUCE_USE_CURL=0
        JUCE_DISPLAY_SPLASH_SCREEN=0)

    target_link_libraries(mint_delay_host PRIVATE
        juce::juce_audio_utils
        juce::juce_audio_processors
        juce::juce_recommended_config_flags
        juce::juce_recommended_warning_flags)

    # 截高级页并验证分页切换，见 tests/echo_shot.cpp。
    juce_add_console_app(mint_delay_shot PRODUCT_NAME "mint_delay_shot")

    target_sources(mint_delay_shot PRIVATE
        tests/echo_shot.cpp
        "${MINT_SHARED_DIR}/mint_ui/MintUi.cpp")

    target_include_directories(mint_delay_shot PRIVATE "${MINT_SHARED_DIR}")

    target_compile_definitions(mint_delay_shot PRIVATE
        "JucePlugin_Name=\"Mintaria Echo\""
        JUCE_MODAL_LOOPS_PERMITTED=1
        JUCE_WEB_BROWSER=0
        JUCE_USE_CURL=0)

    target_link_libraries(mint_delay_shot PRIVATE
        juce::juce_audio_utils
        juce::juce_dsp
        juce::juce_recommended_config_flags
        juce::juce_recommended_warning_flags)
endif()
