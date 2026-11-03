mint_add_plugin(rider
    NAME Helios
    CODE Mrid
    CATEGORY Dynamics
    SOURCES RiderPlugin.cpp)

# 验证用的宿主：能喂侧链、能用 1 采样缓冲区、能在独立音频线程上跑并记录参数通知来自哪个线程。
# mint_probe render 不喂侧链，所以这里需要自己的工具。不进默认构建，只在显式指定时编译。
juce_add_console_app(mint_rider_bench PRODUCT_NAME "mint_rider_bench")
target_sources(mint_rider_bench PRIVATE tests/rider_bench.cpp)
target_compile_definitions(mint_rider_bench PRIVATE
    JUCE_PLUGINHOST_VST3=1
    JUCE_MODAL_LOOPS_PERMITTED=1
    JUCE_WEB_BROWSER=0
    JUCE_USE_CURL=0
    JUCE_DISPLAY_SPLASH_SCREEN=0)
target_link_libraries(mint_rider_bench PRIVATE
    juce::juce_audio_utils
    juce::juce_audio_processors
    juce::juce_recommended_config_flags
    juce::juce_recommended_warning_flags)
set_target_properties(mint_rider_bench PROPERTIES EXCLUDE_FROM_ALL TRUE EXCLUDE_FROM_DEFAULT_BUILD TRUE)
