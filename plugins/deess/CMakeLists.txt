mint_add_plugin(deess
    NAME Medusa
    CODE Mdes
    CATEGORY Dynamics
    SOURCES MedusaPlugin.cpp)

# 单声道、1 采样缓冲区、192 kHz、静音 denormal 这些 mint_probe 没覆盖的边界。不进默认构建。
juce_add_console_app(medusa_edge_host PRODUCT_NAME "medusa_edge_host")
set_target_properties(medusa_edge_host PROPERTIES EXCLUDE_FROM_ALL TRUE)
target_sources(medusa_edge_host PRIVATE tests/edge_host.cpp)
target_compile_definitions(medusa_edge_host PRIVATE
    JUCE_PLUGINHOST_VST3=1
    JUCE_WEB_BROWSER=0
    JUCE_USE_CURL=0)
target_link_libraries(medusa_edge_host PRIVATE
    juce::juce_audio_utils
    juce::juce_recommended_config_flags
    juce::juce_recommended_warning_flags)
