set(MINT_SHARED_DIR "${CMAKE_CURRENT_LIST_DIR}/../shared" CACHE INTERNAL "")

# mint_add_plugin(<slug> NAME Janus CODE Mgat CATEGORY Dynamics SOURCES a.cpp b.cpp)
# 产物名为 "Mintaria <NAME>"，目标名为 mint_<slug>。
function(mint_add_plugin slug)
    cmake_parse_arguments(ARG "" "NAME;CODE;CATEGORY" "SOURCES;LIBS" ${ARGN})

    set(target "mint_${slug}")
    juce_add_plugin(${target}
        COMPANY_NAME "MINT RYU"
        COMPANY_WEBSITE "https://mintaria.mintryu.com/"
        BUNDLE_ID "com.mintryu.pantheon.${slug}"
        PLUGIN_MANUFACTURER_CODE Mntr
        PLUGIN_CODE ${ARG_CODE}
        FORMATS VST3 Standalone
        PRODUCT_NAME "Mintaria ${ARG_NAME}"
        IS_SYNTH FALSE
        NEEDS_MIDI_INPUT FALSE
        NEEDS_MIDI_OUTPUT FALSE
        IS_MIDI_EFFECT FALSE
        EDITOR_WANTS_KEYBOARD_FOCUS FALSE
        COPY_PLUGIN_AFTER_BUILD FALSE
        VST3_CATEGORIES Fx ${ARG_CATEGORY})

    target_sources(${target} PRIVATE
        ${ARG_SOURCES}
        "${MINT_SHARED_DIR}/mint_ui/MintUi.cpp")

    target_include_directories(${target} PRIVATE "${MINT_SHARED_DIR}")

    target_compile_definitions(${target} PUBLIC
        JUCE_WEB_BROWSER=0
        JUCE_USE_CURL=0
        JUCE_VST3_CAN_REPLACE_VST2=0
        JUCE_DISPLAY_SPLASH_SCREEN=0)

    target_link_libraries(${target}
        PRIVATE
            juce::juce_audio_utils
            juce::juce_dsp
            ${ARG_LIBS}
        PUBLIC
            juce::juce_recommended_config_flags
            juce::juce_recommended_lto_flags
            juce::juce_recommended_warning_flags)
endfunction()
