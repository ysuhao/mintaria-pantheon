# Mintaria Lethe。ONNX Runtime 在配置时按固定版本下载并校验哈希，不进仓库；
# 模型在 models/ 下，由 tools/make_stateful_onnx.py 从官方模型生成。
set(LETHE_DEPS_DIR "${CMAKE_BINARY_DIR}/_lethe_deps" CACHE PATH "Where Lethe keeps downloaded dependencies")

set(lethe_ort_version 1.31.1)
set(lethe_ort_zip "${LETHE_DEPS_DIR}/onnxruntime-win-x64-${lethe_ort_version}.zip")
set(lethe_ort_dir "${LETHE_DEPS_DIR}/onnxruntime-win-x64-${lethe_ort_version}")

if(NOT EXISTS "${lethe_ort_dir}/lib/onnxruntime.dll")
    file(DOWNLOAD
        "https://github.com/microsoft/onnxruntime/releases/download/v${lethe_ort_version}/onnxruntime-win-x64-${lethe_ort_version}.zip"
        "${lethe_ort_zip}"
        EXPECTED_HASH SHA256=5869e26bb25314375f2d5ef1877e3ea06b1543d6c710e6e4a1ec2f7177ab4b2c
        SHOW_PROGRESS)
    # pdb 有 400 MB，不解。
    file(ARCHIVE_EXTRACT INPUT "${lethe_ort_zip}" DESTINATION "${LETHE_DEPS_DIR}"
         PATTERNS "*/include/*" "*/lib/onnxruntime.dll" "*/LICENSE" "*/ThirdPartyNotices.txt")
endif()

# RNNoise 静态链接。模型数据是生成的 C 文件，只用小模型：大模型二进制多 9 MB，单帧耗时是 4 倍。
set(lethe_rnnoise_commit 70f1d256acd4b34a572f999a05c87bf00b67730d)
set(lethe_rnnoise_dir "${LETHE_DEPS_DIR}/rnnoise-${lethe_rnnoise_commit}")
if(NOT EXISTS "${lethe_rnnoise_dir}/src/rnnoise_data_little.c")
    file(DOWNLOAD
        "https://github.com/xiph/rnnoise/archive/${lethe_rnnoise_commit}.tar.gz"
        "${LETHE_DEPS_DIR}/rnnoise-src.tar.gz"
        EXPECTED_HASH SHA256=f61ee0b3f4c4cd337303e003d333357c5eaf25ef5d75a742109ee59e9a0a3932)
    file(ARCHIVE_EXTRACT INPUT "${LETHE_DEPS_DIR}/rnnoise-src.tar.gz" DESTINATION "${LETHE_DEPS_DIR}")
    file(DOWNLOAD
        "https://media.xiph.org/rnnoise/models/rnnoise_data-0a8755f8e2d834eff6a54714ecc7d75f9932e845df35f8b59bc52a7cfe6e8b37.tar.gz"
        "${LETHE_DEPS_DIR}/rnnoise-model.tar.gz"
        EXPECTED_HASH SHA256=0a8755f8e2d834eff6a54714ecc7d75f9932e845df35f8b59bc52a7cfe6e8b37)
    file(ARCHIVE_EXTRACT INPUT "${LETHE_DEPS_DIR}/rnnoise-model.tar.gz" DESTINATION "${lethe_rnnoise_dir}"
         PATTERNS "src/rnnoise_data_little.*")
endif()

# 上游的 rnn.c 和模型文件都按 rnnoise_data.h 这个名字包含模型头文件。
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/rnnoise_model/rnnoise_data.h" "#include <rnnoise_data_little.h>")

add_library(lethe_rnnoise STATIC
    "${lethe_rnnoise_dir}/src/denoise.c"
    "${lethe_rnnoise_dir}/src/rnn.c"
    "${lethe_rnnoise_dir}/src/pitch.c"
    "${lethe_rnnoise_dir}/src/kiss_fft.c"
    "${lethe_rnnoise_dir}/src/celt_lpc.c"
    "${lethe_rnnoise_dir}/src/nnet.c"
    "${lethe_rnnoise_dir}/src/nnet_default.c"
    "${lethe_rnnoise_dir}/src/parse_lpcnet_weights.c"
    "${lethe_rnnoise_dir}/src/rnnoise_tables.c"
    "${lethe_rnnoise_dir}/src/rnnoise_data_little.c"
    "${lethe_rnnoise_dir}/src/x86/x86_dnn_map.c"
    "${lethe_rnnoise_dir}/src/x86/x86cpu.c"
    "${lethe_rnnoise_dir}/src/x86/nnet_avx2.c"
    "${lethe_rnnoise_dir}/src/x86/nnet_sse4_1.c")
target_include_directories(lethe_rnnoise
    PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/rnnoise_model" "${lethe_rnnoise_dir}/src" "${lethe_rnnoise_dir}/src/x86"
    PUBLIC "${lethe_rnnoise_dir}/include")
target_compile_definitions(lethe_rnnoise PRIVATE RNN_ENABLE_X86_RTCD _CRT_SECURE_NO_WARNINGS)
target_compile_options(lethe_rnnoise PRIVATE /wd4244 /wd4305 /W1)
set_source_files_properties("${lethe_rnnoise_dir}/src/x86/nnet_avx2.c" PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
set_source_files_properties("${lethe_rnnoise_dir}/src/x86/nnet_sse4_1.c" PROPERTIES COMPILE_DEFINITIONS "__SSE4_1__;__SSE2__;__SSE__")

# onnxruntime.dll 依赖 VC 运行库。放进包里，用户机器上没装 VC 运行库也能加载。
set(CMAKE_INSTALL_SYSTEM_RUNTIME_LIBS_SKIP TRUE)
set(CMAKE_INSTALL_UCRT_LIBRARIES FALSE)
include(InstallRequiredSystemLibraries)
set(lethe_runtime_dlls)
foreach(dll IN LISTS CMAKE_INSTALL_SYSTEM_RUNTIME_LIBS)
    if(dll MATCHES "(msvcp140|msvcp140_1|vcruntime140|vcruntime140_1)\\.dll$")
        list(APPEND lethe_runtime_dlls "${dll}")
    endif()
endforeach()

set(lethe_sources DfEngine.cpp RnnoiseEngine.cpp LetheCore.cpp)

mint_add_plugin(denoise
    NAME Lethe
    CODE Mdns
    CATEGORY Restoration
    SOURCES LethePlugin.cpp ${lethe_sources}
    LIBS lethe_rnnoise)

target_include_directories(mint_denoise PRIVATE "${lethe_ort_dir}/include")

# 把运行库和模型放到二进制旁边。resource_root 是相对二进制的目录。
function(lethe_deploy target resource_root)
    add_custom_command(TARGET ${target} POST_BUILD
        COMMAND "${CMAKE_COMMAND}" -E copy_if_different
                "${lethe_ort_dir}/lib/onnxruntime.dll" ${lethe_runtime_dlls}
                "$<TARGET_FILE_DIR:${target}>"
        COMMAND "${CMAKE_COMMAND}" -E copy_directory
                "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/models"
                "$<TARGET_FILE_DIR:${target}>/${resource_root}/lethe"
        COMMAND "${CMAKE_COMMAND}" -E copy_if_different
                "${lethe_ort_dir}/LICENSE"
                "$<TARGET_FILE_DIR:${target}>/${resource_root}/lethe/LICENSE-onnxruntime.txt"
        COMMAND "${CMAKE_COMMAND}" -E copy_if_different
                "${lethe_ort_dir}/ThirdPartyNotices.txt"
                "$<TARGET_FILE_DIR:${target}>/${resource_root}/lethe/ThirdPartyNotices-onnxruntime.txt"
        COMMAND "${CMAKE_COMMAND}" -E copy_if_different
                "${lethe_rnnoise_dir}/COPYING"
                "$<TARGET_FILE_DIR:${target}>/${resource_root}/lethe/LICENSE-rnnoise.txt"
        COMMAND "${CMAKE_COMMAND}" -E copy_if_different
                "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/THIRD_PARTY.md"
                "$<TARGET_FILE_DIR:${target}>/${resource_root}/lethe/THIRD_PARTY.md"
        VERBATIM)
endfunction()

lethe_deploy(mint_denoise_VST3 "../Resources")
lethe_deploy(mint_denoise_Standalone "Resources")

# 实时压力测试，直接链接核心通路，能读到丢帧和回退的计数。用 --target 单独构建。
juce_add_console_app(lethe_rt_test PRODUCT_NAME "lethe_rt_test")
set_target_properties(lethe_rt_test PROPERTIES EXCLUDE_FROM_ALL TRUE)
target_sources(lethe_rt_test PRIVATE tests/lethe_rt_test.cpp ${lethe_sources})
target_include_directories(lethe_rt_test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}" "${lethe_ort_dir}/include")
target_compile_definitions(lethe_rt_test PRIVATE JUCE_WEB_BROWSER=0 JUCE_USE_CURL=0 JUCE_DISPLAY_SPLASH_SCREEN=0)
target_link_libraries(lethe_rt_test PRIVATE
    lethe_rnnoise
    juce::juce_core
    juce::juce_audio_basics
    juce::juce_recommended_config_flags
    juce::juce_recommended_warning_flags)
lethe_deploy(lethe_rt_test "Resources")
