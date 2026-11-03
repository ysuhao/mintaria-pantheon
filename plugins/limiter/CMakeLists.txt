mint_add_plugin(limiter
    NAME Aegis
    CODE Mlim
    CATEGORY Dynamics
    SOURCES AegisPlugin.cpp)

# 验证工具只在显式指定目标时才构建。
add_subdirectory(tests)
