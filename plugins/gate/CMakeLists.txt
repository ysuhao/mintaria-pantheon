mint_add_plugin(gate
    NAME Janus
    CODE Mgat
    CATEGORY Dynamics
    SOURCES GatePlugin.cpp)

# 测试用宿主：能喂侧链、能改参数、能选 Shell 里的子插件。只在需要时单独构建。
add_subdirectory(tests/host)
