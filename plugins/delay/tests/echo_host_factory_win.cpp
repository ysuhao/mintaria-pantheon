// 单独成文件，避免 windows.h 和 JUCE 的名字冲突。
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <string>
#include <vector>

// 只用 VST3 标准的工厂接口列出类名和类 ID，不实例化任何插件。
// JUCE 的扫描会把外壳里的每个子插件都实例化一遍，遇到一个崩溃的子插件就整个扫不下去。
struct FactoryClass
{
    std::string name, category;
    char cid[16];
};

namespace
{
    struct ClassInfo
    {
        char cid[16];
        int cardinality;
        char category[32];
        char name[64];
    };

    // pluginterfaces/base/ipluginbase.h 里 IPluginFactory 的虚表顺序。
    struct Factory
    {
        virtual long __stdcall queryInterface (const char*, void**) = 0;
        virtual unsigned long __stdcall addRef() = 0;
        virtual unsigned long __stdcall release() = 0;
        virtual long __stdcall getFactoryInfo (void*) = 0;
        virtual int __stdcall countClasses() = 0;
        virtual long __stdcall getClassInfo (int, ClassInfo*) = 0;
    };
}

bool listFactoryClasses (const wchar_t* path, std::vector<FactoryClass>& classes)
{
    auto module = LoadLibraryW (path);
    if (module == nullptr)
        return false;

    if (auto init = (bool (__stdcall*)()) GetProcAddress (module, "InitDll"))
        init();

    auto getFactory = (Factory* (__stdcall*)()) GetProcAddress (module, "GetPluginFactory");
    auto* factory = getFactory != nullptr ? getFactory() : nullptr;
    if (factory == nullptr)
        return false;

    const auto count = factory->countClasses();
    for (int i = 0; i < count; ++i)
    {
        ClassInfo info {};
        if (factory->getClassInfo (i, &info) != 0)
            continue;

        FactoryClass entry;
        entry.name.assign (info.name, strnlen (info.name, sizeof (info.name)));
        entry.category.assign (info.category, strnlen (info.category, sizeof (info.category)));
        memcpy (entry.cid, info.cid, sizeof (entry.cid));
        classes.push_back (entry);
    }
    return true;
}
