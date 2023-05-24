project "04_BottomUpTraversal"
      kind "ConsoleApp"

      targetdir "../bin/%{cfg.buildcfg}"
      location "../build/"

   if os.istarget("windows") then
      links{ "version" }
   end

      includedirs { "../../" }
      includedirs { "../" }
      files { "../../Orochi/**.h", "../../Orochi/**.cpp" }
      files { "../../contrib/**.h", "../../contrib/**.cpp" }
      files { "../common/**.h", "../common/**.cpp" }
      files { "*.h", "*.cpp" }
