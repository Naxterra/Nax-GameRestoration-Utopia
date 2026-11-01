<#
.SYNOPSIS
Translates a game from the user's own copy and builds the native Windows program.

.DESCRIPTION
The script runs the translator on the original program files in the given game folder, writes the C++ modules to
generated\<game>, and compiles them together with the runtime into build\<game>.exe. Both folders stay local,
because their content is derived from the original game.

.EXAMPLE
.\build.ps1 -Game utopia -GameFolder gamedata\utopia
#>
param(
    [ValidateSet("utopia")]
    [string]$Game = "utopia",
    [string]$GameFolder = "",
    [ValidateSet("Release", "Debug")]
    [string]$Configuration = "Release",
    [switch]$Tests
)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot

$games = @{
    utopia = @{
        exe     = "Utopia.exe"
        sources = @("games\utopia\utopia.cpp")
        modules = @(
            @{ file = "i.mdg"; name = "i"; flat = $false },
            @{ file = "u.mdg"; name = "u"; flat = $false },
            @{ file = "beeper.drv"; name = "beeper"; flat = $true },
            @{ file = "adlib.drv"; name = "adlib"; flat = $true },
            @{ file = "mt32.drv"; name = "mt32"; flat = $true }
        )
    }
}
$spec = $games[$Game]

# Load the Visual Studio build environment.
$vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path $vswhere)) { throw "Visual Studio Build Tools are missing. Install them with: winget install Microsoft.VisualStudio.BuildTools" }
$vs = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vs) { throw "The C++ build tools are missing from Visual Studio Build Tools." }
$vcvars = Join-Path $vs "VC\Auxiliary\Build\vcvars64.bat"
cmd /c "`"$vcvars`" >nul && set" | ForEach-Object {
    if ($_ -match "^([^=]+)=(.*)$") { Set-Item -Path "env:$($matches[1])" -Value $matches[2] }
}

# The runtime tests need no game files.
if ($Tests) {
    $testBuild = Join-Path $root "build\tests"
    $testObjects = Join-Path $testBuild "obj"
    New-Item -ItemType Directory -Force $testObjects | Out-Null
    $testSources = @(Get-ChildItem (Join-Path $root "runtime") -Filter *.cpp | Where-Object { $_.Name -notin @("app.cpp", "presenter.cpp") } | ForEach-Object { $_.FullName })
    $testSources += Join-Path $root "tests\cpp\test_runtime.cpp"
    $testExe = Join-Path $testBuild "test_runtime.exe"
    $testFlags = @("/nologo", "/std:c++20", "/EHsc", "/MT", "/W3", "/utf-8", "/DNOMINMAX", "/D_CRT_SECURE_NO_WARNINGS", "/DWIN32_LEAN_AND_MEAN",
        "/DUNICODE", "/D_UNICODE", "/I$root", "/Fo$($testObjects -replace '\\', '/')/", "/Fe$testExe", "/Od")
    & cl.exe @testFlags @testSources /link user32.lib shell32.lib ole32.lib xaudio2.lib
    if ($LASTEXITCODE -ne 0) { throw "Compiling the tests failed." }
    & $testExe
    if ($LASTEXITCODE -ne 0) { throw "Some runtime tests failed." }
}

# The game build needs the user's own copy of the game.
if (-not $GameFolder) { $GameFolder = Join-Path $root "gamedata\$Game" }
if (-not (Test-Path $GameFolder)) {
    if ($Tests) { Write-Host "No game folder at $GameFolder, so only the tests ran."; exit 0 }
    throw "The game folder $GameFolder does not exist. Pass -GameFolder with the folder of your copy of the game."
}
$GameFolder = (Resolve-Path $GameFolder).Path

# Translate the original program files.
$generated = Join-Path $root "generated\$Game"
New-Item -ItemType Directory -Force $generated | Out-Null
$python = (Get-Command python -ErrorAction Stop).Source
foreach ($module in $spec.modules) {
    $source = Join-Path $GameFolder $module.file
    if (-not (Test-Path $source)) { throw "The game folder does not contain $($module.file)." }
    $seeds = Join-Path $root "games\$Game\seeds-$($module.name).txt"
    $arguments = @("-I", (Join-Path $root "tools\recomp\emit.py"), $source, $module.name, (Join-Path $generated "mod_$($module.name).cpp"))
    if ($module.flat) { $arguments += "--flat" }
    if (Test-Path $seeds) { $arguments += @("--seeds", $seeds) }
    & $python @arguments
    if ($LASTEXITCODE -ne 0) { throw "Translating $($module.file) failed." }
}

# Compile.
$build = Join-Path $root "build\$Game"
$objects = Join-Path $build "obj"
New-Item -ItemType Directory -Force $objects | Out-Null
$runtime = Get-ChildItem (Join-Path $root "runtime") -Filter *.cpp | ForEach-Object { $_.FullName }
# Only the modules this run generated are compiled, so a module removed from the list cannot linger.
$modules = $spec.modules | ForEach-Object { Join-Path $generated "mod_$($_.name).cpp" }
$sources = @($runtime) + ($spec.sources | ForEach-Object { Join-Path $root $_ }) + $modules
$optimize = if ($Configuration -eq "Release") { "/O2" } else { "/Od", "/Zi" }
$flags = @("/nologo", "/std:c++20", "/EHsc", "/MT", "/W3", "/wd4102", "/MP", "/bigobj", "/utf-8",
    "/DNOMINMAX", "/D_CRT_SECURE_NO_WARNINGS", "/DWIN32_LEAN_AND_MEAN", "/DUNICODE", "/D_UNICODE", "/I$root", "/Fo$($objects -replace '\\', '/')/", "/Fe$(Join-Path $build $spec.exe)") + $optimize
& cl.exe @flags @sources /link /SUBSYSTEM:WINDOWS d3d12.lib dxgi.lib d3dcompiler.lib user32.lib shell32.lib ole32.lib xaudio2.lib
if ($LASTEXITCODE -ne 0) { throw "Compiling failed." }
Write-Host "Built $(Join-Path $build $spec.exe)"
