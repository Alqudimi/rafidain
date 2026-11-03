#Requires -Version 5.1
# ============================================================================
#  RAFIDAIN — Windows one-shot installer (PowerShell 5.1+)
#  رافدَين — مثبّت كامل لبيئة ويندوز: تثبيت + تهيئة + تشغيل
# ----------------------------------------------------------------------------
#  Realtime Iraqi-dialect voice cloning on seed-uvit-tat-xlsr-tiny (Seed-VC).
#
#  USAGE (PowerShell / Terminal — normal user, no admin needed):
#    powershell -NoProfile -ExecutionPolicy Bypass -File setup_windows.ps1
#    powershell -NoProfile -ExecutionPolicy Bypass -File setup_windows.ps1 -Engine mock -NoTests
#    # curl-style one-liner (auto-clones https://github.com/Alqudimi/rafidain):
#    irm https://raw.githubusercontent.com/Alqudimi/rafidain/main/scripts/install/setup_windows.ps1 -OutFile setup.ps1; powershell -ExecutionPolicy Bypass -File setup.ps1
#
#  OPTIONS:
#    -Source <dir|git-url|zip-url>   project source (default: auto-detect local)
#    -Ref    <branch>                git branch when cloning (default main)
#    -Engine <seedvc|mock>           default: seedvc with GPU, mock without
#    -Device <auto|cuda|cpu>
#    -Port   <n>                     API port (default 8000)
#    -ApiKey <key>                   fixed API key (default: generate random)
#    -NoAuth                         disable API authentication (dev only)
#    -NoModels                       skip the checkpoint pre-download (~1.5 GB)
#    -NoTests                        skip the test suite
#    -Foreground                     run the server attached to this console
#    -Tunnel                         expose a public trycloudflare.com URL
#    -SkipTools                      do not download ffmpeg/python extras
# ============================================================================
[CmdletBinding()]
param(
    [string]$Source,
    [string]$Ref = "main",
    [string]$Engine = "",
    [string]$Device = "",
    [int]$Port = 8000,
    [string]$ApiKey = "",
    [switch]$NoAuth,
    [switch]$NoModels,
    [switch]$NoTests,
    [switch]$Foreground,
    [switch]$Tunnel,
    [switch]$SkipTools
)

$ErrorActionPreference = "Stop"
try { [Console]::OutputEncoding = [Text.Encoding]::UTF8 } catch { }
[Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12

$SeedVcGit   = "https://github.com/Plachtaa/seed-vc"
$SeedVcZip   = "https://github.com/Plachtaa/seed-vc/archive/refs/heads/main.zip"
$RafidainGit = "https://github.com/Alqudimi/rafidain"
$TorchCpuIdx = "https://download.pytorch.org/whl/cpu"
$ReadyTimeoutS = 900

# ------------------------------------------------------------------ helpers
function Write-Step([string]$Msg) { Write-Host ""; Write-Host ("==== " + $Msg) -ForegroundColor Cyan }
function Write-Log([string]$Msg)  { Write-Host ("[..] " + $Msg) -ForegroundColor DarkGray }
function Write-Ok([string]$Msg)   { Write-Host ("[OK] " + $Msg) -ForegroundColor Green }
function Write-Warn2([string]$Msg){ Write-Host ("[!]  " + $Msg) -ForegroundColor Yellow }
function Write-Die([string]$Msg)  {
    Write-Host ("[X]  " + $Msg) -ForegroundColor Red
    Write-Host "     Re-run with tracing:  powershell -NoProfile -ExecutionPolicy Bypass -Command `"& { Set-PSDebug -Trace 1; & '.\setup_windows.ps1' <args> }`"" -ForegroundColor DarkGray
    exit 1
}
function Invoke-Native {   # Invoke-Native <exe> [args...] — safe under EAP=Stop
    param([string]$FilePath, [string[]]$Arguments = @())
    $old = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        if ($Arguments.Count -gt 0) { & $FilePath @Arguments 2>&1 } else { & $FilePath 2>&1 }
    } finally { $ErrorActionPreference = $old }
}
function Invoke-Py([string[]]$ArgList) {          # verbose python run; throws on failure
    Invoke-Native $script:VenvPy $ArgList | ForEach-Object { "$_" }
    if ($LASTEXITCODE -ne 0) { throw ("python " + ($ArgList -join " ") + " failed (exit $LASTEXITCODE)") }
}
function Invoke-PyQ([string[]]$ArgList) {         # quiet python run; throws on failure
    $out = Invoke-Native $script:VenvPy $ArgList
    if ($LASTEXITCODE -ne 0) { throw ("python " + ($ArgList -join " ") + " failed: " + (($out | Select-Object -Last 5) -join " ")) }
    return ($out -join "`n")
}
function Get-Http([string]$Url, [string]$OutFile) {
    # streaming download with a live progress bar (percent + speed + MB)
    try {
        $showBar = $false
        try { $showBar = (-not [Console]::IsOutputRedirected) } catch {}
        $req = [Net.HttpWebRequest]::Create($Url)
        $req.AllowAutoRedirect = $true
        $req.Timeout = 120000
        $resp = $req.GetResponse()
        try {
            $total = [long]$resp.ContentLength
            $inS = $resp.GetResponseStream()
            $outS = [IO.File]::Create($OutFile)
            $buf = New-Object byte[] (1MB)
            $got = [long]0; $t0 = Get-Date
            try {
                while (($n = $inS.Read($buf, 0, $buf.Length)) -gt 0) {
                    $outS.Write($buf, 0, $n)
                    $got += $n
                    if ($showBar -and $total -gt 0) {
                        $pct = [Math]::Round(100.0 * $got / $total, 1)
                        $filled = [Math]::Min(20, [int](20 * $got / $total))
                        $bar = ("#" * $filled) + ("-" * (20 - $filled))
                        $el = ((Get-Date) - $t0).TotalSeconds
                        $mbps = $got / 1MB / [Math]::Max($el, 0.1)
                        Write-Host -NoNewline ("`r[get] |{0}| {1,5}% {2,8:N1}/{3:N1} MB {4,6:N1} MB/s   " -f $bar,$pct,($got/1MB),($total/1MB),$mbps)
                    }
                }
            } finally { $outS.Close(); $inS.Close() }
            if ($showBar -and $total -gt 0) { Write-Host "" }
            if ($total -gt 0 -and $got -ne $total) { throw "incomplete download ($got/$total bytes)" }
        } finally { $resp.Close() }
    } catch {
        # fallback: plain Invoke-WebRequest (native console progress)
        Write-Log ("streaming download failed - falling back to Invoke-WebRequest: " + $_.Exception.Message)
        $ProgressPreference = "Continue"
        Invoke-WebRequest -Uri $Url -OutFile $OutFile -UseBasicParsing
    }
}
function Test-ProjectDir([string]$Path) {
    $toml = Join-Path $Path "pyproject.toml"
    if (-not (Test-Path $toml)) { return $false }
    return ((Get-Content $toml -Raw) -match 'name = "rafidain"')
}
function Get-PyVer([string]$Exe) {
    $v = ((Invoke-Native $Exe @("-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")) -join "").Trim()
    return $v
}

Write-Host ""
Write-Host "  RAFIDAIN — realtime Iraqi voice cloning (seed-uvit-tat-xlsr-tiny) — Windows" -ForegroundColor Cyan
Write-Host "  ==========================================================================" -ForegroundColor Cyan

# --------------------------------------------------------- [1/10] environment
Write-Step "[1/10] Environment probe"
if (-not [Environment]::Is64BitOperatingSystem) { Write-Die "64-bit Windows is required." }
Write-Ok ("Windows PowerShell " + $PSVersionTable.PSVersion)

# --- GPU probe ---------------------------------------------------------------
$GpuPresent = $false; $TorchIndex = ""; $CudaCap = ""
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    $out = Invoke-Native "nvidia-smi"
    if ($LASTEXITCODE -eq 0 -and $out) {
        $line = ($out | Select-String -Pattern "CUDA Version:\s*([\d\.]+)" | Select-Object -First 1)
        if ($line) {
            $CudaCap = $line.Matches[0].Groups[1].Value
            if ([string]::IsNullOrEmpty($CudaCap)) { $CudaCap = "12.4" }
            $parts = $CudaCap.Split("."); $maj = [int]$parts[0]; $min = [int]$parts[1]
            $GpuPresent = $true
            # torch 2.6 ships cu118 / cu124 / cu126 wheels (cu121 does not exist for 2.6)
            if     ($maj -ge 13 -or ($maj -eq 12 -and $min -ge 4)) { $TorchIndex = "https://download.pytorch.org/whl/cu124" }
            elseif ($maj -eq 11 -and $min -ge 8)                    { $TorchIndex = "https://download.pytorch.org/whl/cu118" }
            else { $GpuPresent = $false; Write-Warn2 ("driver CUDA $CudaCap < 11.8 - GPU unusable for torch; using CPU") }
        } else { Write-Warn2 "nvidia-smi present but no CUDA version reported" }
    } else { Write-Warn2 "nvidia-smi failed - continuing without GPU" }
}
if ($GpuPresent) { Write-Ok ("GPU detected (driver CUDA $CudaCap -> " + $TorchIndex.Split("/")[-1] + ")") }
else { Write-Warn2 "no usable GPU - defaulting to the mock engine (use -Engine seedvc for CPU inference)" }

if ([string]::IsNullOrEmpty($Engine)) { if ($GpuPresent) { $Engine = "seedvc" } else { $Engine = "mock" } }
if ($Engine -ne "seedvc" -and $Engine -ne "mock") { Write-Die "-Engine must be 'seedvc' or 'mock'" }
if ([string]::IsNullOrEmpty($Device)) { if ($GpuPresent) { $Device = "auto" } else { $Device = "cpu" } }
if ($Engine -eq "seedvc" -and -not $GpuPresent) {
    Write-Warn2 "seedvc on CPU: works, but streaming latency will exceed call budgets"
}
Write-Ok ("engine=$Engine device=$Device port=$Port")

# --- python discovery ---------------------------------------------------------
$PyExe = ""
$pyLauncher = Get-Command py -ErrorAction SilentlyContinue
if ($pyLauncher) {
    foreach ($v in @("3.12", "3.11", "3.10")) {
        if ($PyExe) { break }
        $exe = ((Invoke-Native $pyLauncher.Source @("-$v", "-c", "import sys; print(sys.executable)")) -join "").Trim()
        if ($LASTEXITCODE -eq 0 -and $exe -and (Test-Path $exe)) {
            $ver = Get-PyVer $exe
            if ($ver -ge "3.10") { $PyExe = $exe }
        }
    }
}
if (-not $PyExe) {
    foreach ($n in @("python", "python3")) {
        if ($PyExe) { break }
        $c = Get-Command $n -ErrorAction SilentlyContinue
        if ($c) {
            $r = Invoke-Native $c.Source @("-c", "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)")
            if ($LASTEXITCODE -eq 0) { $PyExe = $c.Source }
        }
    }
}
if (-not $PyExe) {
    foreach ($v in @("Python312", "Python311", "Python310")) {
        if ($PyExe) { break }
        $p = Join-Path $env:LOCALAPPDATA "Programs\Python\$v\python.exe"
        if (Test-Path $p) {
            $r = Invoke-Native $p @("-c", "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)")
            if ($LASTEXITCODE -eq 0) { $PyExe = $p }
        }
    }
}
if (-not $PyExe -and -not $SkipTools) {
    $wg = Get-Command winget -ErrorAction SilentlyContinue
    if ($wg) {
        Write-Log "installing Python 3.12 via winget (no admin needed)..."
        $null = Invoke-Native $wg.Source @("install", "--id", "Python.Python.3.12", "-e",
                                          "--accept-source-agreements", "--accept-package-agreements")
        $pyLauncher = Get-Command py -ErrorAction SilentlyContinue
        if ($pyLauncher) {
            foreach ($v in @("3.12", "3.11", "3.10")) {
                if ($PyExe) { break }
                $exe = ((Invoke-Native $pyLauncher.Source @("-$v", "-c", "import sys; print(sys.executable)")) -join "").Trim()
                if ($LASTEXITCODE -eq 0 -and $exe -and (Test-Path $exe)) { $PyExe = $exe }
            }
        }
        if (-not $PyExe) {
            $p = Join-Path $env:LOCALAPPDATA "Programs\Python\Python312\python.exe"
            if (Test-Path $p) { $PyExe = $p }
        }
    } else { Write-Warn2 "winget not found" }
}
if (-not $PyExe) {
    Write-Die "Python 3.10+ not found. Install it from https://www.python.org/downloads/ (tick 'Add python.exe to PATH'), open a NEW terminal, and re-run."
}
$PyMajorMinor = Get-PyVer $PyExe
Write-Ok ("Python " + $PyMajorMinor + " (" + $PyExe + ")")

# ----------------------------------------------------------- [2/10] ffmpeg/git
Write-Step "[2/10] System tools (ffmpeg / git)"
$FfBin = ""
$ff = Get-Command ffmpeg -ErrorAction SilentlyContinue
if ($ff) { $FfBin = (Split-Path $ff.Source -Parent); Write-Ok ("ffmpeg already on PATH (" + $ff.Source + ")") }
elseif (-not $SkipTools) {
    $tools = Join-Path (Get-Location) "tools"
    if (-not (Test-Path $tools)) { New-Item -ItemType Directory -Path $tools | Out-Null }
    $ffRoot = Join-Path $tools "ffmpeg"
    $ffExe = Join-Path $ffRoot "bin\ffmpeg.exe"
    if (-not (Test-Path $ffExe)) {
        Write-Log "downloading ffmpeg (gyan.dev essentials build, ~90 MB)..."
        $zip = Join-Path $tools "ffmpeg-essentials.zip"
        try {
            Get-Http "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip" $zip
            $tmp = Join-Path $tools "ffmpeg-extract"
            if (Test-Path $tmp) { Remove-Item $tmp -Recurse -Force }
            Expand-Archive -Path $zip -DestinationPath $tmp -Force
            $found = Get-ChildItem -Path $tmp -Recurse -Filter "ffmpeg.exe" | Select-Object -First 1
            if (-not $found) { throw "ffmpeg.exe not found inside the archive" }
            if (Test-Path $ffRoot) { Remove-Item $ffRoot -Recurse -Force }
            New-Item -ItemType Directory -Path (Join-Path $ffRoot "bin") -Force | Out-Null
            Get-ChildItem -Path $found.Directory.FullName -Filter "*.exe" |
                ForEach-Object { Copy-Item $_.FullName (Join-Path $ffRoot "bin" $_.Name) -Force }
            Remove-Item $tmp -Recurse -Force -ErrorAction SilentlyContinue
            Remove-Item $zip -Force -ErrorAction SilentlyContinue
        } catch {
            Write-Warn2 ("direct ffmpeg download failed: " + $_.Exception.Message)
            $wg = Get-Command winget -ErrorAction SilentlyContinue
            if ($wg) {
                Write-Log "trying winget Gyan.FFmpeg..."
                $null = Invoke-Native $wg.Source @("install", "--id", "Gyan.FFmpeg", "-e",
                                                  "--accept-source-agreements", "--accept-package-agreements")
                Write-Warn2 "winget installed ffmpeg for FUTURE terminals; restart the terminal to pick it up"
            } else {
                Write-Warn2 "ffmpeg could not be installed - voice uploads need it: https://www.gyan.dev/ffmpeg/builds/"
            }
        }
    }
    if (Test-Path $ffExe) {
        $FfBin = Join-Path $ffRoot "bin"
        $env:Path = "$FfBin;$env:Path"
        $userPath = [Environment]::GetEnvironmentVariable("Path", "User")
        if ($userPath -notlike "*$FfBin*") {
            [Environment]::SetEnvironmentVariable("Path", ("$FfBin;" + $userPath), "User")
            Write-Log "added $FfBin to your user PATH (new terminals will see it)"
        }
        Write-Ok ("ffmpeg ready at " + $ffExe)
    }
} else { Write-Warn2 "-SkipTools: ffmpeg not ensured - voice uploads need it" }

$GitExe = Get-Command git -ErrorAction SilentlyContinue
if ($GitExe) { Write-Ok "git present" } else { Write-Log "git not found - Seed-VC will be fetched as a zip instead" }

# ---------------------------------------------------------- [3/10] project source
Write-Step "[3/10] Locating the Rafidain project"
function Get-ProjectFromZip([string]$ZipUrl, [string]$InstallDir) {
    $tmp = Join-Path $env:TEMP ("rafidain-" + [guid]::NewGuid().ToString("N"))
    New-Item -ItemType Directory -Path $tmp | Out-Null
    $zip = Join-Path $tmp "src.zip"
    if ($ZipUrl -match "^https?://") { Get-Http $ZipUrl $zip }
    else { Copy-Item $ZipUrl $zip }
    Expand-Archive -Path $zip -DestinationPath (Join-Path $tmp "x") -Force
    $toml = Get-ChildItem -Path (Join-Path $tmp "x") -Recurse -Filter "pyproject.toml" | Select-Object -First 1
    if (-not $toml) { throw "the zip does not contain a pyproject.toml" }
    if (Test-Path $InstallDir) { Remove-Item $InstallDir -Recurse -Force }
    Move-Item $toml.Directory.FullName $InstallDir
    Remove-Item $tmp -Recurse -Force -ErrorAction SilentlyContinue
}
$Project = ""
try {
    if ($Source) {
        if ($Source -match "^https?://") {
            $InstallDir = Join-Path (Get-Location) "rafidain"
            if ($Source -match "\.zip($|\?)") {
                Write-Log "downloading project zip: $Source"
                Get-ProjectFromZip $Source $InstallDir
            } elseif ($GitExe) {
                Write-Log "git clone: $Source (ref: $Ref)"
                if (Test-Path $InstallDir) { Remove-Item $InstallDir -Recurse -Force }
                $null = Invoke-Native $GitExe.Source @("clone", "--progress", "--depth", "1", "--branch", $Ref, $Source, $InstallDir)
                if ($LASTEXITCODE -ne 0) { throw "git clone failed" }
            } elseif ($Source -match "github\.com/([^/]+)/([^/]+?)(\.git)?/?$") {
                $zipUrl = "https://github.com/" + $Matches[1] + "/" + $Matches[2] + "/archive/refs/heads/" + $Ref + ".zip"
                Write-Log "no git - downloading $zipUrl"
                Get-ProjectFromZip $zipUrl $InstallDir
            } else { throw "URL source requires git, or a .zip URL / github.com repo URL" }
            $Project = $InstallDir
        } elseif (Test-Path $Source -PathType Container) {
            $Project = (Resolve-Path $Source).Path
        } elseif (Test-Path $Source -PathType Leaf -and $Source -match "\.zip$") {
            $InstallDir = Join-Path (Get-Location) "rafidain"
            Get-ProjectFromZip (Resolve-Path $Source).Path $InstallDir
            $Project = $InstallDir
        } else { throw "-Source not found or unsupported (dir, .zip file, git URL, or zip URL): $Source" }
    } else {
        if (Test-ProjectDir (Get-Location).Path) { $Project = (Get-Location).Path }
        elseif (Test-ProjectDir (Join-Path (Get-Location) "rafidain")) { $Project = Join-Path (Get-Location) "rafidain" }
        elseif (Test-ProjectDir (Join-Path (Get-Location) "rafidain-backend")) {
            # legacy layout (pre-repo zips shipped the project under rafidain-backend/)
            $Project = Join-Path (Get-Location) "rafidain-backend"
        }
        elseif (Test-ProjectDir (Join-Path (Get-Location) "rafidain-main")) { $Project = Join-Path (Get-Location) "rafidain-main" }
        elseif (Test-ProjectDir (Join-Path (Get-Location) "rafidain-master")) { $Project = Join-Path (Get-Location) "rafidain-master" }
        elseif ($PSScriptRoot -and (Test-ProjectDir $PSScriptRoot)) { $Project = $PSScriptRoot }
        elseif ($PSScriptRoot -and (Test-ProjectDir (Join-Path $PSScriptRoot "..\.."))) { $Project = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path }
    }
} catch { Write-Die $_.Exception.Message }
if (-not $Project) {
    # curl-style flow: nothing nearby -> clone the official repository
    Write-Log ("project not found nearby - cloning " + $RafidainGit + " (ref: " + $Ref + ")")
    $InstallDir = Join-Path (Get-Location) "rafidain"
    $cloned = $false
    if ($GitExe) {
        if (Test-Path $InstallDir) { Remove-Item $InstallDir -Recurse -Force }
        $null = Invoke-Native $GitExe.Source @("clone", "--progress", "--depth", "1", "--branch", $Ref, $RafidainGit, $InstallDir)
        $cloned = ($LASTEXITCODE -eq 0)
    } else {
        # no git -> GitHub zip archive (extracts to rafidain-<ref>/)
        $zipUrl = $RafidainGit + "/archive/refs/heads/" + $Ref + ".zip"
        try {
            Get-ProjectFromZip $zipUrl $InstallDir
            $cloned = $true
        } catch { $cloned = $false }
    }
    if ($cloned -and (Test-ProjectDir $InstallDir)) {
        $Project = $InstallDir
    } else {
        if (Test-Path $InstallDir) { Remove-Item $InstallDir -Recurse -Force -ErrorAction SilentlyContinue }
        Write-Die @"

could not clone $RafidainGit
  Fix A (local):  clone this repo / extract its zip next to this script, then re-run.
  Fix B (git):    re-run with:  -Source https://github.com/Alqudimi/rafidain
  Fix C (zip):    re-run with:  -Source https://host/rafidain.zip
"@
    }
}
Write-Ok ("project: " + $Project)
Set-Location $Project

# live pip progress in an interactive console; quiet when redirected to logs
$PipV = @("-q")
try { if (-not [Console]::IsOutputRedirected) { $PipV = @("--progress-bar", "on") } } catch {}

# ---------------------------------------------- [4/10] venv + core deps
Write-Step "[4/10] Virtualenv (.venv) + core dependencies"
$VenvPy = Join-Path $Project ".venv\Scripts\python.exe"
if (-not (Test-Path $VenvPy)) {
    $null = Invoke-Native $PyExe @("-m", "venv", (Join-Path $Project ".venv"))
    if ($LASTEXITCODE -ne 0) { Write-Die "python -m venv failed (is the 'py launcher/venv' feature installed?)" }
}
Invoke-PyQ @("-m", "pip", "install", "-q", "--upgrade", "pip", "wheel", "setuptools") | Out-Null
Invoke-PyQ @("-m", "pip", "install") + $PipV + @("--no-input", "-r", "requirements.txt") | Out-Null
Invoke-PyQ @("-m", "pip", "install") + $PipV + @("--no-input", "-e", ".[dev]") | Out-Null
Write-Ok "virtualenv ready: .venv (fastapi/uvicorn/pydantic/soxr/... + dev tooling)"

# ------------------------------------------------------------- [5/10] ML stack
Write-Step ("[5/10] ML stack (engine=" + $Engine + ")")
if ($Engine -eq "seedvc") {
    # torch 2.6.0: supports py3.9-3.13, passes transformers' CVE-2025-32434 gate
    # (XLSR ships pytorch_model.bin only - newer transformers refuse torch<2.6),
    # and honours TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD for Seed-VC's own .pth files.
    $TorchPkgs = @("torch==2.6.0", "torchaudio==2.6.0")
    if ($GpuPresent) {
        Write-Log ("installing " + ($TorchPkgs -join " ") + " from " + $TorchIndex.Split("/")[-1] + " (~2.5 GB, several minutes)")
        Invoke-Py (@("-m", "pip", "install") + $PipV + @("--no-input") + $TorchPkgs + @("--index-url", $TorchIndex))
    } else {
        Write-Log ("installing CPU wheels " + ($TorchPkgs -join " "))
        Invoke-Py (@("-m", "pip", "install") + $PipV + @("--no-input") + $TorchPkgs + @("--index-url", $TorchCpuIdx))
    }
    if ($PyMajorMinor -le "3.12") { Invoke-PyQ @("-m", "pip", "install") + $PipV + @("--no-input", "numpy==1.26.4") | Out-Null }
    Invoke-PyQ @("-m", "pip", "install") + $PipV + @("--no-input", "transformers>=4.40,<5", "librosa>=0.10", "soundfile>=0.12", "huggingface-hub>=0.23", "munch>=4.0", "einops>=0.7", "scipy>=1.10", "descript-audio-codec==1.0.0") | Out-Null
    Write-Ok "torch/transformers/librosa/soundfile ready"
} else {
    Write-Ok "mock engine - ML stack skipped entirely (no torch, no checkpoints)"
}

# ------------------------------------------------------------- [6/10] Seed-VC
Write-Step "[6/10] Fetching the Seed-VC repository (engine code + configs)"
$SvDir = Join-Path $Project "third_party\seed-vc"
if (-not (Test-Path (Join-Path $SvDir "modules\commons.py"))) {
    if (Test-Path $SvDir) { Remove-Item $SvDir -Recurse -Force }
    if ($GitExe) {
        Write-Log "git clone --depth 1 $SeedVcGit"
        $null = Invoke-Native $GitExe.Source @("clone", "--progress", "--depth", "1", $SeedVcGit, $SvDir)
        if ($LASTEXITCODE -ne 0) { Write-Die "git clone of Seed-VC failed" }
    } else {
        Write-Log "downloading Seed-VC zip (no git)..."
        $tools = Join-Path $Project "tools"; if (-not (Test-Path $tools)) { New-Item -ItemType Directory -Path $tools | Out-Null }
        $zip = Join-Path $tools "seed-vc.zip"
        Get-Http $SeedVcZip $zip
        $tmp = Join-Path $tools "seed-vc-extract"
        if (Test-Path $tmp) { Remove-Item $tmp -Recurse -Force }
        Expand-Archive -Path $zip -DestinationPath $tmp -Force
        $inner = Join-Path $tmp "seed-vc-main"
        if (-not (Test-Path $inner)) { Write-Die "unexpected Seed-VC zip layout" }
        New-Item -ItemType Directory -Path (Join-Path $Project "third_party") -Force | Out-Null
        Move-Item $inner $SvDir
        Remove-Item $tmp -Recurse -Force -ErrorAction SilentlyContinue
        Remove-Item $zip -Force -ErrorAction SilentlyContinue
    }
}
if (-not (Test-Path (Join-Path $SvDir "configs\hifigan.yml"))) {
    Write-Die "third_party\seed-vc\configs\hifigan.yml missing - Seed-VC fetch failed?"
}
Write-Ok "third_party\seed-vc ready (modules + configs)"

# ---------------------------------------------------------- [7/10] checkpoints
Write-Step "[7/10] Pre-downloading checkpoints (seed-uvit-tat-xlsr-tiny)"
if ($Engine -eq "seedvc" -and -not $NoModels) {
    Invoke-Py @("scripts\download_models.py", "--cache-dir", "./checkpoints")
    Write-Ok "checkpoints cached in .\checkpoints (~1.5 GB incl. XLSR)"
} elseif ($Engine -eq "seedvc") {
    Write-Warn2 "-NoModels: the server will download checkpoints on first start (slow)"
} else {
    Write-Log "mock engine - checkpoint download skipped"
}

# ------------------------------------------------------------------ [8/10] .env
Write-Step "[8/10] Writing configuration (.env)"
$EnvPath = Join-Path $Project ".env"
if (Test-Path $EnvPath) {
    Copy-Item $EnvPath ($EnvPath + ".bak." + (Get-Date -Format "yyyyMMdd-HHmmss"))
    Write-Log "existing .env backed up"
}
if ([string]::IsNullOrEmpty($ApiKey)) {
    $bytes = New-Object byte[] 24
    ([Security.Cryptography.RNGCryptoServiceProvider]::Create()).GetBytes($bytes)
    $ApiKey = ($bytes | ForEach-Object { $_.ToString("x2") }) -join ""
}
$Fp16 = "true"; if ($Device -eq "cpu") { $Fp16 = "false" }
$AuthLine = 'RAFIDAIN__API_KEYS=["' + $ApiKey + '"]'
if ($NoAuth) { $AuthLine = "RAFIDAIN__API_KEYS=[]" }
$envContent = @"
# Generated by setup_windows.ps1 on $(Get-Date -Format "yyyy-MM-ddTHH:mm:ss")
RAFIDAIN__HOST=0.0.0.0
RAFIDAIN__PORT=$Port
RAFIDAIN__LOG_LEVEL=INFO
RAFIDAIN__LOG_JSON=true
$AuthLine
RAFIDAIN__CORS_ORIGINS=["*"]
RAFIDAIN__ENGINE__IMPL=$Engine
RAFIDAIN__ENGINE__DEVICE=$Device
RAFIDAIN__ENGINE__FP16=$Fp16
SEED_VC_ROOT=./third_party/seed-vc
RAFIDAIN__ENGINE__CACHE_DIR=./checkpoints
RAFIDAIN__ENGINE__EAGER_LOAD=true
RAFIDAIN__ENGINE__WARMUP=true
RAFIDAIN__DEFAULT_PROFILE=balanced
RAFIDAIN__SESSION_TTL_S=300
RAFIDAIN__MAX_SESSIONS=64
RAFIDAIN__SEND_QUEUE_SIZE=64
RAFIDAIN__VAD=off
RAFIDAIN__DATA_DIR=./data
RAFIDAIN__REDIS_URL=
"@
[IO.File]::WriteAllText($EnvPath, $envContent, (New-Object Text.UTF8Encoding($false)))
if ($NoAuth) { Write-Warn2 "authentication DISABLED (-NoAuth) - never do this on a shared machine" }
Write-Ok ".env written (engine=$Engine, device=$Device)"

$RunnerPath = Join-Path $Project "run_server.ps1"
$runner = @'
Set-Location $PSScriptRoot
$env:TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD = "1"   # safe for torch>=2.6 checkpoint loading
$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }
& $py -m rafidain.main
'@
[IO.File]::WriteAllText($RunnerPath, $runner, (New-Object Text.UTF8Encoding($false)))
Write-Ok "run_server.ps1 generated"

# ------------------------------------------------------------------ [9/10] tests
Write-Step "[9/10] Test suite (mock engine - validates the whole stack)"
if (-not $NoTests) {
    $out = Invoke-Native $VenvPy @("-m", "pytest", "-q")
    if ($LASTEXITCODE -ne 0) {
        Write-Host ($out | Select-Object -Last 12) -ForegroundColor Red
        Write-Die "test suite failed - inspect the output above"
    }
    Write-Host ($out | Select-Object -Last 3)
    Write-Ok "all tests passed"
} else { Write-Log "tests skipped (-NoTests)" }

# --------------------------------------------------------------- [10/10] run
Write-Step ("[10/10] Starting the Rafidain server on port " + $Port)
$pidFile = Join-Path $Project "server.pid"
if (Test-Path $pidFile) {
    $oldPid = (Get-Content $pidFile -ErrorAction SilentlyContinue | Select-Object -First 1)
    if ($oldPid) { Stop-Process -Id ([int]$oldPid) -Force -ErrorAction SilentlyContinue }
}
Get-CimInstance Win32_Process -Filter "Name LIKE 'python%'" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -match "rafidain\.main" } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
Start-Sleep -Seconds 1

if ($Foreground) {
    Write-Log "foreground mode - server output follows (Ctrl+C to stop)..."
    $env:TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD = "1"
    Set-Location $Project
    Invoke-Native $VenvPy @("-m", "rafidain.main")
    exit $LASTEXITCODE
}

$serverLog = Join-Path $Project "server.log"
$serverErr = Join-Path $Project "server.err.log"
$proc = Start-Process -FilePath $VenvPy -ArgumentList "-m", "rafidain.main" `
    -WorkingDirectory $Project -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput $serverLog -RedirectStandardError $serverErr
Set-Content -Path $pidFile -Value $proc.Id
Write-Log ("background pid " + $proc.Id + " - waiting for /healthz")
$up = $false
for ($i = 0; $i -lt 60; $i++) {
    try { $null = Invoke-RestMethod "http://127.0.0.1:$Port/healthz" -TimeoutSec 2; $up = $true; break } catch { Start-Sleep -Seconds 1 }
}
if (-not $up) {
    Write-Host "----- server.err.log (tail) -----" -ForegroundColor DarkGray
    if (Test-Path $serverErr) { Get-Content $serverErr -Tail 30 }
    Write-Host "----- server.log (tail) -----" -ForegroundColor DarkGray
    if (Test-Path $serverLog) { Get-Content $serverLog -Tail 30 }
    Write-Die "server did not come up"
}
Write-Ok "/healthz OK"

Write-Log "waiting for /readyz (model load + warmup; first run can take minutes)"
$ready = $false
for ($i = 0; $i -lt $ReadyTimeoutS; $i++) {
    try {
        $r = Invoke-RestMethod "http://127.0.0.1:$Port/readyz" -TimeoutSec 3
        if ($r.ready -eq $true) { $ready = $true; break }
    } catch { }
    Start-Sleep -Seconds 1
}
if ($ready) { Write-Ok "/readyz OK - engine loaded and warmed up" }
else { Write-Warn2 "engine still loading - watch:  Get-Content '$serverErr' -Wait" }

$TunnelUrl = ""
if ($Tunnel) {
    Write-Log "starting public tunnel (-Tunnel)"
    $tools = Join-Path $Project "tools"; if (-not (Test-Path $tools)) { New-Item -ItemType Directory -Path $tools | Out-Null }
    $cfd = Join-Path $tools "cloudflared.exe"
    if (-not (Test-Path $cfd)) {
        Get-Http "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe" $cfd
    }
    $tLog = Join-Path $Project "tunnel.log"
    $tErr = Join-Path $Project "tunnel.err.log"
    Start-Process -FilePath $cfd -ArgumentList "tunnel", "--url", "http://127.0.0.1:$Port", "--no-autoupdate" `
        -WindowStyle Hidden -RedirectStandardOutput $tLog -RedirectStandardError $tErr | Out-Null
    for ($i = 0; $i -lt 60; $i++) {
        $txt = ""
        if (Test-Path $tLog) { $txt += (Get-Content $tLog -Raw -ErrorAction SilentlyContinue) }
        if (Test-Path $tErr) { $txt += (Get-Content $tErr -Raw -ErrorAction SilentlyContinue) }
        if ($txt -match "(https://[a-zA-Z0-9-]+\.trycloudflare\.com)") { $TunnelUrl = $Matches[1]; break }
        Start-Sleep -Seconds 1
    }
    if ($TunnelUrl) { Write-Ok ("public URL: " + $TunnelUrl) } else { Write-Warn2 "tunnel URL not found - see tunnel.log / tunnel.err.log" }
}

# -------------------------------------------------------------------- summary
Write-Host ""
Write-Host "==================================================================" -ForegroundColor DarkGray
Write-Host "  RAFIDAIN IS UP   |   رافدَين جاهز ومشغّل" -ForegroundColor Green
Write-Host "==================================================================" -ForegroundColor DarkGray
Write-Host ("  Local URL  : http://127.0.0.1:" + $Port + "   (docs: /docs  metrics: /metrics)")
if ($TunnelUrl) { Write-Host ("  Public URL : " + $TunnelUrl) }
Write-Host ("  Engine     : " + $Engine + " (device=" + $Device + ", profile=balanced ~430 ms/block on T4)")
if (-not $NoAuth) {
    Write-Host ("  API key    : " + $ApiKey)
    Write-Host  "  Auth       : header  X-API-Key: <key>    websocket  ?token=<key>"
}
Write-Host ("  Project    : " + $Project)
Write-Host  "  Logs       : server.log / server.err.log"
Write-Host ("  Stop       : Stop-Process -Id (Get-Content '" + $Project + "\server.pid')")
Write-Host "==================================================================" -ForegroundColor DarkGray
Write-Host "  TRY IT / جرّب الآن:"
Write-Host ("    curl http://127.0.0.1:" + $Port + "/healthz")
if (-not $NoAuth) {
    Write-Host ("    & '" + $VenvPy + "' scripts\smoke_ws_client.py --url http://127.0.0.1:" + $Port + " --seconds 3 --api-key " + $ApiKey)
    if ($TunnelUrl) { Write-Host ("    & '" + $VenvPy + "' scripts\smoke_ws_client.py --url " + $TunnelUrl + " --seconds 3 --api-key " + $ApiKey) }
} else {
    Write-Host ("    & '" + $VenvPy + "' scripts\smoke_ws_client.py --url http://127.0.0.1:" + $Port + " --seconds 3")
}
Write-Host "==================================================================" -ForegroundColor DarkGray
Write-Host "  LAN access needs a firewall rule (admin PowerShell):" -ForegroundColor DarkGray
Write-Host ("    netsh advfirewall firewall add rule name=`"Rafidain " + $Port + "`" dir=in action=allow protocol=TCP localport=" + $Port) -ForegroundColor DarkGray
Write-Host "  Never clone a voice without the speaker's consent." -ForegroundColor DarkGray
Write-Host "  Iraqi fine-tuning:  & '$VenvPy' finetune\run_finetune.py --help" -ForegroundColor DarkGray
Write-Host "==================================================================" -ForegroundColor DarkGray
