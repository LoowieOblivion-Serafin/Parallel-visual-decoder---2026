<#
===============================================================================
 render_rtx4050.ps1 - Reconstruccion visual EN PARALELO (RTX 4050 / 6 GB)
===============================================================================
 Script de entrada para la maquina del companero. Tras `git pull`:

     1. (una vez) .\setup_env.ps1          # crea env_tesis + torch CUDA
     2. .\exe\render_rtx4050.ps1           # este script: genera reconstrucciones
     3. .\exe\presentacion.ps1             # arma y abre la presentacion HTML

 Corre `phase2.visual_evaluator` (batching GPU + guardado I/O asincrono).
 batch-size 2 = seguro para 6 GB con SD 2.1 unCLIP a 768px en bf16.
 La primera ejecucion descarga ~5 GB de pesos SD al cache HF.

 Todas las rutas son parametros: sobrescribe sin editar el archivo, ej.:
     .\exe\render_rtx4050.ps1 -Subjects CSI1,CSI2 -BatchSize 1 -Limit 20
===============================================================================
#>

[CmdletBinding()]
param(
    # Raiz donde viven los datos. Default: raiz del repo (un nivel sobre exe\).
    [string]   $DataRoot      = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path,
    # Carpeta top de estimulos BOLD5000 (config deriva Scene_Stimuli/Presented_Stimuli).
    [string]   $StimuliRoot   = "",
    # Carpeta con adapter/{sujeto}/embeds_test.pt (salida del Ridge entrenado).
    [string]   $Phase2Outputs = "",
    # Carpeta de salida de reconstrucciones/pares/grid.
    [string]   $EvalOutput    = "",
    # Cache de pesos Hugging Face (SD 2.1 unCLIP).
    [string]   $HfCache       = "",
    # Sujetos a reconstruir.
    [string[]] $Subjects      = @("CSI1"),
    # Tamano de lote GPU. 2 = seguro en 6 GB. Baja a 1 si hay OOM.
    [int]      $BatchSize     = 2,
    # Pasos del scheduler (calidad vs velocidad).
    [int]      $Steps         = 75,
    # Hilos de guardado en disco concurrentes.
    [int]      $SaveWorkers   = 4,
    # Cap de estimulos por sujeto (0 = todos).
    [int]      $Limit         = 0,
    # Forzar CPU (lento, sin GPU). Solo para validar flujo.
    [switch]   $Cpu,
    # Encadenar la presentacion HTML al terminar.
    [switch]   $Gallery
)

$ErrorActionPreference = "Stop"

# --- Defaults derivados de DataRoot ------------------------------------------
if (-not $StimuliRoot)   { $StimuliRoot   = Join-Path $DataRoot "BOLD5000_Stimuli" }
if (-not $Phase2Outputs) { $Phase2Outputs = Join-Path $DataRoot "resultados\phase2_outputs" }
if (-not $EvalOutput)    { $EvalOutput    = Join-Path $DataRoot "output_reconstruccions_test2" }
if (-not $HfCache)       { $HfCache       = Join-Path $DataRoot "models_hf" }

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$srcDir   = Join-Path $repoRoot "src"
$py       = Join-Path $repoRoot "env_tesis\Scripts\python.exe"

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host " Reconstruccion visual PARALELA - RTX 4050 (6 GB)" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan

# --- Verificar entorno virtual -----------------------------------------------
if (-not (Test-Path $py)) {
    Write-Host "ERROR: no existe env_tesis. Corre primero:  .\setup_env.ps1" -ForegroundColor Red
    exit 1
}

# --- Exportar rutas al proceso (config.py las lee via ACECOM_*) ---------------
$env:PYTHONPATH                    = "$srcDir;$($env:PYTHONPATH)"
$env:ACECOM_BOLD5000_STIMULI_ROOT  = $StimuliRoot
$env:ACECOM_PHASE2_OUTPUTS         = $Phase2Outputs
$env:ACECOM_EVAL_OUTPUT            = $EvalOutput
$env:ACECOM_HF_CACHE               = $HfCache

Write-Host "Repo         : $repoRoot"
Write-Host "Stimuli      : $StimuliRoot"
Write-Host "Phase2 out   : $Phase2Outputs"
Write-Host "Eval output  : $EvalOutput"
Write-Host "HF cache     : $HfCache"
Write-Host "Sujetos      : $($Subjects -join ', ')"
Write-Host "batch_size=$BatchSize  steps=$Steps  save_workers=$SaveWorkers  limit=$Limit  cpu=$($Cpu.IsPresent)"

# --- Chequeo rapido de GPU ----------------------------------------------------
& $py -c "import torch,sys; ok=torch.cuda.is_available(); print('CUDA:', ok, (torch.cuda.get_device_name(0) if ok else 'CPU')); sys.exit(0)"
if (-not (Test-Path $StimuliRoot)) {
    Write-Host "AVISO: no encuentro StimuliRoot ($StimuliRoot). El GT no se hallara (pares quedaran sin original)." -ForegroundColor Yellow
}

# --- Reconstruir por sujeto ---------------------------------------------------
Push-Location $srcDir
$fail = 0
foreach ($subj in $Subjects) {
    $embeds = Join-Path $Phase2Outputs "adapter\$subj\embeds_test.pt"
    if (-not (Test-Path $embeds)) {
        Write-Host "[$subj] SALTADO: no existe $embeds (falta entrenar/copiar el adapter)." -ForegroundColor Yellow
        continue
    }

    Write-Host ""
    Write-Host "----------------------------------------------------------------------" -ForegroundColor DarkCyan
    Write-Host " $subj - visual_evaluator SD 2.1 unCLIP (paralelo)" -ForegroundColor DarkCyan
    Write-Host "----------------------------------------------------------------------" -ForegroundColor DarkCyan

    $cmd = @("-m", "phase2.visual_evaluator", "--subject", $subj,
             "--batch-size", $BatchSize, "--steps", $Steps, "--save-workers", $SaveWorkers)
    if ($Limit -gt 0) { $cmd += @("--limit", $Limit) }
    if ($Cpu)         { $cmd += "--cpu" }

    & $py @cmd
    if ($LASTEXITCODE -ne 0) { Write-Host "[$subj] fallo (exit $LASTEXITCODE)" -ForegroundColor Red; $fail++ }
}
Pop-Location

Write-Host ""
Write-Host "======================================================================" -ForegroundColor Cyan
if ($fail -eq 0) {
    Write-Host " Reconstruccion completa. Salida en: $EvalOutput" -ForegroundColor Green
} else {
    Write-Host " Terminado con $fail sujeto(s) en fallo. Revisa el log arriba." -ForegroundColor Yellow
}
Write-Host " Siguiente:  .\exe\presentacion.ps1   (pares GT vs reconstruccion)" -ForegroundColor Green
Write-Host "======================================================================" -ForegroundColor Cyan

# --- Presentacion opcional encadenada ----------------------------------------
if ($Gallery) {
    & (Join-Path $PSScriptRoot "presentacion.ps1") -DataRoot $DataRoot `
        -StimuliRoot $StimuliRoot -EvalOutput $EvalOutput
}

exit $(if ($fail -eq 0) { 0 } else { 1 })
