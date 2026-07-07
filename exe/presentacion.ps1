<#
===============================================================================
 presentacion.ps1 - Presentacion HTML de los pares reconstruidos
===============================================================================
 Arma un unico HTML autocontenido (imagenes embebidas en base64) con las
 parejas [Original] vs [Reconstruccion] y lo abre en el navegador. No necesita
 servidor: el archivo se puede enviar/copiar y abre en cualquier maquina.

 Ejecutar DESPUES de render_rtx4050.ps1:
     .\exe\presentacion.ps1
     .\exe\presentacion.ps1 -Subject CSI1 -Thumb 512
===============================================================================
#>

[CmdletBinding()]
param(
    [string] $DataRoot    = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path,
    [string] $StimuliRoot = "",
    [string] $EvalOutput  = "",
    # Un solo sujeto; vacio = todos los presentes en EvalOutput.
    [string] $Subject     = "",
    # Lado mayor del thumbnail embebido (px). Sube para mas detalle / HTML mas pesado.
    [int]    $Thumb       = 384,
    # Cap de pares por sujeto (0 = todos).
    [int]    $Limit       = 0,
    # No abrir el navegador al terminar.
    [switch] $NoOpen
)

$ErrorActionPreference = "Stop"

if (-not $StimuliRoot) { $StimuliRoot = Join-Path $DataRoot "BOLD5000_Stimuli" }
if (-not $EvalOutput)  { $EvalOutput  = Join-Path $DataRoot "output_reconstruccions_test2" }

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$srcDir   = Join-Path $repoRoot "src"
$py       = Join-Path $repoRoot "env_tesis\Scripts\python.exe"

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host " Presentacion de pares reconstruidos" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan

if (-not (Test-Path $py)) {
    Write-Host "ERROR: no existe env_tesis. Corre primero:  .\setup_env.ps1" -ForegroundColor Red
    exit 1
}
if (-not (Test-Path $EvalOutput)) {
    Write-Host "ERROR: no existe EvalOutput ($EvalOutput). Corre antes:  .\exe\render_rtx4050.ps1" -ForegroundColor Red
    exit 1
}

$env:PYTHONPATH                   = "$srcDir;$($env:PYTHONPATH)"
$env:ACECOM_BOLD5000_STIMULI_ROOT = $StimuliRoot
$env:ACECOM_EVAL_OUTPUT           = $EvalOutput

$cmd = @("-m", "phase2.build_gallery", "--eval-dir", $EvalOutput, "--stimuli-root", $StimuliRoot, "--thumb", $Thumb)
if ($Subject)    { $cmd += @("--subject", $Subject) }
if ($Limit -gt 0){ $cmd += @("--limit", $Limit) }
if ($NoOpen)     { $cmd += "--no-open" }

Push-Location $srcDir
& $py @cmd
$rc = $LASTEXITCODE
Pop-Location

Write-Host "======================================================================" -ForegroundColor Cyan
if ($rc -eq 0) {
    Write-Host " Presentacion lista: $EvalOutput\presentacion_pares.html" -ForegroundColor Green
} else {
    Write-Host " La presentacion fallo (exit $rc). Revisa el log arriba." -ForegroundColor Yellow
}
Write-Host "======================================================================" -ForegroundColor Cyan
exit $rc
