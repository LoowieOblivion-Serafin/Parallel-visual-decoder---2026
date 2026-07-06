# Script de configuración del entorno para Fase 2 (NSD + SD 2.1 unCLIP)
# Ejecuta este script en PowerShell desde la raíz del repositorio

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "Iniciando configuración del entorno virtual 'env_tesis'..." -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan

# 1. Crear entorno virtual si no existe
if (-not (Test-Path "env_tesis")) {
    Write-Host "[1/4] Creando entorno virtual 'env_tesis' con Python 3.12..." -ForegroundColor Yellow
    py -3.12 -m venv env_tesis
    if ($LASTEXITCODE -ne 0) {
        Write-Host "py -3.12 falló o no está disponible. Intentando con 'python' genérico..." -ForegroundColor Yellow
        python -m venv env_tesis
    }
} else {
    Write-Host "[1/4] El entorno virtual 'env_tesis' ya existe." -ForegroundColor Green
}

if (-not (Test-Path "env_tesis\Scripts\pip.exe")) {
    Write-Host "ERROR: No se pudo localizar o crear el entorno virtual correctamente." -ForegroundColor Red
    exit 1
}

# 2. Actualizar pip
Write-Host "[2/4] Actualizando pip en el entorno virtual..." -ForegroundColor Yellow
.\env_tesis\Scripts\python.exe -m pip install --upgrade pip

# 3. Instalar PyTorch con soporte CUDA 12.1
Write-Host "[3/4] Instalando PyTorch y torchvision con CUDA 12.1..." -ForegroundColor Yellow
.\env_tesis\Scripts\pip.exe install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 4. Instalar el resto de dependencias
Write-Host "[4/4] Instalando dependencias de src/requirements_py312.txt..." -ForegroundColor Yellow
.\env_tesis\Scripts\pip.exe install -r src/requirements_py312.txt

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "¡Configuración completada con éxito!" -ForegroundColor Green
Write-Host "El intérprete de VS Code ya está configurado para usar env_tesis." -ForegroundColor Green
Write-Host "======================================================================" -ForegroundColor Cyan
