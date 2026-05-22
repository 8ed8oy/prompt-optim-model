#!/usr/bin/env pwsh
# -*- coding: utf-8 -*-

param(
    [string]$ApiKey,
    [int]$WorkerCount = 4,
    [int]$TargetSizePerWorker = 500,
    [switch]$SkipDataGen = $false
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

function Write-Info {
    param([string]$Message)
    Write-Host "[INFO]  $Message" -ForegroundColor Cyan
}

function Write-Success {
    param([string]$Message)
    Write-Host "[OK]    $Message" -ForegroundColor Green
}

function Write-Warn {
    param([string]$Message)
    Write-Host "[WARN]  $Message" -ForegroundColor Yellow
}

function Write-Err {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor Red
}

function Write-Step {
    param([string]$Message)
    Write-Host "`n===================================================" -ForegroundColor Blue
    Write-Host ">  $Message" -ForegroundColor Blue
    Write-Host "===================================================`n" -ForegroundColor Blue
}

function Normalize-ApiKey {
    param([string]$Value)

    if ([string]::IsNullOrWhiteSpace($Value)) {
        return ""
    }

    $normalized = $Value.Trim()
    if (
        ($normalized.StartsWith('"') -and $normalized.EndsWith('"')) -or
        ($normalized.StartsWith("'") -and $normalized.EndsWith("'"))
    ) {
        $normalized = $normalized.Substring(1, $normalized.Length - 2).Trim()
    }

    return $normalized
}

function Test-ApiKeyFormat {
    param([string]$Value)

    if ([string]::IsNullOrWhiteSpace($Value)) {
        return $false
    }

    return ($Value -match '^sk-[A-Za-z0-9]{16,}$')
}

function Check-Environment {
    Write-Step "Step 1: Environment check"

    try {
        $pythonVersion = & python --version 2>&1
        Write-Success "Python detected: $pythonVersion"
    }
    catch {
        Write-Err "Python not found. Activate your conda environment first."
        exit 1
    }

    $condaEnv = $env:CONDA_DEFAULT_ENV
    if ([string]::IsNullOrWhiteSpace($condaEnv)) {
        Write-Warn "No active conda env detected. Suggested: conda activate prompt-opt"
    }
    else {
        Write-Success "Conda env: $condaEnv"
    }

    $scripts = @(
        "scripts\\data\\generate_data.py",
        "scripts\\data\\merge_clean_data.py",
        "Qwen2.5-7B-train.py",
        "inference.py",
        "scripts\\data\\start_generate_workers.ps1"
    )

    foreach ($script in $scripts) {
        if (Test-Path $script) {
            Write-Success "Found: $script"
        }
        else {
            Write-Err "Missing required script: $script"
            exit 1
        }
    }
}

function Setup-ApiConfig {
    Write-Step "Step 2: API config"

    if ([string]::IsNullOrWhiteSpace($ApiKey)) {
        if (-not [string]::IsNullOrWhiteSpace($env:API_KEY)) {
            $ApiKey = $env:API_KEY
            Write-Info "Reusing API_KEY from current shell."
        }
        else {
            Write-Info "Please input your DeepSeek API key."
            $ApiKey = Read-Host "API_KEY"
        }

        if ([string]::IsNullOrWhiteSpace($ApiKey)) {
            Write-Err "API key cannot be empty."
            exit 1
        }
    }

    $ApiKey = Normalize-ApiKey -Value $ApiKey
    if (-not (Test-ApiKeyFormat -Value $ApiKey)) {
        Write-Err "API key format looks invalid. It should look like sk-xxxxxxxx..."
        exit 1
    }

    $env:API_KEY = $ApiKey
    $env:API_KEYS = $ApiKey
    if ([string]::IsNullOrWhiteSpace($env:BASE_URL)) {
        $env:BASE_URL = "https://api.deepseek.com/v1"
    }
    if ([string]::IsNullOrWhiteSpace($env:MODEL_NAME)) {
        $env:MODEL_NAME = "deepseek-chat"
    }
    $env:HF_ENDPOINT = "https://hf-mirror.com"

    Write-Success "API config complete"
    Write-Info "MODEL_NAME=$($env:MODEL_NAME)"
    Write-Info "BASE_URL=$($env:BASE_URL)"
    Write-Info "API_KEY length=$($env:API_KEY.Length)"
}

function Generate-TrainingData {
    Write-Step "Step 3: Generate training data"

    Write-Info "Starting $WorkerCount workers, each with target $TargetSizePerWorker records"
    Write-Info "Expected total before dedup: $($WorkerCount * $TargetSizePerWorker)"

    & .\scripts\data\start_generate_workers.ps1 `
        -WorkerCount $WorkerCount `
        -TargetSizePerWorker $TargetSizePerWorker `
        -OutputDir ".\data" `
        -ApiKeys "$($env:API_KEY)"

    Write-Warn "Workers are launched. Press Enter after all worker windows finish."
    $null = Read-Host "Press Enter to continue"

    Write-Success "Data generation stage completed"
}

function Merge-CleanData {
    Write-Step "Step 4: Merge and clean data"

    if (-not (Test-Path ".\data")) {
        Write-Warn "Data directory not found: .\\data. Skip merge."
        return
    }

    & python .\scripts\data\merge_clean_data.py `
        --input-dir ".\data" `
        --output ".\data\train_data.cleaned.jsonl"

    if (Test-Path ".\data\train_data.cleaned.jsonl") {
        $fileSize = (Get-Item ".\data\train_data.cleaned.jsonl").Length / 1MB
        Write-Success "Merged file: train_data.cleaned.jsonl ($([math]::Round($fileSize, 2)) MB)"
    }
}

function Train-Model {
    Write-Step "Step 5: Train Qwen2.5-7B"

    if (-not (Test-Path ".\data\train_data.cleaned.jsonl")) {
        Write-Err "Training data missing: train_data.cleaned.jsonl"
        exit 1
    }

    if ((Get-Item ".\data\train_data.cleaned.jsonl").Length -lt 1MB) {
        Write-Warn "Training data is small (< 1MB). Consider generating more data."
    }

    & python .\Qwen2.5-7B-train.py `
        --train-file ".\data\train_data.cleaned.jsonl" `
        --output-dir ".\outputs\qwen25_7b_prompt_optimizer" `
        --max-seq-length 384 `
        --per-device-train-batch-size 2 `
        --gradient-accumulation-steps 8 `
        --num-train-epochs 3 `
        --save-steps 100

    Write-Success "Training completed"
}

function Test-Inference {
    Write-Step "Step 6: Inference test"

    $inferenceChoice = Read-Host "Run inference test now? (y/n)"
    if ($inferenceChoice -eq "y" -or $inferenceChoice -eq "Y") {
        & python .\inference.py
    }
}

function Main {
    Write-Host "`nPipeline: check env -> setup api -> generate -> merge -> train -> inference`n" -ForegroundColor Blue

    try {
        Check-Environment
        Setup-ApiConfig

        if ($SkipDataGen) {
            Write-Warn "SkipDataGen is enabled. Skipping generation stage."
        }
        else {
            Generate-TrainingData
        }

        Merge-CleanData
        Train-Model
        Test-Inference

        Write-Step "All steps finished"
        Write-Success "Pipeline completed successfully"
    }
    catch {
        Write-Err "Pipeline failed: $_"
        exit 1
    }
}

Main
