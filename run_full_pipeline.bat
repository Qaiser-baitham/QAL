@echo off
REM =================================================================
REM  iMC Full Research Pipeline — Unified Run Script
REM  Ag/GeTe/Pt Memristor Compute-in-Memory Framework
REM =================================================================
REM
REM  This script runs the complete pipeline:
REM    1. Device characterization (auto from data/raw_data/*.xlsx)
REM    2. Neural network training (ideal + hardware-aware)
REM    3. Hardware metrics estimation (C*V^2 physics model)
REM    4. Design space exploration (precision/noise sweeps)
REM    5. Evaluation metrics (F1, confusion matrix, class-wise)
REM    6. Publication-quality plots + Excel exports
REM
REM  Usage:
REM    run_full_pipeline.bat                    (interactive mode)
REM    run_full_pipeline.bat --quick            (10 epochs, quick test)
REM    run_full_pipeline.bat --full             (60 epochs, full run)
REM    run_full_pipeline.bat --compare          (model comparison + full)
REM =================================================================

cd /d "%~dp0"

IF "%1"=="--quick" (
    echo [iMC] Running QUICK pipeline (10 epochs, FashionMNIST, MLP)...
    python main.py --non-interactive --dataset FMNIST --model MLP --mode dual --epochs 10 --run-dse --gradient-clip 1.0
    goto :done
)

IF "%1"=="--full" (
    echo [iMC] Running FULL pipeline (60 epochs, FashionMNIST, MLP)...
    python main.py --non-interactive --dataset FMNIST --model MLP --mode dual --epochs 60 --run-dse --gradient-clip 1.0
    goto :done
)

IF "%1"=="--compare" (
    echo [iMC] Running MODEL COMPARISON + FULL pipeline...
    python main.py --non-interactive --dataset FMNIST --model MLP --mode dual --epochs 60 --run-dse --gradient-clip 1.0 --compare-models --comparison-epochs 10
    goto :done
)

echo [iMC] Running in INTERACTIVE mode...
python main.py

:done
echo.
echo =================================================================
echo  Pipeline complete! Check outputs/:
echo    outputs/plots/    — All publication-quality plots
echo    outputs/excel/    — All data in Excel format
echo    outputs/reports/  — Summary reports and artifact index
echo    outputs/checkpoints/ — Model checkpoints for resume
echo =================================================================
pause
