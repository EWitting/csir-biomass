@echo off
REM Batch file to run all three foundation model ensemble experiments in sequence
REM From fastest to slowest (extreme quality last)

echo ========================================
echo Running Foundation Model Ensemble Experiments
echo ========================================
echo.

REM Experiment 1: Fast - 224px resolution (30 min)
echo [1/3] Starting FAST ensemble (224px, 30 min)...
echo ========================================
uv run python train.py model=foundation_ensemble_224_fast
if %errorlevel% neq 0 (
    echo ERROR: Fast ensemble training failed!
    exit /b %errorlevel%
)
echo.
echo [1/3] FAST ensemble completed successfully!
echo.

REM Experiment 2: Best - 384px resolution (1 hour)
echo [2/3] Starting BEST ensemble (384px, 1 hour)...
echo ========================================
uv run python train.py model=foundation_ensemble_384_best
if %errorlevel% neq 0 (
    echo ERROR: Best ensemble training failed!
    exit /b %errorlevel%
)
echo.
echo [2/3] BEST ensemble completed successfully!
echo.

REM Experiment 3: Extreme - 448px resolution (2 hours)
echo [3/3] Starting EXTREME ensemble (448px, 2 hours with auto_stack)...
echo ========================================
uv run python train.py model=foundation_ensemble_448_extreme
if %errorlevel% neq 0 (
    echo ERROR: Extreme ensemble training failed!
    exit /b %errorlevel%
)
echo.
echo [3/3] EXTREME ensemble completed successfully!
echo.

echo ========================================
echo ALL EXPERIMENTS COMPLETED SUCCESSFULLY!
echo ========================================
echo.
echo Summary:
echo - Fast (224px):    1536-dim features, 30 min training
echo - Best (384px):    2304-dim features, 60 min training
echo - Extreme (448px): 3072-dim features, 120 min training with auto_stack
echo.
echo Total estimated time: ~3.5 hours
echo.
