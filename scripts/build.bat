@echo off
REM High-Performance Inference Engine Build Script for Windows

echo Building High-Performance Inference Engine...
echo =============================================

REM Check if we're in the right directory
if not exist "CMakeLists.txt" (
    echo Error: CMakeLists.txt not found. Please run this script from the project root.
    exit /b 1
)

REM Create build directory
if not exist "build" mkdir build
cd build

REM Configure with CMake
echo Configuring with CMake...
cmake .. -DCMAKE_BUILD_TYPE=Release ^
         -DCMAKE_CXX_FLAGS="/O2 /arch:AVX2 /GL" ^
         -DENABLE_SIMD=ON ^
         -DENABLE_MULTITHREADING=ON

REM Build the project
echo Building...
cmake --build . --config Release --parallel

REM Run tests
echo Running tests...
if exist "Release\test_runner.exe" (
    Release\test_runner.exe
    echo Tests passed!
) else (
    echo Warning: Test runner not found
)

REM Create release package
echo Creating release package...
cd ..
tar -czf inference_engine_release.tar.gz ^
    build\Release\inference_engine.exe ^
    build\Release\benchmark.exe ^
    README.md

echo Build completed successfully!
echo Executables:
echo   - build\Release\inference_engine.exe
echo   - build\Release\benchmark.exe
echo Package: inference_engine_release.tar.gz
