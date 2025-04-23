@echo off
echo Installing core dependencies...

REM 필수 패키지 설치
pip install msgpack==1.1.0
pip install msgpack-python==0.5.6
pip install msgpack-rpc-python==0.4.1
pip install numpy==2.0.2
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

REM AirSim 설치 (build isolation 방지)
pip install airsim==1.8.1 --no-build-isolation --no-cache-dir

REM 기타 requirements 설치
pip install -r requirements.txt

echo Installation complete.
pause