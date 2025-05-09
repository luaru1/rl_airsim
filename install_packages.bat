@echo off

set /p user_input=가상환경이 활성화되었는지 확인하세요. (Y/N):
if /i "%user_input%"=="Y" (
  echo 필요한 패키지를 설치합니다...
) else (
  echo 가상환경을 활성화해주세요.
  pause
  exit
)


REM 사전 패키지 설치
pip install msgpack==1.1.0
pip install msgpack-python==0.5.6
pip install msgpack-rpc-python==0.4.1
pip install numpy==2.0.2
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

REM AirSim 설치
pip install airsim==1.8.1 --no-build-isolation --no-cache-dir

REM 기타 패키지 설치
pip install -r requirements.txt

echo 패키지 설치 완료.
pause