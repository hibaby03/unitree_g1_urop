# xr_teleoperate + ZED Mini 2-DoF active stereo camera

Unity 없이 Unitree `xr_teleoperate`의 기존 영상/녹화 파이프라인을 그대로
사용하면서, G1 기본 카메라 입력을 목에 장착된 ZED Mini stereo로 교체한다.

이 구조의 목적은 단순한 원격조종 화면 개선이 아니다. VR 시연 중 작업자가
머리를 움직여 관측 방향을 선택하면 ZED 시야도 함께 움직이고, 그 stereo 영상과
로봇 동작을 같은 episode에 기록하여 imitation-learning policy가 active
perception 전략을 학습할 수 있게 하는 것이다.

설계 근거는 `Active Stereo-Camera Outperforms Multi-Sensor Setup in ACT Imitation
Learning for Humanoid Manipulation`의 active-camera-only(`A`) 구성이다. 논문은
data-limited regime에서 단일 active stereo camera가 복잡한 multi-sensor 구성보다
좋은 robustness/complexity trade-off를 보였고, 같은 head link의 static camera를
함께 쓰면 feature conflict가 발생할 수 있다고 보고한다. 따라서 기본 설정에서는
G1 static head camera와 wrist camera를 모두 비활성화한다.

```text
Quest Meta Browser
  ├─ OpenXR head pose ─ HTTPS/WSS :8012 ─> Host TeleVuer
  │                                           └─ UDP :5005 ─> PC2
  │                                                              └─ U2D2
  │                                                                 ├─ yaw
  │                                                                 └─ pitch
  │
  └─ left/right eye video <──── WebRTC :60001 ─────┐
                                                    │
ZED Mini ─ USB 3 ─> PC2 PyZED ─> rectified left|right frame
                                      ├─ teleimager WebRTC ─> Quest
                                      └─ teleimager ZMQ :55555 ─> Host recorder
```

## 구현 구성

- `pc2/zed_stereo.py`: 한 번의 ZED `grab()`에서 rectified LEFT/RIGHT를 가져와
  가로 `left | right` BGR 프레임으로 결합
- `pc2/zed_teleimager_server.py`: ZED capture를 teleimager의 `head_camera`로
  주입하고 기존 config/ZMQ/WebRTC publisher 재사용
- `pc2/cam_config_zed.yaml`: active-stereo-only 카메라 설정
- `host/run_teleop_with_neck.py`: 기존 teleop 실행과 동시에 raw OpenXR head pose 송신
- `host/head_pose_udp.py`: OpenXR matrix 검증, quaternion 변환, UDP packet 생성
- `pc2/head_pose_receiver.py`: UDP 수신, 최초 pose 중립점, yaw/pitch 제한
- `pc2/dynamixel_neck.py`: U2D2를 통한 두 축 DYNAMIXEL Sync Write
- `PROTOCOL.md`: 고정 48-byte head-pose UDP packet 규격

ZED 송신기는 별도 영상 프로토콜을 만들지 않는다. 최신 `xr_teleoperate`가 사용하는
`teleimager.ImageClient`와 `TeleVuerWrapper`가 서버 설정을 그대로 읽는다.

`binocular: true`, `image_shape: [720, 2560]`이므로:

- Quest/WebRTC: `stereo-left-right`로 왼쪽/오른쪽 눈에 분리 표시
- `xr_teleoperate --record`: 프레임을 가운데에서 나누어 `color_0`(ZED left),
  `color_1`(ZED right)로 저장

## 1. PC2 의존성

ZED Mini를 PC2의 USB 3 포트에 연결한다. PC2의 JetPack과 호환되는 ZED SDK 및
PyZED를 Stereolabs 방식으로 먼저 설치해야 한다. PyZED는 일반적인 PyPI wheel이
아니라 설치된 ZED SDK와 버전이 맞아야 한다.

`xr_teleoperate`가 pin한 버전과 같은 teleimager를 PC2에 설치한다.

```bash
cd ~/xr_teleoperate/teleop/teleimager
python3 -m pip install -e . --no-deps

cd ~/active_camera
python3 -m pip install -r pc2/requirements.txt
```

Host의 TeleVuer 인증서도 기존 teleimager 절차와 동일하게 PC2에 둔다.

```bash
mkdir -p ~/.config/xr_teleoperate
cp cert.pem key.pem ~/.config/xr_teleoperate/
```

설치 확인:

```bash
python3 -c "import pyzed.sl; import teleimager; print('ZED/teleimager OK')"
```

## 2. ZED stereo image server

기존 G1 기본 카메라용 `teleimager-server`는 종료하고 다음 서버 하나만 실행한다.
두 프로세스가 같은 카메라 포트나 WebRTC 포트를 동시에 소유하면 안 된다.

```bash
cd ~/active_camera
python3 pc2/zed_teleimager_server.py
```

ZED가 여러 대라면 ID 또는 serial number로 선택할 수 있다.

```bash
python3 pc2/zed_teleimager_server.py --camera-id 0
# 또는
python3 pc2/zed_teleimager_server.py --serial-number 12345678
```

기본 영상은 ZED HD720 기준 눈당 `1280x720`, 합쳐진 프레임은
`2560x720`, 30 FPS다. `cam_config_zed.yaml`에서 60 FPS로 올릴 수 있지만,
`xr_teleoperate --frequency`가 30이면 episode에는 여전히 약 30 Hz로 샘플된다.
또한 JPEG/ZMQ와 WebRTC encoding 부하가 증가하므로 30 FPS부터 검증한다.

Quest를 연결하기 전에 다음 주소에서 WebRTC 영상이 좌우로 붙어 나오는지 확인한다.

```text
https://192.168.123.164:60001
```

화면의 왼쪽 절반이 ZED LEFT, 오른쪽 절반이 ZED RIGHT여야 한다.

## 3. 목 pose receiver

처음에는 모터 없이 pose만 확인한다.

```bash
cd ~/active_camera
python3 pc2/head_pose_receiver.py --bind 0.0.0.0 --port 5005
```

첫 정상 pose가 중립점이다. 정면을 본 상태에서 receiver를 시작하고, 오른쪽을 보면
양의 yaw, 위를 보면 양의 pitch가 출력되는지 확인한다.

모터 방향과 제한을 확인한 뒤에만 torque를 켠다.

```bash
python3 pc2/head_pose_receiver.py \
  --bind 0.0.0.0 --port 5005 \
  --enable-motor \
  --dxl-device /dev/ttyUSB0 --dxl-baudrate 1000000 \
  --yaw-id 1 --pitch-id 2 \
  --yaw-center 2048 --pitch-center 2048 \
  --yaw-sign 1 --pitch-sign 1 \
  --yaw-limit-deg 80 \
  --pitch-down-limit-deg 35 --pitch-up-limit-deg 45 \
  --smoothing 1.0 \
  --motor-timeout-ms 1000
```

U2D2 USB만으로 모터 전원을 공급하지 말고 2XL430용 외부 전원을 사용한다.

## 4. Host teleoperation과 episode 기록

`host` 폴더를 Host의 `xr_teleoperate` 아래에 복사한다.

```bash
cp -r /path/to/active_camera/host ~/xr_teleoperate/active_camera_host
conda activate tv
cd ~/xr_teleoperate
```

ZED stereo 영상과 robot state/action을 episode로 함께 저장하려면 반드시
`--record`를 사용한다.

```bash
python active_camera_host/run_teleop_with_neck.py \
  --xr-repo . \
  --neck-pose-ip 192.168.123.164 \
  --neck-pose-port 5005 \
  --neck-pose-rate 60 \
  --img-server-ip 192.168.123.164 \
  --arm=G1_29 \
  --ee=dex3 \
  --input-mode=hand \
  --record \
  --task-dir ./utils/data \
  --task-name active_stereo_task
```

launcher는 `--xr-repo`와 `--neck-*`만 소비하고 나머지는 현재 설치된
`teleop_hand_and_arm.py`에 그대로 전달한다. Quest 접속 주소도 기존과 같다.

```text
https://192.168.123.2:8012/?ws=wss://192.168.123.2:8012
```

녹화된 각 timestep에는 ZED left/right가 각각 `color_0`, `color_1`로 저장된다.
G1 static head camera와 wrist camera는 기본 config에서 꺼져 있으므로 논문의
active-stereo-only visual observation과 일치한다.

## 데이터셋 관점에서 중요한 제한

현재 변경으로 stereo observation과 기존 arm/hand state/action은 episode에
저장된다. 그러나 논문 Fig. 3처럼 ACT policy가 추론 시 카메라 자체도 움직이게
하려면 데이터셋의 state/action vector에 다음 2-DoF 값도 포함해야 한다.

- state: encoder에서 읽은 실제 active-camera yaw/pitch joint position
- action: 해당 timestep의 commanded yaw/pitch target

현재 `xr_teleoperate`의 기본 `EpisodeWriter`와 이 폴더의 일방향 pose UDP에는 이
두 값의 feedback/recording 경로가 아직 없다. 따라서 지금 수집한 영상은 active
viewpoint를 포함하지만, 그대로는 policy가 카메라 관절 명령을 출력하도록 학습할
수 없다. 이 항목은 실제 모터 present-position feedback과 EpisodeWriter 확장으로
추가해야 논문의 완전한 `A` policy action/state space가 된다.

## PC2 clock 동기화 원칙

카메라와 두 목 모터가 모두 PC2에 연결되므로, 향후 dataset 동기화의 canonical
clock은 PC2의 `time.monotonic_ns()`로 통일한다. Host wall clock이나 UDP packet의
Unix timestamp를 sensor alignment 기준으로 사용하지 않는다.

현재 기본 세팅은 다음 timestamp 기반을 준비한다.

- ZED: 한 번의 성공한 `grab()` 직후 `latest_pc2_monotonic_ns` 기록
- ZED SDK image timestamp: `latest_zed_image_time_ns`에 진단용으로 별도 보존
- neck target: receiver가 target을 만든 PC2 monotonic timestamp를 로그에 기록
- motor command: 실제 Sync Write 직전 PC2 monotonic timestamp를 로그에 기록

아직 dataset 수집 단계가 아니므로 encoder history buffer, frame/state resampling,
EpisodeWriter 병합은 구현하지 않는다. 이후 수집기를 만들 때 같은 PC2 monotonic
domain에서 `image`, `actual motor state`, `motor command`를 정렬하면 된다.

## 포트

- `8012/TCP`: 기존 Vuer HTTPS/WSS
- `60000/TCP`: teleimager camera-config service
- `60001/TCP/UDP`: ZED head-camera WebRTC signaling/media
- `55555/TCP`: 녹화용 stereo JPEG ZMQ stream
- `5005/UDP`: Host에서 PC2로 보내는 head pose

## 테스트

```bash
cd active_camera
python3 -m unittest discover -s tests -v
```

테스트는 packet, matrix/quaternion, yaw/pitch, motor position 변환과 함께 ZED
좌우 배치 및 active-stereo-only 설정 검증을 수행한다. ZED/Quest/U2D2 실기
end-to-end 동작은 하드웨어에서 별도로 확인해야 한다.
