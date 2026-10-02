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
  │                                           ├─ UDP :5005 ─> PC2 head_pose_receiver
  │                                           │                  └─ U2D2
  │                                           │                     ├─ yaw
  │                                           │                     └─ pitch
  │                                           └─ UDP :5006 <─ 목 명령/엔코더 yaw·pitch
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
- `host/teleop_record_dex3_tactile.py`: G1_29 + Dex3-1 teleop 루프. head pose 송신과
  함께 팔·손·Dex3-1 tactile·목 yaw/pitch를 episode에 녹화
- `host/zmq_stereo_vr.py`: Host가 ZMQ로 받은 ZED 영상을 Vuer로 Quest 양안에 표시하는
  독립 뷰어. 선택적으로 목 추종(3-2절)
- `host/head_pose_udp.py`: OpenXR matrix 검증, quaternion 변환, UDP packet 생성
- `host/neck_state_receiver.py`: PC2가 보낸 목 명령/엔코더 packet 수신과 녹화용 변환
- `host/keyboard_2xl430.py`: Windows에서 2XL430 두 축을 키보드로 움직이는 확인용 도구
  (`host/README_2XL430_KEYBOARD_KO.md`)
- `pc2/head_pose_receiver.py`: UDP 수신, 최초 pose 중립점, yaw/pitch 제한, 목 상태
  피드백 송신
- `pc2/dynamixel_neck.py`: U2D2를 통한 두 축 DYNAMIXEL Sync Write / 엔코더 Sync Read
- `pc2/neck_state_sender.py`: 목 명령/엔코더 상태 UDP packet 생성
- `pc2/frame_timing.py` / `host/frame_timing.py`: ZMQ JPEG에 PC2 캡처 시각 삽입·해석,
  Host↔PC2 monotonic clock offset 측정(UDP 5007), 녹화 item의 `states.timing` 생성
- `PROTOCOL.md`: head-pose(48 byte, Host→PC2)와 neck-state(60 byte, PC2→Host)
  UDP packet, JPEG frame stamp, clock-sync packet 규격

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
  --dxl-device /dev/ttyUSB0 --dxl-baudrate 57600 \
  --yaw-id 5 --pitch-id 6 \
  --yaw-center 2048 --pitch-center 2048 \
  --yaw-sign 1 --pitch-sign 1 \
  --yaw-limit-deg 80 \
  --pitch-down-limit-deg 35 --pitch-up-limit-deg 45 \
  --smoothing 1.0 \
  --motor-timeout-ms 1000
```

모터 위치는 `center + sign × 각도 × 4096/360`이다. 기본 center는 2048이므로
오른쪽/위로 10°면 2048 + 114, 왼쪽/아래로 10°면 2048 − 114 count가 된다. 방향이
반대면 `--yaw-sign -1` 또는 `--pitch-sign -1`을 준다.

`--yaw-id`/`--pitch-id`/`--dxl-baudrate`는 실제 모터 설정과 같아야 한다. 기본값은
실측값(ID 5/6, 57600 bps)이며 `host/keyboard_2xl430.py`와 같다. 모터 ID/baudrate를
모를 때는 DYNAMIXEL SDK의 `ping`으로 여러 baudrate를 훑어 확인한 뒤 그대로 넣는다.

U2D2 USB만으로 모터 전원을 공급하지 말고 2XL430용 외부 전원을 사용한다.

### 목 상태 피드백 (PC2 → Host)

receiver는 pose를 받아 명령할 때마다 두 모터의 Present Position을 Sync Read 한 번으로
읽고, 명령값과 엔코더 값을 60-byte UDP packet으로 Host에 돌려보낸다. 기본 목적지는
head pose를 보낸 주소의 `5006` 포트다.

| 옵션 | 의미 |
| --- | --- |
| `--neck-state-port 5006` | Host 수신 포트 |
| `--neck-state-host <ip>` | 목적지를 직접 지정 (기본: head pose 송신 주소) |
| `--no-neck-state` | 피드백 끄기 |

`--enable-motor` 없이 실행하면 명령값만 보내고 엔코더 값은 비어 있다. 엔코더 읽기나
송신이 실패하면 `{"event":"neck_state_error",...}`를 stderr에 한 번 출력하고, 명령은
계속한다.

## 3-1. G1 없이 영상 + 목만 확인 (`--input-mode=none`)

Quest 헤드셋, PC2의 ZED Mini와 2XL430만으로 실시간 영상과 목 추종을 함께
확인한다. Quest 손 컨트롤러와 G1/Dex3 연결은 필요 없다. Host 실행기는
팔·손 컨트롤러, IK, DDS, 녹화기를 초기화하지 않는다. PC2 코드는 수정하지 않는다.

1. PC2에서 기존 `python3 pc2/zed_teleimager_server.py`를 실행한다.
2. Host에서 아래 명령을 실행하고 Quest에서 기존 Host HTTPS/WSS 주소로 접속해
   VR 세션에 들어간다. 인증서 경로는 기존 영상 확인 때 사용한 값으로 바꾼다.

```bash
cd ~/hckang/xr_teleoperate
python active_camera_host/run_teleop_with_neck.py \
  --xr-repo . \
  --input-mode=none \
  --video-offer-url https://192.168.123.164:60001/offer \
  --video-cert /path/to/cert.pem --video-key /path/to/key.pem \
  --neck-pose-ip 192.168.123.164 --neck-pose-port 5005 --neck-pose-rate 60
```

3. PC2의 다른 터미널에서 먼저 모터 없이 `python3 pc2/head_pose_receiver.py`를
   실행해 머리를 돌릴 때 yaw/pitch가 갱신되는지 확인한다. 중단한 뒤 정면을 보고
   3절의 `--enable-motor` 명령을 실제 ID, baudrate, center, sign에 맞춰 실행한다.
   첫 시험은 yaw/pitch 제한을 각각 15도로 낮춰 확인한다.
4. 머리를 천천히 좌우·위아래로 움직이며 목 추종과 Quest 영상의 연속성을 확인한다.
   녹화는 하지 않는다. Host에서 Ctrl+C로 종료한다.

이 모드는 기존 영상 전용 모드와 같은 legacy TeleVuer API 및 Linux `fork`
실행 환경을 사용한다. `--input-mode=none`은 `--record`, `--arm`, `--ee` 등
전체 teleop 옵션이나 `--video-only`와 함께 사용할 수 없다. 영상 주소와 목 UDP
주소는 각각 `--video-offer-url`, `--neck-pose-ip`로 지정한다.

새 `CAMERA_MOVE`가 250 ms 이상 들어오지 않으면 머리 자세 송신을 멈춘다.
PC2는 마지막 정상 패킷 이후 기존 `--motor-timeout-ms`(기본 1000 ms)가 지나면
토크를 끄고 종료한다. 정상 자세가 한 번도 오지 않은 경우도 PC2 시작 시점부터
타임아웃을 적용하므로 Host/Quest를 먼저 준비한다. 타임아웃 후에는 PC2 목
수신기를 다시 실행해야 한다. 이 검사는 이벤트 수신 중단을 감지하며,
Quest 자체 추적 품질을 별도로 판정하지는 않는다.

기존 `--video-only`는 계속 영상만 표시하며 머리 자세 UDP를 보내지 않는다.
`--input-mode=hand` / `controller` 및 입력 모드 생략은 기존 전체 teleop 경로다.

## 3-2. Host가 받은 영상을 Quest로 보기 (`zmq_stereo_vr.py`)

3-1절은 Quest가 PC2의 WebRTC를 직접 재생한다. 이 뷰어는 녹화 경로와 같은
PC2 ZMQ JPEG(55555)를 Host가 받아 decode한 뒤 Vuer `ImageBackground`의 왼쪽/오른쪽
눈 레이어(`layers=1/2`)로 보낸다. 따라서 Quest에서 보는 영상이 녹화되는 영상과
같다. xr_teleoperate의 TeleVuer, DDS, IK, 녹화기는 띄우지 않으므로 G1 없이 돌릴 수
있고, legacy TeleVuer 버전이나 Linux `fork` 제약도 받지 않는다. 필요한 패키지는
`vuer[all]`, `pyzmq`, `opencv-python`, `numpy`이며 xr_teleoperate의 `tv` 환경에 이미 있다.

```bash
# PC2
python3 pc2/zed_teleimager_server.py

# Host (영상만)
python active_camera_host/zmq_stereo_vr.py --img-server-ip 192.168.123.164

# Host (영상 + 목 추종, PC2에서 head_pose_receiver.py 실행)
python active_camera_host/zmq_stereo_vr.py --img-server-ip 192.168.123.164 \
  --neck-pose-ip 192.168.123.164
```

Quest 브라우저에서 기존과 같은 `https://192.168.123.2:8012/?ws=wss://192.168.123.2:8012`로
접속한다. 인증서는 `--cert/--key`, `XR_TELEOP_CERT/KEY`, `~/.config/xr_teleoperate/` 순으로
찾는다. 2초마다 수신/송신 fps와 영상 지연(PC2 캡처 → Host 시점, clock sync 필요)이
출력된다. Quest 화면까지의 지연은 여기에 Vuer JPEG 재인코딩과 Wi-Fi 전송이 더해진다.

- Vuer 서버가 8012를 쓰므로 teleop/녹화 스크립트와 동시에 실행하지 않는다.
- `--neck-pose-ip`를 주면 `CAMERA_MOVE`를 head-pose packet으로 PC2에 보낸다.
  250 ms 이상 새 pose가 없으면 송신을 멈추므로 Quest가 끊기면 PC2 watchdog이
  토크를 끈다(3-1절과 같은 규칙). 데스크톱 브라우저로 같은 주소에 접속해 시점을
  돌리면 그 카메라 이동도 head pose로 들어가므로, 목 추종 중에는 Quest만 접속한다.
- 화면 크기/거리는 `--distance`(기본 1 m), 화질은 `--jpeg-quality`(기본 80),
  표시 주기는 `--display-fps`(기본 30)로 조정한다. 같은 frame은 다시 보내지 않는다.

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
active-stereo-only visual observation과 일치한다. 이 launcher는 설치된
`teleop_hand_and_arm.py`를 그대로 실행하므로 Dex3-1 tactile과 목 yaw/pitch는
저장하지 않는다.

### 4-1. 팔·손·tactile·목 값을 함께 녹화

`host/teleop_record_dex3_tactile.py`는 G1_29 + Dex3-1 + hand tracking 전용
teleop/녹화 스크립트다. `teleop_hand_and_arm.py`와 같은 모듈(TeleVuerWrapper,
G1_29_ArmIK/Controller, Dex3_1_Controller, ImageClient, EpisodeWriter)로 루프를 직접
돌리며, 기존 launcher처럼 head pose를 PC2로 보낸다. PC2에서는 2·3절의
`zed_teleimager_server.py`와 `head_pose_receiver.py`를 그대로 실행한다.

```bash
cd ~/xr_teleoperate
python active_camera_host/teleop_record_dex3_tactile.py \
  --xr-repo . \
  --img-server-ip 192.168.123.164 \
  --neck-pose-ip 192.168.123.164 --neck-pose-port 5005 --neck-pose-rate 60 \
  --neck-state-port 5006 \
  --record \
  --task-dir ./utils/data --task-name active_stereo_task
```

키는 기존과 같다: `r` 추종 시작, `s` episode 시작/저장, `q` 종료. `--motion`,
`--headless`, `--display-mode`, `--network-interface`, `--frequency`(기본 30)와
`--task-goal/desc/steps`도 받는다. `--arm`, `--ee`, `--input-mode`는 고정이므로 주지
않는다. `--task-dir`는 기존과 같이 `teleop/` 기준 상대 경로다.

각 item에 저장되는 값:

| 키 | 내용 |
| --- | --- |
| `colors.color_0` / `color_1` | ZED left / right |
| `states.left_arm/right_arm` | `qpos` 현재 q, `qvel` 현재 dq |
| `actions.left_arm/right_arm` | `qpos` IK 결과 q, `torque` feed-forward tau |
| `states/actions.left_ee/right_ee` | Dex3-1 손 state/action 각 7개 |
| `tactiles.left_ee/right_ee` | Dex3-1 press sensor별 `pressure`[12], `temperature`[12], `lost`, 수신 후 경과 `age_ms` (한 번도 안 받았으면 `null`) |
| `states.neck` | 엔코더 [yaw, pitch] `qpos`(rad), `position_counts`, `pc2_monotonic_ns`, `age_ms` |
| `actions.neck` | 명령 [yaw, pitch] `qpos`(rad), goal `position_counts`, `pc2_monotonic_ns`, `pose_sequence`, `torque_enabled`, `age_ms` |
| `states.timing` | item 시각과 관측 지연. 아래 표 참고 |

`states.timing` (시각은 모두 ns, 지연은 ms, 모를 때는 `null`):

| 키 | 내용 |
| --- | --- |
| `host_monotonic_ns` | 팔 q를 읽은 직후의 Host `time.monotonic_ns()` (item 기준 시각) |
| `image_frame_sequence` | PC2가 붙인 ZED frame 번호. 연속 item에서 같으면 같은 영상이 재사용된 것 |
| `image_pc2_monotonic_ns` | ZED `grab()` 직후 PC2 monotonic 시각 |
| `image_host_monotonic_ns` | 위 시각을 Host clock으로 옮긴 값 |
| `image_age_ms` | item 기준 시각에 영상이 얼마나 오래됐는지 (캡처 + JPEG + 전송 + 대기) |
| `neck_present_age_ms` / `neck_command_age_ms` | 같은 기준의 목 엔코더 / 명령 지연 |
| `clock_offset_ns` / `clock_uncertainty_ms` | PC2 − Host clock offset과 그 오차 상한(가장 빠른 probe의 RTT/2) |

녹화 루프는 monotonic clock의 고정 주기 스케줄로 돈다. 루프가 늦어지면 누적해서
따라잡지 않고 다시 시작하므로, 실제 간격은 `host_monotonic_ns` 차이로 확인한다.
episode를 저장할 때 영상 지연의 median/p95/max가 로그에 찍힌다.
clock sync는 PC2 `zed_teleimager_server.py`가 UDP 5007로 응답하며, 응답이 없으면
경고 후 지연 값만 `null`로 녹화한다(`--clock-sync-port`, `--no-clock-sync`).

목 yaw는 오른쪽, pitch는 위가 양수이며 center(기본 2048) 기준이다. Host는 녹화
시점에 가장 최근 packet을 넣으므로, 더 정밀한 정렬이 필요하면 `pc2_monotonic_ns`로
보간한다. 목 packet을 아직 받지 못했으면 `neck.qpos`는 빈 리스트다.

Dex3-1 tactile은 `Dex3_1_Controller`와 별도로 `rt/dex3/{left,right}/state`를 콜백
방식으로 구독하여 `HandState_.press_sensor_state`를 읽는다. 한쪽 손 데이터가 끊겨도
다른 쪽은 계속 갱신된다.

> **Host xr_teleoperate 버전 확인 필요.** 이 스크립트는 최신 upstream API
> (`teleimager.ImageClient`, `TeleVuerWrapper(display_mode=, zmq=, webrtc=,
> webrtc_url=, arm_reference_mode=)`)를 기준으로 작성되었다. Host에 예전 TeleVuer
> (`img_shm_name` 기반)가 설치되어 있으면 생성자 호출이 실패하므로, 아래 결과로
> 버전을 먼저 확인한다.
>
> ```bash
> cd ~/hckang/xr_teleoperate
> git log -1 --format='%H %cd'
> grep -n "def __init__" -A3 teleop/televuer/src/televuer/tv_wrapper.py
> ```

## 데이터셋 관점: active-camera state/action

논문 Fig. 3처럼 ACT policy가 추론 시 카메라 자체도 움직이게 하려면 데이터셋의
state/action vector에 다음 2-DoF 값이 포함되어야 한다.

- state: encoder에서 읽은 실제 active-camera yaw/pitch joint position
- action: 해당 timestep의 commanded yaw/pitch target

`host/teleop_record_dex3_tactile.py`와 PC2 neck state 피드백(UDP 5006)으로 이 두
값이 `states.neck`, `actions.neck`에 저장된다(4-1절). 기존
`run_teleop_with_neck.py --record`에는 여전히 목 값이 저장되지 않으므로, policy
학습용 데이터는 4-1절 스크립트로 수집한다.

## 5. ACT 학습과 실행

`training/train_act.py`는 4-1절로 녹화한 task 폴더(`episode_*/data.json`)를 그대로
읽는다. 입력은 ZED `color_0`/`color_1`과 30차원 state(팔 7+7, Dex3-1 7+7, 목 2),
출력은 같은 순서의 action chunk다. `--no-neck`은 목을 빼고 학습한다.

```bash
pip install -r training/requirements.txt
python training/train_act.py \
  --data-dir ~/xr_teleoperate/teleop/utils/data/active_stereo_task \
  --out-dir runs/act_active_stereo
```

`host/deploy_act.py`는 학습된 체크포인트로 팔·손·목을 제어한다. PC2에서는 녹화 때와
같이 `zed_teleimager_server.py`와 `head_pose_receiver.py --enable-motor`를 실행한다.
Quest는 필요 없다. Host의 `tv` 환경에 torch/torchvision이 있어야 하며, `host`만
복사했다면 `--train-script`로 `train_act.py` 경로를 준다.

```bash
cd ~/xr_teleoperate
python active_camera_host/deploy_act.py \
  --xr-repo . \
  --checkpoint /path/to/policy_best.ckpt \
  --train-script /path/to/training/train_act.py \
  --img-server-ip 192.168.123.164 \
  --neck-pose-ip 192.168.123.164 \
  --dry-run
```

키: `r` 시작/재개, `p` 일시정지(마지막 명령 유지), `q` 종료. 처음에는 `--dry-run`으로
명령 없이 예측값만 확인한 뒤 뺀다.

- 팔: `G1_29_ArmController`에 관절 목표와 pinocchio RNEA 중력 보상 torque를 보낸다.
- 손: retargeting 없이 `rt/dex3/{left,right}/cmd`에 관절 목표를 직접 보낸다
  (`Dex3_1_Controller`와 같은 kp 1.5 / kd 0.2).
- 목: yaw/pitch를 OpenXR quaternion으로 바꿔 기존 head-pose packet으로 보낸다. 시작
  시 recenter flag로 identity를 중립점으로 잡으므로 목이 먼저 center로 이동한다.
- 모든 명령은 step마다 `--max-arm-step`(0.05 rad), `--max-hand-step`(0.1 rad),
  `--max-neck-step-deg`(3°)로 제한되어 첫 정책 출력으로도 천천히 들어간다.
- 기본은 ACT temporal ensembling(`--ensemble-k 0.01`)이며, `--no-temporal-agg`로
  chunk를 open-loop로 실행할 수 있다. `--frequency`는 녹화 주파수(30)와 같아야 한다.
- 종료 시 목은 center로 천천히 돌아가고, 팔은 녹화 스크립트와 같이 home으로 이동한다.

### 5-1. 이미 처리한 예외 상황

실제 녹화·실행에서 오류가 날 수 있어 미리 처리한 부분이다.

`training/train_act.py`

- 녹화 중 ZED 프레임이 `None`이면 해당 item이 `colors={}`로 저장된다. 이 프레임은
  앞(없으면 뒤) 프레임 이미지로 채우고 경고만 출력한다.
- `s`로 저장하기 전에 녹화가 죽으면 `data.json`이 닫히지 않아 JSON이 깨진다
  (`EpisodeWriter`는 item을 이어 쓰고 저장 시 `]}`를 붙인다). 이 에피소드는 건너뛴다.
- 목 값이 에피소드 전체에서 비어 있으면(PC2를 `--enable-motor` 없이 실행) 그
  에피소드는 건너뛴다. 일부 timestep만 비어 있으면 앞뒤 값으로 채운다.

`host/deploy_act.py`

- `--dry-run`은 목 packet을 보내지 않아 PC2 목 상태도 오지 않으므로, 목 state를
  center(0)로 두고 진행한다.
- 일시정지 중이나 state/영상이 잠시 비었을 때도 마지막 명령을 계속 다시 보낸다.
- 종료 시 목을 약 1초에 걸쳐 center로 되돌린다. PC2는 자체 속도 제한이 없어
  바로 0을 보내면 목이 튄다.
- `--train-script` 경로가 없으면 바로 명확한 오류를 낸다.

### 5-2. 아직 남은 알려진 문제

아래는 확인됐거나 가능성이 있는 문제로, 아직 코드에 반영하지 않았다. 실기 실행 전에
우회 방법을 따르거나 수정한다.

**`host/deploy_act.py`**

1. **PC2 목 receiver가 먼저 종료된다 (확인됨).** `head_pose_receiver.py --enable-motor`는
   시작 후 `--motor-timeout-ms`(기본 1000 ms) 안에 pose가 오지 않으면 토크를 끄고
   종료한다. deploy는 `r`을 누른 뒤에야 목 packet을 보내므로 README 순서대로 PC2를
   먼저 켜면 receiver가 이미 죽어 있고, `r` 후 3초 뒤 "no arm/hand/neck state"로
   멈춘다.
   - 우회: PC2 receiver에 `--motor-timeout-ms`를 충분히 크게 준다(예: 60000). 이
     값은 통신이 끊겼을 때 토크를 끄는 watchdog이기도 하므로 실행 중에는 목을 계속
     지켜본다.
2. **실행 직후 팔이 0 자세로 빠르게 이동한다 (확인됨).** upstream
   `G1_29_ArmController`는 `q_target = zeros(14)`로 생성되고 250 Hz 제어 스레드를
   바로 시작하며, 속도 제한은 30 rad/s다. 따라서 dry-run이 아니면 `r` 전부터 팔이
   0 자세로 거의 순간 이동한다. 종료 시 `ctrl_dual_arm_go_home`도 같은 속도다.
   녹화 스크립트와 upstream teleop도 동일하게 동작한다.
   - 우회: 로봇을 매단 상태에서 팔 주변을 비우고 실행한다.
3. **팔/손 센서가 끊겨도 마지막 값으로 계속 제어한다 (가능성).** 팔/손 state의 수신
   시각을 검사하지 않는다. ZED 영상은 PC2 캡처 시각 기준 지연이
   `--max-image-age-ms`(기본 200)를 넘으면 policy에 넣지 않고 마지막 명령을
   유지한다. 단, clock sync가 없으면(구버전 PC2 서버, UDP 5007 차단) 영상 끊김도
   감지하지 못하고 시작 시 경고만 한다.
4. **`--query-every`가 chunk 크기보다 크면 IndexError (확인됨).** `--no-temporal-agg`와
   함께 쓸 때만 해당한다. chunk 크기 이하로 준다.

**`training/train_act.py`**

5. **`--resume` 시 정규화 통계와 train/val 분할이 바뀐다 (확인됨).** 통계를
   체크포인트에서 가져오지 않고 현재 데이터로 다시 계산한다. 그 사이 에피소드를
   추가했다면 오류 없이 다른 정규화로 이어서 학습되고, 기존 학습 에피소드가 val로
   넘어갈 수 있다.
   - 우회: 에피소드를 추가했다면 resume하지 말고 새로 학습한다.
6. **에피소드가 4개 이하이면 `policy_best.ckpt`가 생기지 않는다 (확인됨).**
   `round(n × val_ratio)`가 0이 되어 val 세트가 없기 때문이다. 이때는
   `policy_last.ckpt`를 쓴다.

## Host/PC2 clock 동기화

카메라와 두 목 모터는 PC2에, 팔·손·tactile은 Host(DDS)에 있다. 두 PC의 clock은
서로 비교할 수 없으므로 다음 원칙을 따른다.

- 각 PC 안에서는 `time.monotonic_ns()`만 쓴다. wall clock(`time.time()`)과 UDP
  packet의 Unix timestamp는 NTP 보정으로 튈 수 있어 정렬 기준으로 쓰지 않는다.
- PC2 시각은 Host가 측정한 offset(PC2 − Host)으로 Host monotonic clock에 옮긴다.
  Host는 0.25초마다 NTP 방식 probe를 PC2 `zed_teleimager_server.py`(UDP 5007)에
  보내고, 최근 약 8초 안에서 RTT가 가장 짧은 probe의 offset을 쓴다. 비대칭 지연에
  의한 오차는 그 RTT의 절반 이하이며 `clock_uncertainty_ms`로 기록된다. 유선
  LAN에서는 보통 1 ms 미만이다. 창을 짧게 두어 두 PC clock의 drift(수십 ppm)도
  무시할 수준으로 유지한다.
- ZED와 목 프로세스는 같은 PC2 `CLOCK_MONOTONIC`을 쓰므로 ZED 서버의 응답기 하나로
  영상과 목 timestamp를 모두 변환한다. chrony/NTP 설정은 필요 없다.

PC2가 남기는 timestamp:

- ZED: 한 번의 성공한 `grab()` 직후 `latest_pc2_monotonic_ns` 기록
- ZED SDK image timestamp: `latest_zed_image_time_ns`에 진단용으로 별도 보존
- neck target: receiver가 target을 만든 PC2 monotonic timestamp를 로그에 기록
- motor command: 실제 Sync Write 직전 PC2 monotonic timestamp를 로그와 neck-state
  packet(`command_pc2_monotonic_ns`)에 기록
- encoder: Present Position Sync Read 직전 PC2 monotonic timestamp를 neck-state
  packet(`present_pc2_monotonic_ns`)에 기록

- ZED frame: 위 `grab()` 시각과 frame 번호를 ZMQ JPEG의 COM segment에 넣어 Host로
  보낸다. JPEG decoder는 COM segment를 무시하므로 teleimager/WebRTC/Quest와
  기존 `xr_teleoperate` 녹화는 영향이 없다.

녹화 스크립트는 이 값들을 `states.timing`(4-1절)에 Host clock 기준으로 남긴다.
episode의 한 item은 여전히 "item 시각의 최신값"끼리 묶이지만, 각 관측의 실제 지연이
기록되므로 학습 시 지연이 큰 frame을 거르거나 state를 영상 시각으로 보간할 수 있다.
frame/state resampling 자체는 아직 구현하지 않았다.

Host는 영상 JPEG를 직접 decode한다(`ImageClient(request_bgr=False)`).
teleimager의 BGR decoder는 별도 스레드라 `.bgr`가 `.jpg`보다 한 frame 늦을 수
있어서, 영상과 stamp가 같은 frame임을 보장하기 위해서다.

## 포트

- `8012/TCP`: 기존 Vuer HTTPS/WSS
- `60000/TCP`: teleimager camera-config service
- `60001/TCP/UDP`: ZED head-camera WebRTC signaling/media
- `55555/TCP`: 녹화용 stereo JPEG ZMQ stream
- `5005/UDP`: Host에서 PC2로 보내는 head pose
- `5006/UDP`: PC2에서 Host로 돌려보내는 목 명령/엔코더 yaw·pitch
- `5007/UDP`: Host→PC2 clock-sync probe와 응답 (`zed_teleimager_server.py`)

## 테스트

```bash
cd active_camera
python3 -m unittest discover -s tests -v
```

테스트는 packet, matrix/quaternion, yaw/pitch, motor position 변환, 엔코더 Sync Read,
neck-state packet 왕복·UDP 수신, JPEG frame stamp·clock offset 계산과 loopback clock
sync, ZMQ→Vuer 뷰어의 양안 분할·head pose 전달, Dex3-1 tactile 변환, 녹화 state/action 배치와 함께
ZED 좌우 배치 및 active-stereo-only 설정 검증을 수행한다. 하드웨어와
xr_teleoperate 없이 가짜 SDK/DDS 객체로 실행된다. `test_zed_stereo`는 `numpy`가
필요하다. ZED/Quest/U2D2 실기
end-to-end 동작은 하드웨어에서 별도로 확인해야 한다.
