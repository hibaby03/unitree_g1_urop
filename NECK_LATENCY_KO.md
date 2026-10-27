# Quest → 목 모터 지연 계측

제어 알고리즘을 변경하지 않고 ①~⑦을 기록하는 **선택형 계측**이다.
환경변수 `NECK_TRACE_DIR`가 있을 때만 로그·시계 측정이 동작한다.
모터 속도, 가속도, 평활화, 패킷 처리 순서, 엔코더 읽기 순서는 그대로다.
계측과 파일 기록에도 작은 실행 비용은 있으므로 완전히 무부하 측정은 아니다.

## 배치

**Host에는 `host/`만, PC2에는 `pc2/`만 복사한다.** 별도의 공용 코드 폴더는 없다.
Host 폴더는 기존처럼 `active_camera_host/`로 이름을 바꿔도 된다.
추가된 파일의 배치는 다음과 같다.

```text
host/                         → Host Ubuntu에 복사
  neck_trace.py               # 자세 이벤트 ID·시각, 비동기 로그
  neck_clock.py               # PC2 시계 차이 측정 클라이언트
  analyze_neck_latency.py      # 양쪽 로그를 합쳐 CSV·통계 생성
  README_NECK_LATENCY_KO.md    # Host 실행 안내
  ...기존 Host 코드
pc2/                          → PC2 Ubuntu에 복사
  neck_trace.py               # PC2 비동기 로그·계측 포트 설정
  frame_timing.py             # 기존 시계 측정 응답 서버 재사용
  README_NECK_LATENCY_KO.md    # PC2 실행 안내
  ...기존 PC2 코드
```

각 폴더의 계측 코드가 다른 PC의 폴더를 import하지 않는다.
두 `neck_trace.py`는 각 PC에서 독립 실행하기 위한 로컬 모듈이다.
외부 `xr_teleoperate` 파일을 편집할 필요는 없다. 추가 계측 코드는 Python 표준 라이브러리만 사용한다.

계측 가능한 Host 진입점:

- `host/zmq_stereo_vr.py`: ZMQ 영상 + 목 추종
- `host/run_teleop_with_neck.py --input-mode=none`: WebRTC 영상 + 목 추종
- `host/run_teleop_with_neck.py`: 기존 full teleop
- `host/teleop_record_dex3_tactile.py`: 녹화용 teleop

TeleVuer 경로는 기존 Host Linux의 `fork` 실행 방식과 `on_cam_move` API를 요구한다.
녹화/full teleop에서는 설치된 wrapper가 참조하는 TeleVuer 클래스에 실행 중인
프로세스 안에서만 후크를 설치하고, fork 전에 공유 메모리를 만든다.

## 실행

아래는 **각 PC에 복사한 host 또는 pc2 폴더 안**에서 실행한다. 기존 영상 서버는 평소처럼 실행한다.
PC2를 먼저 시작하면 Host의 첫 시계 측정부터 응답할 수 있다.

PC2: 기존 목 수신기 실행에 로그 디렉터리만 추가한다.

```bash
NECK_TRACE_DIR="$PWD/latency_logs/pc2" python3 head_pose_receiver.py \
  --enable-motor \
  --yaw-limit-deg 15 --pitch-down-limit-deg 15 --pitch-up-limit-deg 15
```

Host: 현재 사용하는 진입점 **하나만** 실행한다. 예를 들어 ZMQ 뷰어라면:

```bash
NECK_TRACE_DIR="$PWD/latency_logs/host" python3 zmq_stereo_vr.py
```

WebRTC 목 전용이라면 기존 인증서·영상 설정을 유지해서:

```bash
NECK_TRACE_DIR="$PWD/latency_logs/host" python3 run_teleop_with_neck.py \
  --input-mode=none \
  --video-cert /실제/인증서/cert.pem --video-key /실제/인증서/key.pem
```

full teleop/녹화도 기존 실행 명령 앞에 같은 `NECK_TRACE_DIR=...`를 붙인다.
환경변수는 절대 경로 사용을 권장한다. 일부 launcher는 작업 디렉터리를 바꾼다.
환경변수를 지정하지 않으면 기존 실행처럼 계측 없이 동작한다.

- 제어: 기존 UDP **5005**, 목 상태: 기존 UDP **5006**.
- 이번 시계 측정: PC2 UDP **5010**, Host의 임시 포트로 응답.
- 영상 서버의 기존 시계 측정 UDP **5007**과 별개다. 영상 없이도 측정 가능하다.
- 5010을 변경하려면 양쪽에 같은 `NECK_TRACE_SYNC_PORT`를 지정한다.
- PC2 방화벽은 Host에서 들어오는 해당 UDP 포트와 응답을 허용해야 한다.
- 엔코더 시간을 측정하려면 `--enable-motor`를 쓰고 `--no-neck-state`는 쓰지 않는다.
  기존 코드에서 엔코더 읽기는 목 상태 피드백이 켜졌을 때만 수행된다.
- `--enable-motor`를 빼면 모터를 구동하지 않는 통신·계산 계측이 가능하지만
  실제 직렬 통신/엔코더/물리적 추종 지연은 측정되지 않는다.

수십 초 동안 일정한 좌우·상하 움직임과 잠깐 멈춤을 반복한다.
Ctrl+C로 각각 종료하면 로그가 비워지고 `trace_end`가 기록된다.
강제 종료/전원 차단 시 마지막 일부 기록과 종료 표식이 없을 수 있다.

## 기록 지점과 의미

모든 `*_host_ns`, `*_pc2_ns`는 해당 PC의 `time.monotonic_ns()` 값이다.
정수 나노초로 저장하며 정밀도가 나노초라는 의미는 아니다.

| 단계 | 필드 | 의미 |
| --- | --- | --- |
| ① | `pose_rx_host_ns` | Host CAMERA_MOVE 콜백 진입. Quest 센서 획득 시각은 아님 |
| ② | `send_start_host_ns`, `send_end_host_ns` | UDP sendto 호출 전후 |
| ③ | `rx_pc2_ns` | PC2 recvfrom 반환 직후. 커널 도착 시각은 아님 |
| ④ | `target_ready_pc2_ns` | 중립점·각도 변환·제한·필터 계산 완료 |
| ⑤ | `write_start_pc2_ns`, `write_end_pc2_ns` | write_angles 호출 전후. 각도→카운트 변환 및 Sync Write 포함 |
| ⑥ | `read_start_pc2_ns`, `read_end_pc2_ns` | read_present_angles 호출 전후. 통신 오류에도 기록 |
| ⑦ | `loop_end_pc2_ns` | 피드백·기존 콘솔 출력을 끝낸 시각. 비동기 로그 큐 삽입 직전 |

`pose_id`는 **CAMERA_MOVE 이벤트마다** 증가한다. 머리가 정지해 값이 같아도 새 이벤트면 증가한다.
하나의 이벤트를 60Hz 송신 루프가 여러 번 보내면 `pose_id`는 같고 `packet_seq`는 다르다.
①과 자세 값은 같은 공유 메모리 잠금 안에서 복사한다.
Host 로그는 **실제로 송신에 사용된 이벤트**를 기록한다. 송신 사이에 여러 이벤트가
도착하면 사용되지 않은 이벤트는 별도 로그로 저장하지 않으며 pose_id가 건너뛸 수 있다.

48바이트 제어 패킷과 모터 명령은 기존 규격 그대로다.
양쪽에 이미 있는 `(packet_seq, sender_time_ns)`를 `packet_uid`로 연결한다.
PC2 원시 로그에는 Quest의 pose_id가 없고, **분석된 packets.csv에 Host의 pose_id가 합쳐진다.**
`sender_time_ns`는 원래 패킷의 Unix epoch 값으로, 연결 키로만 사용한다.
지연 계산에 동기화하지 않은 wall clock 차이를 쓰지 않는다.
입력 matrix/quaternion, 최종 목표 각도·카운트, 실제 엔코더 각도·카운트와 오류도 기록한다.
기존 콘솔의 `clock_delta_ms`는 시계 보정 없는 옛 진단 값이므로 이번 지연 판단에는 쓰지 않는다.

## 시계 차이 측정

Host가 0.25초마다 별도 스레드에서 다음 네 시각을 얻어 `clock_sample`에 보관한다.

```text
t0: Host probe 송신 직전
t1: PC2 probe recvfrom 직후
t2: PC2 응답 작성 시각 (송신 직전)
t3: Host 응답 recv 직후

offset = ((t1 - t0) + (t2 - t3)) / 2   # PC2 monotonic - Host monotonic
RTT = (t3 - t0) - (t2 - t1)
PC2 시각을 Host로 변환 = PC2 시각 - offset
```

분석기는 각 패킷 송신 시점 이전 10초 안의 유효 probe 중 RTT가 가장 짧은 것을 쓴다.
유효한 측정이 없으면 PC 간 지연은 빈 값으로 남기며, 같은 PC 내부 구간은 계산한다.
`clock_uncertainty_ms`는 RTT/2다. probe 시점의 안정적인 시계 속도와 양방향 비음수
전달 지연을 전제로 한 범위이며, 장기간 drift까지 포함한 보장 오차는 아니다.
`clock_sample_age_ms`도 같이 확인한다. 이는 OS 시계를 변경하는 동기화가 아니라
두 monotonic 시계 사이의 오프셋 추정이다. 음수 보정 지연도 숨기지 않는다.

## 로그 합치기와 분석

종료 시 각 디렉터리에 `host-<run_id>.jsonl`, `pc2-<run_id>.jsonl`이 남는다.
대응하는 두 파일을 Host로 모은 뒤 Host 폴더 안에서 실행한다.

```bash
python3 analyze_neck_latency.py \
  --host latency_logs/host/host-실제ID.jsonl \
  --pc2 latency_logs/pc2/pc2-실제ID.jsonl \
  --out latency_report_run01
```

`--out`은 새 디렉터리를 지정한다. 기존 결과를 덮어쓰지 않는다.

- `packets.csv`: 패킷별 원시 시각, pose_id, 목표/실제 각도, 오류, 계산된 지연.
- `summary.json`: 각 구간의 min/p50/p95/max, 매칭 수, 시계 측정 유무,
  엔코더 오류 수, 로그 손실 수, 정상 종료 여부.

| 분석 필드 | 해석 |
| --- | --- |
| `host_pose_age_ms` | ①→②. Host가 얼마나 오래된 자세를 송신했는지 |
| `host_send_call_ms` | sendto 호출 비용 |
| `send_to_pc2_dequeue_ms` | ②→③ 보정값. 네트워크 + PC2 수신 큐 + 스케줄링 포함 |
| `pc2_receive_to_target_ms` | ③→④. 디코드/검사/목표 계산 |
| `pc2_target_to_write_ms` | ④→⑤ 시작 |
| `motor_write_ms` | ⑤. 모터 목표 전송 호출 비용 |
| `encoder_read_ms` | ⑥. 엔코더 읽기 비용 |
| `feedback_and_logging_ms` | 엔코더 읽기 후→⑦. 읽기가 없으면 마지막 처리 지점부터 |
| `pc2_service_ms` | ③→⑦. 수신한 패킷 처리 시간 (로그 큐 삽입은 제외) |
| `pose_to_command_sent_ms` | ①→⑤ 종료 보정값. 실제 모터 움직임 완료까지의 시간은 아님 |

60Hz라면 패킷 간격은 약 16.7ms다. `pc2_service_ms`가 지속적으로 이보다 크고
`send_to_pc2_dequeue_ms`가 실행하면서 증가하면 수신 적체의 근거가 된다.
그 값만으로 네트워크와 커널 큐를 분리하지는 못한다. 커널 수신 timestamp는 이번 범위 밖이다.
물리적 추종은 목표 각도와 실제 각도의 **시간 이력**을 비교해야 한다.
한 행의 엔코더 값이 그 행의 목표에 도달한 결과라고 해석하면 안 된다.

로그는 최대 8192개 큐에 넣고 백그라운드에서 저장한다. 큐가 가득 차면 제어 패킷이
아닌 **진단 로그**를 버리고 종료 표식/경고에 개수를 남긴다. 디스크 오류도 경고한다.
정상 종료 표식이 없으면 손실 수는 확정할 수 없다.
로그 불일치는 UDP 손실, 기록 누락, 양쪽 기록 시작·종료 차이 모두 가능하다.

실제 로봇과 Quest에서 아직 지연을 측정하지 않았다. 로컬 테스트는 프로토콜 호환,
네 시각 시계 추정, 콜백 메타데이터, 모의 모터 정상/오류, 로그 합치기를 검증한다.
