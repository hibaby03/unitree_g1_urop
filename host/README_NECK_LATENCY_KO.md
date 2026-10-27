# Host Ubuntu: 목 추종 지연 계측

이 `host/` 폴더 전체만 Host에 복사한다. 기존 `active_camera_host/`라는 이름도 가능하다.
PC2 코드나 별도 공용 폴더는 필요 없다. 기존 영상·TeleVuer 의존성과 인증서는 그대로 사용한다.

이 폴더 안에서 기존 실행 명령 앞에 `NECK_TRACE_DIR`를 붙인다. ZMQ 뷰어 예시:

```bash
NECK_TRACE_DIR="$PWD/latency_logs/host" python3 zmq_stereo_vr.py
```

WebRTC 목 전용이면 다음 명령을 대신 사용한다.

```bash
NECK_TRACE_DIR="$PWD/latency_logs/host" python3 run_teleop_with_neck.py \
  --input-mode=none \
  --video-cert /실제/인증서/cert.pem --video-key /실제/인증서/key.pem
```

full teleop/녹화도 기존 명령에 같은 환경변수를 붙인다. 진입점은 동시에 하나만 실행한다.
`neck_trace.py`가 Quest 자세 이벤트 ID·시각과 송신 전후를 기록하고,
`neck_clock.py`가 PC2 UDP 5010으로 시계 차이를 측정한다.
PC2에서도 계측을 켜야 한다. 포트 변경 시 양쪽 `NECK_TRACE_SYNC_PORT`를 동일하게 설정한다.
TeleVuer 계측은 기존 Ubuntu Host의 multiprocessing `fork`와 `on_cam_move` API를 전제로 한다.

Ctrl+C로 종료한 뒤 PC2의 대응 `pc2-*.jsonl`을 Host로 복사하고 이 폴더에서 분석한다.

```bash
python3 analyze_neck_latency.py \
  --host latency_logs/host/host-실제ID.jsonl \
  --pc2 latency_logs/pc2/pc2-실제ID.jsonl \
  --out latency_report_run01
```

`--out`은 새 디렉터리로 지정한다. `packets.csv`와 `summary.json`이 생성된다.
로그에 시계 측정이 없거나 오래됐으면 PC 간 지연은 비워 두고 PC 내부 구간만 계산한다.
보정된 송신→수신 지연에는 네트워크뿐 아니라 PC2 수신 큐·스케줄링 대기도 포함된다.
목표 전송 완료 시각은 실제 모터가 목표에 도달한 시각이 아니다.
