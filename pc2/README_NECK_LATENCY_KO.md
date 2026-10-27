# PC2 Ubuntu: 목 추종 지연 계측

이 `pc2/` 폴더 전체만 PC2에 복사한다. Host 코드나 별도 공용 폴더는 필요 없다.
기존 DYNAMIXEL SDK와 모터 연결 설정은 그대로 사용한다.

이 폴더 안에서 실행한다. 기존 실행 옵션을 유지하고 `NECK_TRACE_DIR`만 추가하면 된다.
아래는 기존 15도 제한 예시다.

```bash
NECK_TRACE_DIR="$PWD/latency_logs/pc2" python3 head_pose_receiver.py \
  --enable-motor \
  --yaw-limit-deg 15 --pitch-down-limit-deg 15 --pitch-up-limit-deg 15
```

`neck_trace.py`가 PC2 구간별 시각을 비동기로 기록한다.
`frame_timing.py`의 기존 시계 응답 서버를 계측용 UDP **5010**에서 별도로 실행한다.
영상 서버의 UDP 5007과 충돌하지 않으며 영상 서버가 없어도 동작한다.
Host에서 5010으로 들어오는 UDP 요청과 응답을 허용해야 한다.
포트를 변경하려면 양쪽에 같은 `NECK_TRACE_SYNC_PORT` 환경변수를 설정한다.

엔코더 응답 대기를 측정하려면 `--enable-motor`를 사용하고 `--no-neck-state`는 넣지 않는다.
`--enable-motor`를 빼면 모터를 구동하지 않고 통신·목표 계산만 확인할 수 있다.
계측은 제어 순서·속도·가속도·필터 설정을 변경하지 않는다.

Ctrl+C로 종료하면 `latency_logs/pc2/pc2-실제ID.jsonl`이 남는다.
이 파일을 Host로 복사하고 Host의 `analyze_neck_latency.py`로 양쪽 로그를 합친다.
수신 시각은 프로그램의 `recvfrom()` 반환 시각이며 커널 도착 시각은 아니다.
목표 전송 완료는 모터의 실제 도달 완료를 뜻하지 않는다.
