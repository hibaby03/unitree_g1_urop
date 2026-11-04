# 독립 DDS 통신 진단

기존 프로젝트 파일을 수정하지 않고 별도 실행하는 진단 도구입니다.
두 Python 파일을 Host와 PC2의 원하는 폴더에 복사해 사용하세요.
각 컴퓨터에서 `unitree_sdk2py`와 호환되는 `cyclonedds`가 설치된 Python 환경을 사용합니다.
새 설치나 환경 변경은 수행하지 않습니다. `python3`는 해당 환경의 Python을 가리켜야 합니다.

## 1. PC2에서 G1 상태 수신

PC2 터미널에서 실행합니다. 제공된 네트워크 정보에 따라 `eth0`를 사용합니다.

```bash
python3 lowstate_probe.py --interface eth0 --domain 0 --seconds 10
```

- `received`: 이 프로그램의 콜백이 받은 메시지 수
- `motor_count`, `first_motor_q`: 디코딩한 모터 상태의 간단한 확인값
- `decode_errors`: 메시지는 수신했지만 상태 필드 확인에 실패한 횟수
- `last_receive_age_s`: 마지막 수신 이후 경과 시간
- `average_callback_hz`: 대기 시작 이후 평균 콜백 수신률. 로봇의 정확한 발행률이나 패킷 손실률이 아닙니다.
- `PASS`: 유효한 메시지를 수신했고 마지막 수신이 3초 이내이며 필드 확인 오류가 없음
- `FAIL`: 현재 인터페이스/도메인/메시지 타입/로봇 발행 상태에서 확인되지 않음. 네트워크 장애만을 뜻하지 않습니다.

이 코드는 `rt/lowstate`만 구독하며 발행, 모드 변경, 모터 명령을 수행하지 않습니다.
PC2에서 성공해도 손·촉각 토픽이나 motion switcher 서비스까지 검증된 것은 아닙니다.

## 2. Host와 PC2의 무선 DDS 양방향 검사

두 터미널을 몇 초 이내에 시작해 실행 시간이 충분히 겹치게 하세요.
늦게 시작했다면 양쪽을 다시 실행하세요. Host `wlo1`, PC2 `wlan0`는 제공된 정보 기준입니다.

Host:

```bash
python3 wireless_dds_probe.py --role host --interface wlo1 --domain 42 --test-id link_check --seconds 30
```

PC2:

```bash
python3 wireless_dds_probe.py --role pc2 --interface wlan0 --domain 42 --test-id link_check --seconds 30
```

두 컴퓨터의 `--domain`, `--test-id`는 같아야 합니다. 각 역할은 한 프로세스씩 실행하세요.
기본 도메인 42와 전용 `urop_diagnostics/...` 토픽을 사용하므로 로봇 제어 토픽을 발행하지 않습니다.

- Host의 `peer_received > 0`: PC2 → Host 수신 확인
- PC2의 `peer_received > 0`: Host → PC2 수신 확인
- **양쪽 모두 PASS여야 양방향 통신이 확인됩니다.** 한쪽 PASS는 그쪽의 수신 방향만 확인합니다.
- `local_writes_ok`는 로컬 DDS 쓰기 성공이며 상대 수신 성공을 뜻하지 않습니다.
- 기본 판정은 마지막 상대 메시지가 3초 이내인 경우입니다. 낮은 `--hz`에서는 최소 3주기까지 허용합니다.
- 30초 테스트는 연결 확인용이며 제어용 지연·대역폭·장시간 안정성 인증이 아닙니다.
- 도메인 42 성공이 도메인 0 성공을 보장하지 않습니다. 로봇과 같은 도메인 조건까지 비교하려면
  양쪽 명령의 `--domain`을 함께 `0`으로 바꿔 추가 검사할 수 있습니다. 전용 토픽은 유지됩니다.

## 결과 해석

| PC2 G1 수신 | 무선 양방향 | 다음 확인 |
|---|---|---|
| 성공 | 성공 | 두 구간은 독립적으로 동작. G1 DDS를 Host까지 전달할 중계/제어 배치가 필요할 수 있음 |
| 성공 | 실패 | Wi-Fi 구간의 DDS 설정, 양쪽 인터페이스, 검색 패킷·UDP 도달 여부 확인 |
| 실패 | 성공 | PC2 eth0 설정, G1 발행 서비스, 도메인, SDK/IDL 타입 확인 |
| 실패 | 실패 | 내부망과 무선망을 분리해서 각각 진단 |

PC2는 두 네트워크 사이에서 자동 중계하지 않습니다. 이 도구도 중계 기능을 만들지 않습니다.
Host–PC2 무선 연결을 유지하면서 각 구간만 검사합니다.
오류 발생 시 명령, 양쪽 전체 출력, SDK와 Cyclone DDS 버전을 함께 비교하세요.
프로세스 재실행 없이 인터페이스를 바꾸지 마세요. SDK의 DDS 초기화는 프로세스 내에서 재사용될 수 있습니다.

종료 코드: 성공 `0`, 수신 실패 `1`, 초기화/실행 오류 `2`, Ctrl+C `130`.

## 구현 근거

Unitree 공식 SDK의 `ChannelFactoryInitialize`, `ChannelSubscriber`, `ChannelPublisher` 및
공식 사용자 정의 IDL 예제와 같은 API를 사용합니다.

- https://github.com/unitreerobotics/unitree_sdk2_python/tree/master/example/helloworld
- https://github.com/unitreerobotics/unitree_sdk2_python/blob/master/unitree_sdk2py/core/channel.py

로컬 확인은 구문/도움말 및 가짜 DDS 객체를 이용한 실행 검사로 제한됩니다.
실제 네트워크와 로봇 수신은 위 명령을 각 장비에서 실행해 확인해야 합니다.
