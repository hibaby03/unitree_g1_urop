# 2XL430 키보드 단독 테스트 (Windows Host PC)

`keyboard_2xl430.py`는 U2D2와 외부 전원 허브에 연결한 2축 2XL430을 키보드로
조금씩 움직이는 확인용 프로그램이다. 시작 시 토크는 **꺼져** 있으며 `T`를 눌러야
현재 엔코더 위치에서 토크를 켠다. 따라서 프로그램이 시작되었다고 중심점(2048)으로
갑자기 움직이지 않는다.

## 연결 전 확인

1. **전원:** U2D2 USB만으로는 모터를 구동하지 않는다. 전원 허브에서 2XL430의 정격
   전압에 맞는 외부 전원을 공급한다. 전원 극성과 케이블을 먼저 확인한다.
2. **통신:** PC USB → U2D2 → (DYNAMIXEL TTL 버스) → 2XL430으로 연결한다.
   PC의 장치 관리자에서 U2D2의 `COM` 번호를 확인한다.
3. **ID:** 스크립트 기본값은 yaw=1, pitch=2다. 두 축의 ID가 같으면 통신 충돌로
   동작하지 않는다. 2XL430의 두 축 ID가 실제로 1과 2인지 DYNAMIXEL Wizard 2.0에서
   먼저 확인하고, 필요하면 한 축의 ID를 변경한다.
4. 기구 간섭이 없는 자세에서 시작하고, 처음에는 모터/링크를 손으로 잡지 않는다.

## 설치와 실행

PowerShell에서 다음을 실행한다.

```powershell
cd "C:\Users\hibab\OneDrive\바탕 화면\3학년 1학기\urop\active_camera\host"
python -m pip install dynamixel-sdk
python .\keyboard_2xl430.py --port COM3
```

`COM3`은 장치 관리자에서 본 번호로 바꾼다. 2XL430-W250-T의 공장 기본 통신 속도는
`57600`이며, 스크립트도 이 값이 기본이다. DYNAMIXEL Wizard에서 값을 변경한 경우에만
`--baudrate`를 실제 값으로 맞춘다.

처음에는 저속·짧은 이동 한계로 시작한다.

```powershell
python .\keyboard_2xl430.py --port COM3 --step-deg 0.2 --rate-hz 10 --limit-deg 10 --velocity 30 --acceleration 10
```

ID가 다르면 예를 들어 다음처럼 준다.

```powershell
python .\keyboard_2xl430.py --port COM3 --yaw-id 3 --pitch-id 4
```

## 키 조작

| 키 | 동작 |
|---|---|
| `T` | 현재 위치를 목표 위치로 삼아 토크 켜기 |
| `W` / `S` | pitch + / - (길게 누름) |
| `A` / `D` | yaw - / + (길게 누름) |
| `X` | 토크 끄기 (모터가 자유로워짐) |
| `R` | 현재 목표 위치를 소프트웨어 중심으로 재설정 |
| `P` | 현재 엔코더 위치 출력 |
| `Q` 또는 `Esc` | 토크를 끄고 종료 |

방향은 기구 장착 방향에 따라 반대일 수 있다. 그 경우에는 `W/S` 또는 `A/D`를
반대로 사용하면 되고, 위험하면 즉시 `X`를 누르거나 전원을 차단한다.

`W`와 `A/D`를 함께 누르면 두 축 목표를 한 개의 DYNAMIXEL Sync Write 패킷으로
전송하므로 두 축이 함께 움직인다. U2D2 하나와 3핀 TTL 케이블 하나로 두 ID를
동시에 제어하는 것이 정상 구성이다.

`X`, `Q`, `Ctrl+C` 및 통신 오류 후 종료 시에는 스크립트가 두 축의 토크 해제를
시도한다. 단, 물리적인 비상 정지는 전원 차단이 가장 확실하다.
