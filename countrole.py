import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message="SymbolDatabase.GetPrototype")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GLOG_minloglevel'] = '3'
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

import cv2
import mediapipe as mp
import pyautogui
pyautogui.FAILSAFE = False  # フェイルセーフを無効化（画面角でも停止しない）
import time
from collections import deque

# --- マウス設定 ---
MOUSE_SMOOTHING_FRAMES = 5  # スムージングに使うフレーム数
SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()  # 画面サイズ取得
# カメラ座標のマッピング範囲（手の可動域）
CAM_X_MIN, CAM_X_MAX = 0.1, 0.9
CAM_Y_MIN, CAM_Y_MAX = 0.1, 0.9 

# --- メイン処理 ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

# 連打防止フラグ
click_done = False

# 立てている指の本数をカウント（親指以外）
def count_fingers_up(lms):
    count = 0
    # 人差し指
    if lms.landmark[8].y < lms.landmark[6].y:
        count += 1
    # 中指
    if lms.landmark[12].y < lms.landmark[10].y:
        count += 1
    # 薬指
    if lms.landmark[16].y < lms.landmark[14].y:
        count += 1
    # 小指
    if lms.landmark[20].y < lms.landmark[18].y:
        count += 1
    return count

# サムズアップ（親指だけ立てる）判定関数 → 操作モード開始用
def is_thumbs_up(lms, debug=False):
    # 親指が伸びている（先端が付け根より外側に出ている）
    thumb_tip = lms.landmark[4]
    thumb_mcp = lms.landmark[2]
    # 親指の先端と付け根の距離
    thumb_dist = ((thumb_tip.x - thumb_mcp.x)**2 + (thumb_tip.y - thumb_mcp.y)**2)**0.5
    thumb_extended = thumb_dist > 0.06

    # 他の指は曲がっている
    index_down = lms.landmark[8].y > lms.landmark[6].y
    middle_down = lms.landmark[12].y > lms.landmark[10].y
    ring_down = lms.landmark[16].y > lms.landmark[14].y
    pinky_down = lms.landmark[20].y > lms.landmark[18].y

    if debug:
        print(f"  thumb_dist={thumb_dist:.3f}(>{0.06}={thumb_extended}), idx={index_down}, mid={middle_down}, ring={ring_down}, pinky={pinky_down}")

    return thumb_extended and index_down and middle_down and ring_down and pinky_down

# グー（全ての指を曲げた状態）判定関数 → 操作モード終了用
def is_fist(lms):
    # 親指も曲げている（サムズアップとの違い）
    thumb_tip = lms.landmark[4]
    thumb_mcp = lms.landmark[2]
    thumb_dist = ((thumb_tip.x - thumb_mcp.x)**2 + (thumb_tip.y - thumb_mcp.y)**2)**0.5
    thumb_folded = thumb_dist < 0.05  # 親指が曲がっている

    index_down = lms.landmark[8].y > lms.landmark[6].y
    middle_down = lms.landmark[12].y > lms.landmark[10].y
    ring_down = lms.landmark[16].y > lms.landmark[14].y
    pinky_down = lms.landmark[20].y > lms.landmark[18].y
    return thumb_folded and index_down and middle_down and ring_down and pinky_down

# 親指が内側に曲がっているか判定（ピンチモード用）
# 親指の先端が人差し指の付け根より内側（x座標が大きい）ならTrue
def is_thumb_in(lms):
    thumb_tip = lms.landmark[4]
    index_mcp = lms.landmark[5]
    return thumb_tip.x > index_mcp.x

# 手のサイズ（カメラとの距離推定）
# 人差し指付け根(5)から小指付け根(17)の距離で判定。指の形に影響されない
def get_hand_size(lms):
    index_mcp = lms.landmark[5]
    pinky_mcp = lms.landmark[17]
    return ((index_mcp.x - pinky_mcp.x)**2 + (index_mcp.y - pinky_mcp.y)**2)**0.5

# 手のサイズ閾値（これより大きければカメラに近いと判断）
MIN_HAND_SIZE = 0.12  # 調整可能

# マウス用スムージングバッファ
mouse_x_buffer = deque(maxlen=MOUSE_SMOOTHING_FRAMES)
mouse_y_buffer = deque(maxlen=MOUSE_SMOOTHING_FRAMES)

def map_to_screen(cam_x, cam_y):
    """カメラ座標を画面座標にマッピング"""
    norm_x = (cam_x - CAM_X_MIN) / (CAM_X_MAX - CAM_X_MIN)
    norm_y = (cam_y - CAM_Y_MIN) / (CAM_Y_MAX - CAM_Y_MIN)
    norm_x = max(0, min(1, norm_x))
    norm_y = max(0, min(1, norm_y))
    screen_x = int(norm_x * SCREEN_WIDTH)
    screen_y = int(norm_y * SCREEN_HEIGHT)
    return screen_x, screen_y

def smooth_mouse_position(x, y):
    """過去のフレームの平均でスムージング"""
    mouse_x_buffer.append(x)
    mouse_y_buffer.append(y)
    avg_x = sum(mouse_x_buffer) / len(mouse_x_buffer)
    avg_y = sum(mouse_y_buffer) / len(mouse_y_buffer)
    return avg_x, avg_y

# スクロール用
scroll_mode_active = False  # スクロールモードが有効か
EXTEND_THRESHOLD = 0.18   # この距離以上なら伸ばしている → 上スクロール
PINCH_THRESHOLD = 0.06    # この距離以下ならピンチ → 下スクロール
SCROLL_INTERVAL = 0.15    # スクロールの間隔（秒）
last_scroll_time = 0      # 最後にスクロールした時刻
current_scroll_state = None  # 現在のスクロール状態 ("up", "down", None)
current_finger_dist = 0   # 現在の指の距離（表示用）
current_pinch_dist = 0    # 現在のピンチ距離（表示用）

# 操作モード
control_mode = False
last_action_time = 0

# マウスモード
mouse_mode = False
mouse_screen_x, mouse_screen_y = 0, 0

window_name = 'Hand Remote'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 640, 480)
cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)

with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
    while cap.isOpened():
        success, img = cap.read()
        if not success: break
        img = cv2.flip(img, 1)

        results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        now = time.time()

        if results.multi_hand_landmarks:
            for lms in results.multi_hand_landmarks:
                thumb_tip = lms.landmark[4]
                index_base = lms.landmark[5]
                index_tip = lms.landmark[8]
                middle_tip = lms.landmark[12]

                def get_dist(p1, p2):
                    return ((p1.x - p2.x)**2 + (p1.y - p2.y)**2)**0.5

                # --- 手のサイズ（カメラとの距離）をチェック ---
                hand_size = get_hand_size(lms)
                is_close_enough = hand_size > MIN_HAND_SIZE


                # --- 操作モードの切り替え ---
                # カメラに近い状態でサムズアップを出したときだけON
                if is_close_enough and is_thumbs_up(lms):
                    if not control_mode:
                        control_mode = True
                        print(f"Control Mode ON (hand_size={hand_size:.3f})")
                    last_action_time = now

                if is_fist(lms) and control_mode:
                    control_mode = False
                    print("Control Mode OFF (fist)")

                # --- 操作モードがONかつカメラに近いときだけ操作を受け付ける ---
                # サムズアップの時は操作しない（操作モード維持のみ）
                if control_mode and is_close_enough and not is_thumbs_up(lms):

                    # 親指の位置と指の本数を取得
                    thumb_outside = not is_thumb_in(lms)
                    fingers = count_fingers_up(lms)

                    # 中指・薬指・小指が曲がっているか
                    middle_down = lms.landmark[12].y > lms.landmark[10].y
                    ring_down = lms.landmark[16].y > lms.landmark[14].y
                    pinky_down = lms.landmark[20].y > lms.landmark[18].y
                    other_fingers_down = middle_down and ring_down and pinky_down

                    # 人差し指の先端と付け根の距離（伸びているか）
                    finger_dist = get_dist(index_tip, index_base)
                    # 親指と人差し指の先端の距離（ピンチしているか）
                    pinch_dist = get_dist(thumb_tip, index_tip)
                    current_finger_dist = finger_dist
                    current_pinch_dist = pinch_dist

                    # === 上スクロール: 親指が外側 + 人差し指1本 + 指を伸ばす ===
                    if thumb_outside and fingers == 1:
                        scroll_mode_active = True
                        if finger_dist > EXTEND_THRESHOLD:
                            current_scroll_state = "up"
                            if now - last_scroll_time > SCROLL_INTERVAL:
                                pyautogui.scroll(20)
                                print(f">>> Scroll UP (finger={finger_dist:.3f})")
                                last_scroll_time = now
                                last_action_time = now
                        else:
                            current_scroll_state = None
                        click_done = False

                    # === 下スクロール: ピンチ + 中指・薬指・小指が曲がっている ===
                    elif pinch_dist < PINCH_THRESHOLD and other_fingers_down:
                        scroll_mode_active = True
                        current_scroll_state = "down"
                        if now - last_scroll_time > SCROLL_INTERVAL:
                            pyautogui.scroll(-20)
                            print(f">>> Scroll DOWN (pinch={pinch_dist:.3f})")
                            last_scroll_time = now
                            last_action_time = now
                        click_done = False

                    # === 親指が外側 → クリック操作 ===
                    elif thumb_outside:
                        scroll_mode_active = False
                        current_scroll_state = None
                        mouse_mode = False
                        mouse_x_buffer.clear()
                        mouse_y_buffer.clear()

                        # --- 2本指: 左クリック ---
                        if fingers == 2:
                            if not click_done:
                                pyautogui.click()
                                print("Left Click")
                                click_done = True
                                last_action_time = now

                        # --- 3本指: 右クリック ---
                        elif fingers == 3:
                            if not click_done:
                                pyautogui.rightClick()
                                print("Right Click")
                                click_done = True
                                last_action_time = now

                        # --- 0本または4本: リセット ---
                        else:
                            click_done = False

                    # === 親指が内側 → マウス操作 ===
                    else:
                        scroll_mode_active = False
                        current_scroll_state = None

                        # マウス移動モード
                        mouse_mode = True
                        # 親指の付け根（landmark 2）を基準にする
                        thumb_mcp = lms.landmark[2]
                        # スムージング適用
                        smooth_x, smooth_y = smooth_mouse_position(thumb_mcp.x, thumb_mcp.y)
                        # 画面座標にマッピング
                        mouse_screen_x, mouse_screen_y = map_to_screen(smooth_x, smooth_y)
                        # マウス移動
                        pyautogui.moveTo(mouse_screen_x, mouse_screen_y, _pause=False)

                        # ピンチでクリック
                        pinch_dist_index = get_dist(thumb_tip, index_tip)  # 人差し指ピンチ
                        pinch_dist_middle = get_dist(thumb_tip, middle_tip)  # 中指ピンチ

                        # 人差し指ピンチ → 左クリック
                        if pinch_dist_index < 0.05:
                            if not click_done:
                                pyautogui.click()
                                print(f"Left Click (index pinch={pinch_dist_index:.3f})")
                                click_done = True
                                last_action_time = now
                        # 中指ピンチ → 右クリック
                        elif pinch_dist_middle < 0.05:
                            if not click_done:
                                pyautogui.rightClick()
                                print(f"Right Click (middle pinch={pinch_dist_middle:.3f})")
                                click_done = True
                                last_action_time = now
                        else:
                            click_done = False

                mp_drawing.draw_landmarks(img, lms, mp_hands.HAND_CONNECTIONS)

        # 操作モード状態を画面に表示
        status = "CONTROL MODE: ON" if control_mode else "Thumbs up to start"
        color = (0, 255, 0) if control_mode else (128, 128, 128)
        cv2.putText(img, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # 親指の位置によるサブモードを表示
        if control_mode and results.multi_hand_landmarks:
            for lms in results.multi_hand_landmarks:
                if not is_thumbs_up(lms):
                    sub_mode = "MOUSE (thumb in)" if is_thumb_in(lms) else "CLICK/SCROLL (thumb out)"
                    cv2.putText(img, sub_mode, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                    # 指の本数とスクロール状態を表示
                    if not is_thumb_in(lms):
                        fingers = count_fingers_up(lms)
                        cv2.putText(img, f"Fingers: {fingers}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                        # スクロールモードの表示
                        if scroll_mode_active:
                            if current_scroll_state == "up":
                                mode_text = "SCROLLING UP (extend)"
                                mode_color = (0, 255, 0)  # 緑
                            elif current_scroll_state == "down":
                                mode_text = "SCROLLING DOWN (pinch)"
                                mode_color = (0, 165, 255)  # オレンジ
                            else:
                                mode_text = "SCROLL READY"
                                mode_color = (128, 128, 128)  # グレー
                            cv2.putText(img, mode_text, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, mode_color, 2)

                        # マウスモードの表示
                        if mouse_mode:
                            cv2.putText(img, f"MOUSE MODE: ({mouse_screen_x}, {mouse_screen_y})", (10, 150),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 手のサイズ（距離）を表示
        if results.multi_hand_landmarks:
            for lms in results.multi_hand_landmarks:
                hs = get_hand_size(lms)
                dist_status = f"Hand size: {hs:.3f} ({'OK' if hs > MIN_HAND_SIZE else 'TOO FAR'})"
                dist_color = (0, 255, 0) if hs > MIN_HAND_SIZE else (0, 0, 255)
                cv2.putText(img, dist_status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, dist_color, 2)

                # 指の距離とピンチ距離を表示（スクロール用）
                thumb_tip_d = lms.landmark[4]
                index_tip_d = lms.landmark[8]
                index_base_d = lms.landmark[5]
                display_finger_dist = get_dist(index_tip_d, index_base_d)
                display_pinch_dist = get_dist(thumb_tip_d, index_tip_d)
                finger_text = f"Finger: {display_finger_dist:.3f} (ext>{EXTEND_THRESHOLD})"
                pinch_text = f"Pinch: {display_pinch_dist:.3f} (down<{PINCH_THRESHOLD})"
                cv2.putText(img, finger_text, (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.putText(img, pinch_text, (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        cv2.imshow(window_name, img)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()