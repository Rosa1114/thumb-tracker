
import cv2
import mediapipe as mp
import time

def main():
    print("✅ 程序开始运行")
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )
    mp_draw = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 无法打开摄像头")
        exit()

    print("🎥 摄像头已成功打开！")

    left_thumb_points = []
    right_thumb_points = []
    frame_count = 0  # 帧计数器

    while True:
        success, img = cap.read()
        if not success:
            break

        frame_count += 1
        img = cv2.flip(img, 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)
        h, w, _ = img.shape

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_lms, hand_handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                hand_label = hand_handedness.classification[0].label
                landmarks = hand_lms.landmark

                thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
                thumb_ip = landmarks[mp_hands.HandLandmark.THUMB_IP]
                thumb_mcp = landmarks[mp_hands.HandLandmark.THUMB_MCP]
                index_mcp = landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP]

                is_thumb_up = (
                    thumb_tip.y < thumb_ip.y < thumb_mcp.y and
                    abs(thumb_tip.x - index_mcp.x) < 0.1
                )

                if is_thumb_up:
                    cx, cy = int(thumb_tip.x * w), int(thumb_tip.y * h)
                    if hand_label == 'Left':
                        left_thumb_points.append((cx, cy))
                    else:
                        right_thumb_points.append((cx, cy))

                mp_draw.draw_landmarks(img, hand_lms, mp_hands.HAND_CONNECTIONS)

        for i in range(1, len(left_thumb_points)):
            cv2.line(img, left_thumb_points[i - 1], left_thumb_points[i], (0, 255, 0), 2)

        for i in range(1, len(right_thumb_points)):
            cv2.line(img, right_thumb_points[i - 1], right_thumb_points[i], (0, 0, 255), 2)

        cv2.imshow("Thumb Up Tracker - Left vs Right", img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            left_thumb_points.clear()
            right_thumb_points.clear()
            print("🧹 轨迹已清除")
        elif key == ord('s'):
            filename = f"screenshot_{int(time.time())}.jpg"
            cv2.imwrite(filename, img)
            print(f"💾 截图已保存为 {filename}")
        elif key == ord('a'):
            if left_thumb_points:
                left_thumb_points.pop()
                print("↩️ 左手轨迹撤销一步")
            else:
                print("⚠️ 左手没有轨迹可以撤销")
        elif key == ord('d'):
            if right_thumb_points:
                right_thumb_points.pop()
                print("↩️ 右手轨迹撤销一步")
            else:
                print("⚠️ 右手没有轨迹可以撤销")

    cap.release()
    cv2.destroyAllWindows()
    print("👋 文件执行完毕")

if __name__ == "__main__":
    main()
