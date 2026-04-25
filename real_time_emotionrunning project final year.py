import cv2
import time
from deepface import DeepFace

def real_time_emotion_detection(camera_index=0, scale=0.6):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("❌ Cannot open webcam. Check camera index or permissions.")
        return

    print("🎥 Webcam opened. Press 'q' to quit, 's' to save a frame.")
    fps_display_interval = 1
    frame_count = 0
    start_time = time.time()
    last_display_time = start_time
    fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Failed to read frame from webcam.")
            break

        # Resize for faster analysis
        small = cv2.resize(frame, (0, 0), fx=scale, fy=scale)

        try:
            rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

            result = DeepFace.analyze(rgb_small, actions=['emotion'], enforce_detection=False)

            results = result if isinstance(result, list) else [result]

            for r in results:
                emotion = r.get("dominant_emotion", "unknown")
                region = r.get("region", {"x":0,"y":0,"w":0,"h":0})
                x, y, w, h = region["x"], region["y"], region["w"], region["h"]

                if w > 0 and h > 0:
                    sx = int(x / scale)
                    sy = int(y / scale)
                    sw = int(w / scale)
                    sh = int(h / scale)

                    cv2.rectangle(frame, (sx, sy), (sx + sw, sy + sh), (0, 255, 0), 2)
                    cv2.putText(frame, emotion, (sx, sy - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        except Exception:
            pass

        frame_count += 1
        now = time.time()
        if (now - last_display_time) > fps_display_interval:
            fps = frame_count / (now - last_display_time)
            last_display_time = now
            frame_count = 0

        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imshow("🎭 Live Emotion Detection (press q to quit)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            timestamp = int(time.time())
            fname = f"frame_{timestamp}.jpg"
            cv2.imwrite(fname, frame)
            print(f"Saved {fname}")

    cap.release()
    cv2.destroyAllWindows()
    print("👋 Exited successfully.")


if __name__ == "__main__":
    real_time_emotion_detection()
