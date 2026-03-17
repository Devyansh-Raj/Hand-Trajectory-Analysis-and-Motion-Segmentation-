import mediapipe as mp
import cv2

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

vid = cv2.VideoCapture("Reach pick move place retract.mp4")
width = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)

ret, frame =vid.read()
frame = cv2.resize(frame, (640,480))


def main():
    with mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.4,
    ) as hands:
        while True:
            attempt = 0
            success, img = vid.read()
            while  not success and attempt < 5:
                success, img = vid.read()
                attempt += 1
            img = cv2.flip(img, 1)
            h, w, _ = img.shape
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(
                        img,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,

                    )
                    finger_tips ={
                        "thumb": hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP],
                        "index": hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP],
                        "middle": hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP],
                        "ring": hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP],
                        "pinky": hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP],
                    }
                    for name, landmark in finger_tips.items():
                        x, y = int(landmark.x * w), int(landmark.y * h)
                        cv2.putText(
                            img,
                            name,
                            (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 0, 0),
                            1,

                        )
                        cv2.circle(
                            img,
                            (x, y),
                            5,
                            (0, 255, 0),
                            -1

                        )


            cv2.imshow("Image", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break    


if __name__ == "__main__":
    main() 

