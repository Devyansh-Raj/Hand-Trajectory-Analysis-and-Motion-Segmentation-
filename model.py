import mediapipe as mp
import cv2
import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

vid = cv2.VideoCapture("Reach pick move place retract.mp4")
width = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)

ret, frame =vid.read()
frame = cv2.resize(frame, (640,480))


def main():
    trajectory = []
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
            if not success:
                break
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
                    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                    x = wrist.x
                    y = wrist.y
                    z = wrist.z
                    trajectory.append((x, y, z))
      
                    # print(trajectory[-1])
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

    trajectory = np.array(trajectory)
    x = trajectory[:, 0]
    y = trajectory[:, 1]
    z = trajectory[:, 2]
    if len(trajectory) < 5:
        print("Not enough points")
        return
    else:
        t = np.arange(len(x))
        cs_x = CubicSpline(t, x)
        cs_y = CubicSpline(t, y)
        cs_z = CubicSpline(t, z)

        

        t_smooth = np.linspace(0, len(x)-1, 500)
        x_smooth = cs_x(t_smooth)
        y_smooth = cs_y(t_smooth)
        z_smooth = cs_z(t_smooth)

        dx = np.gradient(x_smooth)
        dy = np.gradient(y_smooth)
        dz = np.gradient(z_smooth)

        
        
        velocity = np.sqrt(dx**2 + dy**2 + dz**2)
        velocity_smooth = gaussian_filter1d(velocity, sigma=6)
        acceleration = np.gradient(velocity_smooth)
        peaks, _ = find_peaks(velocity_smooth, height=0.003, distance=40)
        print("Peak values", velocity_smooth[peaks])
        valleys, _ = find_peaks(-velocity_smooth, distance=30, prominence=0.0005)
        key_points = np.sort(valleys)
        indices = np.split(np.arange(len(velocity_smooth)), key_points)
        segments = [velocity_smooth[idx] for idx in indices]
        filtered_segments = []
        filtered_indices = []
        min_length = 20
        for seg, idx in zip(segments, indices):
            
            if len(seg) >= min_length:
                filtered_segments.append(seg)
                filtered_indices.append(idx)
        segments = filtered_segments
        indices = filtered_indices

        for i, seg in enumerate(segments):
            avg_vel = np.mean(seg)
            print(f"Segment {i}: Length = {len(seg)}, Avg Velocity = {avg_vel:.6f}")
        
        Sequence =["Reach", "Pick", "Grasp", "Move", "Release", "Retract"]
        labels = []
        seg_features = []
        for i, (seg, idx) in enumerate(zip(segments, indices)):
            avg_vel = np.mean(seg)
            max_vel = np.max(seg)
            acc_seg = acceleration[idx]
            avg_acc = np.mean(acc_seg)
            z_seg = z_smooth[idx]
            dz = np.mean(np.gradient(z_seg))

            z_seg = z_smooth[idx]
            dz = np.mean(np.gradient(z_seg))
            seg_features.append({
                "avg_vel": avg_vel,
                "max_vel": max_vel,
                "avg_acc": avg_acc,
                "dz": dz,
                "length" : len(seg)
            })

        num_segs = len(seg_features)
        STILL_THRESH =0.0025
        HIGH_VEL_THRESH = 0.006

        for i, feat in enumerate(seg_features):
            pos = i / max(num_segs -1, 1)
            avg_vel = feat["avg_vel"]
            dz = feat["dz"]


            if pos < 0.20:
                labels.append("Reach")
            elif pos < 0.45:
         
                if feat["max_vel"] == max(s["max_vel"] for s in seg_features):
                    labels.append("Pick")
                else:
                    labels.append("Grasp")
            elif pos < 0.65:
                labels.append("Move")
            elif pos < 0.85:
                labels.append("Place")
            else:
                labels.append("Retract")


        
        actions = labels.copy()
       
        print("\nDetected Actions:")
        for i, act in enumerate(actions):
            print(f"Segment {i}: {act}")


        plt.plot(velocity_smooth)
        for peak in peaks:
            plt.axvline(x=peak, color='r', linestyle='--')
        plt.title("Segmented Motion")
        plt.xlabel("Time")
        plt.ylabel("velocity")
        plt.show()


        plt.plot(velocity_smooth)
        plt.title("Smoothed velocity Curve")
        plt.xlabel("Time")
        plt.ylabel("velocity")
        plt.show()


        plt.plot(velocity_smooth)
        plt.plot(peaks, velocity_smooth[peaks],"x" )
        plt.title("Clean Peaks")
        plt.show()

    vid.release()
    cv2.destroyAllWindows()
    plt.close('all')
    
    


if __name__ == "__main__":
    main() 
