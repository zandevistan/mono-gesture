import cv2
import mediapipe as mp
import os
import numpy as np
import math

class BrutalistHandController:
    def __init__(self, image_folder, total_frames=66):
        self.image_folder = image_folder
        self.total_frames = total_frames
        self.images = []
        self.current_frame_idx = 0
        
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        print("INITIALIZING ASSETS...")
        self.load_images()
        print(f"LOADED {len(self.images)} FRAMES SUCCESSFULLY.")

    def load_images(self):
        if not os.path.exists(self.image_folder):
            for i in range(self.total_frames):
                img = np.zeros((400, 400, 3), dtype=np.uint8)
                cv2.putText(img, f"FRAME {i+1}", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5)
                cv2.rectangle(img, (0,0), (400,400), (255,255,255), 10)
                self.images.append(img)
            return

        files = sorted(os.listdir(self.image_folder))
        count = 0
        for file in files:
            if count >= self.total_frames: break
            path = os.path.join(self.image_folder, file)
            img = cv2.imread(path)
            if img is not None:
                img = cv2.resize(img, (500, 500))
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                h, w, _ = img.shape
                cv2.rectangle(img, (0,0), (w, h), (255, 255, 255), 20)
                self.images.append(img)
                count += 1
        
        while len(self.images) < self.total_frames:
            self.images.append(self.images[-1])

    def calculate_openness(self, landmarks):
        wrist = landmarks[0]
        tips = [4, 8, 12, 16, 20]
        mcp_middle = landmarks[9]
        ref_len = math.hypot(mcp_middle.x - wrist.x, mcp_middle.y - wrist.y)
        
        total_dist = 0
        for tip_idx in tips:
            tip = landmarks[tip_idx]
            dist = math.hypot(tip.x - wrist.x, tip.y - wrist.y)
            total_dist += dist
            
        avg_dist = total_dist / 5
        ratio = avg_dist / ref_len
        
        min_ratio = 0.9
        max_ratio = 2.0
        
        openness = (ratio - min_ratio) / (max_ratio - min_ratio)
        return max(0.0, min(1.0, openness))

    def create_brutalist_ui(self, frame_cam, frame_anim, openness_val):
        gray_cam = cv2.cvtColor(frame_cam, cv2.COLOR_BGR2GRAY)
        _, thresh_cam = cv2.threshold(gray_cam, 100, 255, cv2.THRESH_BINARY)
        display_cam = cv2.cvtColor(thresh_cam, cv2.COLOR_GRAY2BGR)
        display_cam = cv2.resize(display_cam, (300, 300))
        cv2.rectangle(display_cam, (0,0), (300,300), (0,0,0), 10)

        h_anim, w_anim, _ = frame_anim.shape
        canvas_h = h_anim + 150
        canvas_w = w_anim + 350
        
        canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255
        canvas[50:50+h_anim, 50:50+w_anim] = frame_anim
        
        cam_x = 50 + w_anim + 20
        cam_y = 50
        canvas[cam_y:cam_y+300, cam_x:cam_x+300] = display_cam
        
        cv2.line(canvas, (cam_x, 370), (cam_x + 300, 370), (0,0,0), 5)
        
        status_text = "OPEN" if openness_val > 0.5 else "CLOSED"
        frame_num_text = f"FRAME: {self.current_frame_idx + 1:02d}/{self.total_frames}"
        percent_text = f"VAL: {int(openness_val * 100)}%"

        font = cv2.FONT_HERSHEY_PLAIN
        cv2.putText(canvas, "SYSTEM_OVERRIDE // HAND_TRACKING", (50, 35), font, 2, (0,0,0), 3)
        cv2.putText(canvas, status_text, (cam_x, 410), font, 4, (0,0,0), 4)
        cv2.putText(canvas, percent_text, (cam_x, 450), font, 2, (0,0,0), 2)
        cv2.putText(canvas, frame_num_text, (50, h_anim + 100), font, 3, (0,0,0), 3)
        
        bar_w = int(openness_val * (w_anim))
        cv2.rectangle(canvas, (50, h_anim + 60), (50 + w_anim, h_anim + 70), (0,0,0), 2)
        cv2.rectangle(canvas, (50, h_anim + 60), (50 + bar_w, h_anim + 70), (0,0,0), -1)

        return canvas

    def run(self):
        cap = cv2.VideoCapture(0)
        
        while True:
            success, img = cap.read()
            if not success: break
            
            img = cv2.flip(img, 1)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.hands.process(img_rgb)
            
            openness = 0.0
            if results.multi_hand_landmarks:
                for hand_lms in results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(img, hand_lms, self.mp_hands.HAND_CONNECTIONS)
                    openness = self.calculate_openness(hand_lms.landmark)
            
            target_idx = int(openness * (self.total_frames - 1))
            diff = target_idx - self.current_frame_idx
            
            if abs(diff) > 0:
                self.current_frame_idx += int(diff * 0.2)
                if diff > 0 and self.current_frame_idx < target_idx: self.current_frame_idx += 1
                elif diff < 0 and self.current_frame_idx > target_idx: self.current_frame_idx -= 1
            
            self.current_frame_idx = max(0, min(self.total_frames - 1, self.current_frame_idx))
            current_anim_frame = self.images[self.current_frame_idx]
            final_ui = self.create_brutalist_ui(img, current_anim_frame, openness)
            
            cv2.imshow("BRUTAL_HAND_CONTROL", final_ui)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = BrutalistHandController("animasi", total_frames=66)
    app.run()
