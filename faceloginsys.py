import cv2
import face_recognition
import numpy as np
import pickle
import os
import time

DATA_FILE = 'known_faces_data.pkl'
FACE_RECOGNITION_TOLERANCE = 0.6
CAMERA_INDEX = 0
REGISTRATION_IMAGE_COUNT = 5
FRAME_THICKNESS = 3
FONT_THICKNESS = 2
FONT_SCALE = 0.6
PROCESS_EVERY_N_FRAMES = 2

def load_known_faces(data_file=DATA_FILE):
    known_face_encodings = []
    known_face_names = []
    if os.path.exists(data_file):
        try:
            with open(data_file, 'rb') as f:
                data = pickle.load(f)
                known_face_encodings = data['encodings']
                known_face_names = data['names']
        except Exception as e:
            known_face_encodings = []
            known_face_names = []
    return known_face_encodings, known_face_names

def save_known_faces(encodings, names, data_file=DATA_FILE):
    data = {'encodings': encodings, 'names': names}
    try:
        with open(data_file, 'wb') as f:
            pickle.dump(data, f)
    except Exception as e:
        pass

def register_new_user(known_face_encodings, known_face_names):
    username = input("Enter username for new registration: ").strip()
    if not username:
        return
    if username in known_face_names:
        return

    input("Press Enter to start...")
    video_capture = cv2.VideoCapture(CAMERA_INDEX)
    if not video_capture.isOpened():
        return

    captured_encodings = []
    images_captured = 0
    last_capture_time = time.time()

    while images_captured < REGISTRATION_IMAGE_COUNT:
        ret, frame = video_capture.read()
        if not ret:
            break
        rgb_frame = frame[:, :, ::-1]
        face_locations = face_recognition.face_locations(rgb_frame)

        if len(face_locations) == 1:
            (top, right, bottom, left) = face_locations[0]
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), FRAME_THICKNESS)
            current_time = time.time()
            if current_time - last_capture_time > 1.0:
                face_encoding = face_recognition.face_encodings(rgb_frame, face_locations)[0]
                captured_encodings.append(face_encoding)
                images_captured += 1
                last_capture_time = current_time
            status_text = f"Capturing: {images_captured}/{REGISTRATION_IMAGE_COUNT}"
            cv2.putText(frame, status_text, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0, 255, 0), FONT_THICKNESS)
        elif len(face_locations) > 1:
            cv2.putText(frame, "Show only ONE face!", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0, 0, 255), FONT_THICKNESS)
        else:
            cv2.putText(frame, "No face detected!", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0, 0, 255), FONT_THICKNESS)

        cv2.imshow('Register User - Press Q to Cancel', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

    if images_captured == REGISTRATION_IMAGE_COUNT:
        if captured_encodings:
            user_encoding = captured_encodings[0]
            known_face_encodings.append(user_encoding)
            known_face_names.append(username)
            save_known_faces(known_face_encodings, known_face_names)

def run_login_system(known_face_encodings, known_face_names):
    if not known_face_encodings:
        return

    video_capture = cv2.VideoCapture(CAMERA_INDEX)
    if not video_capture.isOpened():
        return

    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]

        if process_this_frame:
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            face_names = []
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=FACE_RECOGNITION_TOLERANCE)
                name = "Unknown"
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                face_names.append(name)

        process_this_frame = not process_this_frame

        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, FRAME_THICKNESS)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (255, 255, 255), FONT_THICKNESS)
            if name != "Unknown":
                video_capture.release()
                cv2.destroyAllWindows()
                return name

        cv2.imshow('Video Login System - Press Q to Quit', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()
    return None

def main():
    known_face_encodings, known_face_names = load_known_faces()
    while True:
        print("\n--- Face Login System Menu ---")
        print("1. Register New User")
        print("2. Run Login System")
        print("3. Exit")
        choice = input("Enter your choice: ").strip()
        if choice == '1':
            register_new_user(known_face_encodings, known_face_names)
            known_face_encodings, known_face_names = load_known_faces()
        elif choice == '2':
            logged_in_user = run_login_system(known_face_encodings, known_face_names)
            if logged_in_user:
                print(f"Welcome, {logged_in_user}!")
            else:
                print("Login failed or system exited.")
        elif choice == '3':
            break
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main()
