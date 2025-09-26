import face_recognition
import os
import pickle

def main():
    print("📦 Starting encoding process...")

    known_faces_dir = "student_images"
    known_face_encodings = []
    known_face_names = []

    for person_name in os.listdir(known_faces_dir):
        person_folder = os.path.join(known_faces_dir, person_name)
        if not os.path.isdir(person_folder):
            continue

        for filename in os.listdir(person_folder):
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                image_path = os.path.join(person_folder, filename)
                image = face_recognition.load_image_file(image_path)
                face_encodings = face_recognition.face_encodings(image)

                if face_encodings:
                    known_face_encodings.append(face_encodings[0])
                    known_face_names.append(person_name.strip())
                    print(f"✅ Encoded {filename} for {person_name}")
                else:
                    print(f"⚠️ No face found in {filename}. Skipping.")

    data = {"encodings": known_face_encodings, "names": known_face_names}
    with open("encodings.pickle", "wb") as f:
        pickle.dump(data, f)

    print(f"\n🎉 Encoding complete. {len(set(known_face_names))} students encoded.")

if __name__ == "__main__":
    main()
