import time
from os.path import dirname, join
from PIL import Image
from .face_embedding_manager import FaceEmbeddingManager


def initiation_test():
    manager = FaceEmbeddingManager(mode='stand_alone')


def recognition_test():
    manager = FaceEmbeddingManager()
    pil = Image.open(join(dirname(__file__), '5c4fc7bb1deedf3292ba512a.jpg'))
    cropped_pils = manager.crop_and_align_images([pil])
    embs = manager.embedding_extractor(cropped_pils)
    result = manager.recognize(embs)
    print(result)


def add_and_remove_user_test_and_benchmarking():
    start = time.time()
    manager = FaceEmbeddingManager(mode='stand_alone')
    initiation_time = time.time() - start
    print(f"Time Taken for inititation: {initiation_time}")
    timer = []
    for n in range(11):
        start = time.time()
        reg_pil = Image.open(join(dirname(__file__), 'WaYWfN-Q_400x400.jpg'))
        manager.update_facebank([reg_pil], 'Jessica')
        test_image = Image.open(join(dirname(__file__), 'jessica-jung-for-banila.jpg'))
        cropped_pils = manager.crop_and_align_images([test_image])
        embs = manager.embedding_extractor(cropped_pils)
        result = manager.recognize(embs)
        print(result)
        manager.update_facebank([reg_pil], 'Jessica')
        result = manager.recognize(embs)
        print(result)
        manager.remove_identity('Jessica')
        result = manager.recognize(embs)
        print(result)
        timer.append(time.time() - start)
    print(f"Total time taken: {sum(timer)}")
    print(f"On average one loop: {sum(timer) / len(timer)}")
    print("Excluding the first loop")
    print(f"Total time taken: {sum(timer[1:])}")
    print(f"On average one loop: {sum(timer[1:]) / len(timer[1:])}")


def test_recognition_without_face_detection():
    manager = FaceEmbeddingManager()
    import cv2
    cap = cv2.VideoCapture(0)
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        h, w, c = frame.shape
        if h > w:
            d = h - w
            half_d = d // 2
            frame = frame[half_d: half_d + w, :, :]
        else:
            d = w - h
            half_d = d // 2
            frame = frame[:, half_d: half_d + h, :]
        # Our operations on the frame come here
        pil = Image.fromarray(cv2.resize(frame, (112, 112)))
        try:
            embs = manager.embedding_extractor([pil])
            result = manager.recognize(embs)
            print(result)
        except ValueError as e:
            print(e)
            breakpoint()
        # Display the resulting frame
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


def demo(name=None, photo=None):
    manager = FaceEmbeddingManager(mode='stand_alone')
    if name is not None and photo is not None:
        reg_pil = Image.open(photo)
        manager.update_facebank([reg_pil], name)
    import cv2
    cap = cv2.VideoCapture(0)
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        # Our operations on the frame come here
        try:
            pil = Image.fromarray(frame)
            cropped_pils = manager.crop_and_align_images([pil])
            embs = manager.embedding_extractor(cropped_pils)
            result = manager.recognize(embs)
            print(result)
        except ValueError as e:
            print(e)
        # Display the resulting frame
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    if name is not None and photo is not None:
        manager.remove_identity(name)


if __name__ == "__main__":
    import argparse
    import os
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, help="The path to new user's image", default=None)
    parser.add_argument("--identity", type=str, help="Name of the new user's image", default=None)
    args = parser.parse_args()
    name = args.identity
    photo = args.image
    demo(name, photo)
