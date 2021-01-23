from pathlib import Path

import cv2

PATH = str(Path.cwd() / "data/test.mp4")
LABELS = str(Path.cwd() / "data/test_pred.txt")
OVERLAYED = str(Path.cwd() / "data/test_overlay.mp4")

def overlay():
    vid = cv2.VideoCapture(PATH)

    labels = [float(x) for x in open(LABELS).read().splitlines()]
    print(f"Labels: {len(labels)}")

    overlay = cv2.VideoWriter(OVERLAYED, 0, 1, (640, 480))

    i = 0
    while(i< len(labels)):
        ret, frame = vid.read()

        cv2.putText(frame, f"Vel (m/s): {labels[i]}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_4)
        i+=1

        cv2.imshow('', frame)

        overlay.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break

    vid.release()
    overlay.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    overlay()
