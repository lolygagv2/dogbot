# detect_aruco_id.py
import sys, cv2, numpy as np

def get_det():
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
    try:
        p = cv2.aruco.DetectorParameters(); return d, cv2.aruco.ArucoDetector(d,p)
    except AttributeError:
        p = cv2.aruco.DetectorParameters_create(); return d, p

def detect_ids(img):
    d, det = get_det()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    try: corners, ids, _ = det.detectMarkers(gray)
    except Exception: corners, ids, _ = cv2.aruco.detectMarkers(gray, d, parameters=det)
    out=[]
    if ids is not None:
        for c,id_ in zip(corners, ids.flatten()):
            pts=c[0]; cx,cy=pts[:,0].mean(), pts[:,1].mean()
            out.append((int(id_), float(cx), float(cy)))
    return out

def read_img(path): 
    if path.isdigit():  # webcam
        cap=cv2.VideoCapture(int(path)); ok,frame=cap.read(); cap.release()
        if not ok: raise SystemExit("no frame")
        return frame
    im=cv2.imread(path); 
    if im is None: raise SystemExit("bad path")
    return im

if __name__=="__main__":
    if len(sys.argv)<2: raise SystemExit("usage: python detect_aruco_id.py <image_or_cam_index>")
    img = read_img(sys.argv[1])
    print("markers:", detect_ids(img))  # e.g. [(230, cx, cy)]
