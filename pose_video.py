import cv2
import mediapipe as mp
import numpy as np
import math
import csv
def calculate_angle(a,b,c):
    """
    Angle at point b formed by points a-b-c (in degrees)
    a, b, c are (x, y)
    """
    a=np.array(a)
    b=np.array(b)
    c=np.array(c)

    radians=np.arctan2(c[1]-b[1],c[0]-b[0])-np.arctan2(a[1]-b[1],a[0]-b[0])
    angle=abs(radians*180.0/math.pi)
    if angle>180:
        angle=360-angle
    return angle
def write_csv_header(csv_path,mp_pose):
    with open(csv_path,"w",newline="") as f:
        writer=csv.writer(f)
        header=["frame"]
        for lm in mp_pose.PoseLandmark:
            header+=[f"{lm.name}_x",f"{lm.name}_y",f"{lm.name}_z",f"{lm.name}_vis"]
        writer.writerow(header)
def append_keypoints(csv_path,frame_id,landmarks):
    with open(csv_path, "a", newline="") as f:
        writer=csv.writer(f)
        row=[frame_id]
        for lm in landmarks:
            row+=[lm.x,lm.y,lm.z,lm.visibility]
        writer.writerow(row)
def compute_angles(landmarks,w,h,mp_pose):
    shoulder=(
        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x*w,
        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y*h
    )
    elbow=(
        landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x*w,
        landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y*h
    )
    wrist=(
        landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x*w,
        landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y*h
    )
    elbow_angle=calculate_angle(shoulder,elbow,wrist)
    hip=(
        landmarks[mp_pose.PoseLandmark.LEFT_HIP].x*w,
        landmarks[mp_pose.PoseLandmark.LEFT_HIP].y*h
    )
    knee=(
        landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x*w,
        landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y*h
    )
    ankle=(
        landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x*w,
        landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y*h
    )
    knee_angle=calculate_angle(hip,knee,ankle)

   
    right_shoulder = (
        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x*w,
        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y*h
    )
    shoulder_mid=((shoulder[0]+right_shoulder[0])/2,(shoulder[1] + right_shoulder[1]) / 2)
    hip_angle=calculate_angle(shoulder_mid,hip,knee)

    return elbow_angle,knee_angle,hip_angle

def print_metric(name,values):
    if len(values)==0:
        print(f"{name}:no values")
        return
    rom=max(values)-min(values)
    var=np.var(values)
    print(f"{name} ROM: {int(rom)} deg")
    print(f"{name} Stability (variance): {var:.2f}")
def main():
    input_video="input_video.mp4"      
    output_video="output_pose.mp4"
    csv_path="keypoints.csv"
    mp_pose=mp.solutions.pose
    mp_draw=mp.solutions.drawing_utils
    pose=mp_pose.Pose(min_detection_confidence=0.5,
                        min_tracking_confidence=0.5)
    cap=cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print("Cannot open video:", input_video)
        return
    else:
        print("Video opened:", input_video)

    w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps=cap.get(cv2.CAP_PROP_FPS)
    out=cv2.VideoWriter(
        output_video,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h)
    )


    write_csv_header(csv_path,mp_pose)

    elbow_list,knee_list,hip_list=[],[],[]
    frame_id=0

    while True:
        ret,frame=cap.read()
        if not ret:
            break

        rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        results=pose.process(rgb)

        if results.pose_landmarks:
            lm=results.pose_landmarks.landmark
            append_keypoints(csv_path,frame_id,lm)
            elbow_ang,knee_ang,hip_ang=compute_angles(lm,w,h,mp_pose)
            elbow_list.append(elbow_ang)
            knee_list.append(knee_ang)
            hip_list.append(hip_ang)
            mp_draw.draw_landmarks(frame,results.pose_landmarks,mp_pose.POSE_CONNECTIONS)

    
            cv2.putText(frame,f"Elbow: {int(elbow_ang)} deg",(30,60),
                        cv2.FONT_HERSHEY_SIMPLEX,1, (0, 255, 0),2)
            cv2.putText(frame,f"Knee : {int(knee_ang)} deg", (30,105),
                        cv2.FONT_HERSHEY_SIMPLEX,1, (0, 255, 0),2)
            cv2.putText(frame,f"Hip  : {int(hip_ang)} deg", (30,150),
                        cv2.FONT_HERSHEY_SIMPLEX,1, (0, 255, 0),2)

        out.write(frame)
        cv2.imshow("Pose Output (press q to quit)",frame)

        if cv2.waitKey(1) & 0xFF==ord("q"):
            break

        frame_id+=1

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print("\n Output video saved:",output_video)
    print(" Keypoints CSV saved:",csv_path)

    print("\n--- Movement Metrics Summary ---")
    print_metric("Elbow",elbow_list)
    print_metric("Knee",knee_list)
    print_metric("Hip",hip_list)


if __name__=="__main__":
    main()
