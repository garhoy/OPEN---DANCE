""" Project Done by 
    "DTSKT" y garhoy"""
import cv2
import mediapipe as mp 
import numpy as np
import time
import random
import math
import sys



# Command line argument -----------------
show_calib = False
argumentList = sys.argv[1:]

if argumentList:
    if argumentList.pop() == "calib":
        show_calib = True

if show_calib:
    print("** DEMO CLAIB INPUT **")
else:
    print("** NO DEMO CLAIB INPUT **")
# ---------------------------------------


# Load mediapipe pose model -------------------------------------------------
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.8,min_tracking_confidence=0.8)
# ----------------------------------------------------------------------------
    

# Max values: 1280x720 -----
frame_width = 1280
frame_height = 720
#---------------------------


# Webcam input ---------------------------------------------------
cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FPS, 30.0)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('m','j','p','g'))
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M','J','P','G'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
# --------------------------------------------------------------------
    
    
Start_btn_coord = [ (450, 60), (150, 850) ] 
Quit_btn_coord =  [ (450, 600),(690, 850)] 

Principal_Menu_coord = [(450, 450), (540, 850)]
Quit_btn2_coord = [ (450, 600),(690, 850)] 

Difficulty_Select_coord = [ (450, 60), (150, 850) ] 

Difficulty_Easy_coord =[(450,290),(380,850)]
Difficulty_Medium_coord = [(450,450),(540,850)]
Difficulty_Hard_coord = [(450,600),(690,850)]

color = (85,34,244)

level = 15
wait_time = 10

""" ************************************************************************************************************************ """

# Main menu
# -------------------------------------------------------------------------------------------
def menu(frame):

    start = False

    x, y, _ = frame.shape

    frame = cv2.flip(frame, 1)
    results = pose.process(frame)

    # Draw the pose annotation on the image.
    frame.flags.writeable = True
    mp_drawing.draw_landmarks(frame,results.pose_landmarks,mp_pose.POSE_CONNECTIONS,landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

    # Generate output by blending image with shapes image, using the shapes
    # images also as scaled to limit the blending to those parts
    shapes = np.zeros_like(frame, np.uint8) 
    
    # Quit button
    shapes = rounded_rectangle(shapes, Quit_btn_coord[0], Quit_btn_coord[1], color=color, radius=0.5, thickness=-1)
    # Start button
    shapes = rounded_rectangle(shapes, Start_btn_coord[0], Start_btn_coord[1], color=color, radius=0.5, thickness=-1)
    out = frame.copy()
    alpha = 0.25
    scaled = shapes.astype(bool)
    out[scaled] = cv2.addWeighted(frame, alpha, shapes, 1 - alpha, 0)[scaled]
    

    cv2.putText(out, "Start", ( 605, 115 ), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (74,74,74), 3, cv2.LINE_AA)
    cv2.putText(out, "Start", ( 605, 115 ), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (255,255,255), 1, cv2.LINE_AA)
    
    
    cv2.putText(out, "OPEN DANCE", ( 450, 360 ), cv2.FONT_HERSHEY_SIMPLEX, 
                        2, (74,74,74), 10, cv2.LINE_AA)
    cv2.putText(out, "OPEN DANCE", ( 450, 360 ), cv2.FONT_HERSHEY_SIMPLEX, 
                        2, (85,34,244), 2, cv2.LINE_AA)
    

    cv2.putText(out, "Quit", ( 605, 655 ), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (74,74,74), 3, cv2.LINE_AA)
    cv2.putText(out, "Quit", ( 605, 655 ), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (255,255,255), 1, cv2.LINE_AA)
                
    cv2.imshow("MAIN MENU",out)

    if results.pose_landmarks:
        
        left_index = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_INDEX]

        if left_index.x * y <= Start_btn_coord[1][1] and left_index.x * y >= Start_btn_coord[0][0]:
            if(left_index.y * x <= Start_btn_coord[1][0] and left_index.y * x >= Start_btn_coord[0][1]):
                print("game_started")
                start = True

        if left_index.x * y <= Quit_btn_coord[1][1] and left_index.x * y >= Quit_btn_coord[0][0]:
            if(left_index.y * x <= Quit_btn_coord[1][0] and left_index.y * x >= Quit_btn_coord[0][1]):
                print("game_stopped")
                # release the webcam and destroy all active windows
                cap.release()
                cv2.destroyAllWindows()
                exit()
    
    return start
# -------------------------------------------------------------------------------------------


# Difficulty Menu
# -------------------------------------------------------------------------------------------
def diff(frame):

    global level
    global wait_time

    difficulty = False

    x, y, c = frame.shape

    frame = cv2.flip(frame, 1)
    results = pose.process(frame)

    # Draw the pose annotation on the image.
    frame.flags.writeable = True
    mp_drawing.draw_landmarks(frame,results.pose_landmarks,mp_pose.POSE_CONNECTIONS,landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

    # Generate output by blending image with shapes image, using the shapes
    # images also as scaled to limit the blending to those parts
    shapes = np.zeros_like(frame, np.uint8) 
    
    
    # Easy button
    shapes = rounded_rectangle(shapes,Difficulty_Easy_coord[0],Difficulty_Easy_coord[1],color=color,radius=0.5,thickness=-1)

    # Medium button
    shapes = rounded_rectangle(shapes, Difficulty_Medium_coord[0], Difficulty_Medium_coord[1], color=color, radius=0.5, thickness=-1)
    
    # Hard button
    shapes = rounded_rectangle(shapes, Difficulty_Hard_coord[0], Difficulty_Hard_coord[1], color=color, radius=0.5, thickness=-1)
    

    out = frame.copy()
    alpha = 0.25
    scaled = shapes.astype(bool)
    out[scaled] = cv2.addWeighted(frame, alpha, shapes, 1 - alpha, 0)[scaled]
    

    cv2.putText(out, "SELECT DIFFICULTY", ( 380, 115 ), cv2.FONT_HERSHEY_SIMPLEX, 
                        2, (154,154,154), 3, cv2.LINE_AA)
    cv2.putText(out, "SELECT DIFFICULTY", ( 380, 115 ), cv2.FONT_HERSHEY_SIMPLEX, 
                        2, (255,255,255), 1, cv2.LINE_AA)
    
    
    cv2.putText(out, "Easy", (610, 340), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (74,74,74), 3, cv2.LINE_AA)
    cv2.putText(out, "Easy", ( 610, 340), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (255,255,255), 1, cv2.LINE_AA)

    cv2.putText(out, "Medium", ( 595, 505), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (74,74,74), 3, cv2.LINE_AA)
    cv2.putText(out, "Medium", ( 595, 505 ), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (255,255,255), 1, cv2.LINE_AA)

    cv2.putText(out, "Hard", ( 605, 655 ), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (74,74,74), 3, cv2.LINE_AA)
    cv2.putText(out, "Hard", ( 605, 655 ), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (255,255,255), 1, cv2.LINE_AA)
                
    cv2.imshow("Difficulty Menu",out) 

    if results.pose_landmarks:
        
        left_index = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_INDEX]
            
        if left_index.x * y <= Difficulty_Easy_coord[1][1] and left_index.x * y >= Difficulty_Easy_coord[0][0]:
            if(left_index.y * x <= Difficulty_Easy_coord[1][0] and left_index.y * x >= Difficulty_Easy_coord[0][1]):
                print("Easy has been selected")
                difficulty = True
                wait_time = 15
                level = 20

        if left_index.x * y <= Difficulty_Medium_coord[1][1] and left_index.x * y >= Difficulty_Medium_coord[0][0]:
            if(left_index.y * x <= Difficulty_Medium_coord[1][0] and left_index.y * x >= Difficulty_Medium_coord[0][1]):
                print("Medium has been selected")
                difficulty = True
                wait_time = 10
                level = 15

        if left_index.x * y <= Difficulty_Hard_coord[1][1] and left_index.x * y >= Difficulty_Hard_coord[0][0]:
            if(left_index.y * x <= Difficulty_Hard_coord[1][0] and left_index.y * x >= Difficulty_Hard_coord[0][1]):
                print("Hard has been selected")
                difficulty = True
                wait_time = 5
                level = 8


    return difficulty
# -------------------------------------------------------------------------------------------


# Draws rounded rectangles
# -------------------------------------------------------------------------------------------
def rounded_rectangle(src, top_left, bottom_LEFT, radius=1, color=255, thickness=1, line_type=cv2.LINE_AA):

    #  corners:
    #  p1 - p2
    #  |     |
    #  p4 - p3

    p1 = top_left
    p2 = (bottom_LEFT[1], top_left[1])
    p3 = (bottom_LEFT[1], bottom_LEFT[0])
    p4 = (top_left[0], bottom_LEFT[0])

    height = abs(bottom_LEFT[0] - top_left[1])

    if radius > 1:
        radius = 1

    corner_radius = int(radius * (height/2))

    if thickness < 0:

        #big rect
        top_left_main_rect = (int(p1[0] + corner_radius), int(p1[1]))
        bottom_LEFT_main_rect = (int(p3[0] - corner_radius), int(p3[1]))

        top_left_rect_left = (p1[0], p1[1] + corner_radius)
        bottom_LEFT_rect_left = (p4[0] + corner_radius, p4[1] - corner_radius)

        top_left_rect_LEFT = (p2[0] - corner_radius, p2[1] + corner_radius)
        bottom_LEFT_rect_LEFT = (p3[0], p3[1] - corner_radius)

        all_rects = [
        [top_left_main_rect, bottom_LEFT_main_rect], 
        [top_left_rect_left, bottom_LEFT_rect_left], 
        [top_left_rect_LEFT, bottom_LEFT_rect_LEFT]]

        [cv2.rectangle(src, rect[0], rect[1], color, thickness) for rect in all_rects]

    # draw straight lines
    cv2.line(src, (p1[0] + corner_radius, p1[1]), (p2[0] - corner_radius, p2[1]), color, abs(thickness), line_type)
    cv2.line(src, (p2[0], p2[1] + corner_radius), (p3[0], p3[1] - corner_radius), color, abs(thickness), line_type)
    cv2.line(src, (p3[0] - corner_radius, p4[1]), (p4[0] + corner_radius, p3[1]), color, abs(thickness), line_type)
    cv2.line(src, (p4[0], p4[1] - corner_radius), (p1[0], p1[1] + corner_radius), color, abs(thickness), line_type)

    # draw arcs
    cv2.ellipse(src, (p1[0] + corner_radius, p1[1] + corner_radius), (corner_radius, corner_radius), 180.0, 0, 90, color ,thickness, line_type)
    cv2.ellipse(src, (p2[0] - corner_radius, p2[1] + corner_radius), (corner_radius, corner_radius), 270.0, 0, 90, color , thickness, line_type)
    cv2.ellipse(src, (p3[0] - corner_radius, p3[1] - corner_radius), (corner_radius, corner_radius), 0.0, 0, 90,   color , thickness, line_type)
    cv2.ellipse(src, (p4[0] + corner_radius, p4[1] - corner_radius), (corner_radius, corner_radius), 90.0, 0, 90,  color , thickness, line_type)

    return src
# -------------------------------------------------------------------------------------------


# Computes a random angle for the joints
# -------------------------------------------------------------------------------------------
def rand_dir():
    
    dir = random.randint(0,3)
    
    if dir == 0:
        angle = math.pi/3
    elif dir == 1:
        angle = math.pi/4
    elif dir == 2:
        angle = math.pi/6
    elif dir == 3:
        angle = 0
    
    return angle
# -------------------------------------------------------------------------------------------


# Draws a rectangle at a certain angle
# ------------------------------------------------------------------------------------------- 
def tilted_rect(src, p1, w, h, theta, c):
    
    #  corners:
    #  p1 - p2
    #  |     |
    #  p4 - p3
    
    p2 = np.array([0,0])
    p3 = np.array([0,0])
    p4 = np.array([0,0])
    
    theta = (theta * math.pi)/180

    phi = math.pi/2 - theta
    alfa = phi
    delta = math.pi/2 - phi
    
    p2[0] = int(p1[0] + w*math.cos(theta))
    p2[1] = int(p1[1] + w*math.sin(theta))
    
    p3[0] = int(p2[0] - h*math.sin(delta))
    p3[1] = int(p2[1] + h*math.cos(delta))
    
    p4[0] = int(p1[0] - h*math.cos(alfa))
    p4[1] = int(p1[1] + h*math.sin(alfa))
    
    cv2.line(src, p1, p2, c, 3)
    cv2.line(src, p2, p3, c, 3)
    cv2.line(src, p4, p3, c, 3)
    cv2.line(src, p1, p4, c, 3)
    
    return src
# -------------------------------------------------------------------------------------------


# Checks for joints macthing
# -------------------------------------------------------------------------------------------
def match_joints(joint1, joint2):

    if( ( joint1[0] < (joint2[0] + level ) ) and ( joint1[1] < (joint2[1] + level ) ) and 
        ( joint1[0] > (joint2[0] - level ) ) and ( joint1[1] > (joint2[1] - level ) )    ):
    
        cv2.circle(frame,joint2,20,(0,255,0),-1)
    
        return True
    else:
        return False
# -------------------------------------------------------------------------------------------


# Computes joint coordinates 
# -------------------------------------------------------------------------------------------
def compute_joint(joint):
    
    return np.array([int(joint.x*frame_width), int(joint.y*frame_height)])
# -------------------------------------------------------------------------------------------

# Variables for loops -----
is_menu = True
is_play = True
is_difficulty = True
random.seed(time.time())
# -------------------------

while cap.isOpened():
    
    # ***************************** Calib demo ***************************** #
    
    if show_calib:
    
        go_up1 = True
        go_up2 = True
        go_up3 = True
        go_up4 = True
        
        angle1 = 0
        angle2 = 1.0
        
        angle3 = 0
        angle4 = 1.0
        
        
        scaled_right_shoulder = np.array([0,0])
        scaled_left_shoulder = np.array([0,0])
        
        
        scaled_right_elbow = np.array([0,0])
        scaled_left_elbow = np.array([0,0])
        
        
        scaled_right_wrist = np.array([0,0])
        scaled_left_wrist = np.array([0,0])
        
        
        scaled_right_hip = np.array([0,0])
        scaled_left_hip = np.array([0,0])
        
        
        left_d_se = 0
                
        left_d_ew = 0
        
        right_d_se = 0
        
        right_d_ew = 0
        
        
        while True:
            
            _, frame = cap.read()
            
            
            frame = cv2.flip(frame, 1)
            results = pose.process(frame)

            # Draw the pose annotation on the image.
            frame.flags.writeable = True
            mp_drawing.draw_landmarks(frame,results.pose_landmarks,mp_pose.POSE_CONNECTIONS,landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

            # Generate output by blending image with shapes image, using the shapes
            # images also as scaled to limit the blending to those parts
            shapes = np.zeros_like(frame, np.uint8)
            
            
            
            if results.pose_landmarks:
        
                scaled_right_shoulder = compute_joint(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER])
                scaled_left_shoulder = compute_joint(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER])
                
                
                scaled_right_elbow = compute_joint(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW])
                scaled_left_elbow = compute_joint(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW])
                
                
                scaled_right_wrist = compute_joint(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST])
                scaled_left_wrist = compute_joint(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST])
                
                
                scaled_right_hip = compute_joint(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP])
                scaled_left_hip = compute_joint(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP])
            
                
                
                left_d_se = math.sqrt( pow( (scaled_left_shoulder[0] - scaled_left_elbow[0]), 2) 
                                + pow( (scaled_left_shoulder[1] - scaled_left_elbow[1]), 2))
                
                left_d_ew = math.sqrt( pow( (scaled_left_elbow[0] - scaled_left_wrist[0]), 2) 
                                + pow( (scaled_left_elbow[1] - scaled_left_wrist[1]), 2))
                
                right_d_se = math.sqrt( pow( (scaled_right_shoulder[0] - scaled_right_elbow[0]), 2) 
                                + pow( (scaled_right_shoulder[1] - scaled_right_elbow[1]), 2))
                
                right_d_ew = math.sqrt( pow( (scaled_right_elbow[0] - scaled_right_wrist[0]), 2) 
                                + pow( (scaled_right_elbow[1] - scaled_right_wrist[1]), 2))
            # ---------------------------------------------------------------------------------
            
            
            
            # Mesh above the player ------------------------------------------------
            cv2.line(frame, scaled_right_shoulder, scaled_left_shoulder, color, 15)
            cv2.line(frame, scaled_right_hip, scaled_left_hip, color, 15)
            
            cv2.line(frame, scaled_right_shoulder, scaled_right_hip, color, 15)
            cv2.line(frame, scaled_left_shoulder, scaled_left_hip, color, 15)
            

            
            if round(angle1, 2) == 0.0:
                go_up1 = True
            elif round(angle1, 2) == 1.0:
                go_up1 = False
                
            if go_up1:
                angle1 = angle1 + 0.05
            else:
                angle1 = angle1 - 0.05
                
                
                
            if round(angle2, 2) == 0.0:
                go_up2 = True
            elif round(angle2, 2) == 1.0:
                go_up2 = False
                
            if go_up2:
                angle2 = angle2 + 0.05
            else:
                angle2 = angle2 - 0.05
                
                
            
            if round(angle3, 2) == 0.0:
                go_up3 = True
            elif round(angle3, 2) == 1.0:
                go_up3 = False
                
            if go_up3:
                angle3 = angle3 + 0.05
            else:
                angle3 = angle3 - 0.05
            
            
            
            if round(angle4, 2) == 0.0:
                go_up4 = True
            elif round(angle1, 2) == 1.0:
                go_up4 = False
                
            if go_up4:
                angle4 = angle4 + 0.05
            else:
                angle4 = angle4 - 0.05
                
                
            
            # Computes the stickman joints coordinates ------------------------------------------------
            deltaX_leftElbow = (left_d_se*math.cos(angle1))
            deltaY_leftElbow = (left_d_se*math.sin(angle1))
            
            if deltaX_leftElbow < 0: deltaX_leftElbow = deltaX_leftElbow * (-1)
            
            
            deltaX_leftWrist = (left_d_ew*math.cos(angle2))
            deltaY_leftWrist = (left_d_ew*math.sin(angle2))
            
            if deltaX_leftWrist < 0: deltaX_leftWrist = deltaX_leftWrist * (-1)
            
            
            deltaX_rightElbow = (right_d_se*math.cos(angle3))
            deltaY_rightElbow = (right_d_se*math.sin(angle3))
            
            if deltaX_rightElbow > 0: deltaX_rightElbow = deltaX_rightElbow * (-1)
            
            
            deltaX_rightWrist= (right_d_ew*math.cos(angle4))
            deltaY_rightWrist = (right_d_ew*math.sin(angle4))
            
            if deltaX_rightWrist> 0: deltaX_rightWrist= deltaX_rightWrist* (-1)
            
            
            stickman_left_elbow = np.array( [int(scaled_left_shoulder[0] + deltaX_leftElbow), 
                                                int(scaled_left_shoulder[1] + deltaY_leftElbow)] )
            
            cv2.circle(frame,stickman_left_elbow,20,(0,0,255),-1)
            cv2.line(frame, scaled_left_shoulder, stickman_left_elbow , color, 15)
            
            stickman_left_wrist = np.array( [int(stickman_left_elbow[0] + deltaX_leftWrist), 
                                                int(stickman_left_elbow[1] + deltaY_leftWrist)] )
            
            cv2.circle(frame,stickman_left_wrist,20,(0,0,255),-1)
            cv2.line(frame, stickman_left_elbow, stickman_left_wrist,  color, 15)
            
            stickman_right_elbow = np.array( [int(scaled_right_shoulder[0] + deltaX_rightElbow), 
                                                int(scaled_right_shoulder[1] + deltaY_rightElbow)] )
            
            cv2.circle(frame,stickman_right_elbow,20,(0,0,255),-1)
            cv2.line(frame, scaled_right_shoulder, stickman_right_elbow, color, 15)
            
            stickman_right_wrist = np.array( [int(stickman_right_elbow[0] + deltaX_rightWrist), 
                                                int(stickman_right_elbow[1] + deltaY_rightWrist)] )
            
            cv2.circle(frame,stickman_right_wrist,20,(0,0,255),-1)
            cv2.line(frame, stickman_right_elbow, stickman_right_wrist,  color, 15)
                
                
            
            cv2.imshow("CALIB DEMO",frame)
            
            if cv2.waitKey(1) == ord('q'):
                cv2.destroyAllWindows()
                cap.release()  
                exit()
    
    # ********************************************************************** #
    
    
    # ***************************** Main menu ***************************** #
    
    while is_menu:

        _, frame = cap.read()


        # Displays Menu ------ 
        start = menu(frame)
        # --------------------

        if start: break

        if cv2.waitKey(1) == ord('q'):
            cv2.destroyAllWindows()
            cap.release()  
            exit()

    # ********************************************************************* #


    cv2.destroyAllWindows()


    # ***************************** Difficulty selection ***************************** #

    while is_difficulty:

        _, frame = cap.read()

        # Display Difficulty Settings --
        difficulty = diff(frame)
        #-------------------------------

        if difficulty: break

        if cv2.waitKey(1) == ord('q'):
            cv2.destroyAllWindows()
            cap.release()  
            exit()
        
    # ******************************************************************************** #


    cv2.destroyAllWindows()
    
    
    start_t = time.time()
    angle = 0
    points = 0
    lives = 1
    matchs = 0
    Fin_juego = False
    lives_lost = 0


    # ***************************** Main Play ***************************** #

    while is_play:
        
        _, frame = cap.read()
        
        h, w, _ = frame.shape
        

        # [right wrist, right elbow, left elbow, left wrist]
        joints_flags = np.array([False,False,False,False])
        
        
        frame = cv2.flip(frame, 1)
        results = pose.process(frame)


        # Score, lives and time titles
        # ---------------------------------------------------------------------------------
        cv2.putText(frame, "Score", ( 70, 30 ), cv2.FONT_HERSHEY_SIMPLEX, 
                      1, (74,74,74), 10, cv2.LINE_AA)
        cv2.putText(frame, "Score", ( 70, 30 ), cv2.FONT_HERSHEY_SIMPLEX, 
                      1, (85,34,244), 2, cv2.LINE_AA)

        cv2.putText(frame, str(points), ( 105, 70 ), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (74,74,74), 10, cv2.LINE_AA)
        cv2.putText(frame, str(points), ( 105, 70 ), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (85,34,244), 2, cv2.LINE_AA)



        cv2.putText(frame, "Lives", ( 1100, 30 ), cv2.FONT_HERSHEY_SIMPLEX, 
                      1, (74,74,74), 10, cv2.LINE_AA)
        cv2.putText(frame, "Lives", ( 1100, 30 ), cv2.FONT_HERSHEY_SIMPLEX, 
                      1, (85,34,244), 2, cv2.LINE_AA)

        cv2.putText(frame, str(lives), ( 1130, 70 ), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (74,74,74), 10, cv2.LINE_AA)
        cv2.putText(frame, str(lives), ( 1130, 70 ), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (85,34,244), 2, cv2.LINE_AA)
                    
                    

        cv2.putText(frame, "Time left", ( 585, 30 ), cv2.FONT_HERSHEY_SIMPLEX, 
                     1, (74,74,74), 10, cv2.LINE_AA)
        cv2.putText(frame, "Time left", ( 585, 30 ), cv2.FONT_HERSHEY_SIMPLEX, 
                      1, (85,34,244), 2, cv2.LINE_AA)

        aux_time = int(time.time() - start_t)

        cv2.putText(frame, str(wait_time-aux_time), ( 640, 70 ), cv2.FONT_HERSHEY_SIMPLEX, 
                      1, (74,74,74), 10, cv2.LINE_AA)
        cv2.putText(frame, str(wait_time-aux_time), ( 640, 70 ), cv2.FONT_HERSHEY_SIMPLEX, 
                      1, (85,34,244), 2, cv2.LINE_AA)
        # ---------------------------------------------------------------------------------


        # Draw the pose annotation on the image
        # ---------------------------------------------------------------------------------
        frame.flags.writeable = True
        mp_drawing.draw_landmarks(frame,results.pose_landmarks,mp_pose.POSE_CONNECTIONS,
                                  landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

        # Cheking for the pose 
        if results.pose_landmarks:
    
            scaled_right_shoulder = compute_joint(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER])
            scaled_left_shoulder = compute_joint(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER])
            
            
            scaled_right_elbow = compute_joint(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW])
            scaled_left_elbow = compute_joint(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW])
            
            
            scaled_right_wrist = compute_joint(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST])
            scaled_left_wrist = compute_joint(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST])
            
            
            scaled_right_hip = compute_joint(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP])
            scaled_left_hip = compute_joint(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP])
        
            
            
            left_d_se = math.sqrt( pow( (scaled_left_shoulder[0] - scaled_left_elbow[0]), 2) 
                            + pow( (scaled_left_shoulder[1] - scaled_left_elbow[1]), 2))
            
            left_d_ew = math.sqrt( pow( (scaled_left_elbow[0] - scaled_left_wrist[0]), 2) 
                            + pow( (scaled_left_elbow[1] - scaled_left_wrist[1]), 2))
            
            right_d_se = math.sqrt( pow( (scaled_right_shoulder[0] - scaled_right_elbow[0]), 2) 
                            + pow( (scaled_right_shoulder[1] - scaled_right_elbow[1]), 2))
            
            right_d_ew = math.sqrt( pow( (scaled_right_elbow[0] - scaled_right_wrist[0]), 2) 
                            + pow( (scaled_right_elbow[1] - scaled_right_wrist[1]), 2))
        # ---------------------------------------------------------------------------------
        
        
        
        # Mesh above the player ------------------------------------------------
        cv2.line(frame, scaled_right_shoulder, scaled_left_shoulder, color, 15)
        cv2.line(frame, scaled_right_hip, scaled_left_hip, color, 15)
        
        cv2.line(frame, scaled_right_shoulder, scaled_right_hip, color, 15)
        cv2.line(frame, scaled_left_shoulder, scaled_left_hip, color, 15)
        # ----------------------------------------------------------------------
        
        
        act_time = int(time.time() - start_t)
        
        if act_time == 1:
            
            angle1 = rand_dir()
            angle2 = rand_dir()
            
            angle3 = rand_dir()
            angle4 = rand_dir()
        
        elif act_time > 1:
            
            # Computes the stickman joints coordinates ------------------------------------------------
            deltaX_leftElbow = (left_d_se*math.cos(angle1))
            deltaY_leftElbow = (left_d_se*math.sin(angle1))
            
            if deltaX_leftElbow < 0: deltaX_leftElbow = deltaX_leftElbow * (-1)
            
            
            deltaX_leftWrist = (left_d_ew*math.cos(angle2))
            deltaY_leftWrist = (left_d_ew*math.sin(angle2))
            
            if deltaX_leftWrist < 0: deltaX_leftWrist = deltaX_leftWrist * (-1)
            
            
            deltaX_rightElbow = (right_d_se*math.cos(angle3))
            deltaY_rightElbow = (right_d_se*math.sin(angle3))
            
            if deltaX_rightElbow > 0: deltaX_rightElbow = deltaX_rightElbow * (-1)
            
            
            deltaX_rightWrist= (right_d_ew*math.cos(angle4))
            deltaY_rightWrist = (right_d_ew*math.sin(angle4))
            
            if deltaX_rightWrist> 0: deltaX_rightWrist= deltaX_rightWrist* (-1)
            
            
            stickman_left_elbow = np.array( [int(scaled_left_shoulder[0] + deltaX_leftElbow), 
                                             int(scaled_left_shoulder[1] + deltaY_leftElbow)] )
            
            cv2.circle(frame,stickman_left_elbow,20,(0,0,255),-1)
            cv2.line(frame, scaled_left_shoulder, stickman_left_elbow , color, 15)
            
            stickman_left_wrist = np.array( [int(stickman_left_elbow[0] + deltaX_leftWrist), 
                                             int(stickman_left_elbow[1] + deltaY_leftWrist)] )
            
            cv2.circle(frame,stickman_left_wrist,20,(0,0,255),-1)
            cv2.line(frame, stickman_left_elbow, stickman_left_wrist,  color, 15)
            
            stickman_right_elbow = np.array( [int(scaled_right_shoulder[0] + deltaX_rightElbow), 
                                              int(scaled_right_shoulder[1] + deltaY_rightElbow)] )
            
            cv2.circle(frame,stickman_right_elbow,20,(0,0,255),-1)
            cv2.line(frame, scaled_right_shoulder, stickman_right_elbow, color, 15)
            
            stickman_right_wrist = np.array( [int(stickman_right_elbow[0] + deltaX_rightWrist), 
                                              int(stickman_right_elbow[1] + deltaY_rightWrist)] )
            
            cv2.circle(frame,stickman_right_wrist,20,(0,0,255),-1)
            cv2.line(frame, stickman_right_elbow, stickman_right_wrist,  color, 15)
            # -----------------------------------------------------------------------------------------
            
            
            # Matching the joints ---------------------------------------------------------------------
            
            joints_flags[3] = match_joints(scaled_left_wrist, stickman_left_wrist)
            
            joints_flags[2] = match_joints(scaled_left_elbow, stickman_left_elbow)
            
            joints_flags[1] = match_joints(scaled_right_elbow, stickman_right_elbow)
            
            joints_flags[0] = match_joints(scaled_right_wrist, stickman_right_wrist)   

            if( joints_flags[0] and joints_flags[1] and joints_flags[2] and joints_flags[3] ): # All joints Match
                
                start_t = time.time()
                points = points + (10 * lives)
                matchs = matchs + 1
                
                if matchs == math.pow(2,lives):
                    lives = lives + 1

            elif(aux_time == wait_time):
                
                start_t = time.time()
                
                matchs = 0
                lives = lives - 1
                lives_lost = lives_lost + 1
            
            if(lives <= 0):
                Fin_juego = True
                break
            # -----------------------------------------------------------------------------------------

        cv2.imshow("PLAYING",frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            cap.release()  
            exit()

    # ********************************************************************* #
    
    
    if (Fin_juego):
        cv2.destroyAllWindows()


    # ***************************** Final screen ***************************** #


    while Fin_juego:
        _, frame = cap.read()
        
        x, y, c = frame.shape
        frame = cv2.flip(frame, 1)
        results = pose.process(frame)

        frame.flags.writeable = True
        mp_drawing.draw_landmarks(frame,results.pose_landmarks,mp_pose.POSE_CONNECTIONS,landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

        cv2.putText(frame, "Highest players scores", ( 50, 80 ), cv2.FONT_HERSHEY_SIMPLEX, 
                      1, (74,74,74), 10, cv2.LINE_AA)

        cv2.putText(frame, "Highest players scores", ( 50, 80 ), cv2.FONT_HERSHEY_SIMPLEX, 
                      1, (85,34,244), 2, cv2.LINE_AA)

        cv2.putText(frame, "Ander     -> ", ( 50, 120 ), cv2.FONT_HERSHEY_SIMPLEX, 
                      1, (123,45,33), 2, cv2.LINE_AA)

        cv2.putText(frame, "240", ( 360, 120 ), cv2.FONT_HERSHEY_SIMPLEX, 
                      1, (123,45,33), 2, cv2.LINE_AA)


        cv2.putText(frame, "Javier     -> ", ( 50, 160 ), cv2.FONT_HERSHEY_SIMPLEX, 
                      1, (60,47,33), 2, cv2.LINE_AA)

        cv2.putText(frame, "180", ( 360, 160 ), cv2.FONT_HERSHEY_SIMPLEX, 
                      1, (60,47,33), 2, cv2.LINE_AA)

        # ----------------------------------------------------------------------

        cv2.putText(frame, "Your score", ( 560, 80 ), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (74,74,74), 10, cv2.LINE_AA)

        cv2.putText(frame, "Your score", ( 560, 80 ), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (85,34,244), 2, cv2.LINE_AA)


        cv2.putText(frame, str(points), ( 625, 140 ), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (74,74,74), 10, cv2.LINE_AA)
        cv2.putText(frame, str(points), ( 625, 140 ), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (85,34,244), 2, cv2.LINE_AA)
        # ----------------------------------------------------------------------
        cv2.putText(frame, "Total lives lost", ( 960, 80 ), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (74,74,74), 10, cv2.LINE_AA)

        cv2.putText(frame, "Total lives lost", ( 960, 80 ), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (85,34,244), 2, cv2.LINE_AA)

        
        cv2.putText(frame, str(lives_lost), ( 1070, 140 ), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (74,74,74), 10, cv2.LINE_AA)
        cv2.putText(frame, str(lives_lost), ( 1070, 140 ), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (85,34,244), 2, cv2.LINE_AA)
        # ----------------------------------------------------------------------

        if(points > 240):
            cv2.putText(frame, "CONGRATULATIONS NEW RECORD", ( 110, 360 ), cv2.FONT_HERSHEY_SIMPLEX, 
                    2, (74,74,74), 10, cv2.LINE_AA)
            cv2.putText(frame, "CONGRATULATIONS NEW RECORD", ( 110, 360 ), cv2.FONT_HERSHEY_SIMPLEX, 
                    2, (85,34,244), 2, cv2.LINE_AA)
        elif(points > 180):
            cv2.putText(frame, "Congratulations you surpassed Javier", ( 50, 360 ), cv2.FONT_HERSHEY_SIMPLEX, 
                    2, (74,74,74), 10, cv2.LINE_AA)
            cv2.putText(frame, "Congratulations you surpassed Javier", ( 50, 360 ), cv2.FONT_HERSHEY_SIMPLEX, 
                    2, (85,34,244), 2, cv2.LINE_AA)
        else : 
            cv2.putText(frame, "You can do better", ( 360, 360 ), cv2.FONT_HERSHEY_SIMPLEX, 
                    2, (74,74,74), 10, cv2.LINE_AA)
            cv2.putText(frame, "You can do better", ( 360, 360 ), cv2.FONT_HERSHEY_SIMPLEX, 
                    2, (85,34,244), 2, cv2.LINE_AA)
        # ----------------------------------------------------------------------
        shapes = np.zeros_like(frame, np.uint8) 
        shapes = rounded_rectangle(shapes, Quit_btn2_coord[0], Quit_btn2_coord[1], color=color, radius=0.5, thickness=-1)
        shapes = rounded_rectangle(shapes, Principal_Menu_coord[0], Principal_Menu_coord[1], color=color, radius=0.5, thickness=-1)

        out = frame.copy()
        alpha = 0.25
        scaled = shapes.astype(bool)
        out[scaled] = cv2.addWeighted(frame, alpha, shapes, 1 - alpha, 0)[scaled]
        

        cv2.putText(out, "Principal Menu", ( 535, 505), cv2.FONT_HERSHEY_SIMPLEX, 
                      1, (74,74,74), 3, cv2.LINE_AA)
        cv2.putText(out, "Principal Menu", ( 535, 505 ), cv2.FONT_HERSHEY_SIMPLEX, 
                      1, (255,255,255), 1, cv2.LINE_AA)


        cv2.putText(out, "Quit", ( 610, 655 ), cv2.FONT_HERSHEY_SIMPLEX, 
                            1, (74,74,74), 3, cv2.LINE_AA)
        cv2.putText(out, "Quit", ( 610, 655 ), cv2.FONT_HERSHEY_SIMPLEX, 
                            1, (255,255,255), 1, cv2.LINE_AA)


    
        cv2.imshow("Fin de Partida",out)

        if results.pose_landmarks:
        
            left_index = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_INDEX]

            if left_index.x * y <= Principal_Menu_coord[1][1] and left_index.x * y >= Principal_Menu_coord[0][0]:
                if(left_index.y * x <= Principal_Menu_coord[1][0] and left_index.y * x >= Principal_Menu_coord[0][1]):
                    print("Game_Menu")
                    cv2.destroyAllWindows()
                    break
                    

            if left_index.x * y <= Quit_btn2_coord[1][1] and left_index.x * y >= Quit_btn2_coord[0][0]:
                if(left_index.y * x <= Quit_btn2_coord[1][0] and left_index.y * x >= Quit_btn2_coord[0][1]):
                    print("game_stopped")
                    # release the webcam and destroy all active windows
                    cap.release()
                    cv2.destroyAllWindows()
                    exit()


        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            cap.release()  
            exit()
            
    # ************************************************************************ #