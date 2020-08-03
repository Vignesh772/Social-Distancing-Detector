import numpy as np
import tensorflow as tf
import threading 
import detector

import cv2
from scipy.spatial import distance
refPts=[]
def gui():
    import slider
    


    
def click_and_crop(event, x, y, flags, param):
    global refPts
    image1=first_snap
	
    if(event == cv2.EVENT_LBUTTONDOWN):
        refPts.append([x,y])
        cv2.circle(first_snap, (x,y), 5, (0, 0, 255), -1)
        print(refPts)
        if(len(refPts)==4):
            width=abs(refPts[0][0]-refPts[1][0])
            height=abs(refPts[0][1]-refPts[2][1])
            print(height,width)
            

def perspective(w,h,image1,points_list,refPt):
    
    pts1 = np.float32(refPt)
    pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    
    
    corners=np.array([[[0,0]],[[640,0]],[[0,480]],[[640,480]]])
    corner_homg_points = np.array([[x, y, 1] for [[x, y]] in corners]).T
    transf_corner_points = matrix.dot(corner_homg_points)
    transf_corner_points /= transf_corner_points[2]
    transf_points_corner = np.array([[[x,y]] for [x, y] in transf_corner_points[:2].T])
    
    
    x_min=min(transf_points_corner[0][0][0],transf_points_corner[1][0][0],
          transf_points_corner[2][0][0],transf_points_corner[3][0][0])
    x_max=max(transf_points_corner[0][0][0],transf_points_corner[1][0][0],
          transf_points_corner[2][0][0],transf_points_corner[3][0][0])
    
    y_min=min(transf_points_corner[0][0][1],transf_points_corner[1][0][1],
          transf_points_corner[2][0][1],transf_points_corner[3][0][1])
    y_max=max(transf_points_corner[0][0][1],transf_points_corner[1][0][1],
          transf_points_corner[2][0][1],transf_points_corner[3][0][1])
    
    #print("X_min, Y_min ",x_min,y_min)
    
    
    transf_points_corner[0][0][0]=transf_points_corner[0][0][0]-x_min
    transf_points_corner[1][0][0]=transf_points_corner[1][0][0]-x_min
    transf_points_corner[2][0][0]=transf_points_corner[2][0][0]-x_min
    transf_points_corner[3][0][0]=transf_points_corner[3][0][0]-x_min
    
    transf_points_corner[0][0][1]=transf_points_corner[0][0][1]-y_min
    transf_points_corner[1][0][1]=transf_points_corner[1][0][1]-y_min
    transf_points_corner[2][0][1]=transf_points_corner[2][0][1]-y_min
    transf_points_corner[3][0][1]=transf_points_corner[3][0][1]-y_min
    
    pt1=np.float32([[0,0],[640,0],[0,480],[640,480]])
    pt2=np.float32([[ transf_points_corner[0][0][0], transf_points_corner[0][0][1]],
                   [ transf_points_corner[1][0][0], transf_points_corner[1][0][1]],
                   [ transf_points_corner[2][0][0], transf_points_corner[2][0][1]],
                   [ transf_points_corner[3][0][0], transf_points_corner[3][0][1]]])
    #print("PT2 : ",pt2)
    
    
    matrix1=cv2.getPerspectiveTransform(pt1, pt2)
    result_to_display = cv2.warpPerspective(image1, matrix1, ( int(x_max-x_min),int(y_max-y_min)))
    
            
    #result = cv2.warpPerspective(image1, matrix, (w,h))
    if(len(points_list) > 0):
        points = np.array(points_list)
        homg_points = np.array([[x, y, 1] for [[x, y]] in points]).T
        transf_homg_points = matrix1.dot(homg_points)
        transf_homg_points /= transf_homg_points[2]
        transf_points = np.array([[[x,y]] for [x, y] in transf_homg_points[:2].T])
    else:
        transf_points=[[[]]]
        
        
    #print(transf_points)
    
    
    
    
    
    point1 = np.array([refPt[2][0],refPt[2][1]])
    homg_point1 = [point1[0], point1[1], 1] # homogeneous coords
    transf_homg_point1 = matrix1.dot(homg_point1) # transform
    transf_homg_point1 /= transf_homg_point1[2] # scale
    transf_point1 = transf_homg_point1[:2] # remove Cartesian coords
    
    point2 = np.array([refPt[3][0],refPt[3][1]])
    homg_point2 = [point2[0], point2[1], 1] # homogeneous coords
    transf_homg_point2 = matrix1.dot(homg_point2) # transform
    transf_homg_point2 /= transf_homg_point2[2] # scale
    transf_point2 = transf_homg_point2[:2]
    
    
    distance_btw_road_in_trans=distance.euclidean(transf_point1,transf_point2)
    #result=cv2.resize(result,(640,480))
    result=result_to_display
    return(result,transf_points,distance_btw_road_in_trans,result_to_display)

def video():
    writer = None
    frame_width=1000
    frame_height=700
    
    while True:
        f = open("minValue.txt", "r")

        _,frame=cap.read()
        frame=cv2.resize(frame,(640,480))
        img=cv2.resize(frame,(640,480))

        
       
        
        output_data_location,output_data_confidence,output_data_item=detector.human_detection(frame)
        #print(output_data_location,output_data_confidence,output_data_item)
        points=[]
        points_for_rectangle=[]
        for i in range(0,len(output_data_location)):
            #if(output_data_item[i] == 0):
            cv2.rectangle(img,( int(output_data_location[i][0]) , int(output_data_location[i][1])),
                          ( int(output_data_location[i][2] ), int(output_data_location[i][3]) ),(0,255,0))
            #print(output_data_location[i])  
            points_for_rectangle.append([output_data_location[i][0],output_data_location[i][1],output_data_location[i][2],output_data_location[i][3]])
                
            points.append([[ int(output_data_location[i][0]+ ((-output_data_location[i][0] +  output_data_location[i][2])/2)),output_data_location[i][3] ]])
            cv2.circle(img,(int(points[-1][0][0]),int(points[-1][0][1])),5,(255,0,0))
            

                
        
        
        frame=cv2.resize(frame,(640,480))
        width=abs(refPt[0][0]-refPt[1][0])
        height=abs(refPt[0][1]-refPt[2][1])
        #print(points)
        result,transf_points,distance_btw_road_in_trans,result_to_display=perspective(width,height,frame,points,refPt)
        result=result*0
        
        
        aspect_ratio=width_of_reference_road/distance_btw_road_in_trans

        #min_distance=(distance_btw_road_in_trans/width_of_reference_road)*4
        try:
            min_distance=int(f.read())
        except:
            min_distance=0
        #print("MIN VALLLLLLLLLLLL ",min_distance)
        if(len(points)>0):
            for point in transf_points:
                #print(points)
                cv2.circle(result,(int(point[0][0]),int(point[0][1])),int((min_distance/aspect_ratio)//2),(0,255,0))
                cv2.circle(result,(int(point[0][0]),int(point[0][1])),2,(0,255,0))
    
    
    
                
            
            for i in range(0,transf_points.shape[0]-1):
                for j in range(i+1,transf_points.shape[0]):
    
    
                    
                    d = distance.euclidean(transf_points[i][0], transf_points[j][0])
                    d=d*aspect_ratio
                    #print("DDDDDDDDDD",d)
                    if(d<min_distance):
                        print(min_distance)
                        #cv2.line(result, (int(transf_points[i][0][0]), int(transf_points[i][0][1])), (int(transf_points[j][0][0]), int(transf_points[j][0][1])), (255,252,255), 1) 
                        #result = cv2.putText(result, str(d), (int(transf_points[i][0][0]), int(transf_points[i][0][1])), cv2.FONT_HERSHEY_SIMPLEX ,  
                         #                   1, (255,255,0), 1, cv2.LINE_AA) 
                        color=(0,0,255)
                        
                        cv2.line(img, (int(points[i][0][0]), int(points[i][0][1])), (int(points[j][0][0]), int(points[j][0][1])), color, 1) 
                        cv2.rectangle(img,( int(points_for_rectangle[i][0]) , int(points_for_rectangle[i][1])),
                                  ( int(points_for_rectangle[i][2] ), int(points_for_rectangle[i][3]) ),color)
                        cv2.rectangle(img,( int(points_for_rectangle[j][0]) , int(points_for_rectangle[j][1])),
                                  ( int(points_for_rectangle[j][2] ), int(points_for_rectangle[j][3]) ),color)
                        cv2.circle(result,(int(transf_points[i][0][0]),int(transf_points[i][0][1])),int((min_distance/aspect_ratio)//2),(0,0,255))
                        cv2.circle(result,(int(transf_points[j][0][0]),int(transf_points[j][0][1])),int((min_distance/aspect_ratio)//2),(0,0,255))
                        cv2.circle(result,(int(transf_points[i][0][0]),int(transf_points[i][0][1])),2,(0,0,255))
                        cv2.circle(result,(int(transf_points[j][0][0]),int(transf_points[j][0][1])),2,(0,0,255))
                        
                        img = cv2.putText(img, str(round(d, 2))+' ft', ((int(points[i][0][0])+int(points[j][0][0]))//2,(int(points[i][0][1])+int(points[j][0][1]))//2), cv2.FONT_HERSHEY_SIMPLEX , 0.5, (255,255,0), 1) 
                    else:
                        color=(0,255,0)
                        
        
            min_human_x=min([transf_points[i][0][0] for i in range(0,len(transf_points))])
            max_human_x=max([transf_points[i][0][0] for i in range(0,len(transf_points))])
            min_human_y=min([transf_points[i][0][1] for i in range(0,len(transf_points))])
            max_human_y=max([transf_points[i][0][1] for i in range(0,len(transf_points))])
            #print("Min-max ", min_human_x,max_human_x,min_human_y,max_human_y)
            
            #result=result[int(min_human_y)-min_distance//2:int(max_human_y)+min_distance//2,
             #             int(min_human_x)-min_distance//2:int(max_human_x)+min_distance//2]
            
        result=cv2.resize(result,(frame_width,frame_height))
        cv2.imshow("image", result)

        result_to_display=cv2.resize(result_to_display,(900,900))
        cv2.imshow("Display",result_to_display)

        #print(aspect_ratio)

        
        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter("output.avi", fourcc, 30,
                                     (img.shape[1], img.shape[0]), True)
    
    		
    
        writer.write(img)
        #cv2.namedWindow("image",cv2.CV_WINDOW_AUTOSIZE)
        
        img=cv2.resize(img,(frame_width,frame_height))
        cv2.imshow('frame',img)
        k=cv2.waitKey(1)&0xFF
        if k==27:
            break
    cv2.destroyAllWindows()
    cap.release()



cap=cv2.VideoCapture('input.avi')
#cap=cv2.VideoCapture(0)
width_of_reference_road=20

classes=[]





cv2.namedWindow("image")

cv2.setMouseCallback("image", click_and_crop)
_,first_snap=cap.read()
first_snap=cv2.resize(first_snap,(640,480))
while True:
	
    
    cv2.imshow("image", first_snap)
    key = cv2.waitKey(1) & 0xFF
    if(key==27):
        break
cv2.destroyAllWindows()
    
refPt=refPts
print(refPt)
# creating thread

 
t1 = threading.Thread(target=gui) 
t2 = threading.Thread(target=video) 
  
# starting thread 1 
t1.start() 
# starting thread 2 
t2.start()


# wait until thread 1 is completely executed 
t1.join() 
# wait until thread 2 is completely executed 
t2.join() 


