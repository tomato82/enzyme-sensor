import cv2
import time
import cv2
import numpy as np
import math
import csv
import pyautogui as pag
import tkinter.filedialog
import tkinter as tk
import os
import openpyxl
import sys
wname = 'gluDetect'
cv2.namedWindow(wname)
maxVal = 100
minVal = 200
epsilon = 0.03
font_size = 1
iDir = os.path.abspath(os.path.dirname(__file__))
max_sc = lambda image,x,y:x/image.shape[1] if (image.shape[1]/image.shape[0])>(x/y) else y/image.shape[0]
def maximum(image,x,y):
    f = max_sc(image,x,y)
    return cv2.resize(image,dsize = None,fx = f,fy = f)
def set_ui(img,ui_x,ui_y):
    button = np.full((img.shape[0]+ui_y,img.shape[1]+ui_x),'')
    show_img = np.pad(img,((0,ui_y),(ui_x,0),(0,0)))
    square_button_y = img.shape[0]//4-square_button.shape[0]//2
    rectangle_button_y = img.shape[0]//4*2-rectangle_button.shape[0]//2
    auto_button_y = img.shape[0]//4*3-auto_button.shape[0]//2
    square_button_x = ui_x//2-square_button.shape[1]//2
    rectangle_button_x = ui_x//2-rectangle_button.shape[1]//2
    auto_button_x = ui_x//2-auto_button.shape[1]//2
    show_img[square_button_y:square_button_y+square_button.shape[0],square_button_x:square_button_x+square_button.shape[1]] = square_button
    show_img[rectangle_button_y:rectangle_button_y+rectangle_button.shape[0],rectangle_button_x:rectangle_button_x+rectangle_button.shape[1]] = rectangle_button
    show_img[auto_button_y:auto_button_y+auto_button.shape[0],auto_button_x:auto_button_x+auto_button.shape[1]] = auto_button
    button[square_button_y:square_button_y+square_button.shape[0],square_button_x:square_button_x+square_button.shape[1]] = 'square'
    button[rectangle_button_y:rectangle_button_y+rectangle_button.shape[0],rectangle_button_x:rectangle_button_x+rectangle_button.shape[1]] = 'rectangle'
    button[auto_button_y:auto_button_y+auto_button.shape[0],auto_button_x:auto_button_x+auto_button.shape[1]] = 'auto'
    return show_img,button

def getObject(image,pos):
    debug = image.copy()
    src = pos.astype(np.float32)
    o_width = np.linalg.norm(pos[1] - pos[0])
    o_width = math.floor(o_width)
    o_height = np.linalg.norm(pos[3] - pos[0])
    o_height = math.floor(o_height)
    dst = np.float32([[0, 0],[o_width, 0],[o_width, o_height],[0, o_height]])
    M = cv2.getPerspectiveTransform(src, dst)
    output = cv2.warpPerspective(image, M,(o_width, o_height))
    return output

def FindContours(image,maxVal,minVal):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = (((100<hsv[:,:,0])&(hsv[:,:,0]<150))*255).astype(np.uint8)
    gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, np.ones((15, 15), np.uint8))
    binary = cv2.Canny(gray, maxVal, minVal)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def filterRectangle(cnt,epsilon,min_area):
    arclen = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon * arclen, True)
    approx = approx.reshape((-1,2))
    if len(approx)==4 and cv2.contourArea(cnt) > min_area and (not 1/3<(np.linalg.norm(approx[1] - approx[0])/np.linalg.norm(approx[3] - approx[0]))<3):
        o_width = np.linalg.norm(approx[1] - approx[0])
        o_width = math.floor(o_width)
        o_height = np.linalg.norm(approx[3] - approx[0])
        o_height = math.floor(o_height)
        if o_width>o_height:
            print(approx[1])
            approx[1] = approx[0]+(approx[1]-approx[0])*2
            approx[2] = approx[3]+(approx[2]-approx[3])*2
        if o_height>o_width:
            approx[2] = approx[1]+(approx[2]-approx[1])*2
            approx[3] = approx[0]+(approx[3]-approx[0])*2
        return True,approx
    else:
        return False,None
def detect_gln(object):
    object_hsv = cv2.cvtColor(object,cv2.COLOR_BGR2HSV)
    object_h = object_hsv[:,:,0]
    angle = [(np.rot90(object_h,n)[:object_h.shape[0]//4]).mean() for n in range(0,4)]
    #angle = [((105<(np.rot90(object_h,n)[:object_h.shape[0]//4]))&((np.rot90(object_h,n)[:object_h.shape[0]//4])<105)).sum() for n in range(0,4)]
    angle = np.absolute(np.array(angle)-103)
    angle = np.argmin(angle)
    object = np.rot90(object,angle)
    white = [object[object.shape[0]//4:-object.shape[0]//16,:,index].mean() for index in range(3)]
    glu = [object[-object.shape[0]//16:,:,index].mean() for index in range(3)]
    rgb = [v/v2 for v,v2 in zip(glu,white)]
    result = (max(rgb)-min(rgb))/max(rgb)*76.76
    result = round(result,2)
    return result
def showResult(image,paper,ui_x,ui_y):
    show_img,button = set_ui(image,ui_x,ui_y)
    for index,p in enumerate(paper):
        mp = maximum(p,(show_img.shape[1]-10*(len(paper)+1))//len(paper),(ui_y-30)//2)
        mpx = show_img.shape[1]//(len(paper)+1)*(index+1)-mp.shape[1]//2
        mpy = ui_y//2-5-mp.shape[0]
        show_img[mpy+image.shape[0]:mpy+image.shape[0]+mp.shape[0],mpx:mpx+mp.shape[1]] = mp
        button[mpy+image.shape[0]:mpy+image.shape[0]+mp.shape[0],mpx:mpx+mp.shape[1]] = str(index)
        res = detect_gln(p)
        res_x = mpx
        res_y = ui_y//2+15+image.shape[0]
        show_img = cv2.putText(show_img,
            text=str(res),
            org=(res_x, res_y),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=font_size,
            color=(255, 255, 255),
            thickness=2,
            lineType=cv2.LINE_4)
    return show_img,button


def onMouse(event,x,y,flag,param):
    global detect_mode,start_pos,show_img,img_r,img_e,pos,paper,button
    if event == cv2.EVENT_LBUTTONDOWN:
        if button[y,x]:
            detect_mode = button[y,x]
            if detect_mode == 'a':
                img_r = img_e.copy()
                if len(paper_auto) == 0:
                    for cnt in contours:
                        img_r = cv2.polylines(img_r,[cnt],True,(255,0,0),5)
       #                 _,rec = filterRectangle(cnt,epsilon,0)
                        paper_auto.append(getObject(image,cnt))
                show_img,button = showResult(img_r,paper+paper_auto,ui_x,ui_y)
        elif y <= image.shape[0] and x > ui_x:
            if detect_mode == 's':
                start_pos = [x-ui_x,y]
            elif detect_mode == 'r':
                pos.append([x,y])
                if len(pos) == 4:
                    img_e = img_r
                    object_pos = np.array(pos)
                    object_pos[:,0] -= ui_x
                    paper.append(getObject(image,object_pos))
                    pos = []
                    show_img,button = showResult(img_e,paper,ui_x,ui_y)
    if event == cv2.EVENT_RBUTTONDOWN:
        if str.isalnum(button[y,x]):
            del paper[int(button[y,x])]
            show_img,button = showResult(img_e,paper,ui_x,ui_y)
    elif event == cv2.EVENT_MOUSEMOVE:
        if detect_mode == 'r':
            img_r = img_e.copy()
            pts = np.array(pos+[[x,y]])
            pts[:,0] = pts[:,0] - ui_x
            img_r = cv2.polylines(img_r,[pts],True,(0,0,255))
            show_img,button = showResult(img_r,paper,ui_x,ui_y)
        elif (detect_mode == 's')and (flag == cv2.EVENT_FLAG_LBUTTON) and not start_pos is None:
            img_r = img_e.copy()
            img_r = cv2.rectangle(img_r, start_pos, [max(x-ui_x,0),min(y,image.shape[0])], (255,0,0))
            show_img,button = showResult(img_r,paper,ui_x,ui_y)

    elif (event == cv2.EVENT_LBUTTONUP) and (detect_mode == 's') and not start_pos is None:
        img_e = img_r
        y_pos = [start_pos[1],min(y,image.shape[0])]
        x_pos = [start_pos[0],max(x-ui_x,0)]
        object = image[min(y_pos):max(y_pos),min(x_pos):max(x_pos)]
        paper.append(object)
        show_img,button = showResult(img_e,paper,ui_x,ui_y)
        start_pos = None

def onMouse_photo(event,x,y,flag,param):
    global button_press,photo,i,show_frame
    if photo and event == cv2.EVENT_LBUTTONDOWN and button_point[y,x] == 1:
        button_press = True
def onMouse_select(event,x,y,flag,param):
    global mode,start_time
    if not mode:
        if event == cv2.EVENT_LBUTTONDOWN:
            mode = button_point[y,x]
        elif event == cv2.EVENT_MOUSEMOVE:
            if not button_point[y,x]:
                start_time = None
            else:
                if start_time is None:
                    start_time = time.time()
                else:
                    show_frame[button_point == button_point[y,x]] = base_image[button_point == button_point[y,x]]*(np.sin(time.time()-start_time)+1)/2
button_press = False
mode = ''
file = cv2.imread('data/image/file.png')
cam = cv2.imread('data/image/cam.png')
ui_x,ui_y = 600,400
file = maximum(file,ui_x//2,ui_y)
cam = maximum(cam,ui_x//2,ui_y)
base_image = np.zeros((ui_y,ui_x,3))
button_point = np.full((ui_y,ui_x),'')
file_y = (ui_y-file.shape[0])//2
cam_y = (ui_y-cam.shape[0])//2
file_x = ui_x//4-file.shape[1]//2
cam_x = ui_x//4*3-cam.shape[1]//2
base_image[file_y:file_y+file.shape[0],file_x:file_x+file.shape[1]] = file
button_point[file_y:file_y+file.shape[0],file_x:file_x+file.shape[1]] = 'f'
base_image[cam_y:cam_y+cam.shape[0],cam_x:cam_x+cam.shape[1]] = cam
button_point[cam_y:cam_y+cam.shape[0],cam_x:cam_x+cam.shape[1]] = 'c'
show_frame = base_image.copy()
cv2.setMouseCallback(wname,onMouse_select)
start_time = None
screen_x,screen_y = pag.size()
ui_x = 50
ui_y = 300
while not mode:
    cv2.imshow(wname,show_frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('f'):
        mode = 'f'
        break
    elif key == ord('c'):
        mode = 'c'
        break
    elif key == ord('q'):
        sys.exit()
if mode == 'c':
    i = 0
    camera = cv2.VideoCapture(0)
    ret,frame = camera.read()
    photo = True
    bar_y = 50
    button = cv2.imread('data/image/detect.png')
    button = cv2.resize(button,(button.shape[1]*ui_y//button.shape[0],ui_y))
    button_press = False
    cv2.setMouseCallback(wname,onMouse_photo)
    while not button_press:
        ret,frame = camera.read()
        frame = maximum(frame,screen_x-ui_x,screen_y-ui_y)
        #frame = maximum(frame,1920,700)
        cv2.imwrite('frame.jpg',frame)
        contours = FindContours(frame,maxVal,minVal)
        f = lambda cnt: filterRectangle(cnt,epsilon,frame.shape[0]*frame.shape[1]*0.001)[1]
        contours = list(map(f,contours))
        contours = np.array(list(filter(lambda x:not x is None,contours)))
        show_frame = frame.copy()
        show_frame = cv2.drawContours(show_frame,contours,-1,(255,0,0),5)
        show_frame = np.pad(show_frame,((0,ui_y),(0,0),(0,0)))
        button_x = (frame.shape[1]-button.shape[1])//2
        button_y = frame.shape[0]+ui_y//2-button.shape[0]//2
        change_x = ()
        show_frame[button_y:button_y+button.shape[0],button_x:button_x+button.shape[1]] = button
        #show_frame[]
        button_point = np.zeros((show_frame.shape[0],show_frame.shape[1]))
        button_point[button_y:button_y+button.shape[0],button_x:button_x+button.shape[1]] = 1
        cv2.imshow(wname, show_frame)
    
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    image = frame
    photo = False
    camera.release()
else:
    imageFile = [".png",".jpg",".tiff",".bmp",".dib",".pbm",".pgm",".ppm",".pnm",".sr",".ras",".tiff",".tif",".jp2",".jpeg",".jpe"]
    fTyp = [("Image files","*"+(";*".join(imageFile)))]
    root = tk.Tk()
    root.withdraw()
    root.title('hide')
    selected_file_path = tkinter.filedialog.askopenfilename(filetypes = fTyp,initialdir = iDir)
    image = cv2.imread(selected_file_path)
    image = maximum(image,screen_x-ui_x,screen_y-ui_y)
    contours = FindContours(image,maxVal,minVal)
    f = lambda cnt: filterRectangle(cnt,epsilon,image.shape[0]*image.shape[1]*0.001)[1]
    contours = list(map(f,contours))
    contours = np.array(list(filter(lambda x:not x is None,contours)))


log = []
maximum_x = lambda image,ui_x:cv2.resize(image,(ui_x,image.shape[0]*ui_x//image.shape[1]))
pos = []
paper_auto = []
paper = []
detect = True
show_img = image.copy()
drug = False
detect_mode = 'a'
scale = max_sc(image,screen_x-ui_x,screen_y-ui_y)
#contours = list(map(lambda x:print(x),contours))
start_pos = None
img_e = image.copy()
square_button =  maximum_x(cv2.imread('data/image/square.png'),ui_x)
rectangle_button =  maximum_x(cv2.imread('data/image/rectangle.png'),ui_x)
auto_button =  maximum_x(cv2.imread('data/image/auto.png'),ui_x)
show_img,button = set_ui(image,ui_x,ui_y)
print((button == '').sum())
cv2.setMouseCallback(wname,onMouse)
while True:
    cv2.imshow(wname,show_img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        del pos[-1]
cv2.destroyAllWindows()
outputType = [("",".xlsx")]
output_file = ""
while not output_file:
    output_file = tkinter.filedialog.asksaveasfilename(filetypes = outputType,initialdir = iDir)
if ".xlsx" != output_file[-5:]:
    output_file += ".xlsx"
print(output_file)
f = openpyxl.Workbook()
sheet = f['Sheet']
for i,x in enumerate(list(map(detect_gln,paper))):
    sheet.cell(row=1,
               column=i+1,
               value=x)
f.save(output_file)