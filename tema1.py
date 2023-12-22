import numpy as np
import cv2 as cv
import os

def show_image(title,image):
    image=cv.resize(image,(0,0),fx=0.4,fy=0.4)
    image = cv.resize(image, (600, 600))

    cv.imshow(title,image)
    cv.waitKey(0)
    cv.destroyAllWindows()

def normalize_brightness_color(image, target_brightness):
    # Split the image into color channels
    b, g, r = cv.split(image)

    # Calculate the average brightness level of each color channel
    average_brightness_b = np.mean(b)
    average_brightness_g = np.mean(g)
    average_brightness_r = np.mean(r)

    # Normalize each color channel brightness to the target level
    normalized_b = cv.convertScaleAbs(b, alpha=target_brightness/average_brightness_b, beta=0)
    normalized_g = cv.convertScaleAbs(g, alpha=target_brightness/average_brightness_g, beta=0)
    normalized_r = cv.convertScaleAbs(r, alpha=target_brightness/average_brightness_r, beta=0)

    # Merge the normalized color channels back into an RGB image
    normalized_image = cv.merge([normalized_b, normalized_g, normalized_r])

    return normalized_image

def houghCircles(image):
    gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    image_g_blur = cv.GaussianBlur(gray, (0, 0), 2) 
    equ = cv.equalizeHist(image_g_blur)
    #show_image('equ', equ)
    
    circles = cv.HoughCircles(
        equ,
        cv.HOUGH_GRADIENT,
        dp=1.1,
        minDist=10,
        param1=55,
        param2=31,
        minRadius=10,
        maxRadius=13
    )
    gray = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)

    # Check if circles were found
    if circles is not None:
        circles = np.uint16(np.around(circles))
        
        # Draw the circles on the original image
        for i in circles[0, :]:
            cv.circle(gray, (i[0], i[1]), i[2], (0, 255, 0), 2)  # Green circles

    #show_image('cercuri', gray) 
    return circles  

def houghCircles2(image):
    gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    equ = cv.equalizeHist(gray)
    blurred = cv.GaussianBlur(equ, (5, 5), 3)
    #show_image('cercuri_blur', blurred) 
    
    circles = cv.HoughCircles(
        blurred,
        cv.HOUGH_GRADIENT,
        dp=1.2,
        minDist=20,
        param1=55,
        param2=30,
        minRadius=10,
        maxRadius=20
    )
    gray = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
    
    # Check if circles were found
    if circles is not None:
        circles = np.uint16(np.around(circles))
        
        # Draw the circles on the original image
        for i in circles[0, :]:
            cv.circle(gray, (i[0], i[1]), i[2], (0, 255, 0), 2)  # Green circles
        #show_image('cercuri', gray) 
    else:
        return '0'
    
    return len(circles[0])



def gaseste_tabla2(image):

    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    copie = image.copy()

    image_m_blur = cv.medianBlur(image,3)
    image_g_blur = cv.GaussianBlur(image_m_blur, (0, 0), 13) 
    image_sharpened = cv.addWeighted(image_m_blur, 4, image_g_blur, -5, 0)
    image_copy = image_sharpened
    #show_image('second_sharpedned', image_copy)
    _, thresh2 = cv.threshold(image_copy, 5, 255, cv.THRESH_BINARY)
    #show_image('threshold2', thresh2)

    kernel = np.ones((3, 3), np.uint8)  
    thresh2 = cv.dilate(thresh2, kernel, iterations=4) 
    #show_image('dilate',thresh2)

    kernel = np.ones((2, 2), np.uint8)
    thresh2 = cv.erode(thresh2, kernel, iterations=13)
    #show_image('erode', thresh2)

    kernel = np.ones((5, 5), np.uint8)  
    thresh2 = cv.dilate(thresh2, kernel, iterations=25) 
    #show_image('dilate',thresh2)

    edges =  cv.Canny(thresh2 ,200,500)
    #show_image('edges',edges)
    contours, _ = cv.findContours(edges,  cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    max_area = 0
   
    for i in range(len(contours)):
        if(len(contours[i]) >3):
            possible_top_left = None
            possible_bottom_right = None
            for point in contours[i].squeeze():
                if possible_top_left is None or point[0] + point[1] < possible_top_left[0] + possible_top_left[1]:
                    possible_top_left = point

                if possible_bottom_right is None or point[0] + point[1] > possible_bottom_right[0] + possible_bottom_right[1] :
                    possible_bottom_right = point

            diff = np.diff(contours[i].squeeze(), axis = 1)
            possible_top_right = contours[i].squeeze()[np.argmin(diff)]
            possible_bottom_left = contours[i].squeeze()[np.argmax(diff)]
            if cv.contourArea(np.array([[possible_top_left],[possible_top_right],[possible_bottom_right],[possible_bottom_left]])) > max_area:
                max_area = cv.contourArea(np.array([[possible_top_left],[possible_top_right],[possible_bottom_right],[possible_bottom_left]]))
                top_left = possible_top_left
                bottom_right = possible_bottom_right
                top_right = possible_top_right
                bottom_left = possible_bottom_left

    width = 2000
    height = 2000

    puzzle = np.array([top_left,top_right,bottom_right,bottom_left], dtype = "float32")
    destination_of_puzzle = np.array([[0,0],[width,0],[width,height],[0,height]], dtype = "float32")

    M = cv.getPerspectiveTransform(puzzle,destination_of_puzzle)

    result = cv.warpPerspective(image, M, (width, height))
    result = cv.cvtColor(result,cv.COLOR_GRAY2BGR)
    #show_image('result', result)

    x = 330
    y = 330

    # Crop the image
    #show_image('crop', cropped_image)

    cropped_image = result[y:height-y+10, x+10:width-x-20]
    cropped_image=cv.resize(cropped_image, (1500, 1500))

    return cropped_image

    # houghCircles(thresh)

offset = 20

def gaseste_pozitii(thresh, circles, matrix):
    #show_image('determina_careu', thresh)
    positions = []
    lines = []

    for circle in circles[0]:
        #print(circle)
        line = (circle[0]//100*100, circle[1]//100*100)
        if line not in lines:
            lines.append(line)
    # print(lines)
    for line in lines:
            j = line[0]//100
            i = line[1]//100
            if matrix[i][j] == 'o':
                matrix[i][j] = 'x'
                positions.append((i,j))
    #print(positions)
    if len(positions) == 1:
        i = positions[0][0]
        j = positions[0][1]
        coord = [[1,0], [0,1], [0, -1], [-1, 0]]
        for xy in coord:
            #print(i+xy[0], j+xy[1])
            if i+xy[0] >=0 and i+xy[0] <= 14 and j+xy[1] >= 0 and j+xy[1] <= 14:
                if matrix[i+xy[0]][j+xy[1]] == 'o':
                    y_min = (i+xy[0])*100
                    y_max = (i+xy[0])*100 + 100 - offset
                    x_min = (j+xy[1])*100 + offset
                    x_max = (j+xy[1])*100 + 100 - offset
                    patch = thresh[y_min:y_max, x_min:x_max].copy()
                    
                    medie_patch = np.mean(patch)
                    if medie_patch < 30 :
                        #show_image('patch_gasit', patch)
                        matrix[i+xy[0]][j+xy[1]] = 'x'
                        positions.append((i+xy[0],j+xy[1]))
    else:
        if len(positions) == 0:
            #show_image('thresh_00', thresh)
            found = 0
            for i in range(15):
                for j in range(15):
                    if matrix[i][j] == 'o':
                        #print(i,j)
                        y_min = (i)*100
                        y_max = (i)*100 + 100 - offset
                        x_min = (j)*100 + offset
                        x_max = (j)*100 + 100 - offset
                        patch = thresh[y_min:y_max, x_min:x_max].copy()
                        
                        medie_patch = np.mean(patch)
                        if medie_patch < 30 :
                            #show_image('patch_gasit', patch)
                            matrix[i][j] = 'x'
                            positions.append((i,j))
                            found = 1
                            break
                    if found == 1:
                        break
            if found == 1:
                #print(positions)
                i = positions[0][0]
                j = positions[0][1]
                coord = [[1,0], [0,1], [0, -1], [-1, 0]]
                for xy in coord:
                    #print(i+xy[0], j+xy[1])
                    if i+xy[0] >=0 and i+xy[0] <= 14 and j+xy[1] >= 0 and j+xy[1] <= 14:
                        if matrix[i+xy[0]][j+xy[1]] == 'o':
                            y_min = (i+xy[0])*100
                            y_max = (i+xy[0])*100 + 100 - offset
                            x_min = (j+xy[1])*100 + offset
                            x_max = (j+xy[1])*100 + 100 - offset
                            patch = thresh[y_min:y_max, x_min:x_max].copy()
                            #show_image('patch', patch)
                            medie_patch = np.mean(patch)
                            if medie_patch < 30 :
                                
                                matrix[i+xy[0]][j+xy[1]] = 'x'
                                positions.append((i+xy[0],j+xy[1]))

    return matrix, positions

def determina_piese(img,thresh, positions, lines_horizontal,lines_vertical):
    nr=''
    for pos in positions:
        i=pos[0]
        j=pos[1]
        # print(i,j)
        if lines_vertical[j][0][0]-5 < 0:
            y_min = lines_vertical[j][0][0]
        else:
            y_min = lines_vertical[j][0][0]-5
        y_max = lines_vertical[j+1][1][0]+10

        if lines_horizontal[i][0][1]-5 < 0:
            x_min = lines_horizontal[i][0][1]
        else:
            x_min = lines_horizontal[i][0][1] - 5
        x_max = lines_horizontal[i+1][1][1]+10

        patch = thresh[x_min:x_max, y_min:y_max].copy()
        patch_orig=img[x_min:x_max, y_min:y_max].copy()
        patch_orig= cv.cvtColor(patch_orig,cv.COLOR_BGR2GRAY)

        #Apllying filters to the patch
        image_m_blur = cv.medianBlur(patch_orig,3)
        image_g_blur = cv.GaussianBlur(image_m_blur, (0, 0), 0.8) 

        nr_circles = houghCircles2(patch)
        patch_orig = image_g_blur
        nr += str(nr_circles)
    return nr


lines_horizontal=[]
for i in range(1,1501,99):
    l=[]
    l.append((0,i))
    l.append((1500 ,i))
    lines_horizontal.append(l)
    
lines_vertical=[]
for i in range(0,1501,100):
    l=[]
    l.append((i,0))
    l.append((i,2000))
    lines_vertical.append(l)

# Get the current working directory
current_directory = os.path.dirname(os.path.abspath(__file__))

# Construct a file path relative to the current directory
path1 = os.path.join(current_directory, 'input')


xaxis=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O']

points1 = [(4,4), (5,5), (9,9), (10,10), (6,4), (4,6), (4,8), (4,10), (5,9), (6,10),
           (8,4), (9,5), (10,6), (10,4), (10,8), (8,10) ]
points2 = [(2,4), (4,2), (3,5), (5,3), (2,10), (10,2), (3,9), (9,3), (4,12), (12,4), 
         (11,5), (5,11), (9,11), (11,9), (12,10), (10,12)]
points4 = [(0,3),(0,11),(1,5),(1,9),(3,0),(3,14),(5,1),(5,13),(9,1),(9,13),(11,0),
          (11,14),(13,5),(13,9),(14,3),(14,11)]
points3 = [(0,7),(1,2),(1,12),(2,1),(2,13),(3,3),(3,11),(7,0),(7,14),(11,3),(11,11),
          (12,1),(12,13), (13,2),(13,12),(14,7)]
points5 = [(0,0), (0,14), (14,0), (14,14)]

pozitii_ok = 0
piese_ok = 0
puncte_ok = 0

drum = [-1,1,2,3,4,5,6,0,2,5,3,4,6,2,2,0,3,5,4,1,6,2,4,5,
        5,0,6,3,4,2,0,1,5,1,3,4,4,4,5,0,6,3,5,4,1,3,2,0,
        0,1,1,2,3,6,3,5,2,1,0,6,6,5,2,1,2,5,0,3,3,5,0,6,
        1,4,0,6,3,5,1,4,2,6,2,3,1,6,5,6,2,0,4,0,1,6,4,4,
        1,6,6,3,0]

target_brightness = 70

for i in range(1,6):
    player1 = 0
    player2 = 0
    with open(os.path.join(path1, (str(i) + '_' + 'mutari.txt'))) as file:
        move_lines = file.readlines()
    matrix = np.full((15, 15), 'o', dtype=str)
    matrix_nr = np.full((15,15), 0, dtype=int)
    for j in range(1,21):
        if j < 10:
            nr_img = '0' + str(j)
        else:
            nr_img =  str(j)
        
        path = os.path.join(path1, (str(i) + '_' + nr_img + '.jpg'))
        image = cv.imread(path)

        image = normalize_brightness_color(image, target_brightness)
        #show_image('normalize', image)

        result = gaseste_tabla2(image)

        # copie = result.copy()
        # for line in  lines_vertical : 
        #     cv.line(copie, line[0], line[1], (0, 255, 0), 5)
        # for line in  lines_horizontal : 
        #     cv.line(copie, line[0], line[1], (0, 0, 255), 5)
        #show_image('img',copie)
        
        circles = houghCircles(result)

        image_m_blur = cv.medianBlur(result,3)
        #show_image('blur1', image_m_blur)
        image_g_blur = cv.GaussianBlur(image_m_blur, (0, 0), 5) 
        #show_image('blur2', image_g_blur)
        image_sharpened = cv.addWeighted(image_m_blur, 1.2, image_g_blur, -0.8, 0)
        #show_image('sharpened', image_sharpened)
      
        _, thresh = cv.threshold(image_sharpened, 55, 255, cv.THRESH_BINARY_INV)
        #show_image('thresh', thresh)
        matrix, positions = gaseste_pozitii(thresh, circles, matrix)
        
        piesa = determina_piese(result, thresh, positions, lines_horizontal,lines_vertical)
        piesa_nr = int(piesa)
        player_curent = move_lines[j-1].strip().split()[1]
        
        puncte = 0
        # print(player1, player2)
        # print(player_curent)
        for pos in positions:
            if pos in points1:
                puncte += 1
            elif pos in points2:
                puncte += 2
            elif pos in points3:
                puncte += 3
            elif pos in points4:
                puncte += 4
            elif pos in points5:
                puncte += 5
        
        if piesa_nr // 10 == piesa_nr % 10:
            puncte *= 2

        if piesa_nr // 10 == drum[player1] or piesa_nr % 10 == drum[player1]:
            if player_curent == 'player1':
                puncte += 3
            else:
                player1 = player1 + 3
        
        if piesa_nr // 10 == drum[player2] or piesa_nr % 10 == drum[player2]:
            if player_curent == 'player2':
                puncte += 3
            else:
                player2 = player2 + 3

        # print(puncte)

        if player_curent == 'player1':
            player1 += puncte
        else:
            player2 += puncte


        with open(os.path.join(path1, (str(i) + '_' + nr_img + '.txt'))) as file:
            lines = file.readlines()
        
        mutare = {}
        k=0
        for pos in positions:
            pozitie = str(pos[0]+1) + xaxis[pos[1]]
            mutare[pozitie] = piesa[k:k+1]
            k+=1

        path2 = os.path.join(current_directory, 'results')
        if not os.path.exists(path2):
            os.makedirs(path2)

        with open(os.path.join(path2, (str(i) + '_' + nr_img + '.txt')), 'w') as file:
            for move in mutare.keys():
                file.write(move + ' ' + mutare[move] + '\n')
            file.write(str(puncte) + '\n')

        #print(mutare)

        nr_ok = 1
        poz_ok=1
        for k in range(2):
            line = lines[k].strip().split()
            pozitie = line[0]
            if pozitie not in mutare.keys():
                poz_ok=0
            else:
                if mutare[pozitie] != line[1]:
                    nr_ok = 0
        if poz_ok == 1:
            print('Pozitie ok')
            pozitii_ok += 1
        else:
            #print(positions)
            print(positions)
        if nr_ok == 1:
            print('Numarul piesei ok')
            piese_ok += 1
        if puncte == int(lines[2].strip().split()[0]):
            print('Puncte ok')
            puncte_ok += 1

print("Pozitii corecte: {} din 100".format(pozitii_ok))
print("Piese corecte: {} din 100".format(piese_ok))
print("Puncte corecte: {} din 100".format(puncte_ok))

nota = pozitii_ok * 0.05 + piese_ok * 0.02 + puncte_ok * 0.02 + 2

print(nota)
