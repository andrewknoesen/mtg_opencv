import requests
import cv2
import numpy as np
import pytesseract

# initialize the webcam
cap = cv2.VideoCapture(0)

# load an image of a Magic: The Gathering card as a template
template = cv2.imread("template.jpg", cv2.IMREAD_GRAYSCALE)
w, h = template.shape[::-1]

# configure OCR settings
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # replace with the path to your Tesseract executable
config = ("--oem 1 --psm 7")  # set OCR engine mode and page segmentation mode

while True:
    # capture a frame from the webcam
    ret, frame = cap.read()
    
    # convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # use template matching to detect the card in the frame
    res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.8  # adjust this value to fine-tune the detection
    loc = np.where(res >= threshold)
    for pt in zip(*loc[::-1]):
        # draw a bounding box around the detected card
        cv2.rectangle(frame, pt, (pt[0] + w, pt[1] + h), (0, 255, 0), 2)
        
        # crop the detected card from the frame
        card = frame[pt[1]:pt[1]+h, pt[0]:pt[0]+w]
        
        # use OCR to extract the card name from the cropped image
        card_name = pytesseract.image_to_string(card, config=config)
        
        # use the Scryfall API to get information about the card
        api_url = f"https://api.scryfall.com/cards/named?exact={card_name}"
        api_response = requests.get(api_url)
        card_info = api_response.json()
        
        # print the card information to the console
        print(card_info)
    
    # display the frame with the detected card
    cv2.imshow("Webcam", frame)
    
    # press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release the webcam
cap.release()
cv2.destroyAllWindows()
