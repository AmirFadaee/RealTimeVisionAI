# %%
from ultralytics import YOLO
import cv2

# %%
model = YOLO('yolov8n.pt')

# %%
cam = cv2.VideoCapture(0)

while True:
    ret , frame = cam.read()
    frame = cv2.flip(frame, 1)
    result = model(frame,verbose= False)

    annotated = result[0].plot()
    cv2.imshow('YOLO Object Detection', annotated)

    if cv2.waitKey(1) == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()

# %%
from transformers import BlipProcessor , BlipForConditionalGeneration
import torch , time

# %%
processor = BlipProcessor.from_pretrained('Salesforce/blip-image-captioning-base')
model_blip = BlipForConditionalGeneration.from_pretrained('Salesforce/blip-image-captioning-base')

# %%
cam = cv2.VideoCapture(0)

last_caption = ''
last_time = 0

while True:
    ret, frame = cam.read()
    frame = cv2.flip(frame, 1)
    result = model(frame, verbose= False)

    annotated = result[0].plot()

    if time.time() - last_time > 3 :
        inputs  = processor(images = frame,return_tensors ='pt')
        with torch.no_grad():
            out = model_blip.generate(**inputs,max_new_tokens = 30)

        last_caption = processor.decode(out[0],skip_special_tokens = True)

        last_time = time.time()

    cv2.putText(annotated, last_caption,(10,30) , cv2.FONT_HERSHEY_COMPLEX,0.6,(0,0,255))
    cv2.imshow('Image Captioning Project',annotated)

    if cv2.waitKey(1) == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()


# %%



