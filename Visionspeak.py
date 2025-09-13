
import cv2
from transformers import pipeline

# Set up image captioning and translation tools
describe_scene = pipeline("image-to-text", model="LiquidAI/LFM2-VL-1.6B")
translate_text = pipeline("translation", model="facebook/nllb-200-distilled-600M", src_lang="eng", tgt_lang="fra")

# Start video stream
camera = cv2.VideoCapture(0)

while True:
    ret, frame = camera.read()
    if not ret:
        break

    # Save current frame
    cv2.imwrite("current.jpg", frame)

    # Get description of the scene
    caption = describe_scene("current.jpg")[0]['generated_text']

    # Translate to French
    french_caption = translate_text(caption)[0]['translation_text']

    # Display translated caption on screen
    cv2.putText(frame, french_caption, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow("Live View", frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
