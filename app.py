from ultralytics import YOLO 
import easyocr
import cv2
from flask import Flask, render_template, request, redirect, url_for,jsonify
import os 
import time

app = Flask(__name__)

# Set upload folder
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER




@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Get the uploaded file
        uploaded_file = request.files['image']
        # Check if a file is selected
        if uploaded_file.filename != '':
            # Save the file
            filename = uploaded_file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], "input.jpg")
            uploaded_file.save(filepath)
            # Redirect to success page (you can modify this for further processing)
            return redirect('/result')
    return render_template('index.html')

@app.route('/result/')
def upload_success():
    # file name is the file name that some one uploaded 
    # so the image path will be ./static/uploads/<filename>
    # call your model with keras tf of sklearn
    try:
        model = YOLO('./static/model/yolov8-custom.pt')
        confidence = 0.6 # 0.0-1.0    # render the message that you wanted to show in message variable
        results = model.predict(source = f"./static/uploads/input.jpg", save = True, show = False, conf = confidence)
        print(results)

        img_main = cv2.imread("./static/uploads/input.jpg")
        r = results[0]
        box = r.boxes[0]
        [left, top, right, bottom] = box.xyxy[0]
        left = int(left)
        top = int(top)
        right = int(right)
        bottom = int(bottom)
        cropped_img = img_main[top+1:bottom-1, left+1:right-1]
        output_path = "./static/uploads/processed.jpg"
        cv2.imwrite(output_path, cropped_img)
        print(f"Largest image saved at {output_path}")
    # for all obj
        # for r in results:
        #     boxes = r.boxes
        #     for box in boxes:
        #         # Find Bounding Box
        #         [left, top, right, bottom] = box.xyxy[0]
        #         left = int(left)
        #         top = int(top)
        #         right = int(right)
        #         bottom = int(bottom)
        #         cropped_img = img_main[top+1:bottom-1, left+1:right-1]
        #         output_path = "./static/uploads/processed.jpg"
        #         cv2.imwrite(output_path, cropped_img)
        #         print(f"Largest image saved at {output_path}")
        #         break
        # reader = easyocr.Reader(['bn'], gpu = False)
        # result = reader.readtext("./static/uploads/processed.jpg", detail = 0, paragraph = True)
        # print(result)
        # return render_template('index.html', message=result[0])
        time.sleep(2)
        print("detection and cropping is successfull. ")
        return redirect('/final_result/')
    except Exception as e :
        print(e)
        return render_template('error.html')

@app.route('/final_result/')
def final_result():
    try:
        print("now easy ocr part.")
        reader = easyocr.Reader(['bn'], gpu = False)
        result = reader.readtext("./static/uploads/processed.jpg", detail = 0, paragraph = True)
        print(result)
        return render_template('index.html', message=result[0])
    except Exception as e :
        return render_template('error.html')

@app.route('/dmp_fix/')
def dmp_fix():
    try:
        print("Now easy ocr part.")
        reader = easyocr.Reader(['bn'], gpu = False)
        result = reader.readtext("./static/uploads/processed.jpg", detail = 0, paragraph = True)
        print(result)
        return jsonify(result)
    except Exception as e :
        return jsonify(e)

if __name__ == '__main__':
    app.run(debug=True)
