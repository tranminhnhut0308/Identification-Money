import os
import uuid
import cv2
import numpy as np
import urllib
from gtts import gTTS
from tensorflow import keras
from flask import Flask , render_template   , request , send_file,Response
from playsound import playsound 

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = keras.models.load_model(os.path.join(BASE_DIR , 'vgg_model.h5'))


ALLOWED_EXT = set(['jpg' , 'jpeg' , 'png' , 'jfif'])
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXT
 

def my_predict(filename , model):
    classes =["1 trăm nghìn ","10 nghìn ","1 nghìn ","2 trăm nghìn","2 mươi nghìn","2 nghìn","5 trăm nghìn","5 mươi nghìn ","5 nghìn"]
    P=[]
    anh =cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
    P.append(cv2.resize(anh,(128,128)))
    image_array = np.array(P)
    prediction = model.predict(image_array)
    output = gTTS(classes[np.argmax(prediction)],lang="vi", slow=False)
    output.save("static/mp3/output.mp3")
    return classes[np.argmax(prediction)]

@app.route('/')
def home():
        return render_template("index.html")

@app.route('/success' , methods = ['GET' , 'POST'])
def success():
    error = ''
    target_img = os.path.join(os.getcwd() , 'static/images')
    if request.method == 'POST':
        if (request.files):
            file = request.files['file']
            if file and allowed_file(file.filename):
                file.save(os.path.join(target_img , file.filename))
                img_path = os.path.join(target_img , file.filename)
                img = file.filename

                kq = my_predict(img_path , model)
            else:
                error = "Please upload images of jpg , jpeg and png extension only"

            if(len(error) == 0):
                return  render_template('success.html' , img  = img , predictions = kq)
            else:
                return render_template('index.html' , error = error)

        elif request.form.get("link"):
            link = request.form.get('link')
            try :
                resource = urllib.request.urlopen(link)
                unique_filename = str(uuid.uuid4())
                filename = unique_filename+".jpg"
                img_path = os.path.join(target_img , filename)
                output = open(img_path , "wb")
                output.write(resource.read())
                output.close()
                img = filename

                kq1 = my_predict(img_path , model)

            except Exception as e : 
                print(str(e))
                error = 'This image from this site is not accesible or inappropriate input'

            if(len(error) == 0):
                return  render_template('success.html' , img  = img , predictions = kq1)
            else:
                return render_template('index.html' , error = error) 
            
    else:
        return render_template('index.html')

def generate_frames():
    class_name =["100.000 ","10.000 ","1.000 ","200.000","20.000","2.000","500.000","50.000","5.000"]
    cap = cv2.VideoCapture(0)

    while(True):
        # Capture frame-by-frame
    
        ret, image_org = cap.read()

        if not ret:
            break
        else:
            image_org = cv2.resize(image_org, dsize=None,fx=0.8,fy=0.8)
            # Resize
            image = image_org.copy()
            image = cv2.resize(image, dsize=(128, 128))
            image = np.array(image)
            # image = image.astype('float')*1./255
            # Convert to tensor
            image = np.expand_dims(image, axis=0)

            # Predict
            predict = model.predict(image)
            print("This picture is: ", class_name[np.argmax(predict[0])], (predict[0]))
            print(np.max(predict[0],axis=0))
            
            if (np.max(predict)>=0.8):
                # Show image
                font = cv2.FONT_HERSHEY_SIMPLEX
                org = (50, 50)
                fontScale = 1.5
                color = (0, 255, 0)
                thickness = 2

                cv2.putText(image_org, class_name[np.argmax(predict)], org, font,
                            fontScale, color, thickness, cv2.LINE_AA)
                # text = class_name[np.argmax(predict)]
                # var = gTTS(text = text,lang = "vi") 
                # var.save("static/mp3/vi.mp3") 
                # playsound("static/mp3/vi.mp3")

                # os.remove("static/mp3/vi.mp3")
                
            #     ret,buffer=cv2.imencode('.jpg',image_org)
            #     frame=buffer.tobytes()

            # yield(b'--frame\r\n'
            #         b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            cv2.imshow("Picture", image_org)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows() 
    return render_template("index.html")

    # When everything done, release the capture
    
# @app.route("/camera")
# def camera():
#     return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)


