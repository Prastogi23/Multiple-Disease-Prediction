from flask import Flask , render_template

from flask import request, redirect,Response , url_for
import pickle
import numpy as np
from flask_sqlalchemy import SQLAlchemy
import tensorflow as tf
import os
from tensorflow.keras.models import load_model
import cv2
from PIL import Image
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI']= "sqlite:///db.sqlite3"
app.config['UPLOAD_FOLDER'] = './static/imagedata/eye'
app.config['UPLOAD_FOLDER_2'] = './static/imagedata/pneumonia'
pneumonia_model=load_model('models/model_pneumonia.h5')
eye_model=load_model('models/model_eye.h5')
eye_model.summary()
pneumonia_model.summary()
db = SQLAlchemy()
db.init_app(app)
app.app_context().push()
with app.app_context():
    db.create_all()

ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
class Heart(db.Model):
    __tablename__= 'heart'
    id = db.Column(db.Integer, autoincrement=True, primary_key=True,nullable = False)
    name = db.Column(db.String)
    age = db.Column(db.Float)
    sex = db.Column(db.Float)
    cp = db.Column(db.Float)
    trestbps = db.Column(db.Float)
    chol = db.Column(db.Float)
    fbs = db.Column(db.Float)
    restecg = db.Column(db.Float)
    thalach = db.Column(db.Float)
    exang = db.Column(db.Float)
    oldpeak = db.Column(db.Float)
    slope = db.Column(db.Float)
    ca = db.Column(db.Float)
    thal = db.Column(db.Float)
    target = db.Column(db.Float)


class Liver(db.Model):
    __tablename__= 'liver'
    id = db.Column(db.Integer, autoincrement=True, primary_key=True,nullable = False)
    name = db.Column(db.String)
    age = db.Column(db.Float)
    gender = db.Column(db.Float)
    totalb = db.Column(db.Float)
    directb = db.Column(db.Float)
    alkalinep = db.Column(db.Float)
    alaminea = db.Column(db.Float)
    aspertatea = db.Column(db.Float)
    totalp = db.Column(db.Float)
    albumin = db.Column(db.Float)
    albumingr = db.Column(db.Float)
    target = db.Column(db.Float)


class Kidney(db.Model):
    __tablename__= 'kidney'
    id = db.Column(db.Integer, autoincrement=True, primary_key=True,nullable = False)
    name = db.Column(db.String)
    gender = db.Column(db.String)
    age = db.Column(db.Float)
    blood_pressure = db.Column(db.Float)
    specific_gravity = db.Column(db.Float)
    albumin = db.Column(db.Float)
    sugar = db.Column(db.Float)
    red_blood_cells = db.Column(db.Float)
    pus_cell = db.Column(db.Float)
    pus_cell_clumps = db.Column(db.Float)
    bacteria = db.Column(db.Float)
    blood_glucose = db.Column(db.Float)
    urea = db.Column(db.Float)
    creatinine = db.Column(db.Float)
    sodium = db.Column(db.Float)
    potassium = db.Column(db.Float)
    haemoglobin = db.Column(db.Float)
    packet_cell_vol = db.Column(db.Float)
    white_blood_cells_count = db.Column(db.Float)
    red_blood_cells_count = db.Column(db.Float)
    hypertension = db.Column(db.Float)
    diabetes_mellitus = db.Column(db.Float)
    coronary_artery_disease = db.Column(db.Float)
    apetite = db.Column(db.Float)
    pedia_edema = db.Column(db.Float)
    anaemia = db.Column(db.Float)
    target = db.Column(db.Float)
    
class Eye(db.Model):
    __tablename__= 'eye'
    id = db.Column(db.Integer, autoincrement=True, primary_key=True,nullable = False)
    name = db.Column(db.String)
    gender = db.Column(db.String)
    age = db.Column(db.Float)
    image = db.Column(db.String)
    target = db.Column(db.String)

class Pneumonia(db.Model):
    __tablename__= 'pneumonia'
    id = db.Column(db.Integer, autoincrement=True, primary_key=True,nullable = False)
    name = db.Column(db.String)
    gender = db.Column(db.String)
    age = db.Column(db.Float)
    image = db.Column(db.String)
    target = db.Column(db.String)

'''


if request.method =="GET":
        return render_template("student.html")
    if request.method =="POST":
        name = request.form['studentName']
        admission_no = request.form['AdmissionNO']
        enrollment_no = request.form['enrollmentId']
        dob = request.form['dob']
        gender = request.form['gender']
        mob = request.form['mobile']
        email = request.form['email']
        course = request.form['course']
        section = request.form['section']
        year = request.form['year']
        semester = request.form['semester']
        data = db.session.query(Student).all()
        a = str(data[-1].id)
        a = int(a)+1
        


        print(request.files)
        file = request.files['photo']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = file.filename
            ext = filename.split(".")[-1]
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], str(a)+"."+ext))
            data = Student(name= name, admission_no = admission_no, dob = dob, gender = gender,enrollment_no = enrollment_no , mob=mob, email = email, course = course, section =  section, year = year, semester = semester , image = str(a)+"."+ext)
            db.session.add(data)
            db.session.commit()
            
            with open('dataset_faces.dat', 'rb') as f:
                all_face_encodings = pickle.load(f)
                print(type(all_face_encodings))
                known_face_names = list(all_face_encodings.keys())
                known_face_encodings = np.array(list(all_face_encodings.values()))
            new_image = face_recognition.load_image_file("static/imagedata/"+str(a)+"."+ext)
            new_image_encoding = face_recognition.face_encodings(new_image)[0]
            all_face_encodings[name] = new_image_encoding
            with open('dataset_faces.dat', 'wb') as f:
                pickle.dump(all_face_encodings, f)

            return redirect("/student")
        return None
'''


'''


def pneumoniapredictPage():
    if request.method == 'POST':
        try:
            img = Image.open(request.files['image']).convert('L')
            img.save("uploads/image.jpg")
            img_path = os.path.join(os.path.dirname(__file__), 'uploads/image.jpg')
            os.path.isfile(img_path)
            img = tf.keras.utils.load_img(img_path, target_size=(128, 128))
            img = tf.keras.utils.img_to_array(img)
            img = np.expand_dims(img, axis=0)

            model = tf.keras.models.load_model("models/pneumonia.h5")
            pred = np.argmax(model.predict(img))
        except:
            message = "Please upload an image"
            return render_template('pneumonia.html', message=message)
    return render_template('pneumonia_predict.html', pred=pred)


    '''

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/about/")
def about():
    return render_template("about.html")


@app.route("/contact/")
def contact():
    return render_template("contact.html")


@app.route("/eye/" , methods=["GET","POST"])
def eye():
    if request.method =="GET":
        return render_template("eye.html")
    if request.method =="POST":
        name = request.form['name']
        age = request.form['age']
        gender = request.form['gender']
        data = db.session.query(Eye).all()
        a = str(data[-1].id)
        a = int(a)+1
        


        print(request.files)
        file = request.files['photo']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = file.filename
            ext = filename.split(".")[-1]
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], str(a)+"."+ext))
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], str(a)+"."+ext)
            
            image = Image.open(image_path)

            # Preprocess the image
            img = image.resize((224, 224))
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0)

            # Make predictions
            predictions = eye_model.predict(img_array)
            class_labels = ['cataract', 'diabetic_retinopathy', 'glaucoma', 'normal']
            score = tf.nn.softmax(predictions[0])
            print(score)
            print(predictions)
            accuracy = round(max(predictions[0])*100,2)
            print(accuracy)
            max_index = np.array(predictions).argmax()
            target = class_labels[max_index]
            data = Eye(name = name, age = age, gender = gender, image = str(a)+"."+ext, target = target)
            db.session.add(data)
            db.session.commit()
            print(target)

            if max_index == 3:
                message = "Congratulations "+name+ " You dont have the eye Disease \n Good Luck"
            else:
                message = "Sorry "+name+ " You may have the "+ target+" \n Try to consult the doctor nearby"
            return render_template("result.html",message = message, name_of_model = 'Retinal')
        return None

@app.route("/heart", methods=["GET","POST"])
def heart():
    if request.method =="GET":
        return render_template("heart.html")
    if request.method =="POST":
        form = request.form
        to_predict_dict = request.form.to_dict()
        for key, value in to_predict_dict.items():
            try:
                to_predict_dict[key] = float(value)
            except ValueError:
                to_predict_dict[key] = str(value)

        person_name = to_predict_dict['name']
        del to_predict_dict['name']
        to_predict_list = list(map(float, list(to_predict_dict.values())))
        print(to_predict_dict)
        print(to_predict_list)
        model = pickle.load(open('models/heart.pkl','rb'))
        values = np.asarray(to_predict_list)
        val = model.predict(values.reshape(1, -1))[0]
        new_heart = Heart(name=person_name, age= to_predict_dict['age'], sex=to_predict_dict['sex'], cp = to_predict_dict['cp'], trestbps = to_predict_dict['trestbps'], chol = to_predict_dict['chol'], fbs = to_predict_dict['fbs'], restecg = to_predict_dict['restecg'], thalach = to_predict_dict['thalach'],
        exang = to_predict_dict['exang'], oldpeak = to_predict_dict['oldpeak'], slope = to_predict_dict['slope'], ca = to_predict_dict['ca'], thal = to_predict_dict['thal'], target = val)
        db.session.add(new_heart)
        db.session.commit()
        

        dis =val
        if val == 0:
            message = "Congratulations "+person_name+ " You dont have the Heart Disease \n Good Luck"
        if val == 1:
            message = "Sorry "+person_name+ " You may have the Heart Disease \n Try to consult the doctor nearby"
        return render_template("result.html",message = message, name_of_model = 'Cardiology')

@app.route("/result")
def result():
    return render_template("result.html")


'''


 elif len(values) == 24:
        model = pickle.load(open('models/kidney.pkl','rb'))
        values = np.asarray(values)
        return model.predict(values.reshape(1, -1))[0]
        
        
        '''

'''
@app.route("/login", methods=["GET","POST"])
def login():
    if request.method =="GET":
        return render_template('login_register.html')
    if request.method =="POST":
        form = request.form
        remember= request.form.getlist('remember')
        if len(remember)>0:
            rem = True
        else:
            rem=False

        print(form['username'])
    
        user = User.query.filter_by(username=form['username']).first()
        if user:
            if check_password_hash(user.password, form['password']):
                login_user(user,remember = rem)

                #return redirect('/dashboard')
                return redirect(url_for('login1'))
            
    return render_template('login_register.html', form=form)

'''
@app.route("/department/")
def department():
    return render_template("department.html")


@app.route("/liver/", methods=["GET","POST"])
def liver():
    if request.method =="GET":
        return render_template("liver.html")
    if request.method =="POST":
        form = request.form
        to_predict_dict = request.form.to_dict()
        for key, value in to_predict_dict.items():
            try:
                to_predict_dict[key] = float(value)
            except ValueError:
                to_predict_dict[key] = str(value)
        print(to_predict_dict)
        person_name = to_predict_dict['name']
        del to_predict_dict['name']
        to_predict_list = list(map(float, list(to_predict_dict.values())))
        print(to_predict_dict)
        print(to_predict_list)
        model = pickle.load(open('models/liver.pkl','rb'))
        values = np.asarray(to_predict_list)
        val = model.predict(values.reshape(1, -1))[0]
        new_kidney = Liver(name=person_name, age= to_predict_dict['age'], gender=to_predict_dict['gender'], totalb = to_predict_dict['totalb'], directb = to_predict_dict['directb'], alkalinep = to_predict_dict['alkalinep'], alaminea = to_predict_dict['alaminea'], aspertatea = to_predict_dict['aspertatea'], totalp = to_predict_dict['totalp'],
        albumin = to_predict_dict['albumin'], albumingr = to_predict_dict['albumingr'], target = val)
        db.session.add(new_kidney)
        db.session.commit()
        if val == 2:
            message = "Congratulations "+person_name+ " You dont have the Liver Disease \n Good Luck"
        if val == 1:
            message = "Sorry "+person_name+ " You may have the Liver Disease \n Try to consult the doctor nearby"
        return render_template("result.html",message = message, name_of_model = 'Hepatical')


@app.route("/kidney/", methods=["GET","POST"])
def kidney():
    if request.method =="GET":
        return render_template("kidney.html")
    if request.method =="POST":
        form = request.form
        to_predict_dict = request.form.to_dict()
        for key, value in to_predict_dict.items():
            try:
                to_predict_dict[key] = float(value)
            except ValueError:
                to_predict_dict[key] = str(value)

        person_name = to_predict_dict['name']
        del to_predict_dict['name']
        gender = to_predict_dict['sex']
        del to_predict_dict['sex']
        to_predict_list = list(map(float, list(to_predict_dict.values())))
        print(to_predict_dict)
        print(to_predict_list)
        model = pickle.load(open('models/kidney.pkl','rb'))
        values = np.asarray(to_predict_list)
        val = model.predict(values.reshape(1, -1))[0]
        new_kidney = Kidney(name=person_name, gender = gender ,age= to_predict_dict['age'], blood_pressure=to_predict_dict['blood_pressure'], specific_gravity=to_predict_dict['specific_gravity'], albumin=to_predict_dict['albumin'] ,  sugar=to_predict_dict['sugar'] , red_blood_cells=to_predict_dict['red_blood_cells'],
        pus_cell=to_predict_dict['pus_cell'], pus_cell_clumps=to_predict_dict['pus_cell_clumps'], bacteria=to_predict_dict['bacteria'] , blood_glucose=to_predict_dict['blood_glucose'] , urea=to_predict_dict['urea'] , creatinine=to_predict_dict['creatinine'] , sodium=to_predict_dict['sodium'] , 
        potassium=to_predict_dict['potassium'] , haemoglobin=to_predict_dict['haemoglobin'] , packet_cell_vol=to_predict_dict['packet_cell_vol'] , white_blood_cells_count=to_predict_dict['white_blood_cells_count'], red_blood_cells_count=to_predict_dict['red_blood_cells_count'] , hypertension=to_predict_dict['hypertension'],
        diabetes_mellitus=to_predict_dict['diabetes_mellitus'] , coronary_artery_disease=to_predict_dict['coronary_artery_disease'], apetite=to_predict_dict['apetite'] , pedia_edema=to_predict_dict['pedia_edema'] , anaemia=to_predict_dict['anaemia'] , target= val )

        db.session.add(new_kidney)
        db.session.commit()
        

        if val == 1:
            message = "Congratulations "+person_name+ " You dont have the Chronic Kidney Disease \n Good Luck"
        if val == 0:
            message = "Sorry "+person_name+ " You may have the Chronic Kidney Disease \n Try to consult the doctor nearby"
        return render_template("result.html",message = message, name_of_model = 'Renal')


@app.route("/pneumonia/", methods = ["GET","POST"])
def pneumonia(pneumonia_model = pneumonia_model):
    if request.method =="GET":
        return render_template("pneumonia.html")
    if request.method =="POST":
        name = request.form['name']
        age = request.form['age']
        gender = request.form['gender']
        data = db.session.query(Pneumonia).all()
        a = str(data[-1].id)
        a = int(a)+1
    

        print(request.files)
        file = request.files['photo']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = file.filename
            ext = filename.split(".")[-1]
            file.save(os.path.join(app.config['UPLOAD_FOLDER_2'], str(a)+"."+ext))
            
            image_path = os.path.join(app.config['UPLOAD_FOLDER_2'], str(a)+"."+ext)
            image = Image.open(image_path)

            img = image.resize((224, 224))
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0)

            # Make predictions
            predictions = pneumonia_model.predict(img_array)
            class_labels = ['COVID-19', 'Normal', 'Pneumonia-Bacterial', 'Pneumonia-Viral']
            score = tf.nn.softmax(predictions[0])
            print(score)
            print(predictions)
            accuracy = round(max(predictions[0])*100,2)
            print(accuracy)
            max_index = np.array(predictions).argmax()
            target = class_labels[max_index]
            data = Pneumonia(name= name, age = age, gender = gender, image = str(a)+"."+ext, target = target)
            db.session.add(data)
            db.session.commit()
            print(target)


            if max_index == 1:
                message = "Congratulations "+name+ " You dont have the Pneumonia Disease \n Good Luck"
            else:
                message = "Sorry "+name+ " You may have the "+ target+" \n Try to consult the doctor nearby"
            return render_template("result.html",message = message, name_of_model = 'Respiratory')
        return None


@app.route("/predictor/")
def predictor():
    return render_template("predictor.html")

if __name__ == "__main__":
  app.run(host = '0.0.0.0',debug = True,port = 8080) 