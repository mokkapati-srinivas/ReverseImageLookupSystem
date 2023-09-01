'''
Flask backend for image-to-image search pipeline.
'''

#Import dependencies
import os
import tensorflow as tf
import numpy as np
import pickle
import config as cfg

from model import ImageSearchModel
from inference import simple_inference_with_color_filters

#import Flask dependencies
from flask import Flask, request, render_template, send_from_directory, url_for, redirect

#import database sqlalchemy
from flask_sqlalchemy import SQLAlchemy

#import flask_login dependencies
from flask_login import UserMixin, login_user, LoginManager, login_required, logout_user, current_user
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import InputRequired, Length, ValidationError
from flask_bcrypt import Bcrypt

#Set root dir
APP_ROOT = os.path.dirname(os.path.abspath(__file__))

#Define our model
model = ImageSearchModel(learning_rate=cfg.LEARNING_RATE, image_size=cfg.IMAGE_SIZE, number_of_classes=cfg.NUMBER_OF_CLASSES)

#Start tf.Session()
session = tf.Session()
session.run(tf.global_variables_initializer())
#Restore session
saver = tf.train.Saver()
saver.restore(session, 'saver/model_epoch_5.ckpt')

#Load training set vectors
with open('hamming_train_vectors.pickle', 'rb') as f:
	train_vectors = pickle.load(f)

#Load training set paths
with open('train_images_pickle.pickle', 'rb') as f:
	train_images_paths = pickle.load(f)

with open('color_vectors.pickle', 'rb') as f:
	color_vectors = pickle.load(f)

#Define Flask app
app = Flask(__name__, static_url_path='/static')

db=SQLAlchemy(app)

bcrypt=Bcrypt(app)

app.config['SQLALCHEMY_DATABASE_URI']='sqlite:///database.db'
app.config['SECRET_KEY']='thisisasecretkey'

login_manager=LoginManager()
login_manager.init_app(app)
login_manager.login_view="login"

@login_manager.user_loader
def load_user(user_id):
	return User.query.get(int(user_id))


class User(db.Model, UserMixin):
	id=db.Column(db.Integer, primary_key=True)
	username=db.Column(db.String(20), nullable=False, unique=True)
	password=db.Column(db.String(80), nullable=False)

class Recents(db.Model, UserMixin):
	id=db.Column(db.Integer, primary_key=True)
	username=db.Column(db.String(20), nullable=False)
	image_name=db.Column(db.String(80), nullable=False)

class RegisterForm(FlaskForm):
	username=StringField(validators=[InputRequired(), Length(min=4, max=20)], render_kw={"placeholder": "Username"})
	password=PasswordField(validators=[InputRequired(), Length(min=4, max=20)], render_kw={"placeholder": "Password"})
	submit=SubmitField("Register")

	def validate_username(self, username):
		existing_user_username=User.query.filter_by(username=username.data).first()

		if existing_user_username:
			raise ValidationError("The username already exists. Please choose a different one.")


class LoginForm(FlaskForm):
	username = StringField(validators=[InputRequired(), Length(min=4, max=20)], render_kw={"placeholder": "Username"})
	password = PasswordField(validators=[InputRequired(), Length(min=4, max=20)], render_kw={"placeholder": "Password"})
	submit = SubmitField("Login")

	def validate_username(self, username):
		existing_user_username=User.query.filter_by(username=username.data).first()

		if not existing_user_username:
			raise ValidationError("The username doesn't exist. Please register.")


#Define apps home page
@app.route("/") #www.image-search.com/
@login_required
def index():
	return render_template("index.html")

@app.route("/login", methods=['GET','POST'])
def login():
	form=LoginForm()

	if form.validate_on_submit():
		user=User.query.filter_by(username=form.username.data).first()
		if user:
			if bcrypt.check_password_hash(user.password,form.password.data):
				login_user(user)
				return redirect(url_for('index'))

	return render_template("login.html", form=form)

@app.route("/logout", methods=['GET','POST'])
def logout():
	logout_user()
	return redirect(url_for('login'))

@app.route("/register", methods=['GET','POST'])
def register():
	form=RegisterForm()

	if form.validate_on_submit():
		hashed_password=bcrypt.generate_password_hash(form.password.data)
		new_user=User(username=form.username.data, password=hashed_password)
		db.session.add(new_user)
		db.session.commit()
		return redirect(url_for('login'))

	return render_template("register.html", form=form)

#Define upload function
@app.route("/upload", methods=["POST"])
@login_required
def upload():

	upload_dir = os.path.join(APP_ROOT, "uploads/")

	if not os.path.isdir(upload_dir):
		os.mkdir(upload_dir)

	for img in request.files.getlist("file"):
		img_name = img.filename

		new_recent=Recents(username=current_user.username, image_name=img_name)
		db.session.add(new_recent)
		db.session.commit()

		destination = "/".join([upload_dir, img_name])
		img.save(destination)


	#inference
	result = np.array(train_images_paths)[simple_inference_with_color_filters(model, session, train_vectors, os.path.join(upload_dir, img_name), color_vectors, cfg.IMAGE_SIZE)]

	result_final = []

	for img in result:
		result_final.append("images/"+img.split("/")[-1]) #example: dataset/train/0_frog.png -> [dataset, train, 0_frog.png] -> [-1] = 0_frog.png

	return render_template("result.html", image_name=img_name, result_paths=result_final[:-2]) #added [:-2] just to have equal number of images in the result page per row

@app.route("/recentResult", methods=["POST"])
@login_required
def recentResult():

	upload_dir = os.path.join(APP_ROOT, "uploads/")

	img_name=request.form.get("img_name")

	#inference
	result = np.array(train_images_paths)[simple_inference_with_color_filters(model, session, train_vectors, os.path.join(upload_dir, img_name), color_vectors, cfg.IMAGE_SIZE)]

	result_final = []

	for img in result:
		result_final.append("images/"+img.split("/")[-1]) #example: dataset/train/0_frog.png -> [dataset, train, 0_frog.png] -> [-1] = 0_frog.png

	return render_template("result.html", image_name=img_name, result_paths=result_final[:-2]) #added [:-2] just to have equal number of images in the result page per row

@app.route("/recents", methods=['GET','POST'])
@login_required
def recents():
	recent_images=Recents.query.filter_by(username=current_user.username)

	recent_list=[]
	for r in recent_images:
		if r.image_name not in recent_list:
			recent_list.append(r.image_name)

	return render_template("recents.html", recents=recent_list)


#Define helper function for finding image paths
@app.route("/upload/<filename>")
def send_image(filename):
	return send_from_directory("uploads", filename)

#Start the application

if __name__ == "__main__":
	app.run(port=5000, debug=True)
