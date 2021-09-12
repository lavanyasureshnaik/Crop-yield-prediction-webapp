#-----------------------------------------------extra_modules-----------------------------------------------------------
import os

#-------------------------------------------------model_code------------------------------------------------------------
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_excel('crop_csv_file.xlsx', engine='openpyxl')

data =  data[:5000]
data = data.dropna()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

testing = data
State_Name = le.fit_transform(data.State_Name)
District_Name = le.fit_transform(data.District_Name)
#Crop_Year = le.fit_transform(data.Crop_Year)
crop = le.fit_transform(data.Crop)
Season1 = le.fit_transform(data.Season)
testing['State_Name'] = State_Name
testing['District_Name'] = District_Name
#testing['Crop_Year'] = Crop_Year
testing['Crop'] = crop
testing['Season']  = Season1

from sklearn.model_selection import train_test_split
#from sklearn.cross_validation import train_test_split
X = data.drop(['Crop','Production','Crop_Year',' area'],axis=1)
y = data['Crop']
X_train,X_test,Y_train,Y_test = train_test_split(X,y,test_size=0.1,random_state=100)

print(X.shape)
print(y.shape)
print('model')

import sklearn
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score , classification_report, mean_squared_error, r2_score
forest = RandomForestRegressor(n_estimators=1000,
                               criterion='mse',
                               random_state=1,
                               n_jobs=-1)
forest.fit(X_train, Y_train)
y_train_pred = forest.predict(X_train)
y_test_pred = forest.predict(X_test)

print('MSE train: %.3f, test: %.3f' % (
        mean_squared_error(Y_train, y_train_pred),
        mean_squared_error(Y_test, y_test_pred)))
print('R^2 train: %.3f, test: %.3f' % (
        r2_score(Y_train, y_train_pred),
        r2_score(Y_test, y_test_pred)))
score_rf = r2_score(Y_test, y_test_pred)*100

scores = [score_rf]
algorithms = ["Random forest"]

for i in range(len(algorithms)):
    print("The accuracy score achieved using "+algorithms[i]+" is: "+str(scores[i])+" %")

#print("The accuracy score achieved using Linear SVM is: "+str(score_svm)+" %")
#-------------------------------------------------model_code------------------------------------------------------------
#-------------
#
# -----------------------------------database---------------------------------------------------------------------------
import sqlite3
conn = sqlite3.connect('Crop_Database')
cur = conn.cursor()
try:
   cur.execute('''CREATE TABLE user (
     name varchar(20) DEFAULT NULL,
     email varchar(50) DEFAULT NULL,
     password varchar(20) DEFAULT NULL,
     gender varchar(10) DEFAULT NULL,
     age int(11) DEFAULT NULL)''')

   cur.execute('''CREATE TABLE Government (
        name varchar(20) DEFAULT NULL,
        email varchar(50) DEFAULT NULL,
        password varchar(20) DEFAULT NULL,
        gender varchar(10) DEFAULT NULL,
        age int(11) DEFAULT NULL)''')

   cur.execute('''CREATE TABLE GOODS (
              FARMER varchar(20) DEFAULT NULL,
              LANDAREA varchar(50) DEFAULT NULL,
              CROPGROWN varchar(20) DEFAULT NULL,
              CROPYEILD int(11) DEFAULT 0,
              COSTINIG int(11) DEFAULT 0,
              brougth int(11) DEFAULT 0,
              location varchar(100) DEFAULT NULL)''')
except:
   pass


#------------------------------------------------database---------------------------------------------------------------

from flask import Flask,render_template, url_for,request, flash, redirect, session
app = Flask(__name__)
app.config['SECRET_KEY'] = '881e69e15e7a528830975467b9d87a98'

#-------------------------------------home_page-------------------------------------------------------------------------

@app.route('/')
@app.route('/home')
def home():
   if not session.get('logged_in'):
      return render_template('home.html')
   else:
      return redirect(url_for('user_account'))

@app.route('/home1')
def home1():
   if not session.get('logged_in'):
      return render_template('home1.html')
   else:
      return redirect(url_for('user_account'))

#-------------------------------------home_page-------------------------------------------------------------------------

#-------------------------------------about_page------------------------------------------------------------------------
@app.route("/about")
def about():
   return render_template('about.html')
#-------------------------------------about_page------------------------------------------------------------------------

#-------------------------------------user_login_page-------------------------------------------------------------------
@app.route('/user_login',methods = ['POST', 'GET'])
def user_login():
   conn = sqlite3.connect('Crop_Database')
   cur = conn.cursor()
   if request.method == 'POST':
      email = request.form['email']
      password = request.form['psw']
      print('asd')
      count = cur.execute('SELECT * FROM user WHERE email = "%s" AND password = "%s"' % (email, password))
      print(count)
      #conn.commit()
      #cur.close()
      l = len(cur.fetchall())
      if l > 0:
         flash( f'Successfully Logged in' )
         return render_template('user_account.html')
      else:
         print('hello')
         flash( f'Invalid Email and Password!' )
   return render_template('user_login.html')

# -------------------------------------user_login_page-----------------------------------------------------------------
@app.route('/government_login',methods = ['POST', 'GET'])
def government_login():
   conn = sqlite3.connect('Crop_Database')
   cur = conn.cursor()
   if request.method == 'POST':
      email = request.form['email']
      password = request.form['psw']
      print('asd')
      count = cur.execute('SELECT * FROM Government WHERE email = "%s" AND password = "%s"' % (email, password))
      print(count)
      #conn.commit()
      #cur.close()
      l = len(cur.fetchall())
      if l > 0:
         flash( f'Successfully Logged in' )
         return render_template('government_account.html')
      else:
         print('hello')
         flash( f'Invalid Email and Password!' )
   return render_template('government_login.html')
# -----------------------------------predict_page-----------------------------------------------------------------
@app.route('/Crop_page', methods=['POST', 'GET'])
def Crop_page():
   return render_template('crop_page.html')

#-------------------------------------------------------------------------------------------------------------------

@app.route('/Crop_predict', methods=['POST', 'GET'])
def Crop_predict():
    State_Name = request.form['State_Name']
    District_Name = request.form['District_Name']
    Season = request.form['Season']
    Temperature = request.form['Temperature']
    humidity = request.form['humidity']
    moisture = request.form['moisture']
    if State_Name == 'Andhra Pradesh':
        State_Name = 0
    elif State_Name == 'Arunachal Pradesh':
        State_Name = 1
    elif State_Name == 'Assam':
        State_Name = 2
    elif State_Name == 'Bihar':
        State_Name = 3
    elif State_Name == 'Chandigarh':
        State_Name = 4
    elif State_Name == 'karnataka':
        State_Name = 5
    elif State_Name == 'Andaman and Nicobar Islands':
        State_Name = 6
    elif State_Name == 'MEGHALAYA':
        State_Name = 7
    elif State_Name == 'MANIPUR':
        State_Name = 8
    elif State_Name == 'Chhattisgarh':
        State_Name = 9
    elif State_Name == 'DELHI (UT)':
        State_Name = 10
    if District_Name == 'bagalkot':
        District_Name = 0
    elif District_Name == 'Ballari':
        District_Name = 1
    elif District_Name == 'Belagavi':
        District_Name = 2
    elif District_Name == 'Bengalure Rular':
        District_Name = 3
    elif District_Name == 'Bengalure Urban':
        District_Name = 4
    elif District_Name == 'Bidar':
        District_Name = 5
    elif District_Name == 'chamrajanagar':
        District_Name = 6
    elif District_Name == 'chikballapura':
        District_Name = 7
    elif District_Name == 'chikmangalore':
        District_Name = 8
    elif District_Name == 'chitra durga':
        District_Name = 9
    elif District_Name == 'Davanagere':
        District_Name = 10
    if Season == 'Kharif':
        Season = 0
    elif Season == 'Whole Year':
        Season = 1
    elif Season == 'Autumn':
        Season = 2
    elif Season == 'Rabi':
        Season = 3
    elif Season == 'Summer':
        Season = 4
    elif Season == 'Winter':
        Season = 5
    #if request.method == 'POST':
    my_prediction = forest.predict([[float(State_Name),float(District_Name),float(Season),float(Temperature),float(humidity),
                                    float(moisture)]])
    if 1 == my_prediction[0] or (0 < float(District_Name) and float(District_Name) < 10) :
        flash('wheat , rice , paddy , etc...')
    elif 2 == my_prediction[0] or (10 < float(District_Name) and float(District_Name) < 20):
        flash('coffee , tea , etc...')
    elif 3 == my_prediction[0] or (20 < float(District_Name) and float(District_Name) < 30):
        flash('sugarcane , cotton , etc...')
    else:
        flash('pepper , vanila , etc...')
    return render_template('user_account.html')
# ------------------------------------predict_page-----------------------------------------------------------------
@app.route('/predict', methods=['POST', 'GET'])
def predict():

    return render_template('user_account.html')
#-------------------------------------------------------------------------------------------------------------------
@app.route('/Crop_analyses', methods=['POST', 'GET'])
def Crop_analyses():

    return render_template('user_account.html')
# ------------------------------------predict_page-----------------------------------------------------------------
# ------------------------------------search_page-----------------------------------------------------------------
@app.route('/search')
def search():
   return render_template('search.html')
# ------------------------------------search_page-----------------------------------------------------------------

# -------------------------------------user_register_page-------------------------------------------------------------------------

@ app.route('/user_register', methods=['POST', 'GET'])
def user_register():
   conn = sqlite3.connect('Crop_Database')
   cur = conn.cursor()
   if request.method == 'POST':
      name = request.form['uname']
      email = request.form['email']
      password = request.form['psw']
      gender = request.form['gender']
      age = request.form['age']
      if gender == 0:
          print('male')
      else:
          print('female')
      cur.execute("insert into user(name,email,password,gender,age) values ('%s','%s','%s','%s','%s')" % (name, email, password, gender, age))
      conn.commit()
      # cur.close()
      print('data inserted')
      return redirect(url_for('user_login'))

   return render_template('user_register.html')
# -------------------------------------user_register_page-------------------------------------------------------------------------
@ app.route('/government_register', methods=['POST', 'GET'])
def government_register():
   conn = sqlite3.connect('Crop_Database')
   cur = conn.cursor()
   if request.method == 'POST':
      name = request.form['uname']
      email = request.form['email']
      password = request.form['psw']
      gender = request.form['gender']
      age = request.form['age']

      cur.execute("insert into Government(name,email,password,gender,age) values ('%s','%s','%s','%s','%s')" % (name, email, password, gender, age))
      conn.commit()
      # cur.close()
      print('data inserted into government')
      return redirect(url_for('government_login'))

   return render_template('government_register.html')

# -------------------------------------user_account_page-------------------------------------------------------------------------
@app.route('/user_account',methods = ['POST', 'GET'])
def user_account():
   return render_template('user_account.html')
# -------------------------------------user_account_page-------------------------------------------------------------------------

@app.route('/selling_page', methods=['POST', 'GET'])
def selling_page():
   conn = sqlite3.connect('Crop_Database')
   cur = conn.cursor()
   if request.method == 'POST':
      farmer = request.form['farmer']
      LANDAREA = request.form['LANDAREA']
      CROPGROWN = request.form['CROPGROWN']
      CROPYEILD = request.form['CROPYEILD']
      COSTINIG = request.form['COSTINIG']
      brougth = request.form['brougth']
      location = request.form['location']
      cur.execute("insert into GOODS(FARMER,LANDAREA,CROPGROWN,CROPYEILD,COSTINIG,brougth,location) values ('%s','%s','%s',%d,%d,%d,'%s')" % (farmer, LANDAREA, CROPGROWN, int(CROPYEILD), int(COSTINIG),int(brougth),location))
      conn.commit()
      # cur.close()
      print('data inserted into government')
      return redirect(url_for('government_login'))

   return render_template('selling_page.html')
# -------------------------------------user_logout_page-------------------------------------------------------------------------

@app.route('/Government_buying', methods=['POST', 'GET'])
def Government_buying():
   conn = sqlite3.connect('Crop_Database')
   cur = conn.cursor()
   cur.execute('select * from GOODS')
   a = cur.fetchall()
   return render_template('government_buying.html',q = a)

@app.route('/farmer_list', methods=['POST', 'GET'])
def farmer_list():
   conn = sqlite3.connect('Crop_Database')
   cur = conn.cursor()
   return render_template('government_buying.html')

@app.route("/logout")
def logout():
   session['logged_in'] = False
   return home()

@app.route("/logoutd",methods = ['POST','GET'])
def logoutd():
   return home()# -------------------------------------user_logout_page-------------------------------------------------------------------------


if __name__ == '__main__':
   app.secret_key = os.urandom(12)
   app.run(debug=True)

