import pandas as pd, numpy as np, re

from sklearn.metrics import classification_report, accuracy_score , confusion_matrix
from sklearn.model_selection import train_test_split
import tkinter as tk
from sklearn import svm
from PIL import Image, ImageTk
from tkinter import ttk
from joblib import dump , load
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
from nltk.corpus import stopwords
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
import pickle
import nltk
#######################################################################################################
nltk.download('stopwords')
stop = stopwords.words('english')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
#######################################################################################################
    
root = tk.Tk()
root.title("Spam Review Detection")
w, h = root.winfo_screenwidth(), root.winfo_screenheight()
root.geometry("%dx%d+0+0" % (w, h))
img=ImageTk.PhotoImage(Image.open("s2.jpg"))

img2=ImageTk.PhotoImage(Image.open("s3.png"))

img3=ImageTk.PhotoImage(Image.open("s4.jpg"))


logo_label=tk.Label()
logo_label.place(x=0,y=0)

x = 1

# function to change to next image
def move():
	global x
	if x == 4:
		x = 1
	if x == 1:
		logo_label.config(image=img)
	elif x == 2:
		logo_label.config(image=img2)
	elif x == 3:
		logo_label.config(image=img3)
	x = x+1
	root.after(2000, move)

# calling the function
move()
#background_label.place(x=0, y=0)

###########################################################################################################
lbl = tk.Label(root, text="Spam Review Detection ", font=('times', 35,' bold '), height=1, width=65,bg="#FFBF40",fg="black")
lbl.place(x=0, y=0)
##############################################################################################################################


def Data_Display():
    columns = ['lable', 'text']
    print(columns)

    data1 = pd.read_csv(r"D:/Radhika Bitmap 2023-24/23C9569 Spam Detection/100%code/spam1.csv", encoding='unicode_escape')

    data1.shape

    data1.shape

    data1.head()

    data1

    data1

    article_link = data1.iloc[:, 0]
    headline = data1.iloc[:, 1]
    is_sarcastic = data1.iloc[:, 2]


    display = tk.LabelFrame(root, width=100, height=400, )
    display.place(x=270, y=100)

    tree = ttk.Treeview(display, columns=(
   'lable', 'text'))

    style = ttk.Style()
    style.configure('Treeview', rowheight=40)
    style.configure("Treeview.Heading", font=("Tempus Sans ITC", 15, "bold italic"))
    style.configure(".", font=('Calibri', 10), background="black")
    style.configure("Treeview", foreground='white', background="black")

    tree["columns"] = ("1", "2")
    tree.column("1", width=130)
    tree.column("2", width=150)
    # tree.column("3", width=200)
    # tree.column("3", width=230)

    tree.heading("1", text="lable")
    tree.heading("2", text="text")
   

    treeview = tree

    tree.grid(row=0, column=0, sticky=tk.NSEW)

    print("Data Displayed")

    for i in range(0, 304):
        tree.insert("", 'end', values=(
        article_link[i], headline[i], is_sarcastic[i]))
        i = i + 1
        print(i)

##############################################################################################################


def Train():
    
    result = pd.read_csv(r"D:/Radhika Bitmap 2023-24/23C9569 Spam Detection/100%code/spam1.csv")

    result.head()
        
    result['text_without_stopwords'] = result['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
 ###########################################################################################################################################
    
    def pos(review_without_stopwords):
        return TextBlob(review_without_stopwords).tags
    
    
    os = result.text_without_stopwords.apply(pos)
    os1 = pd.DataFrame(os)
    #
    os1.head()
    
    os1['pos'] = os1['text_without_stopwords'].map(lambda x: " ".join(["/".join(x) for x in x]))
    
    result = result = pd.merge(result, os1, right_index=True, left_index=True)
    result.head()
    result['pos']
    review_train, review_test, label_train, label_test = train_test_split(result['pos'], result['target'],
                                                                              test_size=0.2, random_state=13)
    
    tf_vect = TfidfVectorizer(lowercase=True, use_idf=True, smooth_idf=True, sublinear_tf=False)
    
    X_train_tf = tf_vect.fit_transform(review_train)
    X_test_tf = tf_vect.transform(review_test)
    
    ###########################################################################################################################
   
    
    clf = svm.SVC(C=10, gamma=0.001, kernel='linear')   
    clf.fit(X_train_tf, label_train)
    pred = clf.predict(X_test_tf)
    
    with open('vectorizer.pickle', 'wb') as fin:
        pickle.dump(tf_vect, fin)
    with open('mlmodel.pickle', 'wb') as f:
        pickle.dump(clf, f)
    
    pkl = open('mlmodel.pickle', 'rb')
    clf = pickle.load(pkl)
    vec = open('vectorizer.pickle', 'rb')
    tf_vect = pickle.load(vec)
    
    X_test_tf = tf_vect.transform(review_test)
    pred = clf.predict(X_test_tf)
    
    print(metrics.accuracy_score(label_test, pred))
    
    print(confusion_matrix(label_test, pred))
    
    print(classification_report(label_test, pred))

       
    print("=" * 40)
    print("==========")
    print("Classification Report : ",(classification_report(label_test, pred)))
    print("Accuracy : ",accuracy_score(label_test, pred)*100)
    accuracy = accuracy_score(label_test, pred)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    ACC = (accuracy_score(label_test, pred) * 100)
    repo = (classification_report(label_test, pred))
    
    label4 = tk.Label(root,text =str(repo),width=35,height=10,bg='khaki',fg='black',font=("Tempus Sanc ITC",14))
    label4.place(x=960,y=100)
    
    label5 = tk.Label(root,text ="Accuracy : "+str(ACC)+"%\nModel saved as model.joblib",width=35,height=3,bg='khaki',fg='black',font=("Tempus Sanc ITC",14))
    label5.place(x=960,y=300)
    
    dump (clf,"model.joblib")
    print("Model saved as model.joblib")
    
################################################################################################################################################################

frame = tk.LabelFrame(root,text="Control Panel",width=250,height=450,bd=3,background="black",foreground="white",font=("Tempus Sanc ITC",15,"bold"))
frame.place(x=15,y=100)

entry = tk.Entry(frame,width=18,font=("Times new roman",15,"bold"))
entry.insert(0,"Enter text here...")
entry.place(x=25,y=180)
##############################################################################################################################################################################


def Test():
    predictor = load("model.joblib")
    Given_text = entry.get()
    #Given_text = "the 'roseanne' revival catches up to our thorny po..."
    vec = open('vectorizer.pickle', 'rb')
    tf_vect = pickle.load(vec)
    X_test_tf = tf_vect.transform([Given_text])
    y_predict = predictor.predict(X_test_tf)
    print(y_predict[0])
    if y_predict[0]=="ham":
        label4 = tk.Label(root,text ="ham Review Detected",width=25,height=2,bg='#46C646',fg='black',font=("Tempus Sanc ITC",25))
        label4.place(x=450,y=550)
    else:
        
        label4 = tk.Label(root,text ="spam Review Detected",width=25,height=2,bg='#FF3C3C',fg='black',font=("Tempus Sanc ITC",25))
        label4.place(x=450,y=550)
    
###########################################################################################################################################################
def window():
    root.destroy()
    
button1 = tk.Button(frame,command=Data_Display,text="Data_Display",bg="#E46EE4",fg="white",width=15,font=("Times New Roman",15,"bold"))
button1.place(x=25,y=30)

button2 = tk.Button(frame,command=Train,text="Train",bg="#E46EE4",fg="white",width=15,font=("Times New Roman",15,"bold"))
button2.place(x=25,y=100)

button3 = tk.Button(frame,command=Test,text="Test",bg="#E46EE4",fg="white",width=15,font=("Times New Roman",15,"bold"))
button3.place(x=25,y=250)

button4 = tk.Button(frame,command=window,text="Exit",bg="#E46EE4",fg="white",width=15,font=("Times New Roman",15,"bold"))
button4.place(x=25,y=330)




root.mainloop()