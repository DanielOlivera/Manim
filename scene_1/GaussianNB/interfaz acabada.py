from tkinter import *
from tkinter import messagebox
import pickle

def send_data():
    Cuadrodato1_data= str(Cuadrodato1.get())
    Cuadrodato2_data= str(Cuadrodato2.get())
    Cuadrodato3_data= str(Cuadrodato3.get())
    Cuadrodato4_data= str(Cuadrodato4.get())
    Cuadrodato5_data= str(Cuadrodato5.get())
    Cuadrodato6_data= str(Cuadrodato6.get())
    Cuadrodato7_data= str(Cuadrodato7.get())
    Cuadrodato8_data= str(Cuadrodato8.get())
    Cuadrodato9_data= str(Cuadrodato9.get())
    Cuadrodato10_data= str(Cuadrodato10.get())
    Cuadrodato11_data= str(Cuadrodato11.get())
    Cuadrodato12_data= str(Cuadrodato12.get())
    Cuadrodato13_data= str(Cuadrodato13.get())
    
    print(Cuadrodato1_data, "\t", Cuadrodato2_data, "\t",Cuadrodato3_data, "\t", Cuadrodato4_data, "\t",Cuadrodato5_data, "\t", Cuadrodato6_data, "\t",Cuadrodato7_data, "\t", Cuadrodato8_data, "\t",Cuadrodato9_data, "\t", Cuadrodato10_data, "\t",Cuadrodato11_data, "\t", Cuadrodato12_data, "\t",Cuadrodato13_data)
    ############ pickles ###########################
    filenamerl='modelo_rl'
    loaded_model= pickle.load(open(filenamerl,'rb'))
    filenamesc= 'modelo_sc'
    sc=pickle.load(open(filenamesc,'rb'))
    pred_result=loaded_model.predict(sc.transform([[Cuadrodato1_data,Cuadrodato2_data,Cuadrodato3_data,Cuadrodato4_data,Cuadrodato5_data,Cuadrodato6_data,Cuadrodato7_data,Cuadrodato8_data,Cuadrodato9_data,Cuadrodato10_data,Cuadrodato11_data,Cuadrodato12_data,Cuadrodato13_data]]))
    print("resultado",pred_result)
    if (pred_result==1):
        message="Enfermo"
        messagebox.showinfo(message=message)
    elif(pred_result==0):
        message="Sano"
        messagebox.showinfo(message=message)
    
        

    ###########guardando datos en archivo de texto########
    newfile= open("registrodedatos.txt", "a")
    newfile.write(Cuadrodato1_data)
    newfile.write("\t")
    newfile.write(Cuadrodato2_data)
    newfile.write("\t")
    newfile.write(Cuadrodato3_data)
    newfile.write("\t")
    newfile.write(Cuadrodato4_data)
    newfile.write("\t")
    newfile.write(Cuadrodato5_data)
    newfile.write("\t")
    newfile.write(Cuadrodato6_data)
    newfile.write("\t")
    newfile.write(Cuadrodato7_data)
    newfile.write("\t")
    newfile.write(Cuadrodato8_data)
    newfile.write("\t")
    newfile.write(Cuadrodato9_data)
    newfile.write("\t")
    newfile.write(Cuadrodato10_data)
    newfile.write("\t")
    newfile.write(Cuadrodato11_data)
    newfile.write("\t")
    newfile.write(Cuadrodato12_data)
    newfile.write("\t")
    newfile.write(Cuadrodato13_data)
    newfile.write("\n")
    newfile.close()
    print("Datos registrados".format(Cuadrodato1_data,Cuadrodato2_data,Cuadrodato3_data,Cuadrodato4_data,Cuadrodato5_data,Cuadrodato6_data,Cuadrodato7_data,Cuadrodato8_data,Cuadrodato9_data,Cuadrodato10_data,Cuadrodato11_data,Cuadrodato12_data,Cuadrodato13_data ))
    
    #####################nuevo ingreso de datos##################3
    Cuadrodato1_entry.delete(0,END)
    Cuadrodato2_entry.delete(0,END)
    Cuadrodato3_entry.delete(0,END)
    Cuadrodato4_entry.delete(0,END)
    Cuadrodato5_entry.delete(0,END)
    Cuadrodato6_entry.delete(0,END)
    Cuadrodato7_entry.delete(0,END)
    Cuadrodato8_entry.delete(0,END)
    Cuadrodato9_entry.delete(0,END)
    Cuadrodato10_entry.delete(0,END)
    Cuadrodato11_entry.delete(0,END)
    Cuadrodato12_entry.delete(0,END)
    Cuadrodato13_entry.delete(0,END)
###############creacion de la interfa##################    
myroot= Tk()
myroot.geometry("550x700")
myroot.title("Registro de datos")
myroot.resizable(0,0)
myroot.config(background = "#213141")
main_title = Label(text="Ingreso de datos", font=("Cambria",13), bg="#56CD63",fg="White", width=550, height="2")
main_title.pack()

Cuadrodato1 = Label(text="edad", bg="#FFEEDD")
Cuadrodato1.place(x=22, y=70)
Cuadrodato2 = Label(text="sexo", bg="#FFEEDD")
Cuadrodato2.place(x=22, y=100)
Cuadrodato3 = Label(text="CP", bg="#FFEEDD")
Cuadrodato3.place(x=22, y=130)
Cuadrodato4 = Label(text="TRESTBPS", bg="#FFEEDD")
Cuadrodato4.place(x=22, y=160)
Cuadrodato5 = Label(text="CHOL", bg="#FFEEDD")
Cuadrodato5.place(x=22, y=190)
Cuadrodato6 = Label(text="FBS", bg="#FFEEDD")
Cuadrodato6.place(x=22, y=220)
Cuadrodato7 = Label(text="RESTECG", bg="#FFEEDD")
Cuadrodato7.place(x=22, y=250)
Cuadrodato8 = Label(text="THALACH", bg="#FFEEDD")
Cuadrodato8.place(x=22, y=280)
Cuadrodato9 = Label(text="EXANG", bg="#FFEEDD")
Cuadrodato9.place(x=22, y=310)
Cuadrodato10 = Label(text="OLDPEAK", bg="#FFEEDD")
Cuadrodato10.place(x=22, y=340)
Cuadrodato11 = Label(text="SLOPE", bg="#FFEEDD")
Cuadrodato11.place(x=22, y=370)
Cuadrodato12 = Label(text="CA", bg="#FFEEDD")
Cuadrodato12.place(x=22, y=400)
Cuadrodato13 = Label(text="THAL", bg="#FFEEDD")
Cuadrodato13.place(x=22, y=430)

Cuadrodato1=StringVar()
Cuadrodato2=StringVar()
Cuadrodato3=StringVar()
Cuadrodato4=StringVar()
Cuadrodato5=StringVar()
Cuadrodato6=StringVar()
Cuadrodato7=StringVar()
Cuadrodato8=StringVar()
Cuadrodato9=StringVar()
Cuadrodato10=StringVar()
Cuadrodato11=StringVar()
Cuadrodato12=StringVar()
Cuadrodato13=StringVar()
#########ingreso de datos############33

Cuadrodato1_entry= Entry(textvariable=Cuadrodato1, width="40")
Cuadrodato2_entry= Entry(textvariable=Cuadrodato2, width="40")
Cuadrodato3_entry= Entry(textvariable=Cuadrodato3, width="40")
Cuadrodato4_entry= Entry(textvariable=Cuadrodato4, width="40")
Cuadrodato5_entry= Entry(textvariable=Cuadrodato5, width="40")
Cuadrodato6_entry= Entry(textvariable=Cuadrodato6, width="40")
Cuadrodato7_entry= Entry(textvariable=Cuadrodato7, width="40")
Cuadrodato8_entry= Entry(textvariable=Cuadrodato8, width="40")
Cuadrodato9_entry= Entry(textvariable=Cuadrodato9, width="40")
Cuadrodato10_entry= Entry(textvariable=Cuadrodato10, width="40")
Cuadrodato11_entry= Entry(textvariable=Cuadrodato11, width="40")
Cuadrodato12_entry= Entry(textvariable=Cuadrodato12, width="40")
Cuadrodato13_entry= Entry(textvariable=Cuadrodato13, width="40")

Cuadrodato1_entry.place(x=100, y=70)
Cuadrodato2_entry.place(x=100, y=100)
Cuadrodato3_entry.place(x=100, y=130)
Cuadrodato4_entry.place(x=100, y=160)
Cuadrodato5_entry.place(x=100, y=190)
Cuadrodato6_entry.place(x=100, y=220)
Cuadrodato7_entry.place(x=100, y=250)
Cuadrodato8_entry.place(x=100, y=280)
Cuadrodato9_entry.place(x=100, y=310)
Cuadrodato10_entry.place(x=100, y=340)
Cuadrodato11_entry.place(x=100, y=370)
Cuadrodato12_entry.place(x=100, y=400)
Cuadrodato13_entry.place(x=100, y=430)

###########botonderegistro#####################

boton_envio=Button(myroot, text="guardar info", command=send_data, width="20", heigh="2", bg="#00CD63",curso="hand2")
boton_envio.place(x=100, y=490)
boton_resultado=Button(myroot,text="resultado", command=send_data)
boton_resultado .place(x=100, y=580)
   

myroot.mainloop()