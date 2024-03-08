import tkinter as tk
from tkinter import filedialog
import joblib
import pandas as pd

def load_model():
    global model
    file_path = filedialog.askopenfilename()
    model = joblib.load(file_path)
    status_label.config(text="Model loaded successfully!")

def predict_price():
    try:
        new_data = pd.DataFrame([[float(carlength_entry.get()),
                                  float(carwidth_entry.get()),
                                  float(carheight_entry.get()),
                                  float(curbweight_entry.get()),
                                  float(enginesize_entry.get()),
                                  float(boreratio_entry.get()),
                                  float(stroke_entry.get()),
                                  float(compressionratio_entry.get()),
                                  float(horsepower_entry.get()),
                                  float(peakrpm_entry.get()),
                                  float(citympg_entry.get()),
                                  float(highwaympg_entry.get())]])
        predicted_price = model.predict(new_data)
        result_label.config(text="Predicted price: $%.2f" % predicted_price[0])
    except Exception as e:
        result_label.config(text="Error: " + str(e))

# Create the main window
root = tk.Tk()
root.title("Car Price Predictor")

# Load Model button
load_model_button = tk.Button(root, text="Load Model", command=load_model)
load_model_button.pack(pady=10)

# Status label
status_label = tk.Label(root, text="")
status_label.pack()

# Inputs
carlength_label = tk.Label(root, text="Car Length:")
carlength_label.pack()
carlength_entry = tk.Entry(root)
carlength_entry.pack()

carwidth_label = tk.Label(root, text="Car Width:")
carwidth_label.pack()
carwidth_entry = tk.Entry(root)
carwidth_entry.pack()

carheight_label = tk.Label(root, text="Car Height:")
carheight_label.pack()
carheight_entry = tk.Entry(root)
carheight_entry.pack()

curbweight_label = tk.Label(root, text="Curb Weight:")
curbweight_label.pack()
curbweight_entry = tk.Entry(root)
curbweight_entry.pack()

enginesize_label = tk.Label(root, text="Engine Size:")
enginesize_label.pack()
enginesize_entry = tk.Entry(root)
enginesize_entry.pack()

boreratio_label = tk.Label(root, text="Bore Ratio:")
boreratio_label.pack()
boreratio_entry = tk.Entry(root)
boreratio_entry.pack()

stroke_label = tk.Label(root, text="Stroke:")
stroke_label.pack()
stroke_entry = tk.Entry(root)
stroke_entry.pack()

compressionratio_label = tk.Label(root, text="Compression Ratio:")
compressionratio_label.pack()
compressionratio_entry = tk.Entry(root)
compressionratio_entry.pack()

horsepower_label = tk.Label(root, text="Horsepower:")
horsepower_label.pack()
horsepower_entry = tk.Entry(root)
horsepower_entry.pack()

peakrpm_label = tk.Label(root, text="Peak RPM:")
peakrpm_label.pack()
peakrpm_entry = tk.Entry(root)
peakrpm_entry.pack()

citympg_label = tk.Label(root, text="City MPG:")
citympg_label.pack()
citympg_entry = tk.Entry(root)
citympg_entry.pack()

highwaympg_label = tk.Label(root, text="Highway MPG:")
highwaympg_label.pack()
highwaympg_entry = tk.Entry(root)
highwaympg_entry.pack()

# Predict button
predict_button = tk.Button(root, text="Predict Price", command=predict_price)
predict_button.pack(pady=10)

# Result label
result_label = tk.Label(root, text="")
result_label.pack()

# Run the application
root.mainloop()
