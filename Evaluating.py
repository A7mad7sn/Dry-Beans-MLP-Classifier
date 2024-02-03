import tkinter as tk
from PIL import ImageTk
from GUI import center_window
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import warnings
warnings.simplefilter("ignore")


class Evaluator:
    def __init__(self, model, scaler, y_train, y_test, y_pred_train, y_pred_test):
        self.classes = ['BOMBAY', 'CALI', 'SIRA']
        self.all_features = ['Area', 'Perimeter', 'MajorAxisLength', 'MinorAxisLength', 'roundnes']
        self.feature1 = self.all_features[0]
        self.feature2 = self.all_features[1]
        self.feature3 = self.all_features[2]
        self.feature4 = self.all_features[3]
        self.feature5 = self.all_features[4]

        self.model = model
        self.scaler = scaler
        self.y_train = y_train
        self.y_test = y_test
        self.y_pred_train = y_pred_train
        self.y_pred_test = y_pred_test
        self.gui()

    def accuracy_calculator(self, y_real, y_pred):

        correct_predictions = 0
        total_samples = len(y_real)

        for i in range(total_samples):
            real, pred = y_real[i], y_pred[i]
            if np.array_equal(real, pred):
                correct_predictions += 1

        accuracy = correct_predictions / total_samples
        accuracy = round(accuracy * 100, 2)
        return accuracy

    def confusionMatrix(self, actual, predicted):

        classes = np.unique(actual, axis=0)
        num_classes = len(classes)
        mat = np.zeros((num_classes, num_classes))

        for i in range(len(actual)):
            if np.array_equal(actual[i], predicted[i]) and np.array_equal(actual[i], classes[0]):
                mat[0, 0] += 1
            elif np.array_equal(actual[i], predicted[i]) and np.array_equal(actual[i], classes[1]):
                mat[1, 1] += 1
            elif np.array_equal(actual[i], predicted[i]) and np.array_equal(actual[i], classes[2]):
                mat[2, 2] += 1
            elif np.array_equal(actual[i], classes[0]) and np.array_equal(predicted[i], classes[1]):
                mat[0, 1] += 1
            elif np.array_equal(actual[i], classes[0]) and np.array_equal(predicted[i], classes[2]):
                mat[0, 2] += 1
            elif np.array_equal(actual[i], classes[1]) and np.array_equal(predicted[i], classes[0]):
                mat[1, 0] += 1
            elif np.array_equal(actual[i], classes[1]) and np.array_equal(predicted[i], classes[2]):
                mat[1, 2] += 1
            elif np.array_equal(actual[i], classes[2]) and np.array_equal(predicted[i], classes[0]):
                mat[2, 0] += 1
            elif np.array_equal(actual[i], classes[2]) and np.array_equal(predicted[i], classes[1]):
                mat[2, 1] += 1

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.matshow(mat, cmap="Oranges", alpha=0.3)

        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(x=j, y=i, s=int(mat[i, j]), va='center', ha='center', size='xx-large')

        plt.xlabel('Predicted', fontsize=18)
        plt.xticks([0, 1, 2], self.classes)
        plt.ylabel('Actual', fontsize=18)
        plt.yticks([0, 1, 2], self.classes)
        return fig

    def predict_sample(self):
        if feature1_tbox.get() == '' or feature2_tbox.get() == '' or feature3_tbox.get() == '' or feature4_tbox.get() == '' or feature5_tbox.get() == '':
            tk.messagebox.showinfo(title='Invalid', message='Please fill all required entries')
            return
        feature_1 = float(feature1_tbox.get())
        feature_2 = float(feature2_tbox.get())
        feature_3 = float(feature3_tbox.get())
        feature_4 = float(feature4_tbox.get())
        feature_5 = float(feature5_tbox.get())

        sample = np.array([[feature_1, feature_2, feature_3, feature_4, feature_5]])
        normalized = self.scaler.transform(sample)
        result = self.model.test(normalized)
        predicted_class = 'None'
        if np.array_equal(result[0], np.array([1, 0, 0])):
            predicted_class = 'BOMBAY'
        elif np.array_equal(result[0], np.array([0, 1, 0])):
            predicted_class = 'CALI'
        elif np.array_equal(result[0], np.array([0, 0, 1])):
            predicted_class = 'SIRA'

        tk.messagebox.showinfo(title='Output', message=f"Predicted Class: {predicted_class}")

    def gui(self):
        main = tk.Tk()
        main.title("Evaluation")

        center_window(main, 1300, 772)
        main.resizable(False, False)
        backgroundimg = ImageTk.PhotoImage(file="backgroundforclassifier.jpg")
        canvas = tk.Canvas(main)
        canvas.create_image(0, 0, image=backgroundimg, anchor=tk.NW)
        canvas.pack(fill="both", expand=True)

        evaluating_frame = tk.Frame(canvas, background='')
        evaluating_frame.pack()

        # Training part

        train_frame = tk.Frame(evaluating_frame, background='gold')
        training_label = tk.Label(train_frame,
                                  text="Training",
                                  background='gold',
                                  font=("times", 20, "bold")
                                  )
        accuracy_label = tk.Label(train_frame,
                                  text=f"Accuracy: {self.accuracy_calculator(self.y_train, self.y_pred_train)} %",
                                  background='gold',
                                  font=("times", 15, "bold")
                                  )
        train_cm = self.confusionMatrix(self.y_train, self.y_pred_train)
        train_cm = FigureCanvasTkAgg(train_cm, master=train_frame)
        training_label.pack()
        accuracy_label.pack()
        train_cm.get_tk_widget().pack()
        train_frame.grid(row=0, column=0, padx=25, pady=10)

        # Testing part

        test_frame = tk.Frame(evaluating_frame, background='gold')
        testing_label = tk.Label(test_frame,
                                 text="Testing",
                                 background='gold',
                                 font=("times", 20, "bold")
                                 )
        accuracy_label = tk.Label(test_frame,
                                  text=f"Accuracy: {self.accuracy_calculator(self.y_test, self.y_pred_test)} %",
                                  background='gold',
                                  font=("times", 15, "bold")
                                  )
        test_cm = self.confusionMatrix(self.y_test, self.y_pred_test)
        test_cm = FigureCanvasTkAgg(test_cm, master=test_frame)
        testing_label.pack()
        accuracy_label.pack()
        test_cm.get_tk_widget().pack()
        test_frame.grid(row=0, column=1, padx=25, pady=10)

        # Getting another sample

        title = tk.Label(canvas, text="Predict a sample", background='gold', font=("times", 25, "bold"))
        title.pack(fill="x", padx=25)

        getter_frame = tk.Frame(canvas, background='gold')

        global feature1_tbox
        global feature2_tbox
        global feature3_tbox
        global feature4_tbox
        global feature5_tbox

        feature1_label = tk.Label(getter_frame, text=self.feature1, background='gold', font=("times", 14, "bold"))
        feature2_label = tk.Label(getter_frame, text=self.feature2, background='gold', font=("times", 14, "bold"))
        feature3_label = tk.Label(getter_frame, text=self.feature3, background='gold', font=("times", 14, "bold"))
        feature4_label = tk.Label(getter_frame, text=self.feature4, background='gold', font=("times", 14, "bold"))
        feature5_label = tk.Label(getter_frame, text=self.feature5, background='gold', font=("times", 14, "bold"))

        feature1_tbox = tk.Entry(getter_frame, width=50)
        feature2_tbox = tk.Entry(getter_frame, width=50)
        feature3_tbox = tk.Entry(getter_frame, width=50)
        feature4_tbox = tk.Entry(getter_frame, width=50)
        feature5_tbox = tk.Entry(getter_frame, width=50)

        feature1_label.grid(row=0, column=0)
        feature1_tbox.grid(row=0, column=1)
        feature2_label.grid(row=1, column=0)
        feature2_tbox.grid(row=1, column=1)
        feature3_label.grid(row=2, column=0)
        feature3_tbox.grid(row=2, column=1)
        feature4_label.grid(row=3, column=0)
        feature4_tbox.grid(row=3, column=1)
        feature5_label.grid(row=4, column=0)
        feature5_tbox.grid(row=4, column=1)

        getter_frame.pack()

        predict_button = tk.Button(canvas, text="Predict", command=self.predict_sample, height=2, bg="#5DADE2",
                                   width=32, font=("times", 15, "bold"))
        predict_button.pack(pady=5)

        main.mainloop()
