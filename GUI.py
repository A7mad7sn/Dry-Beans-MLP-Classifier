import tkinter as tk
from tkinter import ttk
from PIL import ImageTk


def center_window(window, width, height):
    screen_width = window.winfo_screenwidth()
    screen_height = window.winfo_screenheight()

    x = (screen_width - width) // 2
    y = (screen_height - height) // 2

    window.geometry(f"{width}x{height}+{x}+{y}")


class GUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Dry Beans Classifier - NN -Task2")
        self.Inputs = []
        center_window(self.root, 500, 300)

        self.root.resizable(False, False)

        self.background_image = ImageTk.PhotoImage(file="Background.jpg")

        canvas = tk.Canvas(self.root)
        canvas.create_image(0, 0, image=self.background_image, anchor=tk.NW)
        canvas.create_text(74, 24, text="Learning Rate:", font=("times", 9, "bold"), fill='White')
        canvas.create_text(70, 52, text="Number of Epochs:", font=("times", 9, "bold"), fill='White')
        canvas.create_text(75, 81, text="Number of Hidden Layers:", font=("times", 9, "bold"), fill='White')
        canvas.create_text(75, 115, text="Neurons in each Layer:", font=("times", 9, "bold"), fill='White')
        canvas.create_text(56, 172, text="Activation Function:", fill="white", font=("times", 9, "bold"))
        canvas.pack(fill="both", expand=True)

        s = ttk.Style()
        s.configure('Wild.TRadiobutton', background="black", foreground='white')

        self.learning_rate_entry = tk.Entry(self.root, width=23)
        self.learning_rate_entry.place(x=155, y=15)

        self.num_of_neuron_entry = tk.Entry(self.root, width=23)
        self.num_of_neuron_entry.place(x=155, y=105)

        self.epochs_entry = tk.Entry(self.root, width=23)
        self.epochs_entry.place(x=155, y=45)

        self.num_of_layers_entry = tk.Entry(self.root, width=23)
        self.num_of_layers_entry.place(x=155, y=75)

        self.num_of_layers_entry.bind("<FocusOut>", self.num_of_layer)
        self.num_of_layers_entry.bind("<Return>", self.num_of_layer)
        # self.enter_of_layer_button = tk.Button(self.root, text="G", width=1, bg="#5DADE2",
        #                                        command=self.num_of_layer)
        # self.enter_of_layer_button.place(x=300, y=72)

        self.varCheck = tk.IntVar()

        self.check_box = tk.Checkbutton(self.root, text="Use Bias?", variable=self.varCheck, onvalue=1, offvalue=0,
                                        background='gold')
        self.check_box.place(x=4, y=210)

        self.var = tk.StringVar()

        self.r1 = ttk.Radiobutton(self.root, text="Sigmoid", variable=self.var, value="sigmoid",
                                  style="Wild.TRadiobutton")
        self.r1.place(x=4, y=185)

        self.r2 = ttk.Radiobutton(self.root, text="Hyperbolic Tangent", variable=self.var, value="hyperbolic_tangent",
                                  style="Wild.TRadiobutton")
        self.r2.place(x=100, y=185)

        self.enter_button = tk.Button(self.root, text="Train", height=2, width=15, bg="#5DADE2",
                                      command=self.get_inputs)
        self.enter_button.place(x=200, y=250)

        self.Inputs = []

        self.entries = []

        self.root.mainloop()

    def num_of_layer(self, _):
        layers = int(self.num_of_layers_entry.get())
        self.num_of_neuron_entry.destroy()
        if self.entries:
            for i in range(len(self.entries)):
                self.entries[i].destroy()
            self.entries = []
        for i in range(layers):
            if i < 11:
                self.num_of_neuron_entry = tk.Entry(self.root, width=4)
                self.num_of_neuron_entry.place(x=155 + i * 30, y=105)
                self.entries.append(self.num_of_neuron_entry)
            elif 10 < i < 27:
                self.num_of_neuron_entry = tk.Entry(self.root, width=4)
                self.num_of_neuron_entry.place(x=4 + (i - 11) * 30, y=135)
                self.entries.append(self.num_of_neuron_entry)

    def get_inputs(self):

        num_of_neurons = []
        for i in self.entries:
            num_of_neurons.append(int(i.get()))

        if (self.num_of_layers_entry.get() == '' or
                self.num_of_neuron_entry.get() == '' or
                self.learning_rate_entry.get() == '' or
                self.epochs_entry.get() == '' or self.var.get() == ''):
            tk.messagebox.showinfo(title='Invalid', message='Please fill all required entries')
            return

        learning_rate = float(self.learning_rate_entry.get())

        epochs_num = int(self.epochs_entry.get())

        if self.varCheck.get() == 1:
            use_bias = True
        else:
            use_bias = False

        name_of_chosen_activation_fn = ''
        if self.var.get() == "sigmoid":
            name_of_chosen_activation_fn = "sigmoid"
        elif self.var.get() == "hyperbolic_tangent":
            name_of_chosen_activation_fn = "hyperbolic_tangent"

        all_inputs = [num_of_neurons, learning_rate, epochs_num, use_bias, name_of_chosen_activation_fn]

        self.Inputs = all_inputs

        self.root.destroy()

        del self
