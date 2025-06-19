import threading
import time

import customtkinter as ctk
from tkinter import messagebox, filedialog, ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from PIL import Image
from draw_based_on_models import draw_crack_path
from predictions_based_on_models import predict_Y1, predict_angle, predict_Y2, predict_T, predict_J

# Configure the customtkinter library
ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")


class SplashScreen:
    def __init__(self, root):
        self.root = root
        self.root.title("Prediction of Crack Path and Fracture Parameters in CTS Specimen")
        self.root.geometry("500x300")
        self.root.resizable(False, False)

        self.frame = ctk.CTkFrame(self.root)
        self.frame.pack(fill="both", expand=True)

        self.label = ctk.CTkLabel(
            self.frame,
            text="Prediction of Crack Path and Fracture Parameters in CTS Specimen",
            font=("Roboto", 16),
            wraplength=400,
            justify="center"
        )
        self.label.pack(expand=True, pady=20)

        button_frame = ctk.CTkFrame(self.frame)
        button_frame.pack(pady=20)

        self.crack_path_button = ctk.CTkButton(
            button_frame, text="Prediction of Crack Path", command=self.start_crack_path, width=200
        )
        self.crack_path_button.pack(side="left", padx=10)

        self.fracture_params_button = ctk.CTkButton(
            button_frame, text="Prediction of Fracture Parameters", command=self.start_fracture_params, width=200
        )
        self.fracture_params_button.pack(side="right", padx=10)

    def start_crack_path(self):
        self.root.destroy()
        MainApp(crack_path=True)

    def start_fracture_params(self):
        self.root.destroy()
        MainApp(crack_path=False)


class MainApp:
    def __init__(self, crack_path):
        self.root = ctk.CTk()
        self.root.title("CTS Prediction App")

        if crack_path:
            CrackPathApp(self.root)
        else:
            FractureParamApp(self.root)

        self.root.mainloop()


class CrackPathApp:
    def __init__(self, root):
        self.root = root
        self.entries = {}
        self.model_type_var = ctk.StringVar(value="DNN")
        self.spinner_running = False

        self.create_input_screen()

    def create_input_screen(self):
        self.clear_window()

        title_label = ctk.CTkLabel(
            self.root, text="Input Parameters for Crack Path", font=("Roboto", 18)
        )
        title_label.pack(pady=10)

        main_frame = ctk.CTkFrame(self.root)
        main_frame.pack(fill="both", expand=True, padx=20, pady=10)

        form_frame = ctk.CTkFrame(main_frame)
        form_frame.pack(side="left", fill="both", expand=True, padx=10, pady=10)

        labels_and_defaults = [
            ("Length of the specimen (L):", "71.4"),
            ("Width of the specimen (W):", "42"),
            ("Vertical distance between pins (2c):", "50.4"),
            ("Horizontal distance between external pins (b):", "25.2"),
            ("Pin diameters (d):", "6.3"),
            ("Initial crack length (a):", "19"),
            ("Loading direction angle (θ):", "45"),
            ("Increment size (Δa):", "2"),
        ]

        for label_text, default_value in labels_and_defaults:
            frame = ctk.CTkFrame(form_frame)
            frame.pack(fill="x", pady=5, padx=10)

            label = ctk.CTkLabel(frame, text=label_text, width=200, anchor="w")
            label.pack(side="left", padx=5)

            entry = ctk.CTkEntry(frame, width=200)
            entry.insert(0, default_value)
            entry.pack(side="right", padx=5)
            self.entries[label_text] = entry

        model_type_frame = ctk.CTkFrame(form_frame)
        model_type_frame.pack(fill="x", pady=10, padx=10)

        model_type_label = ctk.CTkLabel(model_type_frame, text="Model Type:", anchor="w")
        model_type_label.pack(side="left", padx=5)

        for text in ["DNN", "XGBoost", "TabNet"]:
            model_radio = ctk.CTkRadioButton(
                model_type_frame, text=text, variable=self.model_type_var, value=text
            )
            model_radio.pack(side="left", padx=5)

        image_frame = ctk.CTkFrame(main_frame, fg_color="white")
        image_frame.pack(side="right", padx=10, pady=10, fill="y", expand=True)

        image = ctk.CTkImage(Image.open("IMG/app_crack_input.png"), size=(350, 500))
        image_label = ctk.CTkLabel(image_frame, image=image, text="", padx=20, pady=20)
        image_label.pack(fill="both", expand=True)

        button_frame = ctk.CTkFrame(self.root)
        button_frame.pack(pady=20)

        continue_button = ctk.CTkButton(button_frame, text="Continue", command=self.show_plot_screen, width=150)
        continue_button.pack(side="left", padx=10)

        cancel_button = ctk.CTkButton(button_frame, text="Cancel", command=self.return_to_initial_screen, width=150)
        cancel_button.pack(side="right", padx=10)

    def show_plot_screen(self):
        try:
            values = [float(self.entries[label].get()) for label in self.entries]
            chosen_model = self.model_type_var.get()
        except ValueError:
            messagebox.showerror("Input Error", "Please enter valid numeric values for all fields.")
            return

        def run_task():
            try:
                fig = draw_crack_path(*values, chosen_model, True)

                def update_ui():
                    self.clear_window()
                    canvas = FigureCanvasTkAgg(fig, master=self.root)
                    canvas.draw()
                    canvas.get_tk_widget().pack(pady=20)

                    button_frame = ctk.CTkFrame(self.root)
                    button_frame.pack(pady=20)

                    export_button = ctk.CTkButton(button_frame, text="Export Plot", command=lambda: self.export_plot(fig), width=150)
                    export_button.pack(side="left", padx=10)

                    back_button = ctk.CTkButton(button_frame, text="Back", command=self.create_input_screen, width=150)
                    back_button.pack(side="right", padx=10)

                self.root.after(0, update_ui)

            except Exception as e:
                messagebox.showerror("Error", f"An error occurred: {e}")

            finally:
                self.spinner_running = False

        def show_spinner():
            while self.spinner_running:
                self.root.update_idletasks()
                time.sleep(0.1)

        self.clear_window()
        self.spinner_running = True

        self.spinner_label = ctk.CTkLabel(self.root, text="Calculating the crack path...", font=("Roboto", 24, "bold"))
        self.spinner_label.pack(padx=40, pady=40)

        self.progress = ttk.Progressbar(self.root, orient="horizontal", mode="indeterminate")
        self.progress.pack(pady=20, padx=60, fill="x")
        self.progress.start()

        threading.Thread(target=show_spinner, daemon=True).start()
        threading.Thread(target=run_task, daemon=True).start()

    def export_plot(self, fig):
        file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png"), ("All files", "*.*")])
        if file_path:
            fig.savefig(file_path)

    def return_to_initial_screen(self):
        self.root.destroy()
        root = ctk.CTk()
        SplashScreen(root)
        root.mainloop()

    def clear_window(self):
        for widget in self.root.winfo_children():
            widget.destroy()


class FractureParamApp:
    def __init__(self, root):
        self.root = root
        self.entries = {}
        self.model_var = ctk.StringVar(value="DNN")

        self.create_input_screen()

    def create_input_screen(self):
        self.clear_window()

        title_label = ctk.CTkLabel(self.root, text="Input Parameters for Fracture Parameters", font=("Roboto", 18))
        title_label.pack(pady=10)

        main_frame = ctk.CTkFrame(self.root)
        main_frame.pack(fill="both", expand=True, padx=20, pady=10)

        form_frame = ctk.CTkFrame(main_frame)
        form_frame.pack(side="left", padx=20, pady=20, fill="y", expand=True)

        labels_and_defaults = [
            ("crack position (x):", "0.5"),
            ("crack position (y):", "-0.25"),
            ("Theta:", "45"),
            ("Beta:", "1"),
            ("Width of specimen (W):", "42"),
            ("Length of specimen (L):", "71.4"),
        ]

        for label_text, default_value in labels_and_defaults:
            frame = ctk.CTkFrame(form_frame)
            frame.pack(fill="x", pady=5, padx=10)

            label = ctk.CTkLabel(frame, text=label_text, width=200, anchor="w")
            label.pack(side="left", padx=5)

            entry = ctk.CTkEntry(frame, width=200)
            entry.insert(0, default_value)
            entry.pack(side="right", padx=5)
            self.entries[label_text] = entry

        model_frame = ctk.CTkFrame(form_frame)
        model_frame.pack(fill="x", pady=10, padx=10)

        model_label = ctk.CTkLabel(model_frame, text="Model Type:", anchor="w", width=150)
        model_label.pack(side="left", padx=20)

        for text in ["DNN", "XGBoost", "TabNet"]:
            model_radio = ctk.CTkRadioButton(model_frame, text=text, variable=self.model_var, value=text)
            model_radio.pack(side="left", padx=5)

        image_frame = ctk.CTkFrame(main_frame, fg_color="white", corner_radius=10)
        image_frame.pack(side="right", padx=20, pady=20)

        image = ctk.CTkImage(Image.open("IMG/app_preditions.png"), size=(350, 500))
        image_label = ctk.CTkLabel(image_frame, image=image, text="")
        image_label.pack(padx=20, pady=20)

        button_frame = ctk.CTkFrame(self.root)
        button_frame.pack(pady=20)

        continue_button = ctk.CTkButton(button_frame, text="Continue", command=self.show_results_screen, width=150)
        continue_button.pack(side="left", padx=10)

        cancel_button = ctk.CTkButton(button_frame, text="Cancel", command=self.return_to_initial_screen, width=150)
        cancel_button.pack(side="right", padx=10)

    def show_results_screen(self):
        try:
            x = float(self.entries["crack position (x):"].get())
            y = float(self.entries["crack position (y):"].get())
            theta = float(self.entries["Theta:"].get())
            beta = float(self.entries["Beta:"].get())
            width = float(self.entries["Width of specimen (W):"].get())
            length = float(self.entries["Length of specimen (L):"].get())
            model = self.model_var.get()
        except ValueError:
            messagebox.showerror("Input Error", "Please enter valid numeric values for all fields.")
            return

        versions = {
            "YI": {'XGBoost':1, 'DNN':1, 'TabNet':5},
            "YII": {'XGBoost':1, 'DNN':4, 'TabNet':6},
            "angle": {'XGBoost':8, 'DNN':5, 'TabNet':6},
            "T": {'XGBoost':3, 'DNN':5, 'TabNet':6},
            "J": {'XGBoost':1, 'DNN':2, 'TabNet':6},
        }

        results = {
            "YI": predict_Y1(width, length, x/width, y/length, beta, theta, model, versions["YI"][model]),
            "YII": predict_Y2(width, length, x/width, y/length, beta, theta, model, versions["YII"][model]),
            "fracture angle": predict_angle(width, length, x/width, y/length, beta, theta, model, versions["angle"][model]),
            "J-Integral": predict_J(width, length, x/width, y/length, beta, theta, model, versions["J"][model]),
            "T-Stress": predict_T(width, length, x/width, y/length, beta, theta, model, versions["T"][model]),
        }

        self.clear_window()

        title_label = ctk.CTkLabel(self.root, text="Fracture Parameters Results", font=("Roboto", 18))
        title_label.pack(pady=10)

        results_frame = ctk.CTkFrame(self.root)
        results_frame.pack(pady=10, padx=20)

        for key, value in results.items():
            frame = ctk.CTkFrame(results_frame)
            frame.pack(fill="x", pady=5)

            key_label = ctk.CTkLabel(frame, text=f"{key}:", width=100, anchor="w")
            key_label.pack(side="left", padx=10)

            value_label = ctk.CTkLabel(frame, text=f"{round(float(value),2)}", width=100, anchor="e")
            value_label.pack(side="right", padx=10)

        button_frame = ctk.CTkFrame(self.root)
        button_frame.pack(pady=20)

        export_button = ctk.CTkButton(button_frame, text="Export Results", command=lambda: self.export_results(results), width=150)
        export_button.pack(side="left", padx=10)

        new_input_button = ctk.CTkButton(button_frame, text="New Input", command=self.create_input_screen, width=150)
        new_input_button.pack(side="right", padx=10)

    def export_results(self, results):
        file_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
        if file_path:
            with open(file_path, "w") as file:
                for key, value in results.items():
                    file.write(f"{key}: {value}\n")

    def return_to_initial_screen(self):
        self.root.destroy()
        root = ctk.CTk()
        SplashScreen(root)
        root.mainloop()

    def clear_window(self):
        for widget in self.root.winfo_children():
            widget.destroy()


if __name__ == "__main__":
    root = ctk.CTk()
    SplashScreen(root)
    root.mainloop()
