import os
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import json

# Add parent directory and Naive Bayes directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Naive_Bayes.naive_bayes_model import NaiveBayesClassifierWrapper

class FakeNewsGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Fake News Detection System")
        self.root.geometry("800x650")
        
        self.nb_wrapper = None
        self.features_dir = tk.StringVar()
        self.model_type = tk.StringVar(value="Best NB")
        self.available_models = [
            "Best NB", "BNB", "CNB", "MNB", 
            "SVM", "XG Boost", "Logistic Regression"
        ]
        
        self._setup_ui()

    def _setup_ui(self):
        # Create Tab Control
        self.tabControl = ttk.Notebook(self.root)
        
        self.config_tab = ttk.Frame(self.tabControl)
        self.train_tab = ttk.Frame(self.tabControl)
        self.predict_tab = ttk.Frame(self.tabControl)
        self.batch_tab = ttk.Frame(self.tabControl)
        
        self.tabControl.add(self.config_tab, text='Configuration')
        self.tabControl.add(self.train_tab, text='Training')
        self.tabControl.add(self.predict_tab, text='Single Prediction')
        self.tabControl.add(self.batch_tab, text='Batch Evaluation')
        
        self.tabControl.pack(expand=1, fill="both")
        
        self._build_config_tab()
        self._build_train_tab()
        self._build_predict_tab()
        self._build_batch_tab()

    def _build_config_tab(self):
        container = ttk.LabelFrame(self.config_tab, text="Global Settings")
        container.pack(fill="x", padx=10, pady=10)
        
        # Features Directory
        ttk.Label(container, text="Features Output Dir:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        ttk.Entry(container, textvariable=self.features_dir, width=50).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(container, text="Browse", command=self._browse_features).grid(row=0, column=2, padx=5, pady=5)
        
        # Default features dir for convenience
        default_p = r"c:\Master\Repos\Master_Security_Repo\Machine learning\Final Project\Data_preprocessing_and_cleanup\Output\features_out"
        if os.path.exists(default_p):
            self.features_dir.set(default_p)

        # Model Selection
        ttk.Label(container, text="Active Classifier:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        model_dropdown = ttk.Combobox(container, textvariable=self.model_type, values=self.available_models, state="readonly")
        model_dropdown.grid(row=1, column=1, sticky="w", padx=5, pady=5)
        
        ttk.Button(container, text="Load/Init Wrapper", command=self._init_wrapper).grid(row=1, column=2, padx=5, pady=5)

    def _build_train_tab(self):
        ttk.Label(self.train_tab, text="Hyperparameter Tuning & Training (5-Fold CV)").pack(pady=10)
        
        btn_frame = ttk.Frame(self.train_tab)
        btn_frame.pack(pady=5)
        
        ttk.Button(btn_frame, text="Train Selected NB Model", command=self._run_training).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Train All NB Types", command=lambda: self._run_training(all_nb=True)).pack(side="left", padx=5)
        
        # Metrics Display
        self.train_results = tk.Text(self.train_tab, height=20, width=80)
        self.train_results.pack(padx=10, pady=10)

    def _build_predict_tab(self):
        fields_frame = ttk.Frame(self.predict_tab)
        fields_frame.pack(fill="x", padx=10, pady=10)
        
        ttk.Label(fields_frame, text="Title:").grid(row=0, column=0, sticky="nw", pady=5)
        self.pred_title = tk.Text(fields_frame, height=2, width=60)
        self.pred_title.grid(row=0, column=1, pady=5, padx=5)
        
        ttk.Label(fields_frame, text="Text:").grid(row=1, column=0, sticky="nw", pady=5)
        self.pred_text = tk.Text(fields_frame, height=10, width=60)
        self.pred_text.grid(row=1, column=1, pady=5, padx=5)
        
        # Inline Param Overrides
        param_frame = ttk.LabelFrame(self.predict_tab, text="On-the-fly Parameter Overrides (Alpha)")
        param_frame.pack(fill="x", padx=10, pady=5)
        self.custom_alpha = tk.StringVar()
        ttk.Label(param_frame, text="Alpha:").pack(side="left", padx=5)
        ttk.Entry(param_frame, textvariable=self.custom_alpha, width=10).pack(side="left", padx=5)
        
        ttk.Button(self.predict_tab, text="Classify Instance", command=self._run_prediction).pack(pady=10)
        
        self.pred_output = ttk.Label(self.predict_tab, text="Result: ", font=("Helvetica", 12, "bold"))
        self.pred_output.pack(pady=10)

    def _build_batch_tab(self):
        container = ttk.Frame(self.batch_tab)
        container.pack(fill="x", padx=10, pady=10)
        
        ttk.Label(container, text="Select CSV for Batch Predict:").pack(side="left")
        self.batch_csv_path = tk.StringVar()
        ttk.Entry(container, textvariable=self.batch_csv_path, width=50).pack(side="left", padx=5)
        ttk.Button(container, text="Browse", command=self._browse_csv).pack(side="left")
        
        ttk.Button(self.batch_tab, text="Run Evaluation", command=self._run_batch).pack(pady=10)
        
        self.batch_results = tk.Text(self.batch_tab, height=15, width=80)
        self.batch_results.pack(padx=10, pady=10)

    # --- Actions ---
    def _browse_features(self):
        path = filedialog.askdirectory()
        if path:
            self.features_dir.set(path)

    def _browse_csv(self):
        path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if path:
            self.batch_csv_path.set(path)

    def _init_wrapper(self):
        fdir = self.features_dir.get()
        if not os.path.exists(fdir):
            messagebox.showerror("Error", f"Features directory not found:\n{fdir}")
            return
        
        try:
            self.nb_wrapper = NaiveBayesClassifierWrapper(fdir)
            messagebox.showinfo("Success", "Naive Bayes Wrapper initialized.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to initialize: {str(e)}")

    def _run_training(self, all_nb=False):
        if not self.nb_wrapper:
            messagebox.showwarning("Warning", "Initialize the wrapper first in the Configuration tab.")
            return
        
        m_type = self.model_type.get()
        if m_type in ["SVM", "XG Boost", "Logistic Regression"]:
            messagebox.showinfo("Future Feature", f"{m_type} training is not yet implemented. Stay tuned!")
            return

        self.train_results.delete(1.0, tk.END)
        self.train_results.insert(tk.END, "Training started...\n")
        self.root.update_idletasks()
        
        grids = {
            'BNB': {'alpha': [0.1, 1.0, 10.0]},
            'CNB': {'alpha': [0.1, 1.0, 10.0]},
            'MNB': {'alpha': [0.1, 1.0, 10.0]}
        }
        
        model_list = None if all_nb else ([m_type] if m_type != "Best NB" else None)
        
        try:
            best_params, metrics = self.nb_wrapper.train(grids, model_types=model_list)
            
            res_str = "TRAINING COMPLETE\n" + "="*20 + "\n"
            for m, met in metrics.items():
                res_str += f"\nModel: {m}\n"
                res_str += f"- Best Params: {best_params.get(m)}\n"
                res_str += f"- Accuracy:  {met['accuracy']:.4f}\n"
                res_str += f"- Precision: {met['precision']:.4f}\n"
                res_str += f"- Recall:    {met['recall']:.4f}\n"
                res_str += f"- F1-Score:  {met['f1']:.4f}\n"
            
            if self.nb_wrapper.best_model_type:
                res_str += f"\nOVERALL BEST: {self.nb_wrapper.best_model_type}\n"
                
            self.train_results.insert(tk.END, res_str)
        except Exception as e:
            self.train_results.insert(tk.END, f"\nError: {str(e)}")

    def _run_prediction(self):
        if not self.nb_wrapper:
            messagebox.showwarning("Warning", "Initialize/Train model first.")
            return

        title = self.pred_title.get(1.0, tk.END).strip()
        text = self.pred_text.get(1.0, tk.END).strip()
        
        if not title or not text:
            messagebox.showwarning("Input Error", "Please provide both title and text.")
            return

        m_type = self.model_type.get()
        if m_type in ["SVM", "XG Boost", "Logistic Regression"]:
            messagebox.showwarning("Unavailable", f"Classifier {m_type} not yet implemented.")
            return

        # Handle custom parameters
        params = None
        alpha_val = self.custom_alpha.get().strip()
        if alpha_val:
            try:
                params = {'alpha': float(alpha_val)}
            except ValueError:
                messagebox.showerror("Error", "Alpha must be a numeric value.")
                return

        internal_type = 'best' if m_type == "Best NB" else m_type
        
        try:
            pred = self.nb_wrapper.predict_single(title, text, model_type=internal_type, parameters=params)
            label = "REAL NEWS" if pred == 1 else "FAKE NEWS"
            color = "green" if pred == 1 else "red"
            self.pred_output.config(text=f"Result: {label}", foreground=color)
        except Exception as e:
            messagebox.showerror("Prediction Error", str(e))

    def _run_batch(self):
        if not self.nb_wrapper:
            messagebox.showwarning("Warning", "Initialize/Train model first.")
            return

        csv_p = self.batch_csv_path.get().strip()
        if not os.path.exists(csv_p):
            messagebox.showerror("Error", "CSV file not found.")
            return

        m_type = self.model_type.get()
        if m_type in ["SVM", "XG Boost", "Logistic Regression"]:
            messagebox.showwarning("Unavailable", f"Classifier {m_type} not yet implemented.")
            return

        self.batch_results.delete(1.0, tk.END)
        self.batch_results.insert(tk.END, "Processing batch prediction...\n")
        self.root.update_idletasks()

        internal_type = 'best' if m_type == "Best NB" else m_type

        try:
            result = self.nb_wrapper.predict_csv(csv_p, model_type=internal_type)
            
            res_str = f"BATCH COMPLETE (Model: {m_type})\n" + "="*30 + "\n"
            if 'metrics' in result:
                m = result['metrics']
                res_str += f"Accuracy:  {m['accuracy']:.4f}\n"
                res_str += f"Precision: {m['precision']:.4f}\n"
                res_str += f"Recall:    {m['recall']:.4f}\n"
                res_str += f"F1-Score:  {m['f1']:.4f}\n"
            else:
                res_str += f"Processed {len(result['predictions'])} records. (No ground truth labels found in CSV)\n"
            
            self.batch_results.insert(tk.END, res_str)
        except Exception as e:
            self.batch_results.insert(tk.END, f"\nError: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    gui = FakeNewsGUI(root)
    root.mainloop()
