import os
import tkinter as tk
from tkinter import Toplevel, Label, Entry, Button, messagebox
import pandas as pd
import numpy as np
import sweetviz as sv
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from imblearn.over_sampling import SMOTE
from sklearn import metrics
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn import model_selection,svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans

custom_font = ("Helvetica", 12, "bold")
bg_color = "#2C3E50"
fg_color = "#FFFFFF"
df = None  # Initialize DataFrame variable

def create_button(parent, text, command=None, enabled=False,height=8):
    return Button(parent, text=text, fg=fg_color, bg=bg_color, font=custom_font, width=50, height=height, command=command,
                  state=tk.NORMAL if enabled else tk.DISABLED)

def regression():
    global x ,y,x_train, x_test, y_train, y_test
    data = pd.read_csv("Data.csv")
    x = data.iloc[1:, :-1]
    y = data.iloc[1:, -1]
    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.2)
    reg = None
    def on_train_button_click():
        global reg, x_train, y_train
        reg = LinearRegression()
        reg.fit(x_train, y_train)
        messagebox.showinfo("Success", "Model trained successfully.")
        test_button.config(state=tk.NORMAL)

    def on_test_button_click():

        global reg, x_test, y_test
        y_pred = reg.predict(x_test)
        mse = metrics.mean_squared_error(y_test, y_pred)


        # Plot actual vs predicted values
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test, y_pred, color='blue')
        plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--k')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title('Actual vs Predicted')
        plt.show()
        messagebox.showinfo("Test Results","Mean Squared Error: {:.2f}".format(mse))

    frame = tk.Toplevel()
    frame.title("Linear Regression")
    frame.geometry("400x300")
    frame.configure(bg="#292929")


    train_button = Button(frame, text="Train Model", command=on_train_button_click, bg="#7bf542", fg="black",font=("Helvetica", 12, "bold"))
    train_button.pack(pady=10)
    test_button = Button(frame, text="Test Model", command=on_test_button_click, bg="#7bf542", fg="black",font=("Helvetica", 12, "bold"), state=tk.DISABLED)
    test_button.pack(pady=10)

    frame.mainloop()
def clustering():
    def train_model_kmeans(n_clusters):
        global kmeans, x_train
        data = pd.read_csv("Data.csv")
        x_train = data.iloc[:, :-1]

        kmeans = KMeans(n_clusters=n_clusters)
        kmeans.fit(x_train)
        messagebox.showinfo("Success",
                            "KMeans clustering model trained successfully with {} clusters".format(n_clusters))

    def display_clusters():
        global x_train, x_test, y_train, y_test
        global kmeans
        data = pd.read_csv("Data.csv")
        x_train = data.iloc[:, :-1]
        labels = kmeans.labels_

        # Plot the clusters
        plt.scatter(x_train.iloc[:, 0], x_train.iloc[:, 1], c=labels, cmap='viridis')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title('KMeans Clustering')
        plt.show()

    def get_num_clusters():
        try:
            n_clusters = num_clusters_entry.get()
            n_clusters = int(n_clusters)
            train_model_kmeans(n_clusters)
        except Exception as e:
            messagebox.showerror("Error", " An error occurred while training the model: {}".format(str(e)))

    # Create the main frame
    frame7 = tk.Toplevel()
    frame7.title("KMeans Clustering")
    frame7.geometry("400x300")
    frame7.configure(bg="#FFFFFF")

    kmeans = None

    num_clusters_label = Label(frame7, text="Number of Clusters:", fg="#7bf542", bg="#292929",
                               font=("Helvetica", 12, "bold"))
    num_clusters_label.pack(pady=10)

    num_clusters_entry = Entry(frame7, bg="white", fg="black", font=("Helvetica", 10))
    num_clusters_entry.pack(pady=5)

    train_kmeans_button = Button(frame7, text="Train KMeans Model", command=get_num_clusters, bg="#7bf542", fg="black",
                                 font=("Helvetica", 12, "bold"))
    train_kmeans_button.pack(pady=10)

    display_clusters_button = Button(frame7, text="Display Clusters", command=display_clusters, bg="#7bf542",
                                     fg="black", font=("Helvetica", 12, "bold"))
    display_clusters_button.pack(pady=10)

    frame7.mainloop()
def calssification():
    x_train, x_test, y_train, y_test, clf = None, None, None, None, None

    def SVM_done():
        global x_train, x_test, y_train, y_test, clf
        data = pd.read_csv("Data.csv")
        x = data.iloc[2:, :-1]
        y = data.iloc[2:, -1]
        x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.2)
        clf = None

        def train_model_SVM(kernel_type):
            global clf
            clf = svm.SVC(kernel=kernel_type)
            clf.fit(x_train, y_train)
            messagebox.showinfo("Success", "Model trained successfully with {} kernel.".format(kernel_type))
            test_button.config(state=tk.NORMAL)

        def test_model():
            global clf, x_test, y_test
            y_pred = clf.predict(x_test)
            acc = metrics.accuracy_score(y_test, y_pred) * 100
            con_matrix = confusion_matrix(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            messagebox.showinfo("Test Results",
                                "Accuracy: {:.2f}%\nConfusion Matrix:\n{}\nPrecision: {:.2f}\nRecall: {:.2f}\nF1 Score: {:.2f}".format(
                                    acc, con_matrix, precision, recall, f1))

        def get_kernel():
            try:
                kernel_type = kernel_entry.get()
                train_model_SVM(kernel_type)
            except Exception as e:
                messagebox.showerror("Error", " An error occurred while training the model: {}".format(str(e)))

        frame6 = tk.Toplevel()
        frame6.title("Support Vector Machine")
        frame6.geometry("400x300")
        frame6.configure(bg="#292929")  # Set background color

        kernel_label = Label(frame6, text="Enter Kernel Type:", fg="#7bf542", bg="#292929",
                             font=("Helvetica", 12, "bold"))
        kernel_label.pack(pady=10)

        kernel_entry = Entry(frame6, bg="white", fg="black", font=("Helvetica", 10))
        kernel_entry.pack(pady=5)

        train_button = Button(frame6, text="Train Model", command=get_kernel, bg="#7bf542", fg="black",
                              font=("Helvetica", 12, "bold"))
        train_button.pack(pady=10)

        test_button = Button(frame6, text="Test Model", command=test_model, bg="#7bf542", fg="black",
                             font=("Helvetica", 12, "bold"), state=tk.DISABLED)
        test_button.pack(pady=10)

        frame6.mainloop()

    def DECI_done():
        global x_train, x_test, y_train, y_test, clf
        data = pd.read_csv("Data.csv")
        x = data.iloc[2:, :-1]
        y = data.iloc[2:, -1]
        x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.2)
        clf = None

        def train_model_Dt(criterion, max_depth):
            global clf, x_train, y_train
            try:
                clf = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth)
                clf.fit(x_train, y_train)
                messagebox.showinfo("Success", "Model trained successfully with  criterion.andmax_depth")
                test_button.config(state=tk.NORMAL)
            except Exception as e:
                messagebox.showerror("Error", "An error occurred while training the model: {}".format(str(e)))

        def test_model():
            global clf, x_test, y_test
            y_pred = clf.predict(x_test)
            acc = metrics.accuracy_score(y_test, y_pred) * 100
            con_matrix = confusion_matrix(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            messagebox.showinfo("Test Results",
                                "Accuracy: {:.2f}%\nConfusion Matrix:\n{}\nPrecision: {:.2f}\nRecall: {:.2f}\nF1 Score: {:.2f}".format(
                                    acc, con_matrix, precision, recall, f1))

        def get_criterion(max_depth):
            try:
                criterion = criterion_entry.get()
                train_model_Dt(criterion, max_depth=max_depth)
                display_tree()

            except Exception as e:
                messagebox.showerror("Error", " An error occurred while training the model: {}".format(str(e)))

        def display_tree():
            plt.figure(figsize=(20, 10))
            plot_tree(clf, filled=True, feature_names=x.columns, class_names=['0', '1'])
            plt.savefig('decision_tree.png')
            plt.show()

        # Function to get the max depth entered by the user and train the model
        def get_max_depth():
            max_depth = max_depth_entry.get()
            max_depth = int(max_depth)
            get_criterion(max_depth)

        frame6 = tk.Tk()
        frame6.title("NEURAL NETWORK")
        frame6.geometry("400x300")
        frame6.configure(bg="#FFFFFF")  # Set background color
        # Create and pack the widgets
        kernel_label = Label(frame6, text="criterion is:", fg="#7bf542", bg="#292929", font=("Helvetica", 12, "bold"))
        kernel_label.pack(pady=10)

        criterion_entry = Entry(frame6, bg="white", fg="black", font=("Helvetica", 10))
        criterion_entry.pack(pady=5)

        max_depth_label = Label(frame6, text="Enter max depth:", fg="#7bf542", bg="#292929",
                                font=("Helvetica", 12, "bold"))
        max_depth_label.pack(pady=10)

        max_depth_entry = Entry(frame6, bg="white", fg="black", font=("Helvetica", 10))
        max_depth_entry.pack(pady=5)

        train_button = Button(frame6, text="Train Model", command=lambda: get_max_depth(), bg="#7bf542", fg="black",
                              font=("Helvetica", 12, "bold"))
        train_button.pack(pady=10)

        test_button = Button(frame6, text="Test Model", command=test_model, bg="#7bf542", fg="black",
                             font=("Helvetica", 12, "bold"), state=tk.DISABLED)
        test_button.pack(pady=10)

        frame6.mainloop()

    def ANN_done():
        global x_train, x_test, y_train, y_test, clf
        data = pd.read_csv("Data.csv")
        x = data.iloc[2:, :-1]
        y = data.iloc[2:, -1]
        x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.2)
        clf = None

        def train_model_ANN(hidden_layer):
            global clf
            clf = MLPClassifier(hidden_layer_sizes=hidden_layer, activation="relu", learning_rate='constant')
            clf.fit(x_train, y_train)
            messagebox.showinfo("Success", "Model trained successfully with {} hidden_layers.".format(hidden_layer))
            test_button.config(state=tk.NORMAL)

        def test_model():
            global clf, x_test, y_test
            y_pred = clf.predict(x_test)
            acc = metrics.accuracy_score(y_test, y_pred) * 100
            con_matrix = confusion_matrix(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            messagebox.showinfo("Test Results",
                                "Accuracy: {:.2f}%\nConfusion Matrix:\n{}\nPrecision: {:.2f}\nRecall: {:.2f}\nF1 Score: {:.2f}".format(
                                    acc, con_matrix, precision, recall, f1))

        def get_hidden_layer():
            try:
                hidden_layer = hidden_layer_entry.get()
                hidden_layer = int(hidden_layer)
                train_model_ANN(hidden_layer)
            except Exception as e:
                messagebox.showerror("Error", "An error occurred while training the model: {}".format(str(e)))

        frame6 = tk.Toplevel()
        frame6.title("Neural Network")
        frame6.geometry("400x300")
        frame6.configure(bg="#292929")  # Set background color

        kernel_label = Label(frame6, text="Enter Number of Hidden Layers:", fg="#7bf542", bg="#292929",
                             font=("Helvetica", 12, "bold"))
        kernel_label.pack(pady=10)

        hidden_layer_entry = Entry(frame6, bg="white", fg="black", font=("Helvetica", 10))
        hidden_layer_entry.pack(pady=5)

        train_button = Button(frame6, text="Train Model", command=lambda: get_hidden_layer(), bg="#7bf542", fg="black",
                              font=("Helvetica", 12, "bold"))
        train_button.pack(pady=10)

        test_button = Button(frame6, text="Test Model", command=test_model, bg="#7bf542", fg="black",
                             font=("Helvetica", 12, "bold"), state=tk.DISABLED)
        test_button.pack(pady=10)

        frame6.mainloop()

    def KNN_done():
        global x_train, x_test, y_train, y_test, clf
        data = pd.read_csv("Data.csv")
        x = data.iloc[2:, :-1]
        y = data.iloc[2:, -1]
        x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.2)
        clf = None

        def train_model_KNN(n_neighbors1):
            global clf
            clf = KNeighborsClassifier(n_neighbors=n_neighbors1)
            clf.fit(x_train, y_train)
            messagebox.showinfo("Success", "Model trained successfully with {} neighbors.".format(n_neighbors1))
            test_button.config(state=tk.NORMAL)

        def test_model():
            global clf, x_test, y_test
            y_pred = clf.predict(x_test)
            acc = metrics.accuracy_score(y_test, y_pred) * 100
            con_matrix = confusion_matrix(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            messagebox.showinfo("Test Results",
                                "Accuracy: {:.2f}%\nConfusion Matrix:\n{}\nPrecision: {:.2f}\nRecall: {:.2f}\nF1 Score: {:.2f}".format(
                                    acc, con_matrix, precision, recall, f1))

        def get_n_neighbors():
            try:
                n_neighbors = n_neighbors_entry.get()
                n_neighbors = int(n_neighbors)
                train_model_KNN(n_neighbors)
            except Exception as e:
                messagebox.showerror("Error", "An error occurred while training the model: {}".format(str(e)))

        frame6 = tk.Toplevel()
        frame6.title("k-Nearest Neighbor")
        frame6.geometry("400x300")
        frame6.configure(bg="#292929")  # Set background color

        kernel_label = Label(frame6, text="Enter Number of Neighbors:", fg="#7bf542", bg="#292929",
                             font=("Helvetica", 12, "bold"))
        kernel_label.pack(pady=10)

        n_neighbors_entry = Entry(frame6, bg="white", fg="black", font=("Helvetica", 10))
        n_neighbors_entry.pack(pady=5)

        train_button = Button(frame6, text="Train Model", command=get_n_neighbors, bg="#7bf542", fg="black",
                              font=("Helvetica", 12, "bold"))
        train_button.pack(pady=10)

        test_button = Button(frame6, text="Test Model", command=test_model, bg="#7bf542", fg="black",
                             font=("Helvetica", 12, "bold"), state=tk.DISABLED)
        test_button.pack(pady=10)

        frame6.mainloop()

    frame5 = tk.Toplevel()
    frame5.title("ML Project")
    frame5.geometry("600x800")
    frame5.configure(bg="#FFFFFF")  # Set background color

    button1 = create_button(frame5, "Option_1\n\nANN", ANN_done, enabled=True)
    button1.pack(pady=10)
    button2 = create_button(frame5, "Option_2\n\nKNN", KNN_done, enabled=True)
    button2.pack(pady=10)
    button3 = create_button(frame5, "Option_3\n\nDecisionTree", DECI_done, enabled=True)
    button3.pack(pady=10)
    button4 = create_button(frame5, "Option_4\n\nSVM", SVM_done, enabled=True)
    button4.pack(pady=10)

    button5 = Button(frame5, text="Exit", fg="red", bg="#292929", font=("Arial", 12, "bold"), width=15, height=4,
                     command=frame5.quit)
    button5.pack(side=tk.RIGHT, pady=10)

    frame5.mainloop()

def preprocessing():
    global df

    def perform_feature_selection():
        try:
            global df

            def perform_selection(kernel_type):
                df = pd.read_csv("Data.csv")
                y = df.iloc[:, -1]  # Assuming the target variable is in the last column
                X = df.iloc[:, :-1].copy()

                model = svm.SVC(kernel=kernel_type)
                rfe = RFE(model, n_features_to_select=10)
                rfe.fit(X, y)

                feature_mask = rfe.support_
                selected_features = X.columns[feature_mask]



                selected_data = X[selected_features]
                selected_data.loc[:, "outcome"] = y  # Use .loc to explicitly assign values

                selected_data.to_csv("Data.csv", index=False)

                messagebox.showinfo("Success",
                                    f"Feature selection completed. Selected features: {', '.join(selected_features)}")

            def on_select():
                kernel_type = kernel_entry.get()
                perform_selection(kernel_type)

            # Create a frame
            frame = Toplevel()
            frame.title("Feature Selection")
            frame.geometry("400x200")
            frame.configure(bg="#1f1f1f")

            kernel_label = Label(frame, text="Enter Kernel Type (e.g., 'linear', 'rbf'):", fg=fg_color, bg=bg_color, font=custom_font)
            kernel_label.pack(pady=10)
            kernel_entry = Entry(frame, bg=bg_color, fg=fg_color, font=custom_font)
            kernel_entry.pack(pady=5)

            select_button = create_button(frame, "Select Features", on_select, enabled=True)
            select_button.pack(pady=10)

            # Exit button
            exit_button = create_button(frame, "Exit", frame.quit, enabled=True)
            exit_button.pack(pady=10)

            frame.mainloop()

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")

    def standard():
        df1 = pd.read_csv("Data.csv")
        df = df1.iloc[:, :-1]
        x = df1.iloc[:, -1]

        def scale_columns(df):
            # Filter only numeric columns
            numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns

            if len(numeric_columns) == 0:
                messagebox.showerror("Error", "No numeric columns found for standardization.")
                return

            try:
                st = StandardScaler()
                for column in numeric_columns:  # Exclude the last column
                    df[column] = st.fit_transform(df[[column]])
                messagebox.showinfo("Success", "Standardization completed for all numeric columns except the last one.")
                df = pd.concat([df, x], axis=1)
                df.to_csv("Data.csv", index=False)
            except Exception as e:
                messagebox.showerror("Error", f"An error occurred: {e}")

        frame4 = Toplevel()
        frame4.title("Standardization")
        frame4.geometry("400x200")
        frame4.configure(bg="#FFFFFF")

        standardize_button = create_button(frame4, "Scale", lambda: scale_columns(df), enabled=True)
        standardize_button.pack(pady=10)

        exit_button = create_button(frame4, "Exit", frame4.quit, enabled=True)
        exit_button.pack(pady=10)

        frame4.mainloop()

    def imputer():
        df1 = pd.read_csv("Data.csv")
        df = df1.iloc[:, :-1]
        x = df1.iloc[:, -1]

        def impute_columns(df):
            try:
                numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
                columns_to_impute = numeric_columns
                if len(columns_to_impute) == 0:
                    messagebox.showerror("Error", "No numeric columns found for imputation.")
                    return

                imputer = SimpleImputer(strategy="median")
                df[columns_to_impute] = imputer.fit_transform(df[columns_to_impute])
                messagebox.showinfo("Success", "Simple Imputer applied to all numeric columns except the last one.")
                df = pd.concat([df, x], axis=1)
                df.to_csv("Data.csv", index=False)
            except Exception as e:
                messagebox.showerror("Error", f"An error occurred: {e}")

        # Main Frame
        frame4 = Toplevel()
        frame4.title("SimpleImputer")
        frame4.geometry("400x200")
        frame4.configure(bg="#FFFFFF")

        impute_button = create_button(frame4, "Impute", lambda: impute_columns(df), enabled=True)
        impute_button.pack(pady=10)

        exit_button = create_button(frame4, "Exit", frame4.quit, enabled=True)
        exit_button.pack(pady=10)

        frame4.mainloop()

    def one_hot():
        df1 = pd.read_csv("Data.csv")
        df = df1.iloc[:, :-1]
        x = df1.iloc[:, -1]

        def encode_columns(df):
            column_number_str = entry.get()

            if not column_number_str:
                messagebox.showerror("Error", "Please enter a column number.")
                return

            try:
                column_number = int(column_number_str)
                ct = ColumnTransformer([('encoder', OneHotEncoder(), [column_number])], remainder='passthrough')
                df = pd.DataFrame(ct.fit_transform(df))
                messagebox.showinfo("Success", "One-hot encoding completed for column.")
                # Update only numeric columns from df_encoded
                df = pd.concat([df, x], axis=1)
                df.to_csv("Data.csv", index=False)
            except ValueError:
                messagebox.showerror("Error", "Please enter a valid column number.")
            except Exception as e:
                messagebox.showerror("Error", f"An error occurred: {e}")

        # Main Frame
        frame4 = Toplevel()
        frame4.title("One-Hot Encoding")
        frame4.geometry("400x200")
        frame4.configure(bg="#1f1f1f")

        label = Label(frame4, text="Enter Column Number:", fg=fg_color, bg=bg_color, font=custom_font)
        label.pack(pady=10)

        entry = Entry(frame4, bg=bg_color, fg=fg_color, font=custom_font)
        entry.pack(pady=5)

        encode_button = create_button(frame4, "Encode", lambda: encode_columns(df), enabled=True)
        encode_button.pack(pady=5)

        exit_button = create_button(frame4, "Exit", frame4.quit, enabled=True)
        exit_button.pack(pady=5)

        frame4.mainloop()

    def SMOT():
        df1 = pd.read_csv("Data.csv")
        x = df1.iloc[:, :-1]
        y = df1.iloc[:, -1]

        def SMOTE_oversampling():
            try:
                print("Before OverSampling # 1 =", sum(y == 1))
                print("Before OverSampling # 0 =", sum(y == 0))

                sm = SMOTE()
                x_res, y_res = sm.fit_resample(x, y)

                df_y_res = pd.DataFrame(y_res)
                last_column_index = df_y_res.shape[1] - 1
                last_column = df_y_res.iloc[:, last_column_index]

                df = pd.concat([x_res, last_column], axis=1)
                df.to_csv("Data.csv", index=False)

                messagebox.showinfo("Success", "SMOTE DONE")

                print("-----------------------------------------")
                print("After OverSampling # 1 =", sum(y_res == 1))
                print("After OverSampling # 0 =", sum(y_res == 0))
            except Exception as e:
                messagebox.showerror("Error", f"An error occurred: {e}")

        frame4 = Toplevel()
        frame4.title("SMOTE")
        frame4.geometry("400x200")
        frame4.configure(bg="#1f1f1f")

        impute_button = Button(frame4, text="SMOT", fg=fg_color, bg=bg_color, font=custom_font,width=55, height=8, command=SMOTE_oversampling)
        impute_button.pack(pady=10)
        exit_button = Button(frame4, text="Exit", fg=fg_color, bg=bg_color, font=custom_font,width=55, height=8, command=frame4.quit)
        exit_button.pack(pady=10)

        frame4.mainloop()

    def save_file():
        try:
            df = pd.read_csv("Data.csv")
            df.to_csv("Data_copy.csv", index=False)
            button1.config(state=tk.NORMAL)
            button2.config(state=tk.NORMAL)
            button3.config(state=tk.NORMAL)
            button4.config(state=tk.NORMAL)
            button6.config(state=tk.NORMAL)
            messagebox.showinfo("Success", "Data_copy.csv")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred while saving the file: {e}")

    frame3 = tk.Toplevel()
    frame3.title("ML Project")
    frame3.geometry("600x800")
    frame3.configure(bg="#FFFFFF")

    button1 = create_button(frame3, "Option_1\n\nOneHotEncoder", command=lambda: one_hot(), enabled=False,height=6)
    button1.pack(pady=5)
    button2 = create_button(frame3, "Option_2\n\nSimpleImputer", command=lambda: imputer(), enabled=False,height=6)
    button2.pack(pady=5)
    button3 = create_button(frame3, "Option_4\n\nStandardization", command=lambda: standard(), enabled=False,height=6)
    button3.pack(pady=5)
    button4 = create_button(frame3, "Option_4\n\nFeatureSelection", command=lambda: perform_feature_selection(), enabled=False,height=6)
    button4.pack(pady=5)
    button6 = create_button(frame3, "Option_4\n\nSMOT", command=lambda:     SMOT(),enabled=False,height=6)
    button6.pack(pady=5)
    button5 = Button(frame3, text="Copy Before Start", fg=fg_color, bg=bg_color, font=custom_font, width=55, height=8,command=lambda: save_file(), state=tk.NORMAL)
    button5.pack(side=tk.RIGHT, pady=10)

    frame3.mainloop()

def go():
    button1.config(state=tk.NORMAL)
    button2.config(state=tk.NORMAL)
    button3.config(state=tk.NORMAL)
    button4.config(state=tk.NORMAL)

def start():
    frame2 = Toplevel()
    frame2.title("CSV Screen")
    frame2.geometry("750x800")
    frame2.configure(bg="#FFFFFF")

    welcome_label = Label(frame2, text="Welcome to the CSV Screen", fg=fg_color, bg=bg_color, font=custom_font)
    welcome_label.pack(pady=10)

    def csvread():
        global df
        filename = entry.get()
        try:
            df = pd.read_csv(filename)
            done_label.config(text="""WELCOME TO OUR PROGRAM
 We want to inform you about some some points 
   1-Your file is read successfully
   2-Your file is renamed to " Data.csv" to avoid errors
   3-Your Data file is analyzed
   4-Analyzed data is displayed in "your_report.html"


"Notes" :
1-check that you don't have another file with name (Data.csv) 
2-This program doesn't oblige you to do preprocessing if you are sure that your data is clean..
if your data is not clean , certainly do preprocessing")""")
            os.rename(filename, "Data.csv")
            start_button.config(state=tk.NORMAL)
            report = sv.analyze(df)
            report.show_html("your_report.html")

        except FileNotFoundError:
            error_label.config(text="File not found")
            messagebox.showerror("Error", "File not found")

    label = Label(frame2, text="Enter File Name:", fg=fg_color, bg=bg_color, font=custom_font)
    label.pack(pady=10)
    entry = Entry(frame2, bg=bg_color, fg=fg_color, font=custom_font)
    entry.pack(pady=10)
    submit_button = create_button(frame2, "Submit", csvread, enabled=True)
    submit_button.pack(pady=10)
    start_button = create_button(frame2, "Let's Go", command=lambda :go())
    start_button.pack(pady=10)

    done_label = Label(frame2, text="", fg="#0D9873", bg=bg_color, font=custom_font)
    done_label.pack(pady=10)

    error_label = Label(frame2, text="", fg="red", bg=bg_color, font=custom_font)
    error_label.pack(pady=10)

# Main Frame
frame1 = tk.Tk()
frame1.title("ML Project")
frame1.geometry("600x800")
frame1.configure(bg="#FFFFFF")  # Set background color

button1 = create_button(frame1, "Option_1\n\nPreprocessing", command=lambda :preprocessing())
button1.pack(pady=10)
button2 = create_button(frame1, "Option_2\n\nClassification",command=lambda:calssification())
button2.pack(pady=10)
button3 = create_button(frame1, "Option_3\n\nRegression",command=lambda:regression())
button3.pack(pady=10)
button4 = create_button(frame1, "Option_4\n\nClustering", command=lambda:clustering())
button4.pack(pady=10)

button5 = Button(frame1, text="Exit", fg="red", bg="#2C3E50", font=("Arial", 12, "bold"), width=15, height=4,command=frame1.quit)
button5.pack(side=tk.RIGHT, pady=10)

button5 = Button(frame1, text="Rename&Analysis", fg="red", bg="#2C3E50", font=("Arial", 12, "bold"), width=30, height=4,command=lambda:start())
button5.pack(side=tk.LEFT, pady=10)

frame1.mainloop()