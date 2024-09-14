import tkinter as tk
from tkinter import messagebox
import mysql.connector

# Function to handle login
def login():
    username = username_entry.get()
    password = password_entry.get()
    
    # Connecting to the MySQL database
    try:
        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="",
            database="CMLTB_DB"  # Replace with your actual database name
        )
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username=%s AND password=%s", (username, password))
        result = cursor.fetchone()
        
        if result:
            messagebox.showinfo("Login Success", "Welcome, " + username + "!")
        else:
            messagebox.showerror("Login Failed", "Invalid username or password.")
        
    except mysql.connector.Error as err:
        messagebox.showerror("Database Error", f"Error: {err}")
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

# Create the main window
root = tk.Tk()
root.title("Login")

# Create username and password labels and entries
tk.Label(root, text="Username").grid(row=0, column=0, padx=10, pady=10)
username_entry = tk.Entry(root)
username_entry.grid(row=0, column=1, padx=10, pady=10)

tk.Label(root, text="Password").grid(row=1, column=0, padx=10, pady=10)
password_entry = tk.Entry(root, show="*")
password_entry.grid(row=1, column=1, padx=10, pady=10)

# Create login button
login_button = tk.Button(root, text="Login", command=login)
login_button.grid(row=2, column=0, columnspan=2, pady=20)

# Run the application
root.mainloop()
