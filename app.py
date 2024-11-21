import subprocess
import json
import mysql.connector
import sys
from flask import Flask, request, jsonify, render_template, session, redirect, url_for
import bcrypt  # Import bcrypt for password hashing

app = Flask(__name__)
app.secret_key = 'ash'  # Set a secret key for session management

# Database connection function
def fetch_cities():
    db_config = {
        'host': 'localhost',
        'user': 'root',
        'password': '',  # Replace with your MySQL password
        'database': 'travelrecommendation'  # Replace with your database name
    }
    cities = []
    try:
        connection = mysql.connector.connect(**db_config)
        cursor = connection.cursor()
        cursor.execute("SELECT DISTINCT City FROM travel_data")
        cities = [row[0] for row in cursor.fetchall()]
        cursor.close()
        connection.close()
    except Exception as e:
        print(f"Database connection error: {e}")

    return cities

@app.route('/')
def index():
    if 'username' in session:
        return redirect(url_for('travel'))  # Redirect to travel page if authenticated
    else:
        return redirect(url_for('login'))  # Redirect to login if not authenticated

@app.route('/travel')
def travel():
    cities = fetch_cities()  # Fetch the cities for the dropdown
    return render_template('travel.html', cities=cities)  # Pass cities to the template

@app.route('/recommend', methods=['POST'])
def recommend():
    selected_city = request.form['city']
    command = f'python3 /Applications/XAMPP/xamppfiles/htdocs/idbms/model.py "{selected_city}"'

    try:
        # Capture both stdout and stderr
        result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Decode the output from the model
        output = result.stdout.decode('utf-8').strip()

        # Log the raw output for debugging
        print(f"Model output (raw): {repr(output)}")  # Log exact representation
        print(f"Model output (length): {len(output)}")  # Log length for debugging

        # If stderr has any content, log it
        if result.stderr:
            print(f"Error or warnings from the model script: {result.stderr.decode('utf-8')}", file=sys.stderr)

        # Attempt to parse the output as JSON
        recommendations = json.loads(output)

    except subprocess.CalledProcessError as e:
        return jsonify({"error": "Error executing the model script: " + str(e)}), 500
    except json.JSONDecodeError as e:
        return jsonify({"error": "Error parsing JSON output: " + str(e)}), 500

    return jsonify(recommendations)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Connect to the database and verify the user credentials
        db_config = {
            'host': 'localhost',
            'user': 'root',
            'password': '',  # Replace with your MySQL password
            'database': 'travelrecommendation'
        }
        connection = mysql.connector.connect(**db_config)
        cursor = connection.cursor()
        cursor.execute("SELECT password FROM users WHERE username = %s", (username,))
        result = cursor.fetchone()
        connection.close()

        if result:  # Check if user exists
            stored_password = result[0]
            if bcrypt.checkpw(password.encode('utf-8'), stored_password.encode('utf-8')):  # Verify hashed password
                session['username'] = username  # Store username in session
                return redirect(url_for('travel'))  # Redirect to the travel recommendation page
            else:
                return "Invalid username or password", 401
        else:
            return "Invalid username or password", 401

    return render_template('login.html')  # Show the login form

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Hash the password
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

        # Connect to the database and insert the new user
        db_config = {
            'host': 'localhost',
            'user': 'root',
            'password': '',  # Replace with your MySQL password
            'database': 'travelrecommendation'
        }
        connection = mysql.connector.connect(**db_config)
        cursor = connection.cursor()
        
        try:
            cursor.execute("INSERT INTO users (username, password) VALUES (%s, %s)", (username, hashed_password))
            connection.commit()
            return redirect(url_for('login'))  # Redirect to login after successful signup
        except mysql.connector.Error as e:
            print(f"Error: {e}")
            return "Error creating account. Please try again.", 500
        finally:
            cursor.close()
            connection.close()

    return render_template('signup.html')  # Show the signup form

@app.route('/logout', methods=['POST'])
def logout():
    session.pop('username', None)  # Remove the user from the session
    return redirect(url_for('login'))  # Redirect to login page

if __name__ == '__main__':
    app.run(debug=True)
