from flask import Flask, render_template,jsonify, request

from summary import generate_summary

app = Flask(__name__)

@app.route('/',methods = ['GET','POST'])
def index():
    return render_template('index.html')

@app.route('/process_string')
def process_string():
    input_string = request.args.get('input_string', '')
    # Call your summarization function here
    output_string = generate_summary(input_string)
    return render_template('index.html', result=output_string, input_string=input_string)


@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/suggestions')
def suggestions():
    return render_template('suggestions.html')

@app.route('/submit-suggestion', methods=['POST'])
def submit_suggestion():
    name = request.form['name']
    email = request.form['email']
    suggestion = request.form['message']
    # Store the suggestion in a database or file here
    with open('suggestions.txt', 'a') as f:
        f.write(f'{name}, {email}, {suggestion}\n')
    message = '✅Thank you for your suggestion!'
    return render_template('suggestions.html',thankyou=message)

if __name__ == "__main__":
    app.run(debug=True)
