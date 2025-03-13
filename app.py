from flask import Flask, render_template, request
import numpy as np
import pickle


app = Flask(__name__)
model = pickle.load(open('intrusion.pkl', 'rb'))

@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        duration = int(request.form['duration'])
        protocol_type = int(request.form['protocol_type'])
        service = int(request.form['service'])
        flag = int(request.form['flag'])
        src_bytes = int(request.form['src_bytes'])
        dst_bytes = int(request.form['dst_bytes'])


        values = np.array([[duration,protocol_type,service,flag,src_bytes,dst_bytes]])
        prediction = model.predict(values)

        return render_template('result.html', prediction=prediction)
    
if __name__ == "__main__":
    app.run(debug=True)

