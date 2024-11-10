from flask import Flask, request, jsonify
from src.inference import separate_sources

app = Flask(__name__)

@app.route('/separate', methods=['POST'])
def separate():
    # Get the uploaded audio file
    file = request.files['audio']
    
    # Process the file (e.g., save it, separate sources)
    separated_sources = separate_sources(file)
    
    # Send the response back (e.g., separated sources as files or spectrograms)
    return jsonify({"message": "Sources separated successfully!"})

if __name__ == "__main__":
    app.run(debug=True)
