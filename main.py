from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))
from backend.pipeline.rag_engine import get_reranked_chunks, generate_answer

app = Flask(__name__, static_folder='frontend')
CORS(app)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data.get('message', '')
    if not user_message.strip():
        return jsonify({'response': 'Please enter a question.'})
    try:
        top_chunks = get_reranked_chunks(user_message)
        ai_response = generate_answer(top_chunks, user_message)
    except Exception as e:
        ai_response = f"Error: {str(e)}"
    return jsonify({'response': ai_response})

@app.route('/', defaults={'path': 'index.html'})
@app.route('/<path:path>')
def serve_frontend(path):
    return send_from_directory(app.static_folder, path)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
