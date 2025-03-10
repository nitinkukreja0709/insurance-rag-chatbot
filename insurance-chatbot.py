import os
import logging
import argparse
import asyncio
import aiohttp
from flask import Flask, request, jsonify, render_template
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import json
import langdetect

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Initialize global components
try:
    EMBEDDING_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
    QDRANT_CLIENT = QdrantClient("localhost", port=6333)
    OLLAMA_URL = "http://localhost:11434/api/generate"
    logger.info("Initialized embedding model and Qdrant client")
except Exception as e:
    logger.error(f"Initialization failed: {str(e)}")
    raise

# Default collection and model
DEFAULT_COLLECTION = "medical_insurance_qa"
DEFAULT_MODEL = "phi4-mini:latest"  # Adjust if this isn’t the intended model

async def query_ollama(prompt, model=DEFAULT_MODEL):
    """Query Ollama asynchronously with a prompt."""
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(OLLAMA_URL, json={"model": model, "prompt": prompt}, timeout=aiohttp.ClientTimeout(total=3000)) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Ollama returned {response.status}: {error_text}")
                    return f"Error: Ollama API failed with status {response.status}"

                full_response = ""
                async for line in response.content:
                    if line:
                        data = json.loads(line.decode('utf-8'))
                        if "response" in data:
                            full_response += data["response"]
                        if data.get("done", False):
                            break
                
                if not full_response:
                    logger.error("Ollama returned no valid response data")
                    return "Error: No valid response from Ollama"
                full_response = full_response.replace('<think>', '').replace('</think>', '').strip()
                # Removed 20-word limit and "Anything else?"
                logger.info(f"Ollama response for query: {full_response}")
                return full_response
        except aiohttp.ClientConnectorError as e:
            logger.error(f"Ollama connection failed: {str(e)}. Is Ollama running on {OLLAMA_URL}?")
            return "Error: Cannot connect to Ollama"
        except asyncio.TimeoutError:
            logger.error("Ollama query timed out after 3000 seconds")
            return "Error: Ollama took too long to respond"
        except Exception as e:
            logger.error(f"Ollama query failed with unexpected error: {str(e)}")
            return f"Error querying Ollama: {str(e)}"

def get_relevant_context(query, collection_name=DEFAULT_COLLECTION, limit=3):
    """Retrieve relevant chunks from Qdrant."""
    try:
        query_embedding = EMBEDDING_MODEL.encode(query).tolist()
        search_results = QDRANT_CLIENT.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            limit=limit
        )
        contexts = [
            {
                "content": hit.payload.get("content", ""),
                "metadata": hit.payload.get("metadata", {}),
                "score": hit.score
            }
            for hit in search_results if hit.payload and "content" in hit.payload
        ]
        logger.info(f"Found {len(contexts)} relevant contexts for query: {query}")
        for i, ctx in enumerate(contexts, 1):
            logger.info(f"Context {i} (Score: {ctx['score']:.4f}):")
            logger.info(f"  Content: {ctx['content']}")
            logger.info(f"  Metadata: {ctx['metadata']}")
        return contexts
    except Exception as e:
        logger.error(f"Qdrant search failed: {str(e)}")
        return []

async def generate_answer(query, language='en', model=DEFAULT_MODEL):
    """Generate an answer with RAG using chunks."""
    if not query:
        return {"answer": "Sorry, I didn’t catch that.", "contexts": []}

    contexts = get_relevant_context(query)
    if not contexts:
        return {"answer": "No info on that.", "contexts": []}

    context_str = "\n".join([f"Content: {c['content']}\nSource: {c['metadata'].get('source', 'unknown')} (Page {c['metadata'].get('page', 'N/A')})" for c in contexts])
    prompt = f"""You are a friendly medical insurance voice assistant. Use this info:
{context_str}

Question: {query}
Respond naturally, in {language}, like a quick chat. Provide a clear, accurate answer based on the content provided."""

    logger.info(f"Prompt sent to Ollama:\n{prompt}")
    answer = await query_ollama(prompt, model)
    return {"answer": answer, "contexts": contexts}

# Flask Routes
@app.route('/')
def index():
    """Render the voice assistant interface."""
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat API requests with async answer generation."""
    data = request.get_json() or {}
    query = data.get('query', '').strip()
    language = langdetect.detect(query)[:2] if query else 'en'
    model = data.get('model', DEFAULT_MODEL)
    
    if not query:
        return jsonify({"error": "Query is required"}), 400
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    result = loop.run_until_complete(generate_answer(query, language, model))
    loop.close()
    
    return jsonify(result)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Medical Insurance Voice Assistant')
    parser.add_argument('--port', type=int, default=5000, help='Port to run the server on')
    args = parser.parse_args()

    os.makedirs('templates', exist_ok=True)
    with open('templates/index.html', 'w') as f:
        f.write("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Insurance Voice Assistant</title>
    <style>
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }
        body {
            font-family: 'Segoe UI', Arial, sans-serif;
            background: linear-gradient(135deg, #e0eafc, #cfdef3);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }
        .chat-container {
            width: 100%;
            max-width: 700px;
            background: #fff;
            border-radius: 15px;
            box-shadow: 0 5px 25px rgba(0, 0, 0, 0.1);
            overflow: hidden;
            display: flex;
            flex-direction: column;
            height: 80vh;
        }
        .chat-header {
            background: #007bff;
            color: white;
            padding: 15px;
            text-align: center;
            font-size: 20px;
            font-weight: bold;
        }
        .chat-body {
            flex-grow: 1;
            padding: 20px;
            overflow-y: auto;
            background: #f9fafb;
            display: flex;
            flex-direction: column;
            gap: 10px;
        }
        .message {
            padding: 12px 18px;
            border-radius: 10px;
            max-width: 70%;
            word-wrap: break-word;
            animation: fadeIn 0.3s ease;
        }
        .user-message {
            background: #007bff;
            color: white;
            align-self: flex-end;
            text-align: right;
        }
        .bot-message {
            background: #e9ecef;
            color: #333;
            align-self: flex-start;
        }
        .status-message {
            color: #666;
            font-style: italic;
            text-align: center;
            align-self: center;
        }
        .chat-footer {
            padding: 15px;
            background: #fff;
            border-top: 1px solid #ddd;
            display: flex;
            gap: 10px;
            align-items: center;
            justify-content: center;
        }
        #mic-btn {
            padding: 15px;
            background: #28a745;
            color: white;
            border: none;
            border-radius: 50%;
            cursor: pointer;
            font-size: 20px;
            transition: background 0.3s;
        }
        #mic-btn.listening {
            background: #dc3545;
        }
        #text-input {
            flex-grow: 1;
            padding: 12px;
            border: 1px solid #ddd;
            border-radius: 25px;
            outline: none;
            font-size: 14px;
            transition: border-color 0.3s;
            max-width: 300px;
            display: none; /* Hidden by default */
        }
        #text-input:focus {
            border-color: #007bff;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>
</head>
<body>
    <div class="chat-container">
        <div class="chat-header">Insurance Voice Assistant</div>
        <div class="chat-body" id="chat-body">
            <div class="message bot-message">Click the mic to start talking!</div>
        </div>
        <div class="chat-footer">
            <button id="mic-btn" title="Toggle Microphone">🎙️</button>
            <input type="text" id="text-input" placeholder="Type if you prefer...">
        </div>
    </div>

    <script>
        const chatBody = document.getElementById('chat-body');
        const micBtn = document.getElementById('mic-btn');
        const textInput = document.getElementById('text-input');
        const recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
        const synth = window.speechSynthesis;

        recognition.interimResults = true;
        recognition.maxAlternatives = 1;

        const LANGUAGE_MAP = {
            'en': 'en-US',
            'es': 'es-ES',
            'fr': 'fr-FR',
            'hi': 'hi-IN'
        };

        let isListening = false;
        let hasInteracted = false;

        function addMessage(text, isUser = false, isStatus = false) {
            const div = document.createElement('div');
            div.className = `message ${isUser ? 'user-message' : isStatus ? 'status-message' : 'bot-message'}`;
            div.textContent = text;
            chatBody.appendChild(div);
            chatBody.scrollTop = chatBody.scrollHeight;
        }

        function queueSpeech(text, lang, callback) {
    synth.cancel(); // Cancel any ongoing speech
    const sentences = text.match(/[^.!?]+[.!?]*/g) || [text]; // Split into sentences
    let index = 0;

    function speakNext() {
        if (index >= sentences.length) {
            if (callback) callback();
            return;
        }
        const utterance = new SpeechSynthesisUtterance(sentences[index]);
        utterance.lang = lang;
        utterance.rate = 1.0;
        utterance.pitch = 1.0;

        utterance.onend = () => {
            index++;
            speakNext();
        };

        synth.speak(utterance);
    }

    speakNext(); // Start speaking
}


        async function sendQuery(query) {
            addMessage('Thinking...', false, true);
            try {
                const response = await fetch('/api/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ query })
                });
                const data = await response.json();
                const lastStatus = chatBody.querySelector('.status-message:last-child');
                if (lastStatus) lastStatus.remove();
                addMessage(data.answer);
                queueSpeech(data.answer, recognition.lang || 'en-US');
            } catch (error) {
                console.error('API error:', error);
                addMessage('Sorry, something went wrong.');
                queueSpeech('Sorry, something went wrong.', 'en-US');
            }
        }

        function startAssistant() {
            if (!hasInteracted) {
                queueSpeech('Hi! I’m your insurance voice assistant. How can I help?', 'en-US', () => {
                    recognition.start();
                });
                hasInteracted = true;
            } else {
                recognition.start();
            }
        }

        recognition.onstart = () => {
            isListening = true;
            micBtn.classList.add('listening');
            micBtn.textContent = '🔴';
            addMessage('Listening...', false, true);
        };

        recognition.onresult = (event) => {
            let interimTranscript = '';
            let finalTranscript = '';

            for (let i = event.resultIndex; i < event.results.length; i++) {
                const transcript = event.results[i][0].transcript;
                if (event.results[i].isFinal) {
                    finalTranscript += transcript;
                } else {
                    interimTranscript += transcript;
                }
            }

            const lastStatus = chatBody.querySelector('.status-message:last-child');
            if (lastStatus) lastStatus.textContent = interimTranscript || finalTranscript;

            if (finalTranscript) {
                recognition.stop();
                addMessage(finalTranscript, true);
                const langCode = finalTranscript.slice(0, 2).toLowerCase();
                recognition.lang = LANGUAGE_MAP[langCode] || 'en-US';
                sendQuery(finalTranscript);
            }
        };

        recognition.onend = () => {
            if (!isListening) return;
            isListening = false;
            micBtn.classList.remove('listening');
            micBtn.textContent = '🎙️';
            const lastStatus = chatBody.querySelector('.status-message:last-child');
            if (lastStatus) lastStatus.remove();
        };

        recognition.onerror = (event) => {
            addMessage(`Mic error: ${event.error}`, false, true);
            queueSpeech(`Mic error: ${event.error}. Try again?`, 'en-US');
            recognition.stop();
        };

        micBtn.addEventListener('click', () => {
            if (!isListening) {
                navigator.mediaDevices.getUserMedia({ audio: true })
                    .then(() => startAssistant())
                    .catch(err => {
                        addMessage(`Mic access denied: ${err}`);
                        queueSpeech('Please enable your microphone to use me.', 'en-US');
                    });
            } else {
                recognition.stop();
            }
        });

        textInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && textInput.value.trim()) {
                const query = textInput.value.trim();
                addMessage(query, true);
                sendQuery(query);
                textInput.value = '';
            }
        });

        micBtn.addEventListener('dblclick', () => {
            textInput.style.display = textInput.style.display === 'none' ? 'block' : 'none';
        });
    </script>
</body>
</html>
        """)

    logger.info(f"Starting server on port {args.port}")
    app.run(host='0.0.0.0', port=args.port, debug=True)