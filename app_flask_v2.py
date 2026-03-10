from flask import Flask, render_template_string, request, jsonify
import tempfile
import os
from pathlib import Path
from src.nlp.infer_nlp import TfidfNLPInferencer
from src.utils.text_cleaning import clean_email_text
from src.fusion.ensemble import weighted_ensemble

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB max

nlp_infer = TfidfNLPInferencer(model_path="artifacts/nlp_tfidf_lr.joblib")

# Try to load vision model
import traceback
vision_infer = None
try:
    from src.vision.infer_vision import VisionInferencer
    vision_infer = VisionInferencer(model_path="artifacts/vision_mobilenet.keras")
    print("Vision model loaded successfully!")
except Exception as e:
    print("Vision model failed to load:")
    traceback.print_exc()
    vision_infer = None

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Phishing Detector</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }
        .container {
            background: white;
            border-radius: 12px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            max-width: 700px;
            width: 100%;
            padding: 40px;
        }
        h1 { color: #333; margin-bottom: 8px; }
        .subtitle { color: #666; margin-bottom: 30px; font-size: 14px; }
        .tabs { display: flex; gap: 10px; margin-bottom: 30px; border-bottom: 2px solid #e0e0e0; }
        .tab-btn { 
            padding: 12px 20px; background: none; border: none; cursor: pointer; 
            color: #666; font-weight: 500; border-bottom: 3px solid transparent; margin-bottom: -2px;
        }
        .tab-btn.active { color: #667eea; border-bottom-color: #667eea; }
        .tab-content { display: none; }
        .tab-content.active { display: block; }
        .form-group { margin-bottom: 20px; }
        label { display: block; margin-bottom: 8px; color: #333; font-weight: 500; }
        textarea {
            width: 100%; min-height: 140px; padding: 12px;
            border: 2px solid #e0e0e0; border-radius: 8px; font-family: monospace; font-size: 14px;
        }
        textarea:focus { outline: none; border-color: #667eea; }
        .dropzone {
            border: 2px dashed #e0e0e0; border-radius: 8px; padding: 40px; text-align: center;
            cursor: pointer; transition: 0.3s;
        }
        .dropzone:hover { border-color: #667eea; background: rgba(102,126,234,0.05); }
        .dropzone.dragover { border-color: #667eea; background: rgba(102,126,234,0.1); }
        #preview { max-width: 100%; max-height: 300px; margin-top: 15px; border-radius: 8px; display: none; }
        #preview.show { display: block; }
        button {
            width: 100%; padding: 12px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white; border: none; border-radius: 8px; font-size: 16px; font-weight: 600; cursor: pointer;
        }
        button:hover { transform: translateY(-2px); }
        .results { margin-top: 30px; padding-top: 30px; border-top: 2px solid #e0e0e0; display: none; }
        .results.show { display: block; }
        .metrics { display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px; margin-bottom: 20px; }
        .metric-box { background: #f5f5f5; padding: 15px; border-radius: 8px; text-align: center; }
        .metric-box.vision { background: #e3f2fd; }
        .metric-label { color: #666; font-size: 12px; text-transform: uppercase; margin-bottom: 8px; }
        .metric-value { font-size: 24px; font-weight: bold; color: #333; }
        .verdict { margin: 20px 0; padding: 20px; border-radius: 8px; text-align: center; font-size: 18px; font-weight: 600; }
        .verdict.phishing { background: #ffebee; color: #c62828; border-left: 4px solid #c62828; }
        .verdict.legitimate { background: #e8f5e9; color: #2e7d32; border-left: 4px solid #2e7d32; }
        .loading { display: none; text-align: center; color: #667eea; }
        .loading.show { display: block; }
        .error { display: none; background: #ffebee; color: #c62828; padding: 15px; border-radius: 8px; margin: 15px 0; }
        .error.show { display: block; }
        input[type="file"] { display: none; }
        .info { background: #e3f2fd; color: #1565c0; padding: 12px; border-radius: 8px; font-size: 12px; margin-bottom: 15px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Email Phishing Detector</h1>
        <p class="subtitle">Multi-modal analysis system</p>
        
        <div class="tabs">
            <button class="tab-btn active" onclick="switchTab('email', this)">Email Text</button>
            <button class="tab-btn" onclick="switchTab('screenshot', this)">Screenshot</button>
        </div>
        
        <div id="email" class="tab-content active">
            <div class="form-group">
                <label>Paste Email Content</label>
                <textarea id="emailInput" placeholder="Paste email text..."></textarea>
            </div>
            <button onclick="analyzeEmail()">Analyze Email</button>
        </div>
        
        <div id="screenshot" class="tab-content">
            <div class="info">Upload a website screenshot to analyze its appearance.</div>
            <div class="form-group">
                <label>Upload Screenshot</label>
                <div class="dropzone" id="dropzone" onclick="document.getElementById('imageInput').click()">
                    <div style="font-size: 40px; margin-bottom: 10px;"><svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="#667eea" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="7" width="18" height="13" rx="2" ry="2"></rect><circle cx="12" cy="13" r="4"></circle><polyline points="8 7 8 4 16 4 16 7"></polyline></svg></div>
                    <p>Click to upload or drag & drop</p>
                    <p style="font-size: 12px; color: #666;">PNG, JPG, or JPEG</p>
                </div>
                <input type="file" id="imageInput" accept="image/*" onchange="handleImage()">
                <img id="preview" alt="Preview">
            </div>
            <button onclick="analyzeScreenshot()">Analyze Screenshot</button>
        </div>
        
        <div class="loading" id="loading">Analyzing...</div>
        <div class="error" id="error"></div>
        
        <div class="results" id="results">
            <div class="metrics" id="metrics"></div>
            <div class="verdict" id="verdict"></div>
            <p style="color: #666; font-size: 11px; margin-top: 20px;">
                <strong>Disclaimer:</strong> Decision support only. Always verify independently.
            </p>
        </div>
    </div>

    <script>
        function switchTab(tabId, btn) {
            document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
            document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
            document.getElementById(tabId).classList.add('active');
            btn.classList.add('active');
        }
        
        const dz = document.getElementById('dropzone');
        dz.addEventListener('dragover', e => { e.preventDefault(); dz.classList.add('dragover'); });
        dz.addEventListener('dragleave', () => dz.classList.remove('dragover'));
        dz.addEventListener('drop', e => {
            e.preventDefault();
            dz.classList.remove('dragover');
            document.getElementById('imageInput').files = e.dataTransfer.files;
            handleImage();
        });
        
        function handleImage() {
            const file = document.getElementById('imageInput').files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = e => {
                    document.getElementById('preview').src = e.target.result;
                    document.getElementById('preview').classList.add('show');
                };
                reader.readAsDataURL(file);
            }
        }
        
        async function analyzeEmail() {
            const email = document.getElementById('emailInput').value.trim();
            if (!email) { showError('Paste email content'); return; }
            submit('/analyze-email', { email }, true);
        }
        
        async function analyzeScreenshot() {
            const file = document.getElementById('imageInput').files[0];
            if (!file) { showError('Upload a screenshot'); return; }
            const fd = new FormData();
            fd.append('image', file);
            submit('/analyze-screenshot', fd, false);
        }
        
        async function submit(url, data, isJson) {
            document.getElementById('error').classList.remove('show');
            document.getElementById('results').classList.remove('show');
            document.getElementById('loading').classList.add('show');
            
            try {
                const res = await fetch(url, {
                    method: 'POST',
                    ...(isJson ? { headers: {'Content-Type': 'application/json'}, body: JSON.stringify(data) } : { body: data })
                });
                const result = await res.json();
                document.getElementById('loading').classList.remove('show');
                if (!res.ok) { showError(result.error); return; }
                showResults(result);
            } catch (e) {
                document.getElementById('loading').classList.remove('show');
                showError(e.message);
            }
        }
        
        function showResults(data) {
            let html = '';
            if (data.vision_score !== undefined) {
                html = `
                    <div class="metric-box"><div class="metric-label">NLP</div><div class="metric-value">${(data.nlp_score*100).toFixed(1)}%</div></div>
                    <div class="metric-box vision"><div class="metric-label">Vision</div><div class="metric-value">${(data.vision_score*100).toFixed(1)}%</div></div>
                    <div class="metric-box"><div class="metric-label">Final</div><div class="metric-value">${(data.final_score*100).toFixed(1)}%</div></div>
                    <div class="metric-box"><div class="metric-label">Confidence</div><div class="metric-value">${data.confidence}</div></div>
                `;
            } else {
                html = `
                    <div class="metric-box"><div class="metric-label">Phishing Score</div><div class="metric-value">${(data.score*100).toFixed(1)}%</div></div>
                    <div class="metric-box"><div class="metric-label">Confidence</div><div class="metric-value">${data.confidence}</div></div>
                `;
            }
            document.getElementById('metrics').innerHTML = html;
            const v = document.getElementById('verdict');
            const verdict = data.verdict || data.final_verdict;
            if (verdict === 'phishing') {
                v.innerHTML = '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#c62828" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="8" y1="8" x2="16" y2="16"></line><line x1="16" y1="8" x2="8" y2="16"></line></svg> PHISHING';
            } else {
                v.innerHTML = '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#2e7d32" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><polyline points="8 12 12 16 16 8"></polyline></svg> LEGITIMATE';
            }
            v.className = 'verdict ' + verdict;
            document.getElementById('results').classList.add('show');
        }
        
        function showError(msg) {
            document.getElementById('error').textContent = msg;
            document.getElementById('error').classList.add('show');
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/analyze-email', methods=['POST'])
def analyze_email():
    try:
        email = request.json.get('email', '').strip()
        if not email:
            return jsonify({'error': 'Email required'}), 400
        score = nlp_infer.predict_score(clean_email_text(email))
        confidence = 'HIGH' if score > 0.7 or score < 0.3 else 'MEDIUM'
        return jsonify({'score': score, 'verdict': 'phishing' if score >= 0.5 else 'legitimate', 'confidence': confidence})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/analyze-screenshot', methods=['POST'])
def analyze_screenshot():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'Image required'}), 400
        
        file = request.files['image']
        if not file or file.filename == '':
            return jsonify({'error': 'Invalid file'}), 400
        
        # Default to medium confidence on vision
        if not vision_infer:
            return jsonify({'error': 'Vision model not available. Train the MobileNet model first.'}), 503
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
            file.save(tmp.name)
            tmp.close()  # Ensure file is closed before using
            vision_score = vision_infer.predict_score(tmp.name)
            os.unlink(tmp.name)
        
        # Default NLP is neutral (0.5) when only screenshot is analyzed
        result = weighted_ensemble(nlp_score=0.5, vision_score=vision_score, nlp_weight=0.3)
        confidence = 'HIGH' if result.final_score > 0.7 or result.final_score < 0.3 else 'MEDIUM'
        
        return jsonify({
            'nlp_score': 0.5,
            'vision_score': vision_score,
            'final_score': result.final_score,
            'final_verdict': result.verdict,
            'confidence': confidence
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=False, host='127.0.0.1', port=5000)
