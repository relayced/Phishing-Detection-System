# Phishing-Detection-System

Multi-modal phishing detection project using:
- NLP on email text (TF-IDF + Logistic Regression baseline, optional BERT-tiny)
- Vision model on website screenshots (MobileNetV2)
- Weighted ensemble fusion
- Streamlit web app for threat analysis

## Completed Phases

**Phase 1: Data Acquisition**
	- Downloaded phishing email dataset
	- Collected website screenshots

**Phase 2: Data Preprocessing**
	- Cleaned, split, and labeled email and image data

**Phase 3: NLP Model Training**
	- TF-IDF + Logistic Regression (optional BERT-tiny)

**Phase 4: Vision Model Training**
	- MobileNetV2 on screenshots

**Phase 5: Ensemble Fusion**
	- Combined NLP and vision scores for final verdict

**Phase 6: Web App Deployment**
	- Flask app with email and screenshot analysis UI

