Here's your updated project README content with the website link included:

---

# 🔬 Skin Lesion Classifier

A deep learning application for classifying skin lesions using PyTorch and Streamlit, with GradCAM explainability.

🌐 **Live Demo**: [Skin Lesion Classifier Web App](https://skin-lesion-classifier-cwk896vbcp7kbzkzdngtjj.streamlit.app/)

---

## 🚀 Features

* **Deep Learning Classification**: ResNet50-based model for 7 skin lesion types
* **Interactive Web Interface**: Streamlit-powered user-friendly interface
* **Explainable AI**: GradCAM visualizations for model interpretability
* **Production Ready**: Docker support, CI/CD, and cloud deployment ready

---

## 🧬 Skin Lesion Types

* `akiec` - Actinic keratoses and intraepithelial carcinoma
* `bcc` - Basal cell carcinoma
* `bkl` - Benign keratosis-like lesions
* `df` - Dermatofibroma
* `mel` - Melanoma
* `nv` - Melanocytic nevi
* `vasc` - Vascular lesions

---

## ⚡ Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/luffy1501/skin-lesion-classifier.git
cd skin-lesion-classifier
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download Data

```bash
python data/download_data.py
```

### 4. Train Model

```bash
python train.py --metadata_path data/HAM10000_metadata.csv --image_dirs data/ham10000_images_part_1 data/ham10000_images_part_2
```

### 5. Run Application

```bash
streamlit run app.py
```

---

## 🐳 Docker Deployment

### Build and Run

```bash
docker-compose up --build
```

### Training with Docker

```bash
docker-compose --profile training up model-trainer
```

---

## ☁️ Cloud Deployment

### ✅ Streamlit Cloud

1. Push to GitHub
2. Connect repository to Streamlit Cloud
3. Deploy automatically

🔗 **Live App**: [Visit Here](https://skin-lesion-classifier-cwk896vbcp7kbzkzdngtjj.streamlit.app/)

### ☁️ Heroku

```bash
heroku create your-app-name
heroku container:push web
heroku container:release web
```

### ☁️ AWS/GCP/Azure

See `docs/DEPLOYMENT.md` for detailed instructions.

---

## 🗂️ Project Structure

```
skin-lesion-classifier/
├── app.py              # Streamlit application
├── train.py            # Model training
├── models/             # Model architectures
├── data/               # Data handling
├── utils.py            # Utilities
├── requirements.txt    # Dependencies
├── Dockerfile          # Container setup
└── README.md           # Documentation
```

---

## 📊 Model Performance

* **Accuracy**: \~85% on validation set
* **Classes**: 7 skin lesion types
* **Architecture**: ResNet50 with custom classifier
* **Training Data**: HAM10000 dataset (10,015 images)

---

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

---

## 📜 License

This project is licensed under the MIT License - see the `LICENSE` file for details.

---

## ⚠️ Disclaimer

This application is for **educational and research purposes only**. It should **not** be used as a substitute for professional medical diagnosis or treatment.

---

## 🙏 Acknowledgments

* **HAM10000 Dataset**: Tschandl, P., Rosendahl, C. & Kittler, H.
* **PyTorch** team for the deep learning framework
* **Streamlit** team for the web application framework

---

Let me know if you'd like to auto-generate a badge or image banner for the project or make this README stylish with Markdown enhancements!
