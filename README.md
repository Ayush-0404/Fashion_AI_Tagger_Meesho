
---

## 🌐 Live Demo

- **Frontend:** [https://your-frontend-url.onrender.com](https://your-frontend-url.onrender.com)
- **Backend API:** [https://fashion-ai-tagger-meesho.onrender.com](https://fashion-ai-tagger-meesho.onrender.com)

---

## 📸 How It Works

1. **Upload Images:**  
   Drag & drop or select product images.
2. **Analyze:**  
   The app sends images to the backend, which runs the AI model and returns predicted attributes.
3. **View Results:**  
   Instantly see detailed attribute predictions for each image, with a preview and attribute badges.
4. **Export Reports:**  
   Download all results as a PDF or CSV for easy sharing or record-keeping.

---

## 🚀 Quick Start

### **Backend**
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

### **Frontend**
```bash
cd Frontend/meesho-version-re2
npm install
npm run dev
```

---

## ⚙️ Deployment

### **Backend (Render)**
- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn main:app --host 0.0.0.0 --port 10000`
- Python version: 3.9+
- Persistent files: Place all model/data files in the repo.

### **Frontend (Render)**
- Build command: `npm install && npm run build`
- Publish directory: `dist`
- Root directory: `Frontend/meesho-version-re2`

### **CORS**
- Update CORS in `main.py` to allow your frontend URL.

---

## 📂 Project Structure
