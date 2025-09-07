# 🤖 Vision Question Answering (VQA)

This repository contains the source code and deployment for a **Vision Question Answering** system.  
In this project, I focus only on **Yes/No questions**.

---

## 📌 Project Versions

Results of 3 versions can be seen here:

| Version      | Description | Notebook Link |
|--------------|-------------|---------------|
| **TTCS 1**   | Use **Word-based Tokenization** | [Open in Colab ▶️](https://drive.google.com/file/d/1ZCzB-z6afTQSf20TW8cT-1p-vJFJMjjT/view?usp=sharing) |
| **TTCS 2**   | Use **Subword-based Tokenization** | [Open in Colab ▶️](https://drive.google.com/file/d/1R_uZf-KQ6sQoqkFE7I1OyNYvruhplzKV/view?usp=sharing) |
| **TTCS Final** | Apply **Knowledge Distillation** to enhance model performance | [Open in Colab ▶️](https://drive.google.com/file/d/1OLUwuw40nzxglsWtZsymBdz2jwYTm93X/view?usp=sharing) |

---

## 🚀 Live Demo

🎯 Try it out here:  
👉 **[VQA Deployment on Streamlit](https://visiondemo-oweijr291midfr54y.streamlit.app/)**  

---

## ⚙️ Tech Stack

- 🐍 Python 
- 📦 PyTorch
- 🌐 Streamlit (for deployment)  
- 🔤 Word & Subword Tokenization techniques  
- 🎯 Knowledge Distillation  

---

## 📒 Project Notes

- ❌ This project **does not use any pretrained models**.  
- All models are trained from scratch to better analyze the effectiveness of tokenization methods and knowledge distillation.