import streamlit as st
from PIL import Image
import os
import torch
from torchvision import transforms
from model.my_tokenizer import tokenize
from model.student_model import load_student_model

st.set_page_config(
    page_title="VQA",
    page_icon="static/VQA_logo.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.image("static/VQA.png", width=200)

st.title("VQA Model Deployment - YES/NO Questions")

st.divider()

# Set seed
SEED = 42
torch.manual_seed(SEED)

def preprocess_image(img : Image.Image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    return transform(img).unsqueeze(0)

# Classes
classes = {0: "no", 1: "yes"}

# Load model
n_classes = len(classes)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = load_student_model(device)

st.subheader("Select predefined set or Upload and enter your own")
use_predefined_set = st.radio("Choose an option:", ["Predefined Set", "Upload Image and Enter Question"])

image_sets = {
    "Set 5 images": "static/set5/picture5",
    "Set 10 images": "static/set10/picture10",
    "Set 15 images": "static/set15/picture15"
}

question_sets = {
    "Set 5 images": "static/set5/question5.txt",
    "Set 10 images": "static/set10/question10.txt",
    "Set 15 images": "static/set15/question15.txt"
}

answer_sets = {
    "Set 5 images": "static/set5/answer5.txt",
    "Set 10 images": "static/set10/answer10.txt",
    "Set 15 images": "static/set15/answer15.txt"
}

def take_questions(file_path):
    with open(file_path, 'r') as file:
        questions = file.readlines()
    return [question.strip() for question in questions]

def take_answers(file_path):
    with open(file_path, 'r') as file:
        answers = file.readlines()
    return [answer.strip().lower() for answer in answers]

if use_predefined_set == "Predefined Set":
    selected_set = st.selectbox("Choose a predefined set:", list(image_sets.keys()))
    st.divider()
    image_folder = image_sets[selected_set] 
    question_file = question_sets[selected_set]
    answer_file = answer_sets[selected_set]

    image_paths = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.endswith(('jpg', 'jpeg', 'png'))]
    images = [Image.open(img_path).convert("RGB") for img_path in image_paths]
    questions = take_questions(question_file)
    answers = take_answers(answer_file)


    st.markdown("### Result Table")
    
    for img, ques, gt_ans in zip(images, questions, answers):
        img_tensor = preprocess_image(img).to(device)
        quest_vector = tokenize(ques, 20).to(device)

        with torch.no_grad():
            logits, _ = model(img_tensor, quest_vector.unsqueeze(0))
            pred = torch.argmax(logits, dim=1).item()
            answer = "yes" if pred == 1 else "no"

        correct = (answer == gt_ans)
        color = "green" if correct else "red"

        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            st.image(img, width=300)
        with col2:
            st.markdown(
                f"""
                <div style="
                    background-color: #fff3cd;
                    border: 1px solid #ffeeba;
                    border-radius: 5px;
                    padding: 10px;
                    display: inline-block;
                ">
                    <strong>Q:</strong> {ques}
                </div>
                """, unsafe_allow_html=True
            )

        with col3:
            bg_color = "#d4edda" if correct else "#f8d7da"  # xanh nhạt hoặc đỏ nhạt
            border_color = "#c3e6cb" if correct else "#f5c6cb"

            st.markdown(
                f"""
                <div style="
                    background-color: {bg_color};
                    border: 1px solid {border_color};
                    border-radius: 5px;
                    padding: 10px;
                    display: inline-block;
                ">
                    <strong>A:</strong> {answer.capitalize()}
                </div>
                """, unsafe_allow_html=True)

else:
    with st.form("my_form"):
        uploaded_image = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
        question = st.text_area("Enter a question about the picture:",
                                placeholder="Enter what you want to know here and I will show you!",
                                height=100)
        submitted = st.form_submit_button("Ask")

        if submitted and uploaded_image and question:
            st.markdown(
                "<h4 style='color: green; font-weight: bold;'>This is my answer for you, babe!</h4>",
                unsafe_allow_html=True
            )

            image = Image.open(uploaded_image).convert("RGB")
            img_tensor = preprocess_image(image).to(device)
            quest_vector = tokenize(question, 20).to(device)

            with torch.no_grad():
                logits, _ = model(img_tensor, quest_vector.unsqueeze(0))
                pred = torch.argmax(logits, dim=1).item()

            st.image(image, caption="Uploaded Image", width=300)
            st.subheader(f"Question: {question}")
            st.success("Answer: " + ("Yes" if pred == 1 else "No"))
            st.write("Hope I will help you find out the answer!!!")
        else:
            st.warning("You are missing something, please enter full input!")