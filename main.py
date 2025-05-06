import streamlit as st
from PIL import Image
import os
import torch
import math
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


def preprocess_image(img: Image.Image):
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
use_predefined_set = st.radio(
    "Choose an option:", ["Predefined Set", "Upload Image and Enter Question"])

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
    # Save the selected set in session state to keep track of it
    if 'previous_set' not in st.session_state:
        st.session_state.previous_set = None
    
    selected_set = st.selectbox("Choose a predefined set:", list(image_sets.keys()))
    
    # Check if the selected set is different from the previous one
    if st.session_state.previous_set != selected_set:
        # Reset current page to 1
        st.session_state.current_page = 1
        st.session_state.previous_set = selected_set
    
    st.divider()
    image_folder = image_sets[selected_set] 
    question_file = question_sets[selected_set]
    
    # Read the list of images from the folder
    image_paths = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.endswith(('jpg', 'jpeg', 'png'))]
    images = [Image.open(img_path).convert("RGB") for img_path in image_paths]
    questions = take_questions(question_file)
    
    # Read answers if available
    if selected_set in answer_sets:
        answer_file = answer_sets[selected_set]
        answers = take_questions(answer_file)
    else:
        answers = [""] * len(questions)
    
    # Calculate the number of pages based on the selected dataset
    items_per_page = 5
    total_items = len(images)
    total_pages = math.ceil(total_items / items_per_page)
    
    # Save the current page in session state to keep track of it
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 1
    
    # View the current page number and total pages
    if total_pages > 1:
        st.markdown(f"### Result Table (Page {st.session_state.current_page} of {total_pages})")
    else:
        st.markdown("### Result Table")
    
    # Calculate start and end index for the current page
    start_idx = (st.session_state.current_page - 1) * items_per_page
    end_idx = min(start_idx + items_per_page, total_items)
    
    # Get data for the current page
    current_images = images[start_idx:end_idx]
    current_questions = questions[start_idx:end_idx]
    current_answers = answers[start_idx:end_idx]
    
    for img, ques, gt_ans in zip(current_images, current_questions, current_answers):
        img_tensor = preprocess_image(img).to(device)
        quest_vector = tokenize(ques, 20).to(device)

        with torch.no_grad():
            logits, _ = model(img_tensor, quest_vector.unsqueeze(0))
            pred = torch.argmax(logits, dim=1).item()
            answer = "yes" if pred == 1 else "no"

        correct = (answer.lower() == gt_ans.lower())
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
            bg_color = "#d4edda" if correct else "#f8d7da"  # light green or light red
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
    
    # Add navigation buttons if there are more than 1 page
    if total_pages > 1:
        # Fucntion to change page
        def change_page(page_num):
            st.session_state.current_page = page_num
        
        col1, col2, col3 = st.columns([1, 0.25, 1])
        with col2:
            st.write("")
            st.write("")
            container = st.container()
            with container:
                cols = st.columns(total_pages)
                for i in range(1, total_pages + 1):
                    with cols[i-1]:
                        btn_style = "primary" if i == st.session_state.current_page else "secondary"
                        st.button(f"{i}", key=f"page_{i}", on_click=change_page, args=(i,), type=btn_style)

else:
    with st.form("my_form"):
        uploaded_image = st.file_uploader(
            "Upload Image", type=["jpg", "jpeg", "png"])
        question = st.text_area("Enter a question about the picture:",
                                placeholder="Enter what you want to know here and I will show you!",
                                height=100)
        submitted = st.form_submit_button("Ask")

        if submitted and uploaded_image and question:
            st.markdown(
                "<h4 style='color: green; font-weight: bold;'>This is my answer!</h4>",
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
            st.write("Hope I helped you find out what you're looking for!")
        else:
            st.warning("You are missing something, please enter full input!")
