import streamlit as st
import torch
import pandas as pd
import time
from transformers import AutoTokenizer, RobertaForSequenceClassification

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Consumer Complaint AI",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# CUSTOM CSS
# ==========================================================

st.markdown("""
<style>

html, body, [class*="css"]{
    font-family: "Segoe UI", sans-serif;
}

.stApp{
background:
linear-gradient(135deg,#0f172a 0%,#172554 50%,#1e3a8a 100%);
}

.block-container{
padding-top:2rem;
padding-bottom:2rem;
}

section[data-testid="stSidebar"]{
background:#0b1120;
}

section[data-testid="stSidebar"] *{
color:white;
}

.hero{

background:linear-gradient(90deg,#2563eb,#4f46e5);

padding:35px;

border-radius:20px;

color:white;

box-shadow:0px 10px 25px rgba(0,0,0,.35);

margin-bottom:30px;

}

.hero h1{

font-size:42px;

margin-bottom:5px;

}

.hero p{

font-size:18px;

opacity:.9;

}

.metric-card{

background:white;

padding:18px;

border-radius:18px;

box-shadow:0px 8px 20px rgba(0,0,0,.12);

text-align:center;

}

.metric-title{

font-size:15px;

color:gray;

}

.metric-value{

font-size:34px;

font-weight:bold;

color:#2563eb;

}

.result-card{

padding:30px;

border-radius:20px;

color:white;

text-align:center;

font-size:20px;

box-shadow:0px 10px 20px rgba(0,0,0,.30);

}

.stButton>button{

width:100%;

height:60px;

font-size:20px;

font-weight:bold;

border-radius:15px;

border:none;

background:linear-gradient(90deg,#2563eb,#4f46e5);

color:white;

transition:.3s;

}

.stButton>button:hover{

transform:scale(1.02);

}

.footer{

text-align:center;

color:#d1d5db;

margin-top:60px;

font-size:15px;

}

</style>
""", unsafe_allow_html=True)

# ==========================================================
# LOAD MODEL
# ==========================================================

@st.cache_resource
def load_model():

    MODEL_PATH = "saved_model"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    model = RobertaForSequenceClassification.from_pretrained(MODEL_PATH)

    model.eval()

    return tokenizer, model

tokenizer, model = load_model()

# ==========================================================
# LABELS
# ==========================================================

label_mapping = {

0:"Credit Card",

1:"Credit Reporting",

2:"Debt Collection",

3:"Mortgages & Loans",

4:"Retail Banking"

}

icons = {

0:"💳",

1:"📄",

2:"📞",

3:"🏠",

4:"🏦"

}

colors = {

0:"#2563eb",

1:"#16a34a",

2:"#dc2626",

3:"#9333ea",

4:"#ea580c"

}

# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.title("🏦 Consumer Complaint AI")

st.sidebar.success("Fine-tuned RoBERTa")

st.sidebar.divider()

st.sidebar.metric("Accuracy","89%")

st.sidebar.metric("Classes","5")

st.sidebar.metric("Dataset","162K")

st.sidebar.metric("Framework","PyTorch")

st.sidebar.metric("Library","Transformers")

st.sidebar.divider()

st.sidebar.info("""
This application classifies
financial customer complaints
into five categories using a
fine-tuned RoBERTa Transformer.
""")

st.sidebar.divider()

st.sidebar.markdown("### Categories")

st.sidebar.write("💳 Credit Card")

st.sidebar.write("📄 Credit Reporting")

st.sidebar.write("📞 Debt Collection")

st.sidebar.write("🏠 Mortgages & Loans")

st.sidebar.write("🏦 Retail Banking")

# ==========================================================
# HERO
# ==========================================================

st.markdown("""

<div class="hero">

<h1>🏦 Consumer Complaint AI</h1>

<p>
Intelligent Financial Complaint Classification using
a Fine-tuned RoBERTa Transformer Model.
</p>

</div>

""",unsafe_allow_html=True)

# ==========================================================
# DASHBOARD
# ==========================================================

c1,c2,c3,c4 = st.columns(4)

with c1:

    st.markdown("""

<div class="metric-card">

<div class="metric-title">Accuracy</div>

<div class="metric-value">89%</div>

</div>

""",unsafe_allow_html=True)

with c2:

    st.markdown("""

<div class="metric-card">

<div class="metric-title">Model</div>

<div class="metric-value">RoBERTa</div>

</div>

""",unsafe_allow_html=True)

with c3:

    st.markdown("""

<div class="metric-card">

<div class="metric-title">Classes</div>

<div class="metric-value">5</div>

</div>

""",unsafe_allow_html=True)

with c4:

    st.markdown("""

<div class="metric-card">

<div class="metric-title">Dataset</div>

<div class="metric-value">162K</div>

</div>

""",unsafe_allow_html=True)

st.write("")

st.subheader("📝 Enter Customer Complaint")

st.caption("Describe the customer's complaint below. The AI model will predict the most likely complaint category.")

complaint = st.text_area(

"",

height=220,

placeholder="""
Example:

I made my credit card payment on time, but the bank still charged me a late fee.
Customer service has ignored my complaint for several weeks.
"""

)

predict = st.button("🔍 Analyze Complaint",use_container_width=True)

# ==========================================================
# PREDICTION
# ==========================================================

if predict:

    if complaint.strip() == "":

        st.warning("⚠️ Please enter a customer complaint.")

    else:

        with st.spinner("🧠 AI is analyzing your complaint..."):

            time.sleep(1)

            inputs = tokenizer(
                complaint,
                truncation=True,
                padding=True,
                max_length=256,
                return_tensors="pt"
            )

            with torch.no_grad():

                outputs = model(**inputs)

                probabilities = torch.softmax(outputs.logits, dim=1)

                prediction = torch.argmax(probabilities, dim=1).item()

                confidence = probabilities.max().item() * 100

        
        st.write("")
        st.subheader("🎯 Prediction Result")

        st.markdown(
            f"""
            <div class="result-card"
            style="background:{colors[prediction]};">

            <h1 style="font-size:70px;">
            {icons[prediction]}
            </h1>

            <h2 style="font-size:34px;">
            {label_mapping[prediction]}
            </h2>

            <h3>
            Confidence: {confidence:.2f}%
            </h3>

            </div>
            """,
            unsafe_allow_html=True
        )

        st.write("")

        if confidence >= 95:

            st.success("✅ The model is highly confident in this prediction.")

        elif confidence >= 80:

            st.info("ℹ️ The model is confident in this prediction.")

        else:

            st.warning("⚠️ The prediction confidence is relatively low.")

        st.write("")

        st.subheader("📊 Prediction Probabilities")

        for i, probability in enumerate(probabilities[0]):

            left, right = st.columns([5,1])

            with left:

                st.write(
                    f"{icons[i]}  {label_mapping[i]}"
                )

                st.progress(float(probability))

            with right:

                st.write(
                    f"{probability.item()*100:.2f}%"
                )

# ==========================================================
# MODEL PERFORMANCE
# ==========================================================

st.divider()

st.subheader("📈 Model Performance")

c1,c2,c3,c4 = st.columns(4)

with c1:
    st.metric(
        "SimpleRNN",
        "72%"
    )

with c2:
    st.metric(
        "LSTM",
        "85%"
    )

with c3:
    st.metric(
        "GRU",
        "85%"
    )

with c4:
    st.metric(
        "RoBERTa ⭐",
        "89%"
    )

st.write("")

comparison = pd.DataFrame({

"Model":[
"SimpleRNN",
"LSTM",
"GRU",
"RoBERTa"
],

"Accuracy":[
72,
85,
85,
89
]

})

st.bar_chart(
comparison.set_index("Model")
)

# ==========================================================
# EXAMPLES
# ==========================================================

st.divider()

st.subheader("💡 Example Complaints")

col1,col2 = st.columns(2)

with col1:

    st.info("""
💳 **Credit Card**

"I made my payment on time but
was charged a late fee."
""")

    st.info("""
📞 **Debt Collection**

"A debt collector keeps calling
me about a debt that isn't mine."
""")

with col2:

    st.info("""
🏠 **Mortgage**

"My mortgage application was
rejected without explanation."
""")

    st.info("""
🏦 **Retail Banking**

"My checking account was frozen
without notifying me."
""")

# ==========================================================
# ABOUT MODEL
# ==========================================================

st.divider()

st.subheader("🤖 About this AI")

st.write("""
This application uses a **fine-tuned RoBERTa Transformer**
trained on over **162,000 consumer complaints**.

The model classifies complaints into:

- 💳 Credit Card
- 📄 Credit Reporting
- 📞 Debt Collection
- 🏠 Mortgages & Loans
- 🏦 Retail Banking

Compared with the deep learning models
(SimpleRNN, LSTM and GRU), RoBERTa achieved
the highest overall accuracy on the test dataset.
""")

# ==========================================================
# FOOTER
# ==========================================================

st.markdown("""

<div class="footer">

<hr>

<h3>🏦 Consumer Complaint AI</h3>

Fine-tuned RoBERTa Transformer

<br><br>

Built with ❤️ using

<b>PyTorch</b> •
<b>Hugging Face</b> •
<b>Transformers</b> •
<b>Streamlit</b>

<br><br>

Consumer Complaint Classification Project

</div>

""", unsafe_allow_html=True)