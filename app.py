import streamlit as st
import torch
import numpy as np
from torch import nn
from streamlit_lottie import st_lottie
import requests
import json

# Load Lottie animation from local JSON file
def load_lottiefile(filepath: str):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


# -----------------------
# Load Model & Vocab
# -----------------------
chars = sorted(list(set(open("witchy_incantations_200.txt", "r", encoding="utf-8").read().lower())))
char2idx = {ch: i for i, ch in enumerate(chars)}
idx2char = {i: ch for i, ch in enumerate(chars)}
vocab_size = len(chars)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CharLSTM(nn.Module):
    def __init__(self, vocab_size, embed_size=128, hidden_size=256, num_layers=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden):
        x = self.embed(x)
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out[:, -1, :])
        return out, hidden

    def init_hidden(self, batch_size, device):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device),
                torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device))

# Load trained model
model = CharLSTM(vocab_size).to(device)
model.load_state_dict(torch.load("char_lstm_epoch30.pth", map_location=device))
model.eval()

# -----------------------
# Generation function
# -----------------------
def generate_text_rnn(model, start_str, gen_len=300, temperature=0.8):
    input_seq = torch.tensor([char2idx[c] for c in start_str.lower() if c in char2idx]).unsqueeze(0).to(device)

    hidden = model.init_hidden(1, device)
    generated = start_str

    for _ in range(gen_len):
        output, hidden = model(input_seq, hidden)
        probs = torch.softmax(output / temperature, dim=-1).detach().cpu().numpy().squeeze()
        char_idx = np.random.choice(len(chars), p=probs)
        generated += idx2char[char_idx]
        input_seq = torch.tensor([[char_idx]]).to(device)

    return generated

# -----------------------
# Lottie animation helper
# -----------------------
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# -----------------------
# Streamlit UI
# -----------------------
# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config(page_title="🔮 Witchy Spell Generator", page_icon="🕯️", layout="wide")
st.markdown(
    """
    <style>
    .stApp {
        background-image:url('https://images.unsplash.com/photo-1508931133503-b1944a4ecdd5?q=80&w=2070&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D'); 
        background-size: cover;
        color: white;
    }
    .stButton>button {
        background-color: #5c0a0a;
        color: white;
        font-size: 18px;
    }
    </style>
    """, unsafe_allow_html=True
)

st.title("🔮 Witchy Spell Generator")
st.markdown("Generate your own magical incantations powered by an AI RNN trained on witchy poems. 🕯️✨")

# Lottie Animation
halloween_animation = load_lottiefile("f.json")  # path to your downloaded JSON
if halloween_animation:
    st_lottie(halloween_animation, speed=1, height=250)

# -----------------------
# User input for spell
# -----------------------
# -----------------------
# User input for their problem
# -----------------------
user_problem = st.text_area("Tell the AI about your problem or what you wish for:", height=100, placeholder="I want love to bloom...")

length = st.slider("Spell length (characters):", 100, 500, 300)
temperature = st.slider("Creativity (temperature):", 0.3, 1.0, 0.7, 0.05)

if st.button("✨ Generate Spell"):
    if not user_problem.strip():
        st.warning("Please describe your problem or wish!")
    else:
        with st.spinner("Brewing your personalized spell... 🕯️"):
            # Optionally, prepend a magical phrase or emoji
            start_str = f"✨ Spell for: {user_problem}\n"
            spell = generate_text_rnn(model, start_str=start_str, gen_len=length, temperature=temperature)
        st.text_area("Your Personalized Incantation", value=spell, height=300)
        st.balloons()
