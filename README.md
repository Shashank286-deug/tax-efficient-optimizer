# 🏦 Tax-Efficient Portfolio Optimizer

![GitHub License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.9+-blue.svg?logo=python&logoColor=white)
![Build](https://img.shields.io/badge/status-active-brightgreen.svg)
![Fintech](https://img.shields.io/badge/sector-Fintech-orange.svg)

---

### 🌟 Overview
Most portfolio tools ignore the "tax man." This project bridges the gap between **Modern Portfolio Theory** and **Tax Strategy**. By simulating "Tax Drag," it helps investors see their true, post-tax wealth potential.

> **Why this matters:** A portfolio with an 8% return and high turnover can often underperform a 6% "tax-lazy" portfolio over 20 years.

---

### 🚀 Key Capabilities

| Feature | Tech Used | Impact |
| :--- | :---: | :--- |
| **Monte Carlo Simulation** | `NumPy` | Runs 5,000+ iterations to find the Efficient Frontier. |
| **Tax Drag Engine** | `Pandas` | Adjusts returns for capital gains and turnover leakage. |
| **Sentiment Overlay** | `TextBlob` | Scores 24h news headlines to validate technical picks. |
| **Interactive UI** | `Streamlit` | Real-time parameter tuning for diverse tax brackets. |

---

### 🛠 The Tech Stack
<p align="left">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" />
  <img src="https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white" />
</p>

---

### 📈 Financial Logic
The optimizer calculates the **Sharpe Ratio** using the formula:

$$\text{Sharpe Ratio} = \frac{R_p - R_f}{\sigma_p}$$

Where $R_p$ is the **Tax-Adjusted Return**. This ensures your portfolio isn't just "winning" on paper, but winning in your bank account.



---

### 📂 Project Structure
```text
├── app.py              # Main Streamlit interface
├── engine.py           # Quantitative math & simulations
├── sentiment.py        # News API & NLP processing
├── requirements.txt    # Dependencies
└── assets/             # Branding & screenshots
