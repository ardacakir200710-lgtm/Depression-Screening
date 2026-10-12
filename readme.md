# Psychological Risk Screening Model

This project is a Machine Learning API that calculates users' depression risk levels based on the DASS-42 psychometric test.

## Technologies Used
* **Backend:** Python, FastAPI, Uvicorn
* **Machine Learning:** Scikit-learn, Pandas, Joblib (Logistic Regression - Multi-class)
* **Frontend:** HTML/JS (Vanilla)

## How to Run
1. Clone this repository to your local machine.
2. Install the required dependencies: 
   ```bash
   pip install -r requirements.txt
   ```
3. Start the FastAPI server: 
   ```bash
   python -m uvicorn api:app --reload
   ```
4. Open the `index.html` file in your web browser to take the test and see the real-time analysis.

## Disclaimer
*This tool is intended for educational and preliminary screening purposes only. It is not a diagnostic tool. If you are experiencing psychological distress, please consult a qualified healthcare professional.*