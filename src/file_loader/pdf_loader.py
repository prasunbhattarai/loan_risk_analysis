import pypdf
import re
import os
import pandas as pd
def load_pdf(file_path):
    with open(file_path, 'rb') as file:
        reader = pypdf.PdfReader(file)
        text = ''
        for page in reader.pages:
            text += page.extract_text()

    clean_text = re.sub(r'[%$()]', '', text)
    return clean_text

def extract_text(pattern, text):
    match = re.search(pattern, text)
    if match:
        value = match.group(1).replace(",", "").strip()
        try:
            return float(value)
        except ValueError:
            return value  
    return None


def normalize_percentage(value):
    if value is None:
        return None
    return value / 100 
def normalize_binary(value):
    if value is None:
        return None
    return 1 if value > 0 else 0



feature_text = {
    "LoanAmount": "Requested Loan Amount",
    "LoanDuration": "Loan Duration Months",
    "Age": "Age",
    "InterestRate": "Proposed Interest Rate",
    "BaseInterestRate": "Base Interest Rate",
    "AnnualIncome": "Annual Income",
    "MonthlyIncome": "Monthly Income",
    "Experience": "Years of Experience",
    "CreditScore": "Credit Score",
    "LengthOfCreditHistory": "Credit History Years",
    "NetWorth": "Net Worth",
    "TotalAssets": "Total Assets",
    "PaymentHistory": "Overall Payment History",
    "CreditCardUtilizationRate": "Credit Card Utilization Rate",
    "BankruptcyHistory": "Bankruptcy History",
    "PreviousLoanDefaults": "Previous Loan Defaults",
    "EducationLevel": "Highest Education Level",
    "MonthlyDebt": "Total Monthly Debt Service",
    "TotalOutstandingDebt": "Total Outstanding Debt"

}

def prediction_data(file_path):
    text = load_pdf(file_path)

    data ={}

    for col, label in feature_text.items():
        if col in ["EducationLevel"]:
            pattern = rf"{label}\s*([A-Za-z\- ]+)"
            data[col] = extract_text(pattern, text)
        else:
            data[col] = extract_text(rf"{label}\s*([\d.,]+)", text)
    return data


def compute_features(data):
    new_data = data.copy()
    new_data["DebtToIncomeRatio"] = data["MonthlyDebt"] / data["MonthlyIncome"] if data["MonthlyIncome"] else None
    new_data["TotalDebtToIncomeRatio"] = data["TotalOutstandingDebt"] / data["AnnualIncome"] if data["AnnualIncome"] else None
    DROP_COLS = ["MonthlyDebt", "TotalOutstandingDebt"]

    for col in DROP_COLS:
        new_data.pop(col, None)
    new_data["BaseInterestRate"] = normalize_percentage(new_data.get("BaseInterestRate"))
    new_data["CreditCardUtilizationRate"] = normalize_percentage(new_data.get("CreditCardUtilizationRate"))
    new_data["BankruptcyHistory"] = normalize_binary(new_data.get("BankruptcyHistory"))
    new_data["PreviousLoanDefaults"] = normalize_binary(new_data.get("PreviousLoanDefaults"))
    prediction_data = pd.DataFrame({k: [v] for k, v in new_data.items()})
    
    return prediction_data

        

