import pandas as pd

def load_email_data(path="Emails.csv"):
    df = pd.read_csv(path)
    return df
