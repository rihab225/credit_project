# Credit Card Approval Prediction

## Project Overview
The Credit Card Approval Prediction project is a Machine Learning and Flask-based web application that predicts whether a credit card application is likely to be approved or rejected based on applicant details. The project uses data preprocessing, machine learning models, and a user-friendly web interface to provide instant predictions.

## Features
- Data preprocessing and cleaning
- Feature engineering
- Multiple machine learning models
  - Logistic Regression
  - Decision Tree
  - Random Forest
- Model comparison
- Flask web application
- User-friendly HTML interface
- Real-time prediction

## Technologies Used
- Python
- Flask
- NumPy
- Pandas
- Scikit-learn
- Pickle
- HTML
- CSS

## Project Structure

```
Credit_Card_Approval/
│
├── app.py
├── data_preprocessing.py
├── decision_tree_model.py
├── logistic_regression_model.py
├── random_forest_model.py
├── save_model.py
├── model.pkl
├── final_dataset.csv
├── templates/
│   ├── home.html
│   ├── index.html
│   └── result.html
├── README.md
└── requirements.txt
```

## Dataset
The project uses a credit card approval dataset containing applicant information such as:
- Gender
- Income
- Education
- Employment Status
- Family Members
- Housing Type
- Car Ownership
- Property Ownership
- Age
- Occupation

## Machine Learning Models
The following algorithms were implemented:
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier

Among these, the Random Forest model was selected and saved as `model.pkl` for deployment.

## Installation

### Clone the Repository

```bash
git clone https://github.com/your-username/credit_project.git
```

### Navigate to the Project Folder

```bash
cd credit_project
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Application

```bash
python app.py
```

The application will start at:

```
http://127.0.0.1:5000/
```

## How It Works

1. User opens the home page.
2. Clicks on Predict.
3. Enters applicant details.
4. Flask sends the data to the trained model.
5. The model predicts:
   - Credit Card Approved
   - Credit Card Rejected
6. The result is displayed on the web page.

## Future Enhancements
- Improve prediction accuracy
- Deploy the application online
- Add user authentication
- Connect with a database
- Visualize prediction statistics

## Author

BITNI RIHAAB KHANAM

## License

This project is developed for educational and academic purposes.
