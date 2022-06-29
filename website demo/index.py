from flask import Flask, render_template, request
import pickle
import numpy as np

model = pickle.load(open('xgboost.pkl', 'rb'))
app = Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data_arr  = [0 for i in range(0, 82)]
    data_arr[0] = float(request.form['loan_amnt'])
    
    data_arr[1] = float(request.form['int_rate'])
    data_arr[2] = float(request.form['installment'])
    data_arr[3] = float(request.form['annual_inc'])
    data_arr[4] = float(request.form['dti'])
    data_arr[5] = float(request.form['open_acc'])
    data_arr[6] = float(request.form['pub_rec'])
    data_arr[7] = float(request.form['revol_bal'])
    data_arr[8] = float(request.form['revol_util'])
    data_arr[9] = float(request.form['total_acc'])
    categorical_attr = []
    categorical_attr.append(int(request.form['pub_rec_bankruptcies']))
    categorical_attr.append(int(request.form['sub_grade']))
    categorical_attr.append(int(request.form['term']))
    categorical_attr.append(int(request.form['home_ownership']))
    categorical_attr.append(int(request.form['home_ownership']))
    categorical_attr.append(int(request.form['verification_status']))
    categorical_attr.append(int(request.form['initial_list_status']))
    categorical_attr.append(int(request.form['application_type']))
    categorical_attr.append(int(request.form['mort_acc_amnt']))
    categorical_attr.append(int(request.form['zip_code']))

    for cate in categorical_attr:
        if cate != -1:
            data_arr[cate-1] = 1
    arr = np.array([data_arr])
    print(arr)
    pred = model.predict(arr)
    return render_template('after.html', data=pred)
if (__name__):
    app.run(debug=True)






# ['loan_amnt', 'int_rate', 'installment', 'annual_inc', 'loan_status',
#        'dti', 'open_acc', 'pub_rec', 'revol_bal', 'revol_util', 'total_acc',
#        'pub_rec_bankruptcies', 'term_ 60 months', 'sub_grade_A2',
#        'sub_grade_A3', 'sub_grade_A4', 'sub_grade_A5', 'sub_grade_B1',
#        'sub_grade_B2', 'sub_grade_B3', 'sub_grade_B4', 'sub_grade_B5',
#        'sub_grade_C1', 'sub_grade_C2', 'sub_grade_C3', 'sub_grade_C4',
#        'sub_grade_C5', 'sub_grade_D1', 'sub_grade_D2', 'sub_grade_D3',
#        'sub_grade_D4', 'sub_grade_D5', 'sub_grade_E1', 'sub_grade_E2',
#        'sub_grade_E3', 'sub_grade_E4', 'sub_grade_E5', 'sub_grade_F1',
#        'sub_grade_F2', 'sub_grade_F3', 'sub_grade_F4', 'sub_grade_F5',
#        'sub_grade_G1', 'sub_grade_G2', 'sub_grade_G3', 'sub_grade_G4',
#        'sub_grade_G5', 'home_ownership_MORTGAGE', 'home_ownership_NONE',
#        'home_ownership_OTHER', 'home_ownership_OWN', 'home_ownership_RENT',
#        'verification_status_Source Verified', 'verification_status_Verified',
#        'purpose_credit_card', 'purpose_debt_consolidation',
#        'purpose_educational', 'purpose_home_improvement', 'purpose_house',
#        'purpose_major_purchase', 'purpose_medical', 'purpose_moving',
#        'purpose_other', 'purpose_renewable_energy', 'purpose_small_business',
#        'purpose_vacation', 'purpose_wedding', 'initial_list_status_w',
#        'application_type_INDIVIDUAL', 'application_type_JOINT',
#        'mort_acc_amnt_1-6', 'mort_acc_amnt_7-14', 'mort_acc_amnt_more-15',
#        'mort_acc_amnt_unknown', 'zip_code_05113', 'zip_code_11650',
#        'zip_code_22690', 'zip_code_29597', 'zip_code_30723', 'zip_code_48052',
#        'zip_code_70466', 'zip_code_86630', 'zip_code_93700']

