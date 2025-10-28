import streamlit as st
import joblib
import json
import pandas as pd
import numpy as np

# Load models และ feature names
@st.cache_resource
def load_models():
    try:
        model1 = joblib.load('model_stage1_xgboost.pkl')
        model2 = joblib.load('model_stage2_logistic.pkl')
        model3 = joblib.load('model_stage3_logistic.pkl')
        
        with open('feature_names.json', 'r') as f:
            feature_names = json.load(f)
        
        return model1, model2, model3, feature_names
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None

# ข้อมูลตัวอย่างจริงที่รู้เฉลย
def get_real_samples():
    """ข้อมูลจริงจาก dataset แยกตาม AKI class"""
    samples = {
        0: [  # No AKI
            [22.0,1.0,1,1.0,70.0,170.0,24.221453287197235,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,2.0,4.0,1.0,1.0,120.0,90.0,1,80.0,1.0,350.0,350.0,0.0,0.0,0.0,30.0,80.0,240.0,0.0,0.0,0.0,0.0,90.0,52.0,64.66666666666667,0.0,1.0,14.8,4.220000000000001,0.8,136.48,0.0,7.212963],
            [35.0,1.0,2,1.0,75.0,180.0,23.148148148148145,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,5.0,1.0,1.0,260.0,240.0,0,0.0,1.0,5095.0,2100.0,1400.0,700.0,895.0,2100.0,730.0,2265.0,15.0,0.0,0.0,5.0,80.0,30.0,46.666666666666664,0.0,1.0,13.4,3.8,0.9,108.42,2.0,1.376963],
            [51.0,0.0,1,0.0,37.0,145.0,17.598097502972653,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,2.0,1.0,1.0,145.0,90.0,1,80.0,0.0,800.0,800.0,0.0,0.0,0.0,100.0,300.0,400.0,9.0,0.0,0.0,5.0,85.0,45.0,58.33333333333333,0.0,0.0,9.1,4.2,0.8,85.38,0.0,10.67949],
            [60.0,1.0,2,1.0,60.0,165.0,22.03856749311295,1.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,2.0,4.0,1.0,2.0,175.0,140.0,0,0.0,1.0,2426.0,1300.0,0.0,0.0,1225.0,200.0,70.0,2156.0,0.0,957.5,0.0,5.0,75.0,48.0,57.0,0.0,0.0,11.7,2.1,0.9,92.51,3.0,6.233871],
            [69.0,1.0,1,0.0,56.0,165.0,20.569329660238754,1.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,1.0,2.0,1.0,1.0,200.0,140.0,1,130.0,0.0,1000.0,700.0,300.0,0.0,0.0,200.0,340.0,460.0,0.0,0.0,0.0,5.0,88.0,70.0,76.0,0.0,0.0,13.0,5.1,1.1,68.13,0.0,1.180412]
        ],
        1: [  # AKI Stage 1
            [21.0,1.0,0,0.0,55.0,170.0,19.031141868512112,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,5.0,0.0,2.0,165.0,90.0,0,0.0,0.0,200.0,200.0,0.0,0.0,0.0,100.0,145.0,-45.0,0.0,0.0,0.0,0.0,90.0,50.0,63.33333333333333,0.0,0.0,16.2,4.8,1.07,98.7,0.0,0.8953722],
            [71.0,1.0,2,0.0,70.0,167.0,25.099501595611173,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,5.0,0.0,0.0,160.0,120.0,1,75.0,1.0,820.0,820.0,0.0,510.0,0.0,0.0,25.0,795.0,6.0,0.0,0.0,0.0,80.0,50.0,60.0,0.0,0.0,8.8,3.6,1.53,45.08,2.0,6.59292],
            [62.0,0.0,2,0.0,39.0,158.0,15.622496394808522,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,1.0,120.0,80.0,1,80.0,1.0,500.0,500.0,0.0,0.0,0.0,20.0,169.0,311.0,9.0,0.0,0.0,0.0,90.0,50.0,63.33333333333333,0.0,0.0,13.3,3.2,0.4,111.99,0.0,6.705883],
            [54.0,1.0,1,0.0,75.0,174.0,24.772096710265558,1.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,5.0,0.0,0.0,106.0,70.0,1,65.0,1.0,200.0,200.0,0.0,0.0,0.0,50.0,250.0,-100.0,0.0,0.0,0.0,0.0,120.0,80.0,93.33333333333333,0.0,0.0,14.6,4.9,1.15,71.74,0.0,1.25964],
            [40.0,1.0,1,0.0,50.0,165.0,18.36547291092746,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,2.0,165.0,95.0,1,90.0,1.0,800.0,800.0,0.0,0.0,0.0,50.0,50.0,700.0,0.0,0.0,0.0,0.0,110.0,55.0,73.33333333333333,0.0,0.0,14.4,4.8,1.04,86.91,0.0,3.592391]
        ],
        2: [  # AKI Stage 2
            [65.0,0.0,2,1.0,50.0,157.0,20.28479857195018,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,4.0,0.0,1.0,105.0,50.0,1,50.0,1.0,350.0,350.0,0.0,0.0,0.0,10.0,126.0,214.0,18.0,0.0,0.0,0.0,85.0,55.0,65.0,0.0,1.0,13.5,3.7,0.61,95.43,0.0,2.183824],
            [60.0,1.0,2,1.0,50.0,165.0,18.36547291092746,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,5.0,0.0,0.0,120.0,75.0,1,70.0,1.0,1350.0,1050.0,300.0,0.0,0.0,100.0,30.0,1220.0,0.0,10.0,1.0,5.0,70.0,40.0,50.0,0.0,0.0,12.8,2.7,0.87,93.8,0.0,7.116071],
            [71.0,1.0,2,0.0,54.0,150.0,24.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,5.0,0.0,0.0,50.0,25.0,0,0.0,1.0,130.0,130.0,0.0,0.0,0.0,10.0,137.0,-17.0,0.0,0.0,0.0,0.0,110.0,60.0,76.66666666666666,0.0,0.0,11.7,2.7,0.68,96.08,0.0,3.923976],
            [61.0,0.0,1,1.0,60.0,160.0,23.437499999999996,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,2.0,4.0,0.0,1.0,125.0,85.0,1,80.0,1.0,760.0,760.0,0.0,0.0,0.0,100.0,20.0,640.0,0.0,10.0,1.0,5.0,80.0,40.0,53.33333333333333,0.0,0.0,11.7,3.7,0.69,94.26,0.0,3.946809],
            [43.0,1.0,1,1.0,58.0,170.0,20.06920415224914,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,2.0,4.0,1.0,1.0,360.0,300.0,1,210.0,1.0,1250.0,1250.0,0.0,290.0,523.0,600.0,375.0,275.0,3.0,0.0,0.0,0.0,85.0,60.0,68.33333333333333,0.0,0.0,7.4,2.4,0.67,117.68,0.0,4.849673]
        ],
        3: [  # AKI Stage 3
            [87.0,1.0,2,1.0,50.0,160.0,19.531249999999996,1.0,1.0,1.0,0.0,0.0,1.0,1.0,0.0,0.0,1.0,0.0,2.0,4.0,1.0,1.0,260.0,225.0,0,0.0,1.0,2375.0,700.0,600.0,1075.0,0.0,1500.0,285.0,590.0,30.0,10.0,1.0,10.0,68.0,35.0,46.0,0.0,0.0,9.8,3.1199999999999997,2.3,24.61,3.0,7.548077],
            [49.0,1.0,2,1.0,65.0,165.0,23.875114784205696,1.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,2.0,4.0,0.0,0.0,100.0,55.0,1,55.0,1.0,60.0,60.0,0.0,600.0,0.0,50.0,0.0,10.0,0.0,30.0,0.0,0.0,80.0,50.0,60.0,1.0,0.0,7.6,2.7,2.16,34.68,3.0,20.44186],
            [71.0,1.0,2,0.0,45.0,170.0,15.570934256055365,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,2.0,2.0,1.0,1.0,340.0,220.0,1,210.0,1.0,2000.0,1900.0,100.0,0.0,0.0,200.0,350.0,1450.0,15.0,60.0,1.0,5.0,65.0,45.0,51.666666666666664,0.0,0.0,7.6,2.9,1.24,58.12,0.0,10.46154],
            [56.0,0.0,1,1.0,65.0,165.0,23.875114784205696,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,5.0,0.0,1.0,80.0,55.0,1,50.0,1.0,300.0,300.0,0.0,0.0,0.0,50.0,50.0,200.0,0.0,0.0,0.0,0.0,90.0,45.0,60.0,0.0,1.0,12.3,2.9,0.5,144.09,0.0,4.111111],
            [75.0,1.0,2,1.0,60.0,163.0,22.582709172343712,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,2.0,4.0,1.0,1.0,220.0,200.0,1,200.0,1.0,3026.0,540.0,1600.0,280.0,706.0,500.0,395.0,2131.0,39.0,1600.0,1.0,40.0,80.0,30.0,46.666666666666664,1.0,1.0,9.9,1.4,0.5,183.01,3.0,48.5]
        ]
    }
    return samples

def load_real_sample(aki_class, sample_idx):
    """แปลง array เป็น dict และแปลงเป็น type ที่ถูกต้อง"""
    samples = get_real_samples()
    feature_order = ['Age', 'Gender', 'ASAgr', 'Emer_surg', 'BW', 'Height', 'BMI', 'HT', 'DM', 'DLP', 
                     'COPD', 'CAD', 'CVD', 'NSAIDs', 'ACEI', 'ARB', 'Statin', 'Diuretics', 'Dx', 
                     'Type_Op', 'Op_app', 'Side_op', 'Dur_anes', 'Dur_sx', 'One_lung', 'Time_OL', 
                     'Typ_Anal', 'Fluid_ml', 'Crystalloid_ml', 'Total_HES_ml', 'Total_blood_ml', 
                     'FFP_ml', 'Bl_loss', 'Urine', 'fluid_balance', 'Ephedrine', 'Levophed', 
                     'Hypotension', 'Hypotension (mins)', 'LowestSBP', 'LowestDBP', 'Lowest MAP', 
                     'Hypoxemia', 'Hypercarbia', 'Pre Hb', 'Alb', 'PreCr', 'PreGFR', 'offETT', 'NLR1']
    
    values = samples[aki_class][sample_idx]
    data = dict(zip(feature_order, values))
    
    # แปลง type ให้ถูกต้อง - int สำหรับ categorical/count, float สำหรับ continuous
    int_features = ['Age', 'Gender', 'ASAgr', 'Emer_surg', 'HT', 'DM', 'DLP', 'COPD', 'CAD', 'CVD',
                    'NSAIDs', 'ACEI', 'ARB', 'Statin', 'Diuretics', 'Dx', 'Type_Op', 'Op_app', 'Side_op',
                    'Dur_anes', 'Dur_sx', 'One_lung', 'Time_OL', 'Typ_Anal', 'Fluid_ml', 'Crystalloid_ml',
                    'Total_HES_ml', 'Total_blood_ml', 'FFP_ml', 'Bl_loss', 'Urine', 'fluid_balance',
                    'Ephedrine', 'Levophed', 'Hypotension', 'Hypotension (mins)', 'LowestSBP', 'LowestDBP',
                    'Lowest MAP', 'Hypoxemia', 'Hypercarbia', 'PreGFR', 'offETT']
    
    for feature in int_features:
        if feature in data:
            data[feature] = int(data[feature])
    
    return data

# ข้อมูลเดโม่
def get_demo_data():
    return {
        'Age': 65, 'Gender': 1, 'ASAgr': 2, 'Emer_surg': 0, 'BW': 70.0, 'Height': 170.0,
        'BMI': 24.2, 'HT': 1, 'DM': 0, 'DLP': 1, 'COPD': 0, 'CAD': 0, 'CVD': 0,
        'NSAIDs': 0, 'ACEI': 1, 'ARB': 0, 'Statin': 1, 'Diuretics': 0, 'Dx': 1,
        'Type_Op': 2, 'Op_app': 0, 'Side_op': 1, 'Dur_anes': 180, 'Dur_sx': 150,
        'One_lung': 1, 'Time_OL': 90, 'Typ_Anal': 0, 'Fluid_ml': 2000,
        'Crystalloid_ml': 1800, 'Total_HES_ml': 0, 'Total_blood_ml': 0, 'FFP_ml': 0,
        'Bl_loss': 300, 'Urine': 400, 'fluid_balance': 1600, 'Ephedrine': 0,
        'Levophed': 0, 'Hypotension': 0, 'Hypotension (mins)': 0, 'LowestSBP': 110,
        'LowestDBP': 65, 'Lowest MAP': 80, 'Hypoxemia': 0, 'Hypercarbia': 0,
        'Pre Hb': 13.5, 'Alb': 4.0, 'PreCr': 1.0, 'PreGFR': 85, 'offETT': 0, 'NLR1': 3.5
    }

# สุ่มค่า
def randomize_data():
    return {
        'Age': np.random.randint(30, 85),
        'Gender': np.random.randint(0, 2),
        'ASAgr': np.random.randint(0, 4),
        'Emer_surg': np.random.randint(0, 2),
        'BW': round(np.random.uniform(45, 100), 1),
        'Height': round(np.random.uniform(150, 185), 1),
        'BMI': round(np.random.uniform(18, 35), 1),
        'HT': np.random.randint(0, 2),
        'DM': np.random.randint(0, 2),
        'DLP': np.random.randint(0, 2),
        'COPD': np.random.randint(0, 2),
        'CAD': np.random.randint(0, 2),
        'CVD': np.random.randint(0, 2),
        'NSAIDs': np.random.randint(0, 2),
        'ACEI': np.random.randint(0, 2),
        'ARB': np.random.randint(0, 2),
        'Statin': np.random.randint(0, 2),
        'Diuretics': np.random.randint(0, 2),
        'Dx': np.random.randint(0, 3),
        'Type_Op': np.random.randint(0, 6),
        'Op_app': np.random.randint(0, 2),
        'Side_op': np.random.randint(0, 3),
        'Dur_anes': np.random.randint(60, 360),
        'Dur_sx': np.random.randint(50, 300),
        'One_lung': np.random.randint(0, 2),
        'Time_OL': np.random.randint(0, 180),
        'Typ_Anal': np.random.randint(0, 2),
        'Fluid_ml': np.random.randint(500, 4000),
        'Crystalloid_ml': np.random.randint(500, 3500),
        'Total_HES_ml': np.random.randint(0, 1000),
        'Total_blood_ml': np.random.randint(0, 1000),
        'FFP_ml': np.random.randint(0, 800),
        'Bl_loss': np.random.randint(50, 1500),
        'Urine': np.random.randint(100, 1000),
        'fluid_balance': np.random.randint(-500, 3000),
        'Ephedrine': np.random.randint(0, 30),
        'Levophed': np.random.randint(0, 500),
        'Hypotension': np.random.randint(0, 2),
        'Hypotension (mins)': np.random.randint(0, 120),
        'LowestSBP': np.random.randint(70, 140),
        'LowestDBP': np.random.randint(40, 90),
        'Lowest MAP': np.random.randint(50, 100),
        'Hypoxemia': np.random.randint(0, 2),
        'Hypercarbia': np.random.randint(0, 2),
        'Pre Hb': round(np.random.uniform(8, 16), 1),
        'Alb': round(np.random.uniform(2.5, 5.0), 1),
        'PreCr': round(np.random.uniform(0.6, 2.5), 1),
        'PreGFR': np.random.randint(30, 120),
        'offETT': np.random.randint(0, 5),
        'NLR1': round(np.random.uniform(1.5, 10.0), 1)
    }

# Cascade prediction
def cascade_predict(input_data, model1, model2, model3, feature_names):
    df = pd.DataFrame([input_data])
    
    # Stage 1: กรอง No AKI (0) vs มี AKI (1,2,3)
    X1 = df[feature_names['stage1']]
    pred1 = model1.predict(X1)[0]
    prob1 = model1.predict_proba(X1)[0]
    
    results = {
        'stage1': {'prediction': pred1, 'probability': prob1},
        'final_aki': 0  # default = No AKI
    }
    
    # ถ้า Stage 1 = 0 → No AKI, จบ
    if pred1 == 0:
        results['final_aki'] = 0
        return results
    
    # Stage 2: แยก AKI Stage 1 (0) vs AKI Stage 2,3 (1)
    X2 = df[feature_names['stage2']]
    pred2 = model2.predict(X2)[0]
    prob2 = model2.predict_proba(X2)[0]
    results['stage2'] = {'prediction': pred2, 'probability': prob2}
    
    # ถ้า Stage 2 = 0 → AKI Stage 1, จบ
    if pred2 == 0:
        results['final_aki'] = 1
        return results
    
    # Stage 3: แยก AKI Stage 2 (0) vs AKI Stage 3 (1)
    X3 = df[feature_names['stage3']]
    pred3 = model3.predict(X3)[0]
    prob3 = model3.predict_proba(X3)[0]
    results['stage3'] = {'prediction': pred3, 'probability': prob3}
    
    # ถ้า Stage 3 = 0 → AKI Stage 2, ถ้า = 1 → AKI Stage 3
    results['final_aki'] = 2 if pred3 == 0 else 3
    
    return results

# Main App
st.title("🏥 AKI Prediction System (Cascade Model)")
st.markdown("### Postoperative Acute Kidney Injury Prediction")

model1, model2, model3, feature_names = load_models()

if model1 is None:
    st.stop()

# Sidebar controls
st.sidebar.header("Controls")

# ปุ่มโหลดตัวอย่างจริง
st.sidebar.subheader("📁 ตัวอย่างจากข้อมูลจริง (รู้เฉลย)")
aki_class = st.sidebar.selectbox("เลือก AKI Class:", 
                                  [0, 1, 2, 3], 
                                  format_func=lambda x: f"Class {x} ({'No AKI' if x==0 else f'AKI Stage {x}'})")
sample_idx = st.sidebar.selectbox("เลือกตัวอย่างที่:", [0, 1, 2, 3, 4], 
                                   format_func=lambda x: f"ตัวอย่างที่ {x+1}")

if st.sidebar.button("📂 โหลดตัวอย่างจริง", use_container_width=True):
    st.session_state.data = load_real_sample(aki_class, sample_idx)
    st.session_state.true_label = aki_class  # เก็บเฉลยจริง
    st.sidebar.success(f"✅ โหลด: AKI Class {aki_class} (ตัวอย่างที่ {sample_idx+1})")
    
    # Debug: แสดงค่าบางตัว
    st.sidebar.info(f"Age: {st.session_state.data['Age']}, PreCr: {st.session_state.data['PreCr']}, PreGFR: {st.session_state.data['PreGFR']}")

st.sidebar.divider()

if st.sidebar.button("🎲 สุ่มค่า", use_container_width=True):
    st.session_state.data = randomize_data()
    if 'true_label' in st.session_state:
        del st.session_state.true_label  # ลบเฉลยออก

if st.sidebar.button("📋 ใช้ข้อมูลเดโม่", use_container_width=True):
    st.session_state.data = get_demo_data()
    if 'true_label' in st.session_state:
        del st.session_state.true_label  # ลบเฉลยออก

# Initialize data
if 'data' not in st.session_state:
    st.session_state.data = get_demo_data()

# Input Form
st.header("Patient Information")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Demographics")
    st.session_state.data['Age'] = st.number_input("Age (years)", 0, 120, st.session_state.data['Age'])
    st.session_state.data['Gender'] = st.selectbox("Gender", [0, 1], st.session_state.data['Gender'], format_func=lambda x: 'Female' if x == 0 else 'Male')
    st.session_state.data['BW'] = st.number_input("Body Weight (kg)", 0.0, 200.0, float(st.session_state.data['BW']))
    st.session_state.data['Height'] = st.number_input("Height (cm)", 0.0, 250.0, float(st.session_state.data['Height']))
    st.session_state.data['BMI'] = st.number_input("BMI", 0.0, 60.0, float(st.session_state.data['BMI']))

with col2:
    st.subheader("Comorbidities")
    st.session_state.data['ASAgr'] = st.selectbox("ASA Grade", [0, 1, 2, 3], st.session_state.data['ASAgr'])
    st.session_state.data['HT'] = st.selectbox("Hypertension", [0, 1], st.session_state.data['HT'], format_func=lambda x: 'No' if x == 0 else 'Yes')
    st.session_state.data['DM'] = st.selectbox("Diabetes", [0, 1], st.session_state.data['DM'], format_func=lambda x: 'No' if x == 0 else 'Yes')
    st.session_state.data['DLP'] = st.selectbox("Dyslipidemia", [0, 1], st.session_state.data['DLP'], format_func=lambda x: 'No' if x == 0 else 'Yes')
    st.session_state.data['COPD'] = st.selectbox("COPD", [0, 1], st.session_state.data['COPD'], format_func=lambda x: 'No' if x == 0 else 'Yes')

with col3:
    st.subheader("Pre-op Labs")
    st.session_state.data['Pre Hb'] = st.number_input("Pre Hb (g/dL)", 0.0, 20.0, float(st.session_state.data['Pre Hb']))
    st.session_state.data['Alb'] = st.number_input("Albumin", 0.0, 10.0, float(st.session_state.data['Alb']))
    st.session_state.data['PreCr'] = st.number_input("Pre Creatinine", 0.0, 10.0, float(st.session_state.data['PreCr']))
    st.session_state.data['PreGFR'] = st.number_input("Pre GFR", 0, 200, st.session_state.data['PreGFR'])

# Expandable sections for other features
with st.expander("🔧 Surgical Details"):
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.data['Emer_surg'] = st.selectbox("Emergency Surgery", [0, 1], st.session_state.data['Emer_surg'])
        st.session_state.data['Type_Op'] = st.selectbox("Operation Type", [0, 1, 2, 3, 4, 5], st.session_state.data['Type_Op'])
        st.session_state.data['Dur_sx'] = st.number_input("Surgery Duration (min)", 0, 600, st.session_state.data['Dur_sx'])
    with col2:
        st.session_state.data['Op_app'] = st.selectbox("Approach", [0, 1], st.session_state.data['Op_app'])
        st.session_state.data['Side_op'] = st.selectbox("Side", [0, 1, 2], st.session_state.data['Side_op'])
        st.session_state.data['Dur_anes'] = st.number_input("Anesthesia Duration (min)", 0, 600, st.session_state.data['Dur_anes'])

with st.expander("💉 Intraoperative Data"):
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.data['Fluid_ml'] = st.number_input("Total Fluid (mL)", 0, 10000, st.session_state.data['Fluid_ml'])
        st.session_state.data['Bl_loss'] = st.number_input("Blood Loss (mL)", 0, 5000, st.session_state.data['Bl_loss'])
        st.session_state.data['LowestSBP'] = st.number_input("Lowest SBP", 0, 200, st.session_state.data['LowestSBP'])
    with col2:
        st.session_state.data['Urine'] = st.number_input("Urine Output (mL)", 0, 5000, st.session_state.data['Urine'])
        st.session_state.data['Hypotension'] = st.selectbox("Hypotension", [0, 1], st.session_state.data['Hypotension'])
        st.session_state.data['Lowest MAP'] = st.number_input("Lowest MAP", 0, 200, st.session_state.data['Lowest MAP'])

# Predict button
if st.button("🔮 Predict AKI Risk", use_container_width=True, type="primary"):
    with st.spinner("Predicting..."):
        results = cascade_predict(st.session_state.data, model1, model2, model3, feature_names)
    
    # แสดงผลสุดท้ายก่อน (ที่สำคัญที่สุด)
    final_aki = results['final_aki']
    aki_labels = {
        0: ("ไม่มี AKI", "✅ ผู้ป่วยไม่มีภาวะไตวายเฉียบพลัน", "success"),
        1: ("AKI Stage 1", "⚠️ ภาวะไตวายเฉียบพลันระดับเล็กน้อย - ควรติดตามอาการ", "info"),
        2: ("AKI Stage 2", "🔶 ภาวะไตวายเฉียบพลันระดับปานกลาง - ต้องดูแลใกล้ชิด", "warning"),
        3: ("AKI Stage 3", "🔴 ภาวะไตวายเฉียบพลันระดับรุนแรง - ต้องแทรกแซงทันที", "error")
    }
    
    label, message, alert_type = aki_labels[final_aki]
    
    st.header("🎯 ผลการวินิจฉัย")
    st.markdown(f"## **{label}**")
    
    if alert_type == "success":
        st.success(message)
    elif alert_type == "info":
        st.info(message)
    elif alert_type == "warning":
        st.warning(message)
    else:
        st.error(message)
    
    # แสดงการเปรียบเทียบกับเฉลยจริง (ถ้ามี)
    if 'true_label' in st.session_state:
        st.divider()
        true_aki = st.session_state.true_label
        is_correct = (final_aki == true_aki)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🎯 เฉลยจริง", f"AKI Class {true_aki}")
        with col2:
            st.metric("🤖 ทำนาย", f"AKI Class {final_aki}")
        with col3:
            if is_correct:
                st.success("✅ ถูกต้อง!")
            else:
                st.error("❌ ผิด")
    
    # แสดง decision path (สำหรับเข้าใจว่าโมเดลตัดสินใจยังไง)
    st.divider()
    with st.expander("📋 ดูรายละเอียดการตัดสินใจของโมเดล", expanded=True):
        st.markdown("### Decision Path:")
        
        # Stage 1
        pred1 = results['stage1']['prediction']
        prob1 = results['stage1']['probability']
        st.markdown(f"**Step 1:** กรองเบื้องต้น")
        st.markdown(f"- ตัดสินใจ: {'✅ มี AKI → ไป Step 2' if pred1 == 1 else '❌ ไม่มี AKI → จบการวินิจฉัย'}")
        st.markdown(f"- Probability: [No AKI: {prob1[0]:.2%}, มี AKI: {prob1[1]:.2%}]")
        st.markdown(f"- ความมั่นใจ: {prob1[pred1]:.2%}")
        
        # Stage 2
        if 'stage2' in results:
            pred2 = results['stage2']['prediction']
            prob2 = results['stage2']['probability']
            st.markdown(f"**Step 2:** แยกระดับความรุนแรง")
            st.markdown(f"- ตัดสินใจ: {'➡️ AKI รุนแรง (Stage 2-3) → ไป Step 3' if pred2 == 1 else '✅ AKI Stage 1 → จบการวินิจฉัย'}")
            st.markdown(f"- Probability: [Stage 1: {prob2[0]:.2%}, Stage 2-3: {prob2[1]:.2%}]")
            st.markdown(f"- ความมั่นใจ: {prob2[pred2]:.2%}")
        
        # Stage 3
        if 'stage3' in results:
            pred3 = results['stage3']['prediction']
            prob3 = results['stage3']['probability']
            st.markdown(f"**Step 3:** จำแนกระดับสุดท้าย")
            st.markdown(f"- ตัดสินใจ: {'🔴 AKI Stage 3' if pred3 == 1 else '🔶 AKI Stage 2'}")
            st.markdown(f"- Probability: [Stage 2: {prob3[0]:.2%}, Stage 3: {prob3[1]:.2%}]")
            st.markdown(f"- ความมั่นใจ: {prob3[pred3]:.2%}")