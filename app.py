import streamlit as st
import joblib
import json
import pandas as pd
import numpy as np

# Load models และ feature names
@st.cache_resource
def load_models():
    """โหลดโมเดลและ feature names ทั้งหมด"""
    try:
        model1 = joblib.load('model_stage1_xgboost.pkl')
        model2 = joblib.load('model_stage2_logistic.pkl')
        model3 = joblib.load('model_stage3_logistic.pkl')
        
        # (ข้อควรระวัง: feature_names.json ต้องมีโครงสร้าง 
        # {"stage1": [...], "stage2": [...], "stage3": [...]})
        with open('feature_names.json', 'r') as f:
            feature_names = json.load(f)
        
        return model1, model2, model3, feature_names
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.error("กรุณาตรวจสอบว่าไฟล์ .pkl 3 ไฟล์ และ feature_names.json อยู่ในโฟลเดอร์เดียวกับแอป")
        return None, None, None, None

# ข้อมูลตัวอย่างจริงที่รู้เฉลย
def get_real_samples():
    """ข้อมูลจริงจาก dataset แยกตาม AKI class"""
    # (โค้ดส่วนนี้ยาวมาก ขออนุญาตย่อไว้ แต่ในไฟล์จริงของคุณต้องใส่ให้ครบ)
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
    # (ลำดับนี้ต้อง "ตรง" กับข้อมูลใน get_real_samples() เป๊ะๆ)
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
    # (นี่คือ List ที่คุณต้องเช็คให้ตรงกับ data type ที่โมเดลคาดหวัง)
    int_features = ['Age', 'Gender', 'ASAgr', 'Emer_surg', 'HT', 'DM', 'DLP', 'COPD', 'CAD', 'CVD',
                    'NSAIDs', 'ACEI', 'ARB', 'Statin', 'Diuretics', 'Dx', 'Type_Op', 'Op_app', 'Side_op',
                    'Dur_anes', 'Dur_sx', 'One_lung', 'Time_OL', 'Typ_Anal', 'Fluid_ml', 'Crystalloid_ml',
                    'Total_HES_ml', 'Total_blood_ml', 'FFP_ml', 'Bl_loss', 'Urine', 'fluid_balance',
                    'Ephedrine', 'Levophed', 'Hypotension', 'Hypotension (mins)', 'LowestSBP', 'LowestDBP',
                    # 'Lowest MAP' มักจะเป็น float
                    'Hypoxemia', 'Hypercarbia', 'PreGFR', 'offETT']
    
    float_features = ['BW', 'Height', 'BMI', 'Lowest MAP', 'Pre Hb', 'Alb', 'PreCr', 'NLR1']

    for feature in int_features:
        if feature in data:
            # ใช้ .round() ก่อน .astype(int) เพื่อจัดการ float ที่อาจเพี้ยน
            data[feature] = int(round(data[feature]))
            
    for feature in float_features:
        if feature in data:
            data[feature] = float(data[feature])
            
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
        'Bl_loss': 300, 'Urine': 400, 'fluid_balance': 1300, # 2000 - 300 - 400
        'Ephedrine': 0, 'Levophed': 0, 'Hypotension': 0, 'Hypotension (mins)': 0, 
        'LowestSBP': 110, 'LowestDBP': 65, 'Lowest MAP': 80.0, # 65 + (1/3)*(110-65)
        'Hypoxemia': 0, 'Hypercarbia': 0,
        'Pre Hb': 13.5, 'Alb': 4.0, 'PreCr': 1.0, 'PreGFR': 85, 'offETT': 0, 'NLR1': 3.5
    }

# สุ่มค่า
def randomize_data():
    # (ฟังก์ชันนี้เหมือนเดิม ไม่ต้องแก้)
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
    # (ฟังก์ชันนี้เหมือนเดิม ไม่ต้องแก้)
    # (ตรวจสอบให้แน่ใจว่า feature_names.json มี key 'stage1', 'stage2', 'stage3')
    df = pd.DataFrame([input_data])
    
    # Stage 1: กรอง No AKI (0) vs มี AKI (1,2,3)
    # (โมเดล 1 ถูกเทรนให้ทาย 0=NoAKI, 1=AKI)
    try:
        X1 = df[feature_names['stage1']]
        pred1 = model1.predict(X1)[0]
        prob1 = model1.predict_proba(X1)[0]
    except KeyError:
        st.error("Error: 'feature_names.json' ไม่มี key 'stage1' หรือ feature ไม่ตรงกัน")
        return None
    except Exception as e:
        st.error(f"Error Model 1: {e}")
        return None
    
    results = {
        'stage1': {'prediction': pred1, 'probability': prob1},
        'final_aki': 0  # default = No AKI
    }
    
    # ถ้า Stage 1 = 0 → No AKI, จบ
    if pred1 == 0:
        results['final_aki'] = 0
        return results
    
    # Stage 2: แยก AKI Stage 1 (0) vs AKI Stage 2,3 (1)
    # (โมเดล 2 ถูกเทรนให้ทาย 0=AKI 1, 1=[2,3])
    # ‼️‼️ "แก้ไข" การแปลผล pred2 ‼️‼️
    # (ต้องเช็คว่าโมเดล 2 ของคุณทาย 0=Stage1 หรือ 1=Stage1)
    # (โค้ดเก่าของคุณ: 0=Stage1, 1=Stage2-3)
    try:
        X2 = df[feature_names['stage2']]
        pred2 = model2.predict(X2)[0]
        prob2 = model2.predict_proba(X2)[0]
    except KeyError:
        st.error("Error: 'feature_names.json' ไม่มี key 'stage2' หรือ feature ไม่ตรงกัน")
        return None
    except Exception as e:
        st.error(f"Error Model 2: {e}")
        return None

    results['stage2'] = {'prediction': pred2, 'probability': prob2}
    
    # (โค้ดเก่าของคุณ: pred2 == 0 คือ Stage 1)
    if pred2 == 0: 
        results['final_aki'] = 1
        return results
    
    # Stage 3: แยก AKI Stage 2 (0) vs AKI Stage 3 (1)
    # (โมเดล 3 ถูกเทรนให้ทาย 0=Stage2, 1=Stage3)
    if model3 is not None:
        try:
            X3 = df[feature_names['stage3']]
            pred3 = model3.predict(X3)[0]
            prob3 = model3.predict_proba(X3)[0]
        except KeyError:
            st.error("Error: 'feature_names.json' ไม่มี key 'stage3' หรือ feature ไม่ตรงกัน")
            return None
        except Exception as e:
            st.error(f"Error Model 3: {e}")
            return None
            
        results['stage3'] = {'prediction': pred3, 'probability': prob3}
        
        results['final_aki'] = 2 if pred3 == 0 else 3
    else:
        # กรณีไม่มี Model 3 (เทรนไม่ผ่าน)
        results['final_aki'] = 2 # ให้ทายเป็น 2 (ปลอดภัยกว่า)

    return results

# Initialize session state
def init_session_state():
    """Initialize all session state variables"""
    if 'initialized' not in st.session_state:
        default_data = get_demo_data()
        for key, value in default_data.items():
            st.session_state[key] = value
        st.session_state.initialized = True
        
        # (คำนวณค่า derived ทันที)
        st.session_state['BMI'] = st.session_state['BW'] / ((st.session_state['Height'] / 100) ** 2)
        st.session_state['Lowest MAP'] = st.session_state['LowestDBP'] + (1/3) * (st.session_state['LowestSBP'] - st.session_state['LowestDBP'])
        st.session_state['fluid_balance'] = st.session_state['Fluid_ml'] - st.session_state['Bl_loss'] - st.session_state['Urine']


# -----------------------------------
# Main App
# -----------------------------------
st.set_page_config(layout="wide") # ‼️ (เพิ่ม) ใช้พื้นที่เต็มจอ
st.title("🏥 AKI Prediction System (Cascade Model)")
st.markdown("### Postoperative Acute Kidney Injury Prediction")

model1, model2, model3, feature_names = load_models()

if model1 is None:
    st.error("ไม่สามารถโหลดโมเดลได้ กรุณาตรวจสอบไฟล์")
    st.stop()

# Initialize session state
init_session_state()

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
    real_data = load_real_sample(aki_class, sample_idx)
    for key, value in real_data.items():
        st.session_state[key] = value
    st.session_state.true_label = aki_class
    
    # ‼️ (เพิ่ม) คำนวณค่า derived ใหม่ทันที
    st.session_state['BMI'] = st.session_state['BW'] / ((st.session_state['Height'] / 100) ** 2) if st.session_state['Height'] > 0 else 0
    st.session_state['Lowest MAP'] = st.session_state['LowestDBP'] + (1/3) * (st.session_state['LowestSBP'] - st.session_state['LowestDBP']) if st.session_state['LowestSBP'] > st.session_state['LowestDBP'] else st.session_state['LowestDBP']
    st.session_state['fluid_balance'] = st.session_state['Fluid_ml'] - st.session_state['Bl_loss'] - st.session_state['Urine']

    st.sidebar.success(f"✅ โหลด: AKI Class {aki_class} (ตัวอย่างที่ {sample_idx+1})")
    st.sidebar.info(f"Age: {st.session_state['Age']}, PreCr: {st.session_state['PreCr']}, PreGFR: {st.session_state['PreGFR']}")
    st.rerun()

st.sidebar.divider()

if st.sidebar.button("🎲 สุ่มค่า", use_container_width=True):
    random_data = randomize_data()
    for key, value in random_data.items():
        st.session_state[key] = value
    if 'true_label' in st.session_state:
        del st.session_state.true_label
        
    # ‼️ (เพิ่ม) คำนวณค่า derived ใหม่ทันที
    st.session_state['BMI'] = st.session_state['BW'] / ((st.session_state['Height'] / 100) ** 2) if st.session_state['Height'] > 0 else 0
    st.session_state['Lowest MAP'] = st.session_state['LowestDBP'] + (1/3) * (st.session_state['LowestSBP'] - st.session_state['LowestDBP']) if st.session_state['LowestSBP'] > st.session_state['LowestDBP'] else st.session_state['LowestDBP']
    st.session_state['fluid_balance'] = st.session_state['Fluid_ml'] - st.session_state['Bl_loss'] - st.session_state['Urine']
        
    st.rerun()

if st.sidebar.button("📋 ใช้ข้อมูลเดโม่", use_container_width=True):
    demo_data = get_demo_data()
    for key, value in demo_data.items():
        st.session_state[key] = value
    if 'true_label' in st.session_state:
        del st.session_state.true_label
    
    # ‼️ (เพิ่ม) คำนวณค่า derived ใหม่ทันที (แม้ว่า demo จะมีให้แล้ว แต่ทำเพื่อความชัวร์)
    st.session_state['BMI'] = st.session_state['BW'] / ((st.session_state['Height'] / 100) ** 2) if st.session_state['Height'] > 0 else 0
    st.session_state['Lowest MAP'] = st.session_state['LowestDBP'] + (1/3) * (st.session_state['LowestSBP'] - st.session_state['LowestDBP']) if st.session_state['LowestSBP'] > st.session_state['LowestDBP'] else st.session_state['LowestDBP']
    st.session_state['fluid_balance'] = st.session_state['Fluid_ml'] - st.session_state['Bl_loss'] - st.session_state['Urine']
        
    st.rerun()

# ---
# Input Form
# ---
st.header("Patient Information")

# ‼️ (ลบ) ส่วนคำนวณค่า derived จากตรงนี้ (ย้ายไปไว้ใน init / button) ‼️

# แสดงข้อมูลสำคัญด้านบน
col1, col2, col3 = st.columns(3)

# ‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️
# START: ส่วนที่แก้ไข Widget ทั้งหมด
# (เปลี่ยน key='...' และลบ st.session_state[...] = ... ออก)
# ‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️

with col1:
    st.subheader("Demographics")
    st.number_input("Age (years)", 0, 120, key='Age')
    st.selectbox("Gender", [0, 1], format_func=lambda x: 'Female' if x == 0 else 'Male', key='Gender')
    st.number_input("Body Weight (kg)", 0.0, 200.0, key='BW')
    st.number_input("Height (cm)", 0.0, 250.0, key='Height')
    st.metric("BMI (auto)", f"{st.session_state['BMI']:.2f}")

with col2:
    st.subheader("Comorbidities")
    st.selectbox("ASA Grade", [0, 1, 2, 3], key='ASAgr')
    st.selectbox("Hypertension", [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes', key='HT')
    st.selectbox("Diabetes", [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes', key='DM')
    st.selectbox("Dyslipidemia", [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes', key='DLP')
    st.selectbox("COPD", [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes', key='COPD')

with col3:
    st.subheader("Pre-op Labs")
    st.number_input("Pre Hb (g/dL)", 0.0, 20.0, key='Pre Hb')
    st.number_input("Albumin", 0.0, 10.0, key='Alb')
    st.number_input("Pre Creatinine", 0.0, 10.0, key='PreCr')
    st.number_input("Pre GFR", 0, 200, key='PreGFR')

# แสดงฟีเจอร์ที่เหลือทั้งหมดใน expander
with st.expander("🔧 ฟีเจอร์ทั้งหมด (50 features)", expanded=False):
    st.markdown("### ข้อมูลโรคประจำตัวและยา")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.selectbox("CAD", [0, 1], key='CAD', format_func=lambda x: 'No' if x == 0 else 'Yes')
        st.selectbox("CVD", [0, 1], key='CVD', format_func=lambda x: 'No' if x == 0 else 'Yes')
        st.selectbox("NSAIDs", [0, 1], key='NSAIDs', format_func=lambda x: 'No' if x == 0 else 'Yes')
    with col2:
        st.selectbox("ACEI", [0, 1], key='ACEI', format_func=lambda x: 'No' if x == 0 else 'Yes')
        st.selectbox("ARB", [0, 1], key='ARB', format_func=lambda x: 'No' if x == 0 else 'Yes')
        st.selectbox("Statin", [0, 1], key='Statin', format_func=lambda x: 'No' if x == 0 else 'Yes')
    with col3:
        st.selectbox("Diuretics", [0, 1], key='Diuretics', format_func=lambda x: 'No' if x == 0 else 'Yes')
        st.selectbox("Diagnosis", [0, 1, 2], key='Dx')
        st.number_input("NLR1", 0.0, 100.0, key='NLR1')
    
    st.markdown("### ข้อมูลการผ่าตัด")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.selectbox("Emergency Surgery", [0, 1], key='Emer_surg', format_func=lambda x: 'No' if x == 0 else 'Yes')
        st.selectbox("Operation Type", [0, 1, 2, 3, 4, 5], key='Type_Op')
        st.selectbox("Approach", [0, 1], key='Op_app', format_func=lambda x: 'Open' if x == 0 else 'VATS/RATS') # (ตัวอย่างการใส่ format_func)
        st.selectbox("Side", [0, 1, 2], key='Side_op')
    with col2:
        st.number_input("Anesthesia Duration (min)", 0, 1000, key='Dur_anes')
        st.number_input("Surgery Duration (min)", 0, 1000, key='Dur_sx')
        st.selectbox("One Lung", [0, 1], key='One_lung', format_func=lambda x: 'No' if x == 0 else 'Yes')
        st.number_input("Time OL (min)", 0, 500, key='Time_OL')
    with col3:
        st.selectbox("Type Analgesia", [0, 1], key='Typ_Anal')
        st.number_input("Off ETT (days)", 0, 30, key='offETT')
    
    st.markdown("### ข้อมูลสารน้ำและเลือด")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.number_input("Total Fluid (mL)", 0, 10000, key='Fluid_ml')
        st.number_input("Crystalloid (mL)", 0, 10000, key='Crystalloid_ml')
    with col2:
        st.number_input("Total HES (mL)", 0, 5000, key='Total_HES_ml')
        st.number_input("Total Blood (mL)", 0, 5000, key='Total_blood_ml')
    with col3:
        st.number_input("FFP (mL)", 0, 5000, key='FFP_ml')
        st.number_input("Blood Loss (mL)", 0, 5000, key='Bl_loss')
    with col4:
        st.number_input("Urine Output (mL)", 0, 5000, key='Urine')
        st.metric("Fluid Balance (auto)", f"{st.session_state['fluid_balance']:.0f} mL")
    
    st.markdown("### ข้อมูลยาและความดัน")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.number_input("Ephedrine (mg)", 0, 100, key='Ephedrine')
        st.number_input("Levophed (mcg)", 0, 5000, key='Levophed')
        st.selectbox("Hypotension", [0, 1], key='Hypotension', format_func=lambda x: 'No' if x == 0 else 'Yes')
    with col2:
        st.number_input("Hypotension Duration (min)", 0, 500, key='Hypotension (mins)')
        st.number_input("Lowest SBP", 0, 200, key='LowestSBP')
        st.number_input("Lowest DBP", 0, 150, key='LowestDBP')
    with col3:
        st.metric("Lowest MAP (auto)", f"{st.session_state['Lowest MAP']:.2f}")
        st.selectbox("Hypoxemia", [0, 1], key='Hypoxemia', format_func=lambda x: 'No' if x == 0 else 'Yes')
        st.selectbox("Hypercarbia", [0, 1], key='Hypercarbia', format_func=lambda x: 'No' if x == 0 else 'Yes')

# ‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️
# END: ส่วนที่แก้ไข Widget ทั้งหมด
# ‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️


# Predict button
if st.button("🔮 Predict AKI Risk", use_container_width=True, type="primary"):
    # รวบรวมข้อมูลจาก session_state
    # (สร้าง input_data จาก session_state โดยตรง)
    input_data = {key: st.session_state[key] for key in get_demo_data().keys()}
    
    # (อัปเดตค่า auto-calculated เผื่อมีการแก้ไข)
    try:
        input_data['BMI'] = input_data['BW'] / ((input_data['Height'] / 100) ** 2)
        input_data['Lowest MAP'] = input_data['LowestDBP'] + (1/3) * (input_data['LowestSBP'] - input_data['LowestDBP'])
        input_data['fluid_balance'] = input_data['Fluid_ml'] - input_data['Bl_loss'] - input_data['Urine']
    except ZeroDivisionError:
        st.error("Error: Height หรือ BW ห้ามเป็น 0")
        st.stop()
    except Exception as e:
        st.error(f"Error calculating derived fields: {e}")
        st.stop()

    with st.spinner("Predicting..."):
        results = cascade_predict(input_data, model1, model2, model3, feature_names)
    
    if results:
        # แสดงผลสุดท้ายก่อน (ที่สำคัญที่สุด)
        final_aki = results['final_aki']
        aki_labels = {
            0: ("ไม่มี AKI (No AKI)", "✅ ผู้ป่วยไม่มีภาวะไตวายเฉียบพลัน", "success"),
            1: ("AKI Stage 1", "⚠️ ภาวะไตวายเฉียบพลันระดับเล็กน้อย (Mild) - ควรติดตามอาการ", "info"),
            2: ("AKI Stage 2", "🔶 ภาวะไตวายเฉียบพลันระดับปานกลาง (Moderate) - ต้องดูแลใกล้ชิด", "warning"),
            3: ("AKI Stage 3", "🔴 ภาวะไตวายเฉียบพลันระดับรุนแรง (Severe) - ต้องแทรกแซงทันที", "error")
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
        
        # แสดง decision path
        st.divider()
        with st.expander("📋 ดูรายละเอียดการตัดสินใจของโมเดล", expanded=True):
            st.markdown("### Decision Path:")
            
            # Stage 1
            pred1 = results['stage1']['prediction']
            prob1 = results['stage1']['probability']
            st.markdown(f"**Step 1: Gatekeeper (0 vs AKI)**")
            st.markdown(f"- โมเดล 1 (XGBoost) ตัดสินใจ: {'✅ **มี AKI** → ไป Step 2' if pred1 == 1 else '❌ **ไม่มี AKI** → จบการวินิจฉัย'}")
            st.markdown(f"- *Prob: [No AKI: {prob1[0]:.2%}, มี AKI: {prob1[1]:.2%}]*")
            
            # Stage 2
            if 'stage2' in results:
                pred2 = results['stage2']['prediction']
                prob2 = results['stage2']['probability']
                st.markdown(f"**Step 2: Triage (Stage 1 vs Stage 2-3)**")
                st.markdown(f"- โมเดล 2 (Logistic) ตัดสินใจ: {'➡️ **AKI รุนแรง (Stage 2-3)** → ไป Step 3' if pred2 == 1 else '✅ **AKI Stage 1** → จบการวินิจฉัย'}")
                st.markdown(f"- *Prob: [Stage 1: {prob2[0]:.2%}, Stage 2-3: {prob2[1]:.2%}]*")
            
            # Stage 3
            if 'stage3' in results:
                pred3 = results['stage3']['prediction']
                prob3 = results['stage3']['probability']
                st.markdown(f"**Step 3: Specialist (Stage 2 vs Stage 3)**")
                st.markdown(f"- โมเดล 3 (Logistic) ตัดสินใจ: {'🔴 **AKI Stage 3**' if pred3 == 1 else '🔶 **AKI Stage 2**'}")
                st.markdown(f"- *Prob: [Stage 2: {prob3[0]:.2%}, Stage 3: {prob3[1]:.2%}]*")
