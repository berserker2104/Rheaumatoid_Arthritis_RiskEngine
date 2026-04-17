import streamlit as st
import os 
import pandas as pd 
import numpy as np 
import pickle 
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt 
from sklearn.metrics import roc_curve , roc_auc_score

st.set_page_config(
    page_title="Autoimmune Risk Engine",
    layout="wide"
)
@st.cache_resource
def load_models():
    base_path = "models"
    with open(os.path.join(base_path,"XGBClassifier.pkl") ,"rb") as f:
        xgb_1= pickle.load(f)
    with open(os.path.join(base_path,"lr_model.pkl"),"rb") as f:
        clf=pickle.load(f)
    with open(os.path.join(base_path,"ra_scaler.pkl"),"rb") as f:
        scaler=pickle.load(f)
    with open(os.path.join(base_path,"Gradient_Boosting.pkl"), "rb") as f:
        gbm=pickle.load(f)
    """with open("prs_clac.pkl","rb") as f:
        prs_weights = pickle.load(f)"""
    with open (os.path.join(base_path,"prs_meta.pkl"), "rb") as f:
        prs_meta= pickle.load(f)
    return xgb_1,clf,scaler,gbm,"""prs_weights""",prs_meta

@st.cache_data
def load_test_data():
    base_path = "data"
    x_test = pd.read_csv(os.path.join(base_path,"data_x_test.csv"))
    y_test = pd.read_csv(os.path.join(base_path,"data_y_test.csv"))
    return x_test,y_test
xgb_1,clf,scaler,gbm,prs_weights,prs_meta= load_models()
x_test,y_test=load_test_data()
FEATURE_NAMES =["PRS_std","age","bmi","sex","smoker"]

"""def calculate_prs_from_genotype(geno_df):
    betas=pd.Series(prs_weights)
    missing = set(betas.index) - set(geno_df.columns)
    geno_aligned=geno_df.reindex(columns=betas.index,fill_values=0)
    prs_raw=gen
    prs_std=(prs_raw - prs_mets["mean"]) / prs_meta["std"]
    return prs_std, len(missing)"""
st.sidebar.header("Patient CLinical Profile")
age=st.sidebar.slider("age(years)",18,85,45,
                      help="RA risk peaks between 40-70 years")
bmi=st.sidebar.slider("BMI(kg/m2)",16.0,50.0,27.5,step=0.1,
                      help="Obesity increases inflammation risk")
sex=st.sidebar.selectbox("Biological Sex",["Female","Male"],
                         help="Females have 2-3* higher RA risk")
smoker= st.sidebar.selectbox("Smoking Status", ["No","Yes"],
                             help="Smoking is the strongest environmental RA trigger")
model_choice=st.sidebar.radio("Prediction Model",["XGBoost","Logistic_Regression","Gradient_Boosting"])

st.sidebar.markdown("---")
st.sidebar.subheader("Genetic Risk Score (PRS)")
prs_mode= st.sidebar.radio("PRS Input Method",["Enter PRS directly","Calculate from Genotyoe Matrix"])
prs_std=None
n_pateints=1
if prs_mode=="Enter PRS directly":
    prs_std=st.sidebar.slider("PRS_std value",min_value=-3.0,max_value=3.0,value=0.0,step=0.01,
                              help="0 = population average| +2 =high genetic risk | -2 = low genetic risk")
    st.sidebar.caption("Tip: Value above +1.5 indicate elevated genomic risk")
    n_patients=1
else:
    st.sidebar.markdown("**Upload patient genotype matrix CSV**")
    st.sidebar.caption("Rows = patients | Columns =SNP IDs (e.g. rs2476601)")
    uploaded_file= st.sidebar.file_uploader("Choose CSV file",type="csv")
    if uploaded_file is not None:
        geno_df=pd.read_csv(uploaded_file)
        prs_std_arr,missing=calculator_prs_from_genotypeI(geno_df)
        n_patients=len(gen_df)
        st.sidebar.success(f" {n_patients}patient(s) loaded")
        if missing > 0:
            st.sidebar.warning(f"{missing}SNPs missing - defaulted to 0")
        st.sidebar.metric("PRS_std(Patient 1)"
                          f"{prs_std_arr[0]:.4f}"
                         )
    else:
        st.sidebar.info("upload a genotype CSV to proceed")
st.title("Autoimmune Risk Engine")
st.markdown("**Rheumatoid Arthritis Genomic Risk Prediction**")
st.markdown("Comnining polygenic risk scores with clinical covarities")
st.markdown("---")
if prs_mode=="Calculate from Genotype Matrix" and uploaded_file is None:
    st.info("Please upload a genotype matrix CSV from the sidebar to get predictions.")
    st.stop()
if model_choice=="XGBoost":
    model=xgb_1
elif model_choice== "Gradient_Boosting":
    model=gbm
else:
    model=clf
    
FEATURE_NAMES = ["PRS_std","age","sex","bmi","smoker"]
def build_and_predict(prs_val):
    scaled_ab= scaler.transform(pd.DataFrame([{"age":age,"bmi":bmi}]))
    age_scaled=scaled_ab[0][0]
    bmi_scaled=scaled_ab[0][1]
    
    row=pd.DataFrame([{
        "PRS_std": prs_val,
        "age":age_scaled,
        "sex":1 if sex=="Male" else 0,
        "bmi":bmi_scaled,
        #"sex":1 if sex=="Male" else 0,
        "smoker":1 if smoker=="Yes" else 0,
    }])
    #scaled_ab=scaler.transform(raw_row)
    #scaled_df=pd.DataFrame(scaled_ab,columns=["age","bmi"])
    row=row[FEATURE_NAMES]
    #print(model.features_names)
    prob = model.predict_proba(row)[0][1]
    return prob, row
if prs_mode=="Enter PRS directly":
    risk_prob,patient_scaled=build_and_predict(prs_std)
    risk_pct= round(risk_prob*100,1)

    if risk_prob <0.3:
        color="#2ecc71"; label = " low Risk"
    elif risk_prob<0.6:
        color="#f39c12"; label = "Moderate Risk"
    else:
        color= "#e74c3c";label="High RIsk"
    col1,col2,col3=st.columns([1,2,1])
    with col2:
        st.markdown(
            f"""
            <div style='text-align:center; padding:30px; border-radius:14px;
                         background-color:#1a1a2e; border: 3px solid {color}'>
                <h1 style='color:{color}; font-size:64px; margin:0'>{risk_pct}%</h1>
                <h3 style='color:{color}; margin:8px 0'>{label}</h3>
                <p style='color:#aaa; margin:4px 0'>Model: {model_choice}</p>
                <p style='color:#aaa; margin0'>PRS_std:{prs_std:.4f}</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    st.markdown("---")
    col_shap,col_roc=st.columns(2)
    with col_shap:
        st.subheader("SHAP - Why This Prediction?")
        patient_scaled_df=pd.DataFrame(patient_scaled,columns=FEATURE_NAMES)

        if model_choice=="XGBoost":
            st.warning("SHAP temporarily unavilable for XGBOOST model")#explainer =shap.TreeExplainer(xgb_1.get_booster()
        else:
            explainer = shap.LinearExplainer(clf,x_test)
        shap_vals=explainer(patient_scaled_df)
        fig, _ =plt.subplots()
        shap.plots.waterfall(shap_vals[0], max_display=10 , show=False)
        st.pyplot(plt.gcf(),use_container_width=True)
        plt.clf();plt.close()
    with col_roc:
        st.subheader("ROC Curve - Test Set ")
        x_test_scaled=x_test.copy()
        scaled_vals=scaler.transform(x_test[["age","bmi"]])
        x_test_scaled["age"]=scaled_vals[:,0]
        x_test_scaled["bmi"]=scaled_vals[:,1]
        #X_ts_scaler.transform(x_test)
        y_prob_ts=model.predict_proba(x_test_scaled[FEATURE_NAMES])[:,1]
        auc=roc_auc_score(y_test,y_prob_ts)
        fpr,tpr,_=roc_curve(y_test,y_prob_ts)

        fig_roc,ax=plt.subplots(figsize=(5,4))
        ax.plot(fpr,tpr,color="royalblue", lw=2,label=f"AUC={auc:.3f}")
        ax.plot([0,1],[0,1],"k--",lw=1,)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC -- {model_choice}")
        ax.legend(loc="lower right")
        #ax.grid(alpha=0.3)
        #fig_roc.tight_layout()
        st.pyplot(fig_roc,use_container_width=True)
        plt.close()
else:
    st.subheader("Prediction -All Patients")
    result=[]
    for i in range(n_patients):
        prob,_=build_and_predict(prs_std_arr[i])
        results.append({
            "Patients": i+1,
            "PRS_std": round(float(prs_std_arr[i],4)),
            "age": age,
            #"bmi":bmi,
            "sex":sex,
            "bmi":bmi,
            "Smoker": smoker,
            "Risk %": round(prob * 100,1),
            "Risk Lable": "High" if prob >= 0.6 else"Moderate" if prob>= 0.3 else"low"
        })
    results_df=pd.DataFrame(results)
    st.dataframe(results_df,use_container_width=True,hide_index=True)
    st.markdown("---")
    st.subheader("SHAP Explanation -Patient 1")
    _, p1_scaled= build_and_predict(prs_std_arr[0])
    p1_scaled_df=pd.DataFrame(p1_scaled,columns=FEATURE_NAMES)

    if model_choice=="XGBoost":
        explainer=shap.TreeExplainer(xgb_1)
    else:
        explainer=shap.LinearExpaliner(clf,pd.DataFrame(scaler.transform(x_test),columns=FEATURE_NAMES))
    shap_vals= explainer(p1_scaled_df)
    shap.plots.waterfall(shap_vals[0],max_display=10,show=False)
    st.pyplot(plt.gcf(),use_container_width=True)
    plt.clf();plt.close()

st.markdown("---")
st.subheader("Model Comparison")
x_ts=scaler.transform(x_test)
auc_xgb_1=roc_auc_score(y_test,xgb_1.predict_proba(x_ts)[:,1])
auc_clf=rocauc_score(y_test,clf.predict_proba(x_ts)[:,1])
st.dataFrame(pd.DataFrame({
    "Model": ["XGBoost", "Logistic_Regression", "Gradient_Boosting"],
    "AUC": [round(auc_cgb_1,3),round(auc_clf,3)],
    "Imbalance Fix": ["scale_pos_weight=6","SMOTE"],
    "SHAP Type":    ["TreeExplainer","LinearExplainer"]
}),use_container_width=True,hide_index=True)

st.markdown("---")
st.caption("Autoimmune Risk Engine v1,0 | scikit-learn XGBoost SHAP Streamlit")

        
