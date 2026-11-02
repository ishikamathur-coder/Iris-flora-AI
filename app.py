import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

st.set_page_config(page_title="IRIS FLORA AI", page_icon="🪻", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@500;600;700&display=swap');
.stApp{background:#050505;color:white;font-family:'DM Sans',sans-serif}
.block-container{max-width:1250px;padding-top:1.2rem}
h1,h2,h3{color:#C8A2F0!important;font-family:'Playfair Display',serif!important}
p,li,label{color:white!important}
strong,b{color:#C8A2F0!important}
.section-label{color:#00E5FF;letter-spacing:3px;font-size:12px;text-transform:uppercase}
.section-title{color:#C8A2F0;font-size:2.4rem;font-family:'Playfair Display',serif;margin:.4rem 0 1rem}
.hero{min-height:420px;border:1px solid #3B0A6B;border-radius:25px;display:flex;align-items:center;padding:3rem;background-image:linear-gradient(90deg,rgba(5,5,5,.94),rgba(5,5,5,.3)),url("https://commons.wikimedia.org/wiki/Special:FilePath/Purple_iris_flower.JPG");background-size:cover;background-position:center}
.hero h1{font-size:clamp(3rem,7vw,6rem);color:white!important;line-height:1}
.hero h1 span{color:#C8A2F0}
.hero p{max-width:600px;line-height:1.7}
.pill{display:inline-block;border:1px solid #00E5FF;color:#00E5FF;padding:.45rem .8rem;border-radius:30px;font-size:.7rem;letter-spacing:2px}
.metric,.glass-card,.species-card{background:linear-gradient(145deg,#171020,#090909);border:1px solid #3B0A6B;border-radius:18px;padding:1.2rem;height:100%}
.metric .value{font-size:1.8rem;color:#00E5FF;font-weight:bold}
.metric .label{font-size:.75rem;color:#C8A2F0;text-transform:uppercase}
.species-card{min-height:365px}
.species-card img{width:100%;height:190px;object-fit:cover;object-position:center;border-radius:12px;display:block}
[data-testid="stDialog"] [data-testid="stDialogContent"]{background:#FFFFFF!important}
[data-testid="stDialog"] [data-testid="stMarkdownContainer"],
[data-testid="stDialog"] [data-testid="stMarkdownContainer"] p,
[data-testid="stDialog"] [data-testid="stMarkdownContainer"] strong,
[data-testid="stDialog"] [data-testid="stMarkdownContainer"] h1,
[data-testid="stDialog"] [data-testid="stMarkdownContainer"] h2,
[data-testid="stDialog"] [data-testid="stMarkdownContainer"] h3,
[data-testid="stDialog"] [data-testid="stCaptionContainer"]{color:#102A43!important}
[data-testid="stDialog"] [data-testid="stImage"] img{width:100%;height:300px;object-fit:cover;object-position:center;border-radius:12px}

.stButton>button{background:linear-gradient(90deg,#3B0A6B,#7040A0)!important;color:white!important;border:1px solid #C8A2F0!important;border-radius:30px!important;font-weight:bold!important}
.stButton>button:hover{background:#00E5FF!important;color:#050505!important}
[data-testid="stMetric"]{background:#15101C;border:1px solid #3B0A6B;border-radius:14px;padding:14px}
[data-testid="stMetricLabel"]{color:#C8A2F0!important}[data-testid="stMetricValue"]{color:white!important}
hr{border-color:#3B0A6B!important}footer,#MainMenu{visibility:hidden}
header[data-testid="stHeader"] {
    background: #050505 !important;
}

[data-testid="stAppViewContainer"] {
    background: #050505 !important;
}

[data-testid="stAppViewContainer"] {
    padding-top: 0 !important;
}

[data-testid="stAppViewBlockContainer"] {
    padding-top: 4rem !important;
    overflow: visible !important;
}
[data-testid="stAppViewBlockContainer"] {
    padding-top: 3rem !important;
    overflow: visible !important;
}

[data-testid="stAppViewContainer"] {
    overflow: visible !important;
}

div[data-testid="stMarkdown"] {
    overflow: visible !important;
}

div[data-testid="stMarkdown"] h1 {
    line-height: 1.3 !important;
    overflow: visible !important;
}

/* Navy-blue typography inside the flower details popup */
[data-testid="stDialog"] [data-testid="stMarkdownContainer"],
[data-testid="stDialog"] [data-testid="stMarkdownContainer"] p,
[data-testid="stDialog"] [data-testid="stMarkdownContainer"] strong,
[data-testid="stDialog"] [data-testid="stMarkdownContainer"] b,
[data-testid="stDialog"] [data-testid="stCaptionContainer"] {
    color: #142B4A !important;
}
[data-testid="stDialog"] h1,
[data-testid="stDialog"] h2,
[data-testid="stDialog"] h3 {
    color: #142B4A !important;
}

/* Header brand: prevent IRIS.AI text from being clipped */
.brand-header {
    display: flex !important;
    justify-content: space-between !important;
    align-items: center !important;
    width: 100%;
    min-height: 76px;
    padding: 12px 0 18px !important;
    margin: 0 0 4px 0 !important;
    overflow: visible !important;
    box-sizing: border-box !important;
}
.brand-logo {
    display: inline-block !important;
    font-family: 'Playfair Display', serif !important;
    font-size: 1.5rem !important;
    font-weight: 600 !important;
    letter-spacing: 3px !important;
    line-height: 2.15 !important;
    padding: 8px 6px 12px !important;
    margin: 0 !important;
    height: auto !important;
    min-height: 58px !important;
    overflow: visible !important;
    white-space: nowrap !important;
    box-sizing: content-box !important;
}
.brand-tagline {
    color: #C8A2F0;
    letter-spacing: 2px;
    font-size: .7rem;
    line-height: 1.6;
    padding: 4px 0;
}
[data-testid="stMarkdown"]:has(.brand-header),
[data-testid="stMarkdown"]:has(.brand-logo) {
    overflow: visible !important;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def build_model():
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
    model = RandomForestClassifier(n_estimators=150,random_state=42)
    model.fit(X_train,y_train)
    return iris,X,y,X_train,X_test,y_train,y_test,model,model.predict(X_test)

iris,X,y,X_train,X_test,y_train,y_test,model,y_pred = build_model()
accuracy = accuracy_score(y_test,y_pred)
names = list(iris.target_names)

species = [
    {
        "title": "Iris Setosa",
        "scientific": "Iris setosa",
        "common": "Bristle-pointed Iris",
        "desc": "A delicate wild iris admired for its graceful violet-blue petals and distinctive floral structure. It is native to northern regions and is commonly found in moist habitats, including wet meadows and coastal areas.",
        "short": "Recognised by its distinctive flower structure and relatively small petals.",
        "img": "https://upload.wikimedia.org/wikipedia/commons/a/a7/Irissetosa1.jpg"
    },
    {
        "title": "Iris Versicolor",
        "scientific": "Iris versicolor",
        "common": "Blue Flag Iris",
        "desc": "An elegant perennial flower featuring shades of blue and violet, often accented with delicate yellow markings. It commonly grows in wetlands, marshes and along the edges of freshwater habitats.",
        "short": "Known for blue-violet flowers with distinctive veining and yellow markings.",
        "img": "https://upload.wikimedia.org/wikipedia/commons/2/27/Blue_Flag%2C_Ottawa.jpg"
    },
    {
        "title": "Iris Virginica",
        "scientific": "Iris virginica",
        "common": "Virginia Iris",
        "desc": "A graceful flowering plant known for its soft blue-violet petals and natural beauty. It is commonly found in marshes, wet meadows and other freshwater habitats.",
        "short": "A wetland iris with graceful blue-violet flowers.",
        "img": "https://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg"
    }
]

@st.dialog("Iris Flower Details")
def show_species_details(index):
    flower = species[index]
    st.image(flower["img"], use_container_width=True)
    st.markdown(
        f"""
        <div style="color:#102A43;line-height:1.7;">
            <h2 style="color:#102A43!important;margin-bottom:1rem;">{flower['title']}</h2>
            <p style="color:#102A43!important;"><strong style="color:#102A43!important;">Scientific Name :</strong> <em>{flower['scientific']}</em></p>
            <p style="color:#102A43!important;"><strong style="color:#102A43!important;">Common Name :</strong> {flower['common']}</p>
            <p style="color:#102A43!important;"><strong style="color:#102A43!important;">About the Flower :</strong> {flower['desc']}</p>
        </div>
        """,
        unsafe_allow_html=True
    )


pages = ["🏠 Home","🌸 Introduction","🌺 Iris Species","🤖 AI Prediction","📊 Data Visualization","🧠 Model Performance","✨ Conclusion"]
if "page_choice" not in st.session_state:
    st.session_state.page_choice = pages[0]

st.markdown("""<div class="brand-header">
<div class="brand-logo">IRIS<span style="color:#00E5FF">.</span>AI</div>
<div class="brand-tagline">DISCOVER · CLASSIFY · UNDERSTAND</div>
</div>""", unsafe_allow_html=True)
page = st.radio("Navigation",pages,horizontal=True,label_visibility="collapsed",key="page_choice")
st.markdown("---")

if page == pages[0]:
    st.markdown("""<div class="hero"><div><span class="pill">MACHINE LEARNING · SPECIES RECOGNITION</span>
    <h1>A world<br>in <span>every petal.</span></h1>
    <p>Meet IRIS AI — an interactive flower classification experience. Explore the Iris family and use four botanical measurements to predict a species with a trained Random Forest model.</p></div></div>""",unsafe_allow_html=True)
    st.write("")
    cols=st.columns(4)
    for col,val,label in zip(cols,["150","3","4",f"{accuracy*100:.1f}%"],["Dataset samples","Iris species","Input measurements","Test accuracy"]):
        with col: st.markdown(f'<div class="metric"><div class="value">{val}</div><div class="label">{label}</div></div>',unsafe_allow_html=True)
    st.write("")
    st.markdown('<div class="section-label">WELCOME TO IRIS AI</div><div class="section-title">Explore. Learn. Predict.</div>',unsafe_allow_html=True)
    st.write("Use the navigation bar above to explore the project, discover Iris species, try AI prediction and view model results.")

elif page == pages[1]:
    st.markdown('<div class="section-label">01 / ABOUT THE PROJECT</div><div class="section-title">Understanding the science behind every petal.</div>',unsafe_allow_html=True)
    st.write("""IRIS AI is a Machine Learning-based flower classification system designed to identify three Iris flower species using their physical characteristics.
    The system analyzes four measurements and uses a trained Random Forest Classifier to predict the flower species.""")
    a,b,c=st.columns(3)
    for col,n,label in [(a,"150","Total Dataset Samples"),(b,"3","Classification Classes"),(c,"4","Input Features")]:
        with col: st.markdown(f'<div class="metric"><div class="value">{n}</div><div class="label">{label}</div></div>',unsafe_allow_html=True)
    st.markdown("---")
    st.markdown('<div class="section-label">02 / PROBLEM STATEMENT</div><div class="section-title">The challenge of flower classification.</div>',unsafe_allow_html=True)
    st.write("Develop a Machine Learning pipeline capable of classifying Iris flowers into Setosa, Versicolor and Virginica based on four physical measurements.")
    st.markdown("---")
    st.markdown('<div class="section-label">03 / PROPOSED SOLUTION</div><div class="section-title">An intelligent approach to classification.</div>',unsafe_allow_html=True)
    st.write("IRIS AI uses a Random Forest algorithm trained on the Iris dataset. Users enter four measurements through sliders, and the model predicts the corresponding species with class probabilities.")
    st.markdown("---")
    st.markdown('<div class="section-label">04 / FEATURE ANALYSIS</div><div class="section-title">Four measurements. Three species.</div>',unsafe_allow_html=True)
    x1,x2=st.columns(2)
    with x1:
        st.markdown("### Sepal Length"); st.write("Measures the length of the outer floral structure.")
        st.markdown("### Sepal Width"); st.write("Measures the width of the outer floral structure.")
    with x2:
        st.markdown("### Petal Length"); st.write("Measures the length of the flower petal.")
        st.markdown("### Petal Width"); st.write("Measures the width of the flower petal.")

elif page == pages[2]:
    st.markdown('<div class="section-label">EXPLORE THE SPECIES</div><div class="section-title">Three species. One intelligent model.</div>',unsafe_allow_html=True)
    cols = st.columns(3)
    for index, (col, flower) in enumerate(zip(cols, species)):
        with col:
            st.markdown(
                f"""<div class="species-card">
                    <img src="{flower['img']}">
                    <h3>{flower['title']}</h3>
                    <p>{flower['short']}</p>
                </div>""",
                unsafe_allow_html=True
            )
            if st.button("View flower details ↗", key=f"species_details_{index}", use_container_width=True):
                show_species_details(index)
    st.info("Select any flower's details button to explore its botanical name, common name and description.")

elif page == pages[3]:
    st.markdown('<div class="section-label">AI CLASSIFICATION LAB</div><div class="section-title">Let the measurements tell the story.</div>',unsafe_allow_html=True)
    st.write("Adjust the four measurements below, then run the classifier. Measurements are in centimetres (cm).")
    left,right=st.columns([1.05,.95])
    with left:
        st.markdown('<div class="glass-card">',unsafe_allow_html=True)
        st.markdown("### Flower measurements")
        c1,c2=st.columns(2)
        with c1:
            sl=st.slider("Sepal length (cm)",4.0,8.0,5.1,.1)
            sw=st.slider("Sepal width (cm)",2.0,4.5,3.5,.1)
        with c2:
            pl=st.slider("Petal length (cm)",1.0,7.0,1.4,.1)
            pw=st.slider("Petal width (cm)",.1,2.5,.2,.1)
        predict=st.button("✦ CLASSIFY FLOWER",use_container_width=True)
        st.markdown("</div>",unsafe_allow_html=True)
    with right:
        st.markdown('<div class="glass-card"><h3>How prediction works</h3><p><b>01 — Input</b><br>Enter four flower measurements.</p><p><b>02 — Analyse</b><br>Random Forest compares the values with learned patterns.</p><p><b>03 — Predict</b><br>The app returns a species and class probabilities.</p></div>',unsafe_allow_html=True)
    if predict:
        values=np.array([[sl,sw,pl,pw]])
        pred=int(model.predict(values)[0])
        probs=model.predict_proba(values)[0]
        st.markdown('<div class="section-label">YOUR RESULT</div><div class="section-title">Classification complete.</div>',unsafe_allow_html=True)
        a,b=st.columns([.8,1.2])
        with a:st.image(species[pred]["img"], use_container_width=True)
        with b:
            st.markdown(f'<div class="glass-card"><div class="section-label">Predicted species</div><h2>{species[pred]["title"]}</h2><h2>{max(probs)*100:.1f}%</h2><p>Highest class probability (model confidence estimate)</p></div>',unsafe_allow_html=True)
            for name,p in zip(names,probs): st.progress(float(p),text=f"{name.title()} — {p*100:.1f}%")
        st.caption("The percentage is the model's predicted class probability, not a guarantee of correctness.")

elif page == pages[4]:
    st.markdown('<div class="section-label">DATA EXPLORATION</div><div class="section-title">Visualising the Iris dataset.</div>',unsafe_allow_html=True)
    st.markdown("### Iris Species Distribution")
    st.bar_chart(pd.Series([50,50,50],index=["Setosa","Versicolor","Virginica"]),height=280,color="#8c42c7")
    st.markdown("### Sepal Length vs Petal Length")
    chart=pd.DataFrame({"Sepal Length":X.iloc[:,0],"Petal Length":X.iloc[:,2],"Species":[names[i].title() for i in y]})
    st.scatter_chart(chart,x="Sepal Length",y="Petal Length",color="Species",height=350)

elif page == pages[5]:
    st.markdown('<div class="section-label">BEHIND THE INTELLIGENCE</div><div class="section-title">Model performance.</div>',unsafe_allow_html=True)
    t1,t2,t3=st.tabs(["Evaluation","Confusion matrix","Dataset preview"])
    with t1:
        a,b=st.columns([.7,1.3])
        with a:
            st.metric("Test accuracy",f"{accuracy*100:.2f}%")
            st.write("**Algorithm:** Random Forest")
            st.write("**Train-test split:** 80% / 20%")
            st.write("**Random state:** 42")
        with b:
            report=classification_report(y_test,y_pred,target_names=names,output_dict=True)
            st.dataframe(pd.DataFrame(report).T.round(2),use_container_width=True)
    with t2:
        cm=confusion_matrix(y_test,y_pred)
        fig,ax=plt.subplots(figsize=(6,4))
        fig.patch.set_facecolor("#101014"); ax.set_facecolor("#101014")
        ax.imshow(cm,cmap="Purples")
        ax.set_xticks(range(3),[n.title() for n in names],color="white")
        ax.set_yticks(range(3),[n.title() for n in names],color="white")
        ax.set_xlabel("Predicted label",color="white"); ax.set_ylabel("True label",color="white")
        ax.set_title("Confusion Matrix",color="white")
        for i in range(3):
            for j in range(3): ax.text(j,i,str(cm[i,j]),ha="center",va="center",color="white")
        ax.tick_params(colors="white")
        st.pyplot(fig); plt.close(fig)
    with t3:
        preview=X.copy(); preview["species"]=[names[i] for i in y]
        st.dataframe(preview.head(10),use_container_width=True)

elif page == pages[6]:
    st.markdown('<div class="section-label">MACHINE LEARNING WORKFLOW</div><div class="section-title">From raw data to intelligent predictions.</div>',unsafe_allow_html=True)
    steps=[("01","Data Collection","Load the Iris dataset containing 150 flower samples."),("02","Data Preparation","Organize four measurement features and species labels."),("03","Train-Test Split","Divide data into 80% training and 20% testing."),("04","Model Training","Train a Random Forest Classifier."),("05","Prediction","Predict flower species using the trained model."),("06","Evaluation","Evaluate using accuracy and a confusion matrix.")]
    for n,title,desc in steps:
        with st.container(border=True):
            st.markdown(f"### {n} — {title}"); st.write(desc)
    st.markdown("---")
    st.markdown('<div class="section-label">TECHNOLOGY STACK</div><div class="section-title">Technologies powering IRIS AI.</div>',unsafe_allow_html=True)
    tech=pd.DataFrame({"Technology":["Python","Scikit-learn","Pandas","NumPy","Matplotlib","Seaborn","Joblib","Streamlit","Jupyter Notebook"],"Purpose":["Core Programming Language","Machine Learning Model","Data Handling","Numerical Operations","Data Visualization","Statistical Visualization","Model Saving and Loading","Web Application Interface","Model Training Environment"]})
    st.dataframe(tech,use_container_width=True,hide_index=True)
    st.markdown("---")
    st.markdown('<div class="section-label">MODEL AND ALGORITHM</div><div class="section-title">Powered by Random Forest.</div>',unsafe_allow_html=True)
    st.write("Random Forest is an ensemble Machine Learning algorithm that combines multiple Decision Trees to generate predictions.")
    st.markdown("**Why Random Forest?**\n- Combines multiple Decision Trees.\n- Uses majority voting.\n- Provides stable classification results.\n- Helps reduce overfitting compared with a single Decision Tree.")
    st.write("**Training Dataset:** Iris Dataset  |  **Training Data:** 80% (120 samples)  |  **Testing Data:** 20% (30 samples)  |  **Classes:** 3 Iris Species")
    st.markdown("---")
    st.markdown('<div class="section-label">PROJECT CONCLUSION</div><div class="section-title">Turning measurements into meaningful predictions.</div>',unsafe_allow_html=True)
    st.write("IRIS AI demonstrates how Machine Learning can classify flower species through numerical measurements. It combines dataset exploration, model training, interactive predictions and performance evaluation in one application.")
    st.markdown("### Future Scope")
    st.write("- Experiment with additional classification algorithms.\n- Improve performance through parameter tuning.\n- Develop more interactive visualizations.\n- Expand the application with additional datasets.")

st.markdown("---")
st.markdown("""<div style="text-align:center;padding:1.2rem 0"><div style="font-family:'Playfair Display',serif;font-size:1.5rem;letter-spacing:3px">IRIS<span style="color:#00E5FF">.</span>AI</div><p style="color:#aaa;font-size:.8rem">An educational Machine Learning project · Built with Python & Streamlit</p></div>""",unsafe_allow_html=True)
