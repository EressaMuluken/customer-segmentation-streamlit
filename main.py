import streamlit as st 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
from scipy.stats import gaussian_kde 
import numpy as np 
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score, silhouette_score
from sklearn.preprocessing import StandardScaler
import joblib 
st.set_page_config(
    initial_sidebar_state='locked',
    layout='wide'
)
for key in st.session_state: 
  st.session_state[key] = st.session_state[key] 
if 'current_model' not in st.session_state:
    st.session_state['current_model'] = {
        'KMeans':joblib.load('./models/kmeans_model.pkl'), 
        'GMM': joblib.load('./models/GMM_model.pkl')
        }
if 'current_scaler' not in st.session_state:
    st.session_state['current_scaler'] = joblib.load('./models/scaler.pkl')
if 'pca_data' not in st.session_state:
    st.session_state['pca_data'] = None
if 'segmentation_results' not in st.session_state:
    st.session_state['segmentation_results'] = {}
if 'settings' not in st.session_state: 
    st.session_state['settings'] = {}
if 'data' not in st.session_state: 
  st.session_state['data'] = None
if 'features' not in st.session_state: 
    st.session_state['features'] = ['Age', 'Income', 'Total_Spending',
            'NumWebPurchases','NumStorePurchases',
            'NumWebVisitsMonth','Recency']
st.session_state['dataset_name'] = 'Consumer Data'
colors = {
    "Numeric": "#4F8BF9",
    "Categorical": "#8B5CF6",
    "Boolean": "#10B981",
    "Datetime": "#F59E0B",
}
col_plot_exclude = [
    "ID", 
    "Year_Birth", 
    'Dt_Customer'
]
if 'drop_click' not in st.session_state: 
  st.session_state['drop_click'] = False
@st.cache_resource
def standardize_data(x):
    scaler = StandardScaler()
    return scaler.fit_transform(x)
@st.cache_resource
def create_pca(x):
    x = x.copy()
    x_scaled = standardize_data(x)
    pca = PCA(n_components = 2)
    pca_data = pca.fit_transform(x_scaled)
    return pca_data
@st.cache_resource
def train_model(df, features, algorithm, settings, scaling=True):
    x = df[features].copy()
    evaluation_metrics = []
    if scaling: 
        x_scaled = standardize_data(x)
    else:
        x_scaled = x
    if algorithm == 'K-Means': 
        model = KMeans(n_clusters=settings['K-Means']['n_clusters'],init=settings['K-Means']['init'], random_state=random_state)
        
    elif algorithm == 'GMM':
        model = GaussianMixture(n_components=settings['GMM']['n_clusters'], covariance_type=settings['GMM']['covariance_type'], random_state=random_state)
    
    labels = model.fit_predict(x_scaled)
    if algorithm == 'K-Means':
        evaluation_metrics.append({
        "Model": "K-Means",
        "Clusters": settings['K-Means']['n_clusters'],
        "Silhouette Score": silhouette_score(x_scaled, labels),
        "Davies-Bouldin Index": davies_bouldin_score(x_scaled, labels),
        "Calinski-Harabasz Score": calinski_harabasz_score(x_scaled, labels),
        "Inertia": model.inertia_,
    })
    elif algorithm == 'GMM':
        evaluation_metrics.append({
    "Model": "GMM",
    "Clusters": settings['GMM']['n_clusters'],
    "Silhouette Score": silhouette_score(x_scaled, labels),
    "Davies-Bouldin Index": davies_bouldin_score(x_scaled, labels),
    "Calinski-Harabasz Score": calinski_harabasz_score(x_scaled, labels),
    "Inertia": None,
    "AIC": model.aic(x_scaled),
    "BIC": model.bic(x_scaled)
    })
    
    return model, labels, evaluation_metrics
#@st.cache_resource
def plot_segmentation(pca_data, segmentation_result):
    n_plots = len(segmentation_result)
    fig, axes = plt.subplots(1, n_plots, figsize=(10 * n_plots, 6))
    sns.set_theme(
    style="white",
    font="sans serif",
    rc={
        "font.size": 12,
        "axes.titlesize": 16,
        "axes.titleweight": "bold",
        "axes.labelsize": 13,
        "axes.labelweight": "bold",
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
    }
    )
    fig.patch.set_alpha(0)
    if n_plots == 1:
        axes = [axes]
        
    for ax, (algorithm, eval_result) in zip(axes, segmentation_result.items()):
        ax.patch.set_alpha(0)
        sns.scatterplot(
            x = pca_data[:,0],
            y = pca_data[:,1],
            hue = eval_result[0]['labels'],
            palette='Set1',
            ax = ax
        )
        ax.set_title(f'{algorithm} Segementation result')
        ax.set_xlabel('PCA1')
        ax.set_ylabel('PCA2')
    plt.tight_layout()
    
    return fig
def create_stComponent(df, feature): 
    dtype = df[feature].dtype
    if pd.api.types.is_integer_dtype(dtype):
        return st.number_input(feature, value=int(df[feature].median()))
    elif pd.api.types.is_float_dtype(dtype): 
        return st.number_input(feature, value=float(df[feature].median()))
    elif pd.api.types.is_bool_dtype(dtype): 
        return st.checkbox(feature)
    elif pd.api.types.is_datetime64_any_dtype(df[feature]):
        date_values = pd.to_datetime(df[feature], dayfirst=True, errors="coerce").dropna()
        return st.date_input(feature, value=date_values.median().date())
    else: 
        options = df[feature].dropna().unique()
        return st.selectbox(feature, options=options)
@st.cache_resource
def classify_data(series): 
  series = series.dropna()
  if pd.api.types.is_bool_dtype(series): 
    return 'categorical'
  if pd.api.types.is_numeric_dtype(series): 
    n_unique = series.nunique()
    n = len(series)
    if n_unique <=10:
      return 'discrete'
    if pd.api.types.is_integer_dtype(series):
      if n_unique / n < 0.05:
        return 'discrete'
    return 'continuous'
  return 'categorical'
@st.cache_resource
def plot_distribution(data, column_name): 
  if column_name in col_plot_exclude: 
      return st.markdown(f"""
                         <p style="color:gray;text-align:center;">--------</p>
                         """, unsafe_allow_html=True)
  series = data[column_name].dropna()
  dtype = classify_data(series=series)
  if dtype == 'continuous': 
    fig, ax = plt.subplots(figsize=(3, 0.8))

    ax.hist(
        series,
        bins=20,
        density=True,
        alpha=0.6
    )

    # KDE only if there is enough variation
    if series.nunique() > 1:

        kde = gaussian_kde(series)

        x = np.linspace(
            series.min(),
            series.max(),
            300
        )

        ax.plot(
            x,
            kde(x),
            linewidth=2
        )
        #ax.tick_params(axis="x", labelrotation=90)
        ax.axis('off')
        fig.patch.set_alpha(0)
        ax.patch.set_alpha(0)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)     
  else:
    counts = series.value_counts().head(15)
    fig, ax = plt.subplots(figsize=(3,0.8))
    ax.bar(
      counts.index.astype(str),
      counts.values, 
      alpha=0.7
    )
    #ax.tick_params(axis="x", labelrotation=90)
    ax.axis('off')
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
@st.cache_resource
def drop_null(data: pd.DataFrame):
  st.session_state['data'] = data.dropna(inplace=True)
  st.session_state['drop_click'] = True
@st.cache_resource
def data_desc(data:pd.DataFrame):
  result = {
    "rows": len(data), 
    "colums": len(data.columns),
    "missing cells": data.isna().any(axis=1).sum(),
    "memory": data.memory_usage(deep=True).sum() / (1024 ** 2)
  }  
@st.cache_resource
def get_column_type(data):

    if pd.api.types.is_bool_dtype(data):
        return "Boolean"

    elif pd.api.types.is_numeric_dtype(data):
        return "Numeric"

    elif pd.api.types.is_datetime64_any_dtype(data):
        return "Datetime"

    else:
        return "Categorical"
@st.cache_resource
def get_type(column_name: str, data: pd.Series):

    column_type = get_column_type(data)

    missing = round(data.isna().mean() * 100, 2)
    unique = data.nunique()

    if column_type == "Numeric":
        statistic = round(data.mean(), 2)

    elif column_type == "Categorical":
        mode = data.mode()
        statistic = mode.iloc[0] if not mode.empty else None

    elif column_type == "Boolean":
        statistic = data.mode().iloc[0] if not data.mode().empty else None

    elif column_type == "Datetime":
        statistic = data.min()

    else:
        statistic = None

    return [
        column_name,
        column_type,
        missing,
        unique,
        statistic
    ]
@st.cache_resource
def count_types(data:pd.DataFrame): 
  result = []
  for col in data.columns:
    result.append(get_column_type(data[col]))
  type_count = Counter(result)
  return type_count
@st.cache_resource
def generate_gradient(data:pd.DataFrame):
  d_types = count_types(data)
  total = sum(d_types.values())
  gradient_parts = []
  current = 0
  for dtype, count in d_types.items(): 
    percentage = count / total * 100
    next_position = current + percentage
    color = colors[dtype]
    gradient_parts.append(
      f"{color} {current}%, {color} {next_position}%"
    )
    current = next_position
  gradient = ','.join(gradient_parts)
  return gradient
@st.cache_resource
def generate_labels(data:pd.DataFrame):
  
  d_types = count_types(data)
  total = sum(d_types.values())
  labels = ""
  for d_type, count in d_types.items():
    percentage = count / total * 100
    color = colors[d_type]
    labels += f""" 
    <div style="
    display:flex;
    align-items:center;
    gap:6px;
    "> 
    <span style="
    display:inline-block;
    width:12px;
    height:12px;
    background:{color};
    border-radius:3px;
    "></span>
     <span style="
     padding: 3px 8px;
     border-radius:5px;
     background:{color}20;
     font-size:12px;
     white-space:nowrap;
     ">
     {count} {d_type}
     </span>
    </div>
    """
  return labels
@st.cache_resource
def draw_featue_correlation(data:pd.DataFrame, features):
    sns.set_style("white")
    corr = data[features].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(4,3))
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    cbar=False,
    ax=ax
    )
    ax.set_xticklabels(
    ax.get_xticklabels(),
    fontsize=6,
    fontfamily="sans serif",
    rotation=45,
    ha="right"
    )
    ax.set_yticklabels(
        ax.get_yticklabels(),
        fontsize=6,
        fontfamily="sans serif"
    )
    ax.xaxis.set_visible(False)
    for text in ax.texts:
        text.set_fontsize(6)
    st.pyplot(fig)
    plt.close(fig)
@st.cache_resource
def draw_corr(data:pd.DataFrame):
  corr = data.drop(columns=['ID', 'Year_Birth', 'Dt_Customer']).corr(numeric_only=True)
  n = len(data.columns)
  fig, ax = plt.subplots(figsize=(max(6, n * 0.7), max(5, n * 0.7)))

  im = ax.imshow(
      corr,
      cmap="coolwarm",
      vmin=-1,
      vmax=1
  )

  # Axis labels
  ax.set_xticks(range(len(corr.columns)))
  ax.set_yticks(range(len(corr.columns)))

  ax.set_xticklabels(corr.columns, rotation=90, ha="right")
  ax.set_yticklabels(corr.columns)

  # Put correlation values inside each square
  for i in range(len(corr)):
      for j in range(len(corr)):
          ax.text(
              j,
              i,
              f"{corr.iloc[i, j]:.1f}",
              ha="center",
              va="center"
          )

  # Color bar
  #fig.colorbar(im, ax=ax)
  ax.patch.set_alpha(0)
  fig.patch.set_alpha(0)
  plt.tight_layout()

  st.pyplot(fig)
  plt.close(fig)
@st.cache_data
def load_data(path): 
    return pd.read_csv(path)
data = load_data('./data/data_modified.csv')
model_evluation_metrics = load_data('./data/model_evaluation.csv')
st.session_state['data'] = data

features = data.columns

st.markdown("""
<style>

/* ---------- Global typography ---------- */

html, body, [class*="css"] {
    font-family: Inter, sans-serif;
}

h1 {
    font-size: 32px !important;
    font-weight: 700 !important;
    letter-spacing: -0.5px;
}

h2 {
    font-size: 24px !important;
    font-weight: 650 !important;
}

h3 {
    font-size: 18px !important;
    font-weight: 600 !important;
}

p {
    font-size: 15px;
    font-weight: 400;
    color: #475569;
}



/* ---------- Main content ---------- */

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
}

div.stTabs {
    font-size:30px;
    font-weight: 500;
}

/*------unique------*/
div[role="tab"] p {
    font-size: 18px !important;
    font-weight: 600 !important;
    text-transform: uppercase;
}
div[data-testid='stExpander'] p {
    font-size: 14px !important;
    font-weight: 600 !important;
    text-transform: uppercase;
}
div[data-testid='stButton'] button p {
    font-size: 16px !important;
    font-weight: 700 !important;
    color: white !important;
}
</style>
""", unsafe_allow_html=True)

with st.sidebar: 
    st.markdown("""
                    <p style="font-weight:600;color:gray;font-size:14px;">TRAIN A NEW MODEL</p>
                    """, unsafe_allow_html=True)
    with st.expander('Settings', expanded=True): 
        st.markdown("""
                    <p style="font-weight:600;color:gray;font-size:14px;">ALGORITHM</p>
                    """, unsafe_allow_html=True)
        algorithms = st.multiselect('Algorithms', options=['K-Means', 'GMM'],label_visibility='collapsed', placeholder='Select a model' )
        if 'K-Means' in algorithms: 
            with st.container(border=True):
                st.markdown("""
                        <p style="font-weight:600;color:gray;font-size:14px;">K-Means</p>
                        """, unsafe_allow_html=True)
                st.markdown("""
                        <p style="font-weight:600;color:gray;font-size:12px;">Cluster Size</p>
                        """, unsafe_allow_html=True)
                n_clusters_kmeans = st.slider('segments_kmeans', min_value=2, max_value=10, value=4, width='stretch', label_visibility='collapsed')
                st.markdown("""
                        <p style="font-weight:600;color:gray;font-size:12px;">Initialization</p>
                        """, unsafe_allow_html=True)
                init = st.selectbox('K-Means: Initialization', options=['k-means++', 'random'], label_visibility='collapsed')
                st.markdown("""
                        <p style="font-weight:600;color:gray;font-size:12px;">Random State</p>
                        """, unsafe_allow_html=True)
                random_state = st.slider('random_state_kmeans', min_value=1, max_value=100, value=42, width='stretch', label_visibility='collapsed')

            st.session_state.settings['K-Means'] = {
                    'n_clusters': n_clusters_kmeans, 
                    'init': init,
                    'random_state': random_state
            }
        if 'GMM' in algorithms: 
            with st.container(border=True):
                st.markdown("""
                        <p style="font-weight:600;color:gray;font-size:14px;">GAUSSIAN MIXTURE(GMM)</p>
                        """, unsafe_allow_html=True)
                st.markdown("""
                        <p style="font-weight:600;color:gray;font-size:12px;">CLUSTER SIZE</p>
                        """, unsafe_allow_html=True)
                n_components_gmm = st.slider('segments_gmm', min_value=2, max_value=10, value=4, width='stretch', label_visibility='collapsed')
                st.markdown("""
                        <p style="font-weight:600;color:gray;font-size:12px;">COVARIANCE TYPE</p>
                        """, unsafe_allow_html=True)
                covariance_type = st.selectbox('Covariance type', options=['full','tied','diag','spherical'], label_visibility='collapsed')
                st.markdown("""
                        <p style="font-weight:600;color:gray;font-size:12px;">Random State</p>
                        """, unsafe_allow_html=True)
                random_state = st.slider('random_state_gmm', min_value=1, max_value=100, value=42, width='stretch', label_visibility='collapsed')

            st.session_state.settings['GMM'] = {
                'n_clusters': n_components_gmm, 
                'covariance_type': covariance_type,
                'random_state': random_state
            }
        st.markdown("""
                    <p style="font-weight:600;color:gray;font-size:14px;">FEATURES</p>
                    """, unsafe_allow_html=True)
        selected_features = st.multiselect('Features', options=features.tolist(),label_visibility='collapsed', default=list(st.session_state.features), placeholder='Select features')
        st.session_state['features'] = selected_features
        st.markdown("""
                    <p style="font-weight:600;color:gray;font-size:14px;">PREPROCESSING</p>
                    """, unsafe_allow_html=True)
        #scaling = st.checkbox('Scaling', value=True, label_visibility='collapsed')
        st.selectbox('Scaling', options=['Standardize',], label_visibility='collapsed')
        st.markdown("""
                    <div style="width:100%;height:50px;">
                    </div>
                    """, unsafe_allow_html=True)
        if st.button('Run Segmentation', use_container_width=True, type='primary'):
            with st.status("Training segmentation models...", expanded=True) as status:
                if not algorithms: 
                    status.update(
                    label="Please Select a Model to continue!",
                    state="error",
                    expanded=False
                     )
                else: 
                    pca_data = create_pca(data[list(st.session_state['features'])])
                    results = {}
                    st.session_state.current_model = {}
                    for algorithm in algorithms:
                        st.write(f"Training {algorithm}...")
                        model, labels, eval_metrics = train_model(data,st.session_state['features'],
                                                                algorithm=algorithm, settings=st.session_state['settings'])
                        results[algorithm]={
                            'model': model,
                            'labels': labels,
                            'evaluation_metrics': eval_metrics
                        },
                        joblib.dump(model, f'./trained_new_models/{algorithm}.pkl')
                        st.session_state.current_model[algorithm] = model
                    st.session_state.pca_data = pca_data
                    st.session_state.segmentation_results = results 
                    status.update(
                    label="Training completed!",
                    state="complete",
                    expanded=False
                )
        st.markdown("""
                    <div style="width:100%;height:50px;">
                    </div>
                    """, unsafe_allow_html=True)

tab_pred, tab_data = st.tabs(['Predict', 'Data'])
with tab_data:
    st.markdown(f"""
              <div style="display:flex;justify-content:space-between;align-items:center;width:100%;height:30px; border-bottom:1px solid #D2D2D2; padding-right:0px; margin-bottom:4px;font-weight:800; font-size:12px; text-transform:uppercase;">
              <p style="margin-bottom:0rem;">
              <span style="margin-right:4px;">{st.session_state['dataset_name']}.csv</span> / <span style="margin-left:4px;">Data Analysis</span>
              </p>
              <div style="display:flex; gap: 1rem;">
              <p style="margin-bottom:2px;">
              {len(data)} rows
              </p>
                <p style="margin-bottom:0rem;">
              {len(data.columns)} cols
              </p>
                <p style="margin-bottom:0rem;">
              {data.isna().any(axis=1).sum() / len(data):.1%} missing
              </p>
                <p style="margin-bottom:0rem;">
              {data.duplicated().sum()} duplicates
              </p>
              </div>
              </div>
              """, unsafe_allow_html=True)
    hcol1, _ = st.columns([3,1])
    with hcol1: 
        st.markdown(
                "<p style='font-size:20px; font-weight:600;margin-bottom:4px;'>Descriptive Statistics</p>"
                , unsafe_allow_html=True)
    # with hcol2:
    #     hhcol1, hhcol2 = st.columns(2)
    #     with hhcol1:
    #         if st.button('Export profile', type='secondary', use_container_width=True):
    #             st.write('hello') 
    #     with hhcol2:
    #         if st.button('Continue to Cleaning :material/arrow_right_alt:',type='primary', use_container_width=True):
    #             st.write('hello')
    stcol = st.columns(5)
    with stcol[0]:
        with st.container(height=120, border=True):
            st.markdown( f"<p style='font-size:14px; font-weight:700;margin-top:0px; margin-bottom:0px;color:gray;'>ROWS</p>"
                    , unsafe_allow_html=True)
            st.markdown( f"<p style='font-size:24px; font-weight:700;margin-top:0px;margin-bottom:0px;'>{len(data)}</p>"
                    , unsafe_allow_html=True)
            st.caption(f'{data.duplicated().sum()} exact duplicates')
    with stcol[1]:
        with st.container(height=120, border=True):
            st.markdown( f"<p style='font-size:14px; font-weight:700;margin-top:0px; margin-bottom:0px;color:gray;'>COLUMNS</p>"
                    , unsafe_allow_html=True)
            st.markdown( f"<p style='font-size:24px; font-weight:700;margin-top:0px;margin-bottom:0px;'>{len(data.columns)}</p>"
                    , unsafe_allow_html=True)
            numeric = data.select_dtypes(include="number").shape[1]
            datetime = data.select_dtypes(include="datetime").shape[1]
            categorical = data.select_dtypes(include=["object", "category"]).shape[1]

            type_summary = (
            f"{numeric} numeric · "
            f"{categorical} categorical · "
            f"{datetime} datetime"
            )
            st.caption(type_summary)
    with stcol[2]:
        with st.container(height=120, border=True):
            st.markdown( f"<p style='font-size:14px; font-weight:700;margin-top:0px; margin-bottom:0px;color:gray'>MISSING CELLS</p>"
                    , unsafe_allow_html=True)
            st.markdown( f"<p style='font-size:24px; font-weight:700;margin-top:0px;margin-bottom:0px'>{data.isna().any(axis=1).sum()/len(data):.2%}</p>"
                    , unsafe_allow_html=True)
            st.caption(f'{data.isna().any(axis=1).sum()} of {len(data)} cells') 
    with stcol[3]:
        with st.container(height=120, border=True):
            st.markdown( f"<p style='font-size:14px; font-weight:700;margin-top:0px; margin-bottom:0px;color:gray'>MEMORY</p>"
                    , unsafe_allow_html=True)
            st.markdown( f"<p style='font-size:24px; font-weight:700;margin-top:0px;margin-bottom:0px'>{data.memory_usage(deep=True).sum()/(1024 **2):.2f} MB</p>"
                    , unsafe_allow_html=True)
            st.caption('In-memory as float64/object')     
    with stcol[4]:
        with st.container(height=120, border=True):
            st.markdown( f"<p style='font-size:14px; font-weight:700;margin-top:0px; margin-bottom:0px;color:gray'>TARGET BALANCE</p>"
                    , unsafe_allow_html=True)
            st.markdown( f"<p style='font-size:24px; font-weight:700;margin-top:0px;margin-bottom:0px'>{73} / {23}</p>"
                    , unsafe_allow_html=True)
            st.caption(f'churn = {0} / churn = {1} ({1:.2%})') 
    with st.container(border=True):
        st.markdown("""
                    <p style='font-size:16px;font-weight:600;margin-bottom:4px;'>Column type breakdown</p>
                    <hr style='margin-top:16px; margin-bottom:16px; padding:0px;'/>
                    """
                    , unsafe_allow_html=True)
        st.markdown(
            f"""
            <div style="
                width: 100%;
                height: 10px;
                border-radius: 6px;
                background: linear-gradient(
                    to right,
                    {generate_gradient(data)}
                );
            "></div>
            """,
            unsafe_allow_html=True
        )
        st.markdown(
    f"""
    <div style="
        display: flex;
        gap:20px;
        width: 100%;
        margin-top: 6px;
        margin-bottom: 12px
    ">
        {generate_labels(data)}
    </div>
    """,
    unsafe_allow_html=True
)

    st.markdown(f"<span style='font-size:14px; font-weight:700;'>Per-column summary</span><span style='font-size:12px; font-weight:500; color:gray; margin-left:8px'>{len(st.session_state['data'].columns)} columns</span>", unsafe_allow_html=True)  
    st.markdown("<hr style='margin-top:0px; margin-bottom:0px'/>", unsafe_allow_html=True)   
    mcol1, mcol2 = st.columns(2)
    with mcol1:
        title_header = st.columns(6) 
        title = ['COLUMN', 'TYPE', 'MISSING', 'UNIQUE', 'MEAN/TOP', 'DIST.']  
        with st.container(height=670, border=True):
            for i in range(len(title)):
                with title_header[i]:
                    st.markdown(f"<p style='padding-left:12px;font-size:14px; font-weight:600;color:gray; margin-top:4px;margin-bottom:4px;'>{title[i]}</p>", unsafe_allow_html=True)
            st.markdown("<hr style='margin-top:0px; margin-bottom:0px;'/>", unsafe_allow_html=True)  
            for i in range(len(data.columns)):
                for j, col in enumerate(st.columns(6)):
                    result = get_type(data.columns[i], data[data.columns[i]]) 
                    with col:
                        if j == 0: 
                            st.markdown(f"<p style='padding-right:8px;font-size:14px; font-weight:600; white-space: nowrap;overflow: hidden;text-overflow: ellipsis;'>{result[j]}</p>", unsafe_allow_html=True)
                        elif j == 1: 
                            st.markdown(f"<span style='display:inline-block;min-width:60px;font-size:12px; font-weight:600;padding:2px 8px;background-color:{colors[result[j]]}; border: 1px solid #ddd; border-radius:10px;text-align:center;justify-content:center;'>{result[j][:3]}</span>", unsafe_allow_html=True)
                        elif j == 2:
                            misscol1, misscol2 = st.columns(2)
                            with misscol1: 
                                st.progress(int(result[j]))
                            with misscol2:
                                st.markdown(f"<span style='font-size:12px; font-weight:600; color:gray'>{result[j]} %</span>", unsafe_allow_html=True)
                        elif j == 5:
                            plot_distribution(data, data.columns[i])
                        else:
                            st.markdown(f"<p style='font-size:16px; font-weight:600; color:gray;text-align:center'>{result[j]}</p>", unsafe_allow_html=True)
                st.markdown(f"<hr style='margin-top:0px; margin-bottom:-4px;'></>", unsafe_allow_html=True)  
    with mcol2:
        with st.container(height=700,autoscroll=True, border=False):
            draw_corr(data)
    #st.markdown(f"<span style='font-size:14px; font-weight:700;'>Raw Data</span>", unsafe_allow_html=True)  
    #st.markdown("<hr style='margin-top:0px; margin-bottom:0px'/>", unsafe_allow_html=True)   

    #st.dataframe(data)
    #st.markdown(f"<span style='font-size:14px; font-weight:700;'>Descriptive Summary</span>", unsafe_allow_html=True)  
    #st.dataframe(data.describe())

with tab_pred:
    feature_col, parameter_col = st.columns(2)
    with feature_col:
        draw_featue_correlation(data, st.session_state.features)
    with parameter_col:
        with st.container(border=True): 
            par_left, par_right = st.columns(2)
            input_data = {}
            for i, feature in enumerate(st.session_state.features):
                if i%2 == 0:
                    with par_left:
                       value = create_stComponent(data, feature)
                else: 
                    with par_right:
                        value = create_stComponent(data, feature)
                input_data[feature] = value
            if st.button('RUN SIMULATION', width='stretch', type='primary'):
                input_data_df = pd.DataFrame([input_data])
                input_data_scaled = st.session_state.current_scaler.transform(input_data_df)
                for algorithm, trained_model in st.session_state.current_model.items():
                    pred_segment = trained_model.predict(input_data_scaled)[0]
                    st.success(f'Customer is assigned to Cluster {pred_segment} using {algorithm}')
    with st.expander('Segmentation Performance', expanded=True):
        if 'segmentation_results' in st.session_state and st.session_state.segmentation_results != {}:
            data_pca = st.session_state.pca_data
            results = st.session_state.segmentation_results
            evaluation_df = pd.DataFrame(
                    [
                        metric
                        for algorithm_result in results.values()
                        for metric in algorithm_result[0]['evaluation_metrics']
                    ]
                )
            st.dataframe(evaluation_df)
            fig = plot_segmentation(data_pca, results)
            st.pyplot(fig)
        else: 
            st.dataframe(model_evluation_metrics)
            st.image('./notebooks/kmeans_gmm_segmentation_pca.png', width='stretch')