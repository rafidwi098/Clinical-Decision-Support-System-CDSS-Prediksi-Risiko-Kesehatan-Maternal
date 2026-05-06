import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    classification_report, 
    confusion_matrix, 
    ConfusionMatrixDisplay,
    roc_curve, 
    auc,
    precision_recall_curve,
    average_precision_score
)
from sklearn.inspection import permutation_importance

st.set_page_config(page_title="CDSS - Maternal Health Risk", layout="wide")
st.title("🩺 Clinical Decision Support System (CDSS)")
st.subheader("Prediksi Risiko Kesehatan Maternal")

# 1. Load Data
st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload dataset (.csv / .txt)", type=["csv", "txt"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, sep=",")
    df.columns = [c.replace('\ufeff', '').strip() for c in df.columns]
    
    st.write("### Preview Dataset Observasi Klinis", df.head())

    if 'RiskLevel' not in df.columns:
        st.error("Kolom 'RiskLevel' tidak ditemukan. Pastikan format dataset benar.")
        st.stop()

    target_col = 'RiskLevel'
    all_features = [col for col in df.columns if col != target_col]

    st.session_state['df_raw'] = df
    st.session_state['all_features'] = all_features
    st.session_state['target_col_raw'] = target_col
    
    start_training = st.sidebar.button("Mulai Training")

    # 2. Auto-Feature Selection & Training
    if start_training:
        with st.spinner('Mencari kombinasi fitur terbaik (Proses ini dioptimasi agar memori tidak penuh)...'):

            y = df[target_col]
            le = LabelEncoder()
            y_enc = le.fit_transform(y)
            
            eval_models = [
                Pipeline([("scaler", StandardScaler()), ("lr", LogisticRegression(class_weight='balanced', max_iter=500, random_state=42))]),
                RandomForestClassifier(n_estimators=50, random_state=42, class_weight='balanced'),
                GradientBoostingClassifier(n_estimators=50, random_state=42)
            ]
            
            combo_results = []
            best_acc = 0
            best_features = []

            # Mengecek semua kemungkinan kombinasi 
            for r in range(3, len(all_features) + 1):
                for combo in combinations(all_features, r):
                    features = list(combo)
                    X_subset = df[features]
                    
                    # Rata-rata akurasi CV (n_jobs=1 agar RAM server tidak penuh)
                    model_scores = []
                    for model in eval_models:
                        scores = cross_val_score(model, X_subset, y_enc, cv=5, scoring='accuracy', n_jobs=1)
                        model_scores.append(scores.mean())
                    
                    mean_acc = np.mean(model_scores)
                    
                    combo_results.append({
                        'Kombinasi Parameter': ", ".join(features), 
                        'Jumlah Parameter': r, 
                        'Akurasi (Ensemble CV)': mean_acc
                    })
                    
                    if mean_acc > best_acc:
                        best_acc = mean_acc
                        best_features = features

            combo_df = pd.DataFrame(combo_results).sort_values(by='Akurasi (Ensemble CV)', ascending=False).reset_index(drop=True)
            
            st.session_state['combo_df'] = combo_df
            st.session_state['best_features'] = best_features
            st.session_state['best_acc_selection'] = best_acc

        with st.spinner('Melakukan Training & Tuning pada Fitur Terbaik...'):
            X_best = df[best_features]
            X_train, X_test, y_train, y_test = train_test_split(
                X_best, y_enc, test_size=0.30, random_state=42, stratify=y_enc
            )
            models_configs = {
                "Logistic Regression": {
                    "model": Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42))]),
                    "params": {'clf__C':[0.1, 1, 10]}
                },
                "SVM (RBF)": {
                    "model": Pipeline([("scaler", StandardScaler()), ("clf", SVC(kernel="rbf", probability=True, class_weight='balanced', random_state=42))]),
                    "params": {'clf__C': [0.1, 1, 10], 'clf__gamma': ['scale', 'auto']}
                },
                "Random Forest": {
                    "model": RandomForestClassifier(class_weight='balanced', random_state=42),
                    "params": {'n_estimators':[100, 200, 300], 'max_depth': [None, 10, 20]}
                },
                "Gradient Boosting": {
                    "model": GradientBoostingClassifier(random_state=42),
                    "params": {'n_estimators': [100, 200], 'learning_rate': [0.05, 0.1]}
                }
            }

            results = {}
            fitted_models = {}

            for name, config in models_configs.items():
                # n_jobs=1 wajib digunakan agar Cloud RAM stabil
                grid = GridSearchCV(config["model"], config["params"], cv=5, scoring='accuracy', n_jobs=1)
                grid.fit(X_train, y_train)
                
                best_model = grid.best_estimator_
                fitted_models[name] = best_model
                
                # Evaluasi pada Test Data Unseen
                pred = best_model.predict(X_test)
                results[name] = {
                    "Accuracy": accuracy_score(y_test, pred),
                    "F1_macro": f1_score(y_test, pred, average="macro")
                }

            perf_df = pd.DataFrame(results).T.sort_values("Accuracy", ascending=False)
            
            st.session_state['best_model'] = fitted_models[perf_df.index[0]]
            st.session_state['le'] = le
            st.session_state['perf_df'] = perf_df
            st.session_state['best_name'] = perf_df.index[0]
            st.session_state['X_test'] = X_test
            st.session_state['y_test'] = y_test

            st.success("Seleksi Fitur dan Training Selesai!")


    if 'df_raw' in st.session_state and 'best_model' in st.session_state:
        # Tab Navigasi
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "EDA (Data Asli)", 
            "Feature Selection", 
            "Performa Model", 
            "Matriks Evaluasi", 
            "Prediksi CDSS"
        ])

        # TAB 1
        with tab1:
            st.header("1. Eksplorasi Data")
            df_eda = st.session_state['df_raw']
            features_all = st.session_state['all_features']
            target_eda = st.session_state['target_col_raw']

            col_left, col_right = st.columns([1.5, 1])

            with col_left:
                st.markdown("**1. Pengecekan Missing Values**")
                missing_vals = df_eda[features_all].isnull().sum()
                if missing_vals.sum() == 0:
                    st.success("Tidak ada missing values dalam dataset.")
                else:
                    st.warning("Ditemukan missing values:")
                    st.dataframe(missing_vals[missing_vals > 0].rename("Missing Values"))

                st.markdown("**2. Statistik Deskriptif**")
                st.dataframe(df_eda[features_all].describe())

            with col_right:
                st.markdown("**3. Distribusi Kelas Target (Risk Level)**")
                risk_counts = df_eda[target_eda].value_counts()
                
                fig_risk, ax_risk = plt.subplots(figsize=(6, 4))
                colors = []
                for label in risk_counts.index:
                    if str(label).lower() == 'high risk': colors.append('red')
                    elif str(label).lower() == 'mid risk': colors.append('orange')
                    else: colors.append('green')
                    
                ax_risk.bar(risk_counts.index, risk_counts.values, color=colors)
                ax_risk.set_title("Distribusi Tingkat Risiko", fontsize=14)
                ax_risk.set_ylabel("Jumlah Pasien")
                plt.xticks(rotation=0)
                st.pyplot(fig_risk)

            st.markdown("---")
            st.markdown("**4. Distribusi Data & Outlier**")
            
            for feature in features_all:
                fig_dist, (ax_hist, ax_box) = plt.subplots(2, 1, figsize=(10, 6), gridspec_kw={"height_ratios": (.8, .2)}, sharex=True)
                fig_dist.subplots_adjust(hspace=0.05) 

                sns.histplot(data=df_eda, x=feature, kde=True, ax=ax_hist, color='dodgerblue', edgecolor='black', alpha=0.6, line_kws={'linewidth': 2})
                ax_hist.set_title(f"Distribusi dan Deteksi Outlier: {feature}", fontsize=14, fontweight='bold')
                ax_hist.set_ylabel("Frekuensi")
                ax_hist.set_xlabel("") 
                ax_hist.grid(axis='y', linestyle='--', alpha=0.6)
                
                sns.boxplot(x=df_eda[feature], ax=ax_box, color='lightgray', flierprops=dict(marker='o', markerfacecolor='red', markersize=6, linestyle='none'))
                ax_box.set_xlabel(f"Nilai {feature}", fontsize=12)
                
                st.pyplot(fig_dist)

            st.markdown("---")
            st.markdown("**5. Matriks Korelasi**")
            
            corr_matrix = df_eda[features_all].corr(numeric_only=True)
            fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
            cax = ax_corr.matshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
            fig_corr.colorbar(cax, shrink=0.8)

            ticks = np.arange(0, len(features_all), 1)
            ax_corr.set_xticks(ticks)
            ax_corr.set_yticks(ticks)
            ax_corr.xaxis.tick_top()
            ax_corr.set_xticklabels(features_all, rotation=45, ha='left', fontsize=11)
            ax_corr.set_yticklabels(features_all, fontsize=11)

            for i in range(len(features_all)):
                for j in range(len(features_all)):
                    val = corr_matrix.iloc[i, j]
                    text_color = "black" if abs(val) < 0.5 else "white"
                    bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.7) if abs(val) < 0.5 else None
                    ax_corr.text(j, i, f"{val:.2f}", ha='center', va='center', color=text_color, fontsize=11, fontweight='bold', bbox=bbox_props)

            st.pyplot(fig_corr)

            st.markdown("---")
            st.markdown("**6. Signifikansi Parameter Seluruh Fitur**")
      
            X_all_eda = df_eda[features_all]
            y_all_eda = LabelEncoder().fit_transform(df_eda[target_eda])
            rf_base = RandomForestClassifier(random_state=42, class_weight='balanced')
            rf_base.fit(X_all_eda, y_all_eda)
            
            with st.spinner('Menghitung signifikansi parameter...'):
                perm_all = permutation_importance(rf_base, X_all_eda, y_all_eda, n_repeats=10, random_state=42, scoring="accuracy")
                imp_df_all = pd.DataFrame({
                    "Feature": features_all,
                    "Importance_mean": perm_all.importances_mean
                }).sort_values("Importance_mean", ascending=True)

                fig_imp_all, ax_imp_all = plt.subplots(figsize=(8, 5))
                y_pos_all = np.arange(len(imp_df_all))
                ax_imp_all.barh(y_pos_all, imp_df_all["Importance_mean"], align='center', color='teal', edgecolor='black')
                ax_imp_all.set_yticks(y_pos_all)
                ax_imp_all.set_yticklabels(imp_df_all["Feature"])
                ax_imp_all.set_xlabel("Penurunan Akurasi jika Fitur Dihilangkan")
                ax_imp_all.set_title("Tingkat Kepentingan Seluruh Parameter Asli")
                st.pyplot(fig_imp_all)

        if 'combo_df' in st.session_state:
            best_features = st.session_state['best_features']
            
            # TAB 2
            with tab2:
                st.header("Hasil Auto-Feature Selection (Ensemble Evaluator)")                
                st.success(f"**Kombinasi Terbaik ({len(best_features)} Parameter):** {', '.join(best_features)}")
                st.info(f"**Rata-rata Ensemble Cross-Validation:** {st.session_state['best_acc_selection'] * 100:.2f}%")

                col_table, col_empty = st.columns([2, 1])
                with col_table:
                    st.markdown("**Kombinasi Parameter Berdasarkan Akurasi Lintas Model**")
                    st.dataframe(st.session_state['combo_df'].head(10).style.highlight_max(subset=['Akurasi (Ensemble CV)'], color='lightgreen'))

                st.markdown("---")
                st.markdown(f"### Visualisasi Hubungan Antar Parameter Terpilih")
                df_pairplot = df_eda[best_features + [target_eda]].copy()
                palette_dict = {}
                for val in df_pairplot[target_eda].unique():
                    val_lower = str(val).lower()
                    if 'low' in val_lower:
                        palette_dict[val] = '#5c81ea'  
                    elif 'mid' in val_lower:
                        palette_dict[val] = '#fadd4b'  
                    elif 'high' in val_lower:
                        palette_dict[val] = '#fc4f4f'  
                    else:
                        palette_dict[val] = 'gray'
                        
                fig_pair = sns.pairplot(
                    df_pairplot,
                    vars=best_features,
                    hue=target_eda,
                    corner=True, 
                    diag_kind='kde', 
                    palette=palette_dict,
                    plot_kws={'alpha': 0.7, 's': 50, 'edgecolor': 'w'} 
                )
                
                fig_pair.fig.suptitle("Interaksi Klasifikasi pada Parameter Kombinasi Terbaik", y=1.02, fontsize=14, fontweight='bold')
                st.pyplot(fig_pair.fig)

                st.markdown("---")
                st.markdown(f"### Distribusi & Outlier Individual ({len(best_features)})")
                
                for feature in best_features:
                    fig_dist, (ax_hist, ax_box) = plt.subplots(2, 1, figsize=(10, 6), gridspec_kw={"height_ratios": (.8, .2)}, sharex=True)
                    fig_dist.subplots_adjust(hspace=0.05) 
                    
                    sns.histplot(data=df_eda, x=feature, kde=True, ax=ax_hist, color='mediumseagreen', edgecolor='black', alpha=0.6, line_kws={'linewidth': 2})
                    ax_hist.set_title(f"Distribusi dan Outlier: {feature}", fontsize=14, fontweight='bold')
                    ax_hist.set_ylabel("Frekuensi")
                    ax_hist.set_xlabel("") 
                    ax_hist.grid(axis='y', linestyle='--', alpha=0.6)
                    
                    sns.boxplot(x=df_eda[feature], ax=ax_box, color='lightcoral', flierprops=dict(marker='o', markerfacecolor='red', markersize=6, linestyle='none'))
                    ax_box.set_xlabel(f"Nilai {feature}", fontsize=12)
                    
                    st.pyplot(fig_dist)

                st.markdown("---")
                st.markdown("### Matriks Korelasi Parameter Terpilih")
                if len(best_features) > 1:
                    corr_matrix_best = df_eda[best_features].corr(numeric_only=True)
                    fig_corr_best, ax_corr_best = plt.subplots(figsize=(8, 6))
                    cax_best = ax_corr_best.matshow(corr_matrix_best, cmap='coolwarm', vmin=-1, vmax=1)
                    fig_corr_best.colorbar(cax_best, shrink=0.8)

                    ticks_best = np.arange(0, len(best_features), 1)
                    ax_corr_best.set_xticks(ticks_best)
                    ax_corr_best.set_yticks(ticks_best)
                    ax_corr_best.xaxis.tick_top()
                    ax_corr_best.set_xticklabels(best_features, rotation=45, ha='left', fontsize=11)
                    ax_corr_best.set_yticklabels(best_features, fontsize=11)

                    for i in range(len(best_features)):
                        for j in range(len(best_features)):
                            val = corr_matrix_best.iloc[i, j]
                            text_color = "black" if abs(val) < 0.5 else "white"
                            bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.7) if abs(val) < 0.5 else None
                            ax_corr_best.text(j, i, f"{val:.2f}", ha='center', va='center', color=text_color, fontsize=11, fontweight='bold', bbox=bbox_props)

                    st.pyplot(fig_corr_best)
                else:
                    st.info("Kombinasi terbaik hanya terdiri dari 1 parameter, sehingga matriks korelasi tidak dapat ditampilkan.")

                st.markdown("---")
                best_name = st.session_state['best_name']
                st.markdown(f"### Signifikansi Parameter Terpilih ({best_name})")
                
                with st.spinner('Menghitung signifikansi parameter klinis...'):
                    best_model = st.session_state['best_model']
                    X_test = st.session_state['X_test']
                    y_test = st.session_state['y_test']
                    
                    perm_best = permutation_importance(best_model, X_test, y_test, n_repeats=10, random_state=42, scoring="accuracy")
                    imp_df_best = pd.DataFrame({
                        "Feature": best_features,
                        "Importance_mean": perm_best.importances_mean
                    }).sort_values("Importance_mean", ascending=True)

                    fig_imp_best, ax_imp_best = plt.subplots(figsize=(8, 5))
                    y_pos_best = np.arange(len(imp_df_best))
                    ax_imp_best.barh(y_pos_best, imp_df_best["Importance_mean"], align='center', color='coral', edgecolor='black')
                    ax_imp_best.set_yticks(y_pos_best)
                    ax_imp_best.set_yticklabels(imp_df_best["Feature"])
                    ax_imp_best.set_xlabel("Penurunan Akurasi jika Fitur Dihilangkan")
                    ax_imp_best.set_title("Tingkat Kepentingan Parameter Klinis Terpilih")
                    st.pyplot(fig_imp_best)

            # TAB 3
            with tab3:
                st.header("Performa Model")
                perf_df = st.session_state['perf_df']
                st.dataframe(perf_df.style.highlight_max(axis=0, color='lightgreen'))

                col1, col2 = st.columns(2)
                with col1:
                    fig_acc, ax_acc = plt.subplots(figsize=(6, 4))
                    x_pos = np.arange(len(perf_df.index))
                    bars1 = ax_acc.bar(x_pos, perf_df["Accuracy"], color='mediumseagreen')
                    ax_acc.set_xticks(x_pos)
                    ax_acc.set_xticklabels(perf_df.index, rotation=45, ha='right')
                    ax_acc.set_ylabel("Accuracy Score")
                    ax_acc.set_title("Perbandingan Akurasi")
                    for bar in bars1:
                        yval = bar.get_height()
                        ax_acc.text(bar.get_x() + bar.get_width()/2, yval + 0.01, round(yval, 3), ha='center', va='bottom', fontsize=9)
                    st.pyplot(fig_acc)

                with col2:
                    fig_f1, ax_f1 = plt.subplots(figsize=(6, 4))
                    bars2 = ax_f1.bar(x_pos, perf_df["F1_macro"], color='dodgerblue')
                    ax_f1.set_xticks(x_pos)
                    ax_f1.set_xticklabels(perf_df.index, rotation=45, ha='right')
                    ax_f1.set_ylabel("F1-macro Score")
                    ax_f1.set_title("Perbandingan F1-macro")
                    for bar in bars2:
                        yval = bar.get_height()
                        ax_f1.text(bar.get_x() + bar.get_width()/2, yval + 0.01, round(yval, 3), ha='center', va='bottom', fontsize=9)
                    st.pyplot(fig_f1)

            # TAB 4
            with tab4:
                best_name = st.session_state['best_name']
                best_model = st.session_state['best_model']
                le = st.session_state['le']
                X_test = st.session_state['X_test']
                y_test = st.session_state['y_test']

                st.header(f"Evaluasi Keseluruhan: {best_name}")
                pred_best = best_model.predict(X_test)
                y_score = best_model.predict_proba(X_test)
                classes_idx = np.arange(len(le.classes_))
                y_test_bin = label_binarize(y_test, classes=classes_idx)

                col_rep, col_cm = st.columns([1, 1])
                with col_rep:
                    st.markdown("**1. Laporan Klasifikasi**")
                    report_dict = classification_report(y_test, pred_best, target_names=le.classes_, output_dict=True)
                    report_df = pd.DataFrame(report_dict).transpose()
                    st.dataframe(report_df.style.format(precision=3))

                with col_cm:
                    st.markdown("**2. Confusion Matrix**")
                    cm = confusion_matrix(y_test, pred_best)
                    fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
                    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
                    disp.plot(ax=ax_cm, cmap='Blues', colorbar=False) 
                    st.pyplot(fig_cm)

                st.markdown("---")
                col_roc, col_pr = st.columns(2)
                with col_roc:
                    st.markdown("**3. ROC Curve**")
                    fig_roc, ax_roc = plt.subplots(figsize=(6, 5))
                    colors = ['crimson', 'gold', 'forestgreen']
                    for i, color in zip(classes_idx, colors):
                        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
                        roc_auc = auc(fpr, tpr)
                        ax_roc.plot(fpr, tpr, color=color, lw=2, label=f'ROC {le.classes_[i]} (AUC = {roc_auc:.2f})')
                    ax_roc.plot([0, 1],[0, 1], 'k--', lw=2) 
                    ax_roc.set_xlabel('False Positive Rate')
                    ax_roc.set_ylabel('True Positive Rate')
                    ax_roc.legend(loc="lower right")
                    st.pyplot(fig_roc)

                with col_pr:
                    st.markdown("**4. Precision-Recall Curve**")
                    fig_pr, ax_pr = plt.subplots(figsize=(6, 5))
                    for i, color in zip(classes_idx, colors):
                        precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_score[:, i])
                        pr_auc = average_precision_score(y_test_bin[:, i], y_score[:, i])
                        ax_pr.plot(recall, precision, color=color, lw=2, label=f'PR {le.classes_[i]} (AP = {pr_auc:.2f})')
                    ax_pr.set_xlabel('Recall')
                    ax_pr.set_ylabel('Precision')
                    ax_pr.legend(loc="lower left")
                    st.pyplot(fig_pr)

            # TAB 5
            with tab5:
                st.header("Simulasi Clinical Decision Support System (CDSS)")
                st.write("Sistem hanya menanyakan **parameter klinis yang paling esensial** berdasarkan seleksi")
                
                user_inputs = {}
                cols = st.columns(3) 
                for i, feature in enumerate(best_features):
                    with cols[i % 3]:
                        if feature == 'Age':
                            user_inputs[feature] = st.number_input(f"{feature} (Usia - Tahun)", min_value=0, max_value=120, value=25)
                        elif feature == 'SystolicBP':
                            user_inputs[feature] = st.number_input(f"{feature} (Sistolik - mmHg)", min_value=50, max_value=250, value=120)
                        elif feature == 'DiastolicBP':
                            user_inputs[feature] = st.number_input(f"{feature} (Diastolik - mmHg)", min_value=30, max_value=150, value=80)
                        elif feature == 'BS':
                            user_inputs[feature] = st.number_input(f"{feature} (Gula Darah - mmol/L)", min_value=0.0, max_value=50.0, value=7.0, step=0.1)
                        elif feature == 'BodyTemp':
                            user_inputs[feature] = st.number_input(f"{feature} (Suhu - °F)", min_value=90.0, max_value=110.0, value=98.0, step=0.1)
                        elif feature == 'HeartRate':
                            user_inputs[feature] = st.number_input(f"{feature} (Detak Jantung - bpm)", min_value=30, max_value=200, value=75)
                        else:
                            user_inputs[feature] = st.number_input(f"{feature}", value=0.0)

                st.markdown("---")
                st.info("Sistem beroperasi di bawah Protokol Etik. Algoritma bertindak sebagai *Second Opinion*. Keputusan klinis mutlak ada pada DPJP.")
                consent_checked = st.checkbox("Saya menyatakan bahwa data rekam medis adalah anonim & sesuai dengan persetujuan pasien.")

                if consent_checked:
                    if st.button("Jalankan Prediksi Risiko & Eksekusi Protokol Medis"):
                        input_df = pd.DataFrame([user_inputs])
                        model = st.session_state['best_model']
                        le = st.session_state['le']
                        
                        pred_class_idx = model.predict(input_df)[0]
                        pred_label = le.inverse_transform([pred_class_idx])[0]

                        st.markdown("---")
                        st.markdown("### HASIL PREDIKSI & STATUS TRIASE:")
                        if pred_label == 'high risk':
                            st.error(f"🚨 **STATUS: {pred_label.upper()} (KODE MERAH) - POTENSI KOMPLIKASI TINGGI**")
                        elif pred_label == 'mid risk':
                            st.warning(f"⚠️ **STATUS: {pred_label.upper()} (KODE KUNING) - PERLU OBSERVASI LANJUTAN**")
                        else:
                            st.success(f"✅ **STATUS: {pred_label.upper()} (KODE HIJAU) - KONDISI KLINIS STABIL**")

                        st.markdown("---")
                        st.markdown("### PROTOKOL PENANGANAN KLINIS :")
                        
                        warnings_found = False
                        sys_bp = user_inputs.get('SystolicBP', None)
                        dia_bp = user_inputs.get('DiastolicBP', None)
                        if sys_bp is not None and dia_bp is not None:
                            if sys_bp >= 160 or dia_bp >= 110:
                                st.error(f"**🔴 Hipertensi Berat ({sys_bp}/{dia_bp} mmHg): Ancaman Preeklampsia Berat**\n*   **Kegawatdaruratan:** Rujuk CITO ke VK IGD. Siapkan MgSO4 & antihipertensi IV.\n*   **Lab:** Cek Protein Urine, Darah Lengkap, SGOT/SGPT.")
                                warnings_found = True
                            elif sys_bp >= 140 or dia_bp >= 90:
                                st.warning(f"**🟡 Hipertensi Ringan-Sedang ({sys_bp}/{dia_bp} mmHg): Indikasi Preeklampsia Ringan**\n*   **Medis:** Observasi ketat 2-4 jam. Wajib cek Protein Urine.")
                                warnings_found = True
                            elif sys_bp < 90 or dia_bp < 60:
                                st.warning(f"**🟡 Hipotensi ({sys_bp}/{dia_bp} mmHg): Risiko Penurunan Perfusi/Syok**\n*   **Tindakan:** Posisi Left Lateral Decubitus, rehidrasi IV Ringer Laktat jika perlu.")
                                warnings_found = True

                        if 'BS' in user_inputs:
                            bs = user_inputs['BS']
                            if bs >= 11.1:
                                st.error(f"**🔴 Hiperglikemia Ekstrem ({bs} mmol/L): Diabetes Gestasional Tidak Terkontrol**\n*   **Tindakan:** Konsultasi Sp.PD/Sp.OG untuk terapi Insulin. Jadwalkan USG cek Makrosomia.")
                                warnings_found = True
                            elif bs >= 7.8:
                                st.warning(f"**🟡 Risiko Gangguan Toleransi Glukosa ({bs} mmol/L)**\n*   **Tindakan:** Jadwalkan OGTT 75 gram. Rujuk Gizi Klinik untuk MNT.")
                                warnings_found = True
                            elif bs < 4.0:
                                st.warning(f"**🟡 Hipoglikemia Terdeteksi ({bs} mmol/L)**\n*   **Tindakan:** Berikan 15g karbohidrat kerja cepat, re-evaluasi 15 menit.")
                                warnings_found = True

                        if 'BodyTemp' in user_inputs:
                            temp = user_inputs['BodyTemp']
                            if temp >= 100.4:
                                st.error(f"**🔴 Hipertermia ({temp} °F): Indikasi Infeksi Sistemik**\n*   **Tindakan:** Screening Sepsis, cek Leukosit, Paracetamol IV, pertimbangkan antibiotik profilaksis.")
                                warnings_found = True

                        if 'HeartRate' in user_inputs:
                            hr = user_inputs['HeartRate']
                            if hr > 120:
                                st.error(f"**🔴 Takhikardia Berat ({hr} bpm): Indikasi Kompensasi Syok/Tiroid**\n*   **Tindakan:** Pasang O2, EKG 12-lead, USG FAST obstetri.")
                                warnings_found = True
                            elif hr > 100:
                                st.warning(f"**🟡 Takhikardia Ringan ({hr} bpm)**\n*   **Tindakan:** Istirahatkan pasien, anamnesis nyeri/cemas, ulangi TTV 30 menit.")
                                warnings_found = True

                        if 'Age' in user_inputs:
                            age = user_inputs['Age']
                            if age < 18:
                                st.warning(f"**🟡 Kehamilan Remaja (Usia {age} thn): Risiko Preeklampsia & BBLR**\n*   **Tindakan:** ANC ketat, edukasi nutrisi (Zat besi, Folat).")
                                warnings_found = True
                            elif age >= 35:
                                st.warning(f"**🟡 Usia Maternal Lanjut ({age} thn)**\n*   **Tindakan:** Tawarkan skrining NIPT atau USG Fetomaternal.")
                                warnings_found = True

                        if not warnings_found:
                            st.info("✅ **Semua parameter terpilih berada pada batas aman fisiologis.**\n*   Lanjutkan asuhan rutin ANC (K1-K6) dan suplementasi gizi.")

    elif not start_training and 'df_raw' in st.session_state:
         st.info("Silakan klik 'Mulai Training' pada menu di sebelah kiri untuk memproses data.")

else:
    st.info("Silakan upload dataset (Maternal Health Risk) untuk memulai program.")
