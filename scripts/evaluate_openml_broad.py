import os
import sys
import time
import torch
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from evaluate import clear_gpu

# Patch TabPFN for Zero-Shot ISAB
import tabpfn.architectures.tabpfn_v2 as tabpfn_v2
import tabpfn.architectures.tabpfn_v2_5 as tabpfn_v2_5
import tabpfn.architectures.tabpfn_v2_6 as tabpfn_v2_6
from zsisab.engine import AlongColumnAttentionTwoPass
from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion

os.environ["TABPFN_TOKEN"] = "tabpfn_sk_WDvw1MHEYQRQz8NKJBMqEFoink8X-sagyYRMKWM8Vo4"

# Ignore sklearn and openml warnings for clean output
warnings.filterwarnings('ignore')

# Dataset list (OpenML CC-18 subset)
DATASETS = [
    ('breast-cancer', 13),
    ('credit-g', 31),
    ('diabetes', 37),
    ('vehicle', 54),
    ('kc2', 1063),
    ('pc1', 1068),
    ('haberman', 43),
    ('blood-transfusion', 1464)
]

def load_and_prep(data_id):
    data = fetch_openml(data_id=data_id, as_frame=True)
    X = data.data
    y = data.target
    # Handle categoricals natively for TabPFN
    for col in X.columns:
        if X[col].dtype == 'category' or X[col].dtype == 'object':
            X[col] = X[col].astype('category').cat.codes
    
    # Fill NAs
    X = X.fillna(0)
    
    # Binary classification check
    le = LabelEncoder()
    y = le.fit_transform(y)
    
    if len(set(y)) > 2:
        # For simplicity in ROC AUC across many datasets, we'll binarize multiclass
        y = (y > 0).astype(int)
        
    return train_test_split(X, y, test_size=0.33, random_state=42)

def evaluate_broad():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running Broad OpenML Evaluation on {device}")
    
    results = []
    
    for name, data_id in DATASETS:
        print(f"\\n--- Evaluating Dataset: {name} ---")
        try:
            X_train, X_test, y_train, y_test = load_and_prep(data_id)
            
            # 1. Vanilla TabPFN
            clear_gpu()
            importlib = __import__("importlib")
            importlib.reload(tabpfn_v2_5)
            
            clf_vanilla = TabPFNClassifier.create_default_for_version(ModelVersion.V2_5, device=device)
            clf_vanilla.fit(X_train, y_train)
            probs_v = clf_vanilla.predict_proba(X_test)
            preds_v = clf_vanilla.predict(X_test)
            acc_v = accuracy_score(y_test, preds_v)
            auc_v = roc_auc_score(y_test, probs_v[:, 1])
            results.append({'Dataset': name, 'Model': 'Vanilla TabPFN', 'Accuracy': acc_v, 'ROC AUC': auc_v})
            
            # 2. Zero-Shot ISAB
            clear_gpu()
            class TempISAB(AlongColumnAttentionTwoPass):
                def __init__(self, *args, **kwargs):
                    kwargs["num_prototypes"] = 128
                    super().__init__(*args, **kwargs)
            
            tabpfn_v2.AlongColumnAttention = TempISAB
            tabpfn_v2_5.AlongColumnAttention = TempISAB
            tabpfn_v2_6.AlongColumnAttention = TempISAB
            
            original_load_v2_5 = tabpfn_v2_5.TabPFNV2p5.load_state_dict
            tabpfn_v2_5.TabPFNV2p5.load_state_dict = lambda self, sd, strict=True, assign=False: original_load_v2_5(self, sd, strict=False, assign=assign)
            
            clf_isab = TabPFNClassifier.create_default_for_version(ModelVersion.V2_5, device=device)
            clf_isab.fit(X_train, y_train)
            probs_i = clf_isab.predict_proba(X_test)
            preds_i = clf_isab.predict(X_test)
            acc_i = accuracy_score(y_test, preds_i)
            auc_i = roc_auc_score(y_test, probs_i[:, 1])
            results.append({'Dataset': name, 'Model': 'Zero-Shot ISAB', 'Accuracy': acc_i, 'ROC AUC': auc_i})
            
            print(f"Vanilla: Acc={acc_v:.4f}, AUC={auc_v:.4f}")
            print(f"ZS-ISAB: Acc={acc_i:.4f}, AUC={auc_i:.4f}")
            
        except Exception as e:
            print(f"Failed on {name}: {str(e)}")
            continue

    if results:
        df = pd.DataFrame(results)
        df.to_csv("broad_evaluation_results.csv", index=False)
        print("\\nResults saved to broad_evaluation_results.csv")
        
        # Plotting
        os.makedirs("assets", exist_ok=True)
        plt.figure(figsize=(14, 6))
        sns.barplot(data=df, x='Dataset', y='Accuracy', hue='Model')
        plt.title('Zero-Shot Accuracy Comparison Across Broad OpenML Suite (RTX 3090 Ti, 24GB)')
        plt.ylabel('Zero-Shot Accuracy')
        plt.ylim(0, 1.05)
        plt.xticks(rotation=45)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig('assets/broad_evaluation_accuracy.png', dpi=300)
        plt.close()
        print("Generated assets/broad_evaluation_accuracy.png")

if __name__ == "__main__":
    evaluate_broad()
