import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs, make_moons
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (confusion_matrix, accuracy_score, roc_auc_score, 
                             classification_report, roc_curve, precision_recall_curve)
import xgboost as xgb
from scipy.stats import multivariate_normal
import warnings
warnings.filterwarnings('ignore')

class RealFakeClassifier:
    """
    A comprehensive classifier for distinguishing between real and synthetic data.
    Supports multiple data generation methods, dimensionalities, and ML models.
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        np.random.seed(random_state)
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}
        
    def generate_real_data(self, method='blobs', n_samples=1000, n_features=2, **kwargs):
        """
        Generate real data using various methods.
        
        Parameters:
        - method: 'blobs', 'moons', 'multivariate_normal'
        - n_samples: number of samples
        - n_features: number of features (2 for visualization, 128+ for high-dim)
        - kwargs: additional parameters for specific methods
        """
        if method == 'blobs':
            centers = kwargs.get('centers', min(4, max(2, n_features//10)))
            cluster_std = kwargs.get('cluster_std', 1.0)
            X, _ = make_blobs(n_samples=n_samples, centers=centers, 
                            n_features=n_features, cluster_std=cluster_std,
                            random_state=self.random_state)
        
        elif method == 'moons':
            if n_features != 2:
                print(f"Warning: make_moons only supports 2D. Using 2 features instead of {n_features}")
                n_features = 2
            noise = kwargs.get('noise', 0.1)
            X, _ = make_moons(n_samples=n_samples, noise=noise, 
                            random_state=self.random_state)
        
        elif method == 'multivariate_normal':
            # Create a realistic covariance structure
            mean = kwargs.get('mean', np.zeros(n_features))
            if 'cov' in kwargs:
                cov = kwargs['cov']
            else:
                # Generate a positive definite covariance matrix
                A = np.random.randn(n_features, n_features)
                cov = np.dot(A, A.T) + np.eye(n_features) * 0.1
            
            X = np.random.multivariate_normal(mean, cov, n_samples)
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return X
    
    def generate_fake_data(self, n_samples=1000, n_features=2, method='uniform', **kwargs):
        """
        Generate fake data using different distributions.
        
        Parameters:
        - method: 'uniform', 'gaussian_different', 'random_noise'
        - n_samples: number of samples
        - n_features: number of features
        """
        if method == 'uniform':
            low = kwargs.get('low', -3)
            high = kwargs.get('high', 3)
            X = np.random.uniform(low, high, size=(n_samples, n_features))
        
        elif method == 'gaussian_different':
            # Different mean and covariance from real data
            mean = kwargs.get('mean', np.ones(n_features) * 2)
            std = kwargs.get('std', 0.5)
            X = np.random.normal(mean, std, size=(n_samples, n_features))
        
        elif method == 'random_noise':
            # Pure random noise
            X = np.random.randn(n_samples, n_features) * kwargs.get('scale', 2)
        
        else:
            raise ValueError(f"Unknown fake data method: {method}")
        
        return X
    
    def create_dataset(self, real_method='blobs', fake_method='uniform', 
                      n_samples=1000, n_features=2, test_size=0.2, **kwargs):
        """
        Create a complete labeled dataset with real and fake data.
        """
        print(f"Generating dataset with {n_features}D data...")
        print(f"Real data method: {real_method}, Fake data method: {fake_method}")
        
        # Generate real data (label = 1)
        X_real = self.generate_real_data(method=real_method, n_samples=n_samples//2, 
                                       n_features=n_features, **kwargs)
        y_real = np.ones(len(X_real))
        
        # Generate fake data (label = 0)
        X_fake = self.generate_fake_data(method=fake_method, n_samples=n_samples//2, 
                                       n_features=n_features, **kwargs)
        y_fake = np.zeros(len(X_fake))
        
        # Combine datasets
        X = np.vstack([X_real, X_fake])
        y = np.hstack([y_real, y_fake])
        
        # Shuffle the data
        indices = np.random.permutation(len(X))
        X, y = X[indices], y[indices]
        
        # Split into train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        # Store dataset info
        self.dataset_info = {
            'real_method': real_method,
            'fake_method': fake_method,
            'n_features': n_features,
            'n_samples': n_samples,
            'train_size': len(X_train),
            'test_size': len(X_test)
        }
        
        return X_train, X_test, y_train, y_test
    
    def prepare_models(self):
        """Initialize different ML models for comparison."""
        self.models = {
            'Logistic Regression': LogisticRegression(random_state=self.random_state, max_iter=1000),
            'SVM': SVC(probability=True, random_state=self.random_state),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=self.random_state),
            'XGBoost': xgb.XGBClassifier(random_state=self.random_state, eval_metric='logloss')
        }
    
    def train_and_evaluate(self, X_train, X_test, y_train, y_test, scale_data=True):
        """
        Train all models and evaluate their performance.
        """
        # Scale data if requested
        if scale_data:
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
        else:
            X_train_scaled = X_train
            X_test_scaled = X_test
        
        self.prepare_models()
        
        print(f"\nTraining and evaluating {len(self.models)} models...")
        print("=" * 60)
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            
            # Train model
            model.fit(X_train_scaled, y_train)
            
            # Predictions
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            
            # Metrics
            accuracy = accuracy_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            conf_matrix = confusion_matrix(y_test, y_pred)
            
            # Cross-validation score
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='roc_auc')
            
            # Store results
            self.results[name] = {
                'model': model,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba,
                'accuracy': accuracy,
                'roc_auc': roc_auc,
                'confusion_matrix': conf_matrix,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
            
            print(f"Accuracy: {accuracy:.4f}")
            print(f"ROC AUC: {roc_auc:.4f}")
            print(f"CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        return X_test_scaled, y_test
    
    def plot_results(self, X_test, y_test, save_plots=False):
        """
        Create comprehensive visualizations of results.
        """
        n_models = len(self.results)
        
        # If 2D data, create decision boundary plots
        if self.dataset_info['n_features'] == 2:
            fig = plt.figure(figsize=(20, 15))
            
            # Data distribution plot
            plt.subplot(3, 3, 1)
            scatter = plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='RdYlBu', alpha=0.7)
            plt.colorbar(scatter)
            plt.title('True Data Distribution\n(Red=Fake, Blue=Real)')
            plt.xlabel('Feature 1')
            plt.ylabel('Feature 2')
            
            # Decision boundary plots for each model
            for idx, (name, result) in enumerate(self.results.items()):
                plt.subplot(3, 3, idx + 2)
                self._plot_decision_boundary(X_test, result['model'], name)
        
        else:
            fig = plt.figure(figsize=(16, 12))
        
        # Performance comparison plots
        if self.dataset_info['n_features'] == 2:
            plt.subplot(3, 3, 6)
        else:
            plt.subplot(2, 3, 1)
        
        # Accuracy comparison
        accuracies = [result['accuracy'] for result in self.results.values()]
        model_names = list(self.results.keys())
        bars = plt.bar(model_names, accuracies, color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
        plt.title('Model Accuracy Comparison')
        plt.ylabel('Accuracy')
        plt.xticks(rotation=45)
        for bar, acc in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{acc:.3f}', ha='center', va='bottom')
        
        # ROC AUC comparison
        if self.dataset_info['n_features'] == 2:
            plt.subplot(3, 3, 7)
        else:
            plt.subplot(2, 3, 2)
        
        roc_aucs = [result['roc_auc'] for result in self.results.values()]
        bars = plt.bar(model_names, roc_aucs, color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
        plt.title('Model ROC AUC Comparison')
        plt.ylabel('ROC AUC')
        plt.xticks(rotation=45)
        for bar, auc in zip(bars, roc_aucs):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{auc:.3f}', ha='center', va='bottom')
        
        # ROC Curves
        if self.dataset_info['n_features'] == 2:
            plt.subplot(3, 3, 8)
        else:
            plt.subplot(2, 3, 3)
        
        colors = ['blue', 'red', 'green', 'orange']
        for (name, result), color in zip(self.results.items(), colors):
            fpr, tpr, _ = roc_curve(y_test, result['y_pred_proba'])
            plt.plot(fpr, tpr, color=color, label=f"{name} (AUC = {result['roc_auc']:.3f})")
        
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Confusion matrices
        if self.dataset_info['n_features'] == 2:
            plt.subplot(3, 3, 9)
        else:
            plt.subplot(2, 3, 4)
        
        # Show confusion matrix for best performing model
        best_model = max(self.results.items(), key=lambda x: x[1]['roc_auc'])
        conf_matrix = best_model[1]['confusion_matrix']
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
        plt.title(f'Confusion Matrix - {best_model[0]}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Cross-validation scores
        if self.dataset_info['n_features'] != 2:
            plt.subplot(2, 3, 5)
            cv_means = [result['cv_mean'] for result in self.results.values()]
            cv_stds = [result['cv_std'] for result in self.results.values()]
            plt.errorbar(range(len(model_names)), cv_means, yerr=cv_stds, 
                        fmt='o', capsize=5, capthick=2)
            plt.title('Cross-Validation ROC AUC Scores')
            plt.ylabel('CV ROC AUC')
            plt.xticks(range(len(model_names)), model_names, rotation=45)
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(f'real_fake_classifier_results_{self.dataset_info["n_features"]}D.png', 
                       dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def _plot_decision_boundary(self, X, model, title):
        """Plot decision boundary for 2D data."""
        h = 0.02
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))
        
        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        mesh_points_scaled = self.scaler.transform(mesh_points)
        Z = model.predict_proba(mesh_points_scaled)[:, 1]
        Z = Z.reshape(xx.shape)
        
        plt.contourf(xx, yy, Z, alpha=0.6, cmap='RdYlBu')
        plt.title(f'{title}')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
    
    def print_detailed_results(self):
        """Print detailed performance metrics for all models."""
        print("\n" + "="*80)
        print("DETAILED PERFORMANCE RESULTS")
        print("="*80)
        
        print(f"\nDataset Information:")
        print(f"- Real data method: {self.dataset_info['real_method']}")
        print(f"- Fake data method: {self.dataset_info['fake_method']}")
        print(f"- Number of features: {self.dataset_info['n_features']}")
        print(f"- Total samples: {self.dataset_info['n_samples']}")
        print(f"- Training samples: {self.dataset_info['train_size']}")
        print(f"- Test samples: {self.dataset_info['test_size']}")
        
        for name, result in self.results.items():
            print(f"\n{'-'*50}")
            print(f"MODEL: {name}")
            print(f"{'-'*50}")
            print(f"Accuracy: {result['accuracy']:.4f}")
            print(f"ROC AUC: {result['roc_auc']:.4f}")
            print(f"Cross-validation AUC: {result['cv_mean']:.4f} ± {result['cv_std']:.4f}")
            
            print(f"\nConfusion Matrix:")
            print(f"                Predicted")
            print(f"              Fake  Real")
            print(f"Actual Fake   {result['confusion_matrix'][0,0]:4d}  {result['confusion_matrix'][0,1]:4d}")
            print(f"       Real   {result['confusion_matrix'][1,0]:4d}  {result['confusion_matrix'][1,1]:4d}")
            
            # Calculate additional metrics
            tn, fp, fn, tp = result['confusion_matrix'].ravel()
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            print(f"\nAdditional Metrics:")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1-Score: {f1:.4f}")

def run_comprehensive_tests():
    """Run multiple test scenarios to validate the classifier."""
    print("RUNNING COMPREHENSIVE REAL vs FAKE DATA CLASSIFICATION TESTS")
    print("="*70)
    
    test_cases = [
        # 2D visualizable cases
        {
            'name': '2D Blobs vs Uniform',
            'real_method': 'blobs',
            'fake_method': 'uniform',
            'n_features': 2,
            'n_samples': 1000
        },
        {
            'name': '2D Moons vs Gaussian',
            'real_method': 'moons',
            'fake_method': 'gaussian_different',
            'n_features': 2,
            'n_samples': 1000
        },
        {
            'name': '2D Multivariate Normal vs Random Noise',
            'real_method': 'multivariate_normal',
            'fake_method': 'random_noise',
            'n_features': 2,
            'n_samples': 1000
        },
        # High-dimensional cases
        {
            'name': '128D Blobs vs Uniform',
            'real_method': 'blobs',
            'fake_method': 'uniform',
            'n_features': 128,
            'n_samples': 2000
        },
        {
            'name': '64D Multivariate Normal vs Gaussian Different',
            'real_method': 'multivariate_normal',
            'fake_method': 'gaussian_different',
            'n_features': 64,
            'n_samples': 1500
        }
    ]
    
    all_results = []
    
    for i, test_case in enumerate(test_cases):
        print(f"\n{'='*70}")
        print(f"TEST CASE {i+1}: {test_case['name']}")
        print(f"{'='*70}")
        
        # Initialize classifier
        classifier = RealFakeClassifier(random_state=42)
        
        # Create dataset
        X_train, X_test, y_train, y_test = classifier.create_dataset(
            real_method=test_case['real_method'],
            fake_method=test_case['fake_method'],
            n_features=test_case['n_features'],
            n_samples=test_case['n_samples']
        )
        
        # Train and evaluate
        X_test_scaled, y_test = classifier.train_and_evaluate(X_train, X_test, y_train, y_test)
        
        # Print detailed results
        classifier.print_detailed_results()
        
        # Create visualizations
        classifier.plot_results(X_test_scaled, y_test)
        
        # Store results for comparison
        best_model = max(classifier.results.items(), key=lambda x: x[1]['roc_auc'])
        all_results.append({
            'test_case': test_case['name'],
            'best_model': best_model[0],
            'best_auc': best_model[1]['roc_auc'],
            'best_accuracy': best_model[1]['accuracy']
        })
    
    # Summary of all test cases
    print(f"\n{'='*70}")
    print("SUMMARY OF ALL TEST CASES")
    print(f"{'='*70}")
    
    for result in all_results:
        print(f"{result['test_case']:<35} | Best: {result['best_model']:<18} | "
              f"AUC: {result['best_auc']:.4f} | Acc: {result['best_accuracy']:.4f}")

if __name__ == "__main__":
    # Run comprehensive tests
    run_comprehensive_tests()
    
    print(f"\n{'='*70}")
    print("TESTING COMPLETE!")
    print("All test cases have been executed with detailed analysis.")
    print(f"{'='*70}")
    