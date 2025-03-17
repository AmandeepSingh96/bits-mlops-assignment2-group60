import os
import pandas as pd
import sweetviz as sv
from torchvision import datasets, transforms

# Define directories
import os

# Handle paths differently for Colab
# Define BASE_DIR safely (compatible with both local and Colab)
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Works in local scripts
except NameError:
    BASE_DIR = "/content"  # Default to Colab path if __file__ is not available

DATA_DIR = os.path.join(BASE_DIR, "data")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)



def load_fashion_mnist():
    transform = transforms.ToTensor()
    train_dataset = datasets.FashionMNIST(root=os.path.join(DATA_DIR, "raw"), train=True, download=True,
                                          transform=transform)

    # Convert to Pandas DataFrame
    images = train_dataset.data.numpy().reshape(-1, 28 * 28)  # Flatten 28x28 images
    labels = train_dataset.targets.numpy()
    df = pd.DataFrame(images)
    df['label'] = labels

    return df


def generate_sweetviz_report(df):
    report = sv.analyze(df, pairwise_analysis="off")  # Disable pairwise correlations for faster execution
    report_path = os.path.join(REPORTS_DIR, "sweetviz_report.html")
    report.show_html(report_path)
    print(f"Sweetviz report saved to {report_path}")