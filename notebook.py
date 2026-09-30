# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "kagglehub==1.0.2",
#     "nbformat==5.11.1",
#     "numpy==2.5.3",
#     "ruff==0.16.9",
#     "scikit-learn==1.9.1",
#     "skorch==1.4.0",
# ]
# ///

import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium", auto_download=["html"])

with app.setup(hide_code=True):
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    import sklearn
    import torch
    import skorch
    import time
    from torch.utils.data import Subset, DataLoader
    from torchvision import datasets, transforms
    from skorch import NeuralNetClassifier
    from pathlib import Path
    from PIL import Image
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import accuracy_score, confusion_matrix
    import seaborn as sns

    if torch.cuda.is_available():
        device = torch.device("cuda")
    # elif torch.is_vulkan_available:
    #     device = torch.device("vulkan")
    else:
        device = torch.device("cpu")
    print("Device:", device)

    sns.color_palette("colorblind")


@app.cell(hide_code=True)
def _(classes):
    def overview_and_plot_cm(
        acc: float,
        best_params,
        cm: confusion_matrix,
        label: str,
        n_cls: int = 2,
    ):
        print(
            f"{label} — Acc: {(100 * acc):.2f}% | Err: {100 * (1 - acc):.2f}%"
        )
        if best_params:
            print(f"Best params: {best_params}")

        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        _fig, _ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            cm_norm,
            annot=True,
            fmt=".1%",
            cmap="Blues",
            xticklabels=classes,
            yticklabels=classes,
            cbar=True,
            ax=_ax,
        )
        _ax.set_title(f"Confusion Matrix - {label}")
        _ax.set_ylabel("True Label")
        _ax.set_xlabel("Predicted Label")
        plt.tight_layout()
        return _fig

    return (overview_and_plot_cm,)


@app.cell(hide_code=True)
def _():
    mo.md("""
    # Machine Learning Project - Pizza or not pizza?

    > * Roberto Pio Iannello
    > * Computer Engineering
    > * s5201538@studenti.unige.it
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Prepairing Data
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Defining Transformations
    There will be 2 pipelines, the first one with traditional Machine Learning and a Convolutional Neural Network from scratch while the second pipeline will involve transfer learning from models trained on ImageNet, for this reason, for the first part we'll use images resized as $64\times 64$ to avoid long computational times since for RGB images of this dimesnions we will have $64\times 64 \times 3 = 12288$ features and for images of the dimensions for `ResNet` and models based on ImageNet are $224 \times 224 \times 3 = 150528$ features
    """)
    return


@app.cell
def _():
    # Values from ImageNet training
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    # CLASSICAL_ML and CNN from scratch pipeline
    IMG_SIZE_64 = (64, 64)  # Define image sizes as tuple
    transf_64 = transforms.Compose(
        [
            # No data Augmentation here
            transforms.Resize(IMG_SIZE_64),
            transforms.ToTensor(),
            # Normalization here isn't necessary since we don't use models pretrained on imagenet
            # We have added this normalization for coherency
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    # Transfer Learning pipeline
    IMG_SIZE_224 = (224, 224)  # Define image size as tuple
    # Prepairing training dataset
    training_transf_224 = transforms.Compose(
        [
            # DATA AUGMENTATION
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            # ADAPT FOR NN
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    # Prepairing test dataset, here we do not have to apply data augmentation
    testing_transf_224 = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return testing_transf_224, training_transf_224, transf_64


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Import and set up dataset
    """)
    return


@app.cell
def _():
    import kagglehub

    # Download latest version
    DATA_DIR = (
        Path(kagglehub.dataset_download("carlosrunner/pizza-not-pizza"))
        / "pizza_not_pizza"
    )

    dataset = datasets.ImageFolder(root=DATA_DIR)
    classes = dataset.classes
    targets = np.array(dataset.targets)

    print("Dataset: ", dataset)
    NUM_CLASSES = len(classes)
    print(NUM_CLASSES, "classes: ", classes)

    _unique_classes, _counts = np.unique(targets, return_counts=True)
    for _cls, _count in zip(_unique_classes, _counts):
        print(f"\t{classes[_cls]}: {_count} images")
    print(f"Targets ({len(targets)}):  {targets}")
    return DATA_DIR, NUM_CLASSES, classes, dataset, targets


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The dataset consists of 1966 images, with 983 images for each class, we can now apply splitting and transformations
    """)
    return


@app.cell
def _(DATA_DIR, targets, testing_transf_224, training_transf_224, transf_64):
    from sklearn.model_selection import train_test_split

    # Define the training/test indexes to have the same for both versions of the dataset
    idx_train, idx_test = train_test_split(
        np.arange(
            len(targets)
        ),  # np.arange(len) generates an array [0,...,len-1]
        test_size=0.2,
        random_state=11,
        stratify=targets,  # to apply labels
    )

    YL = targets[idx_train]  # Labels for elements in the train set
    YT = targets[idx_test]  # Labels for elemebts in the test set

    # Define the datasets with transformations
    dataset_train_64 = datasets.ImageFolder(root=DATA_DIR, transform=transf_64)
    dataset_test_64 = datasets.ImageFolder(root=DATA_DIR, transform=transf_64)
    dataset_train_224 = datasets.ImageFolder(
        root=DATA_DIR, transform=training_transf_224
    )
    dataset_test_224 = datasets.ImageFolder(
        root=DATA_DIR, transform=testing_transf_224
    )
    return (
        YL,
        YT,
        dataset_test_224,
        dataset_test_64,
        dataset_train_224,
        dataset_train_64,
        idx_test,
        idx_train,
        train_test_split,
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    To load data in RAM with numpy we define the following
    """)
    return


@app.function
def load_numpy(dataset, indexes, batch_size=256):
    """Loads a subset in RAM as a  float32 (N, C, H, W)."""
    sub = Subset(dataset, indexes)
    loader = DataLoader(sub, batch_size=batch_size, shuffle=False)
    batches = [xb.numpy() for xb, _ in loader]
    return np.concatenate(batches).astype(np.float32)


@app.cell
def _(
    dataset_test_224,
    dataset_test_64,
    dataset_train_224,
    dataset_train_64,
    idx_test,
    idx_train,
):
    X_train_64 = load_numpy(dataset_train_64, idx_train)
    X_test_64 = load_numpy(dataset_test_64, idx_test)
    X_train_224 = load_numpy(dataset_train_224, idx_train)
    X_test_224 = load_numpy(dataset_test_224, idx_test)
    return X_test_224, X_test_64, X_train_224, X_train_64


@app.cell
def _(classes, dataset):
    # Display 4 sample images from the dataset in a 2x2 grid
    _fig, _axes = plt.subplots(2, 2, figsize=(8, 8))
    _axes = _axes.flatten()

    # Randomly select 4 images from the dataset
    indices = np.random.choice(len(dataset), size=4, replace=False)

    # Define a transform to convert PIL images to tensors
    _to_tensor = transforms.ToTensor()

    for i, _idx in enumerate(indices):
        img, label = dataset[_idx]
        # Convert PIL image to tensor first
        img = _to_tensor(img)
        _img_np = img.permute(1, 2, 0).numpy()
        _img_np = np.clip(_img_np, 0, 1)

        _axes[i].imshow(_img_np)
        _axes[i].set_title(f"{classes[label]}", fontsize=14, fontweight="bold")
        _axes[i].axis("off")

    _fig.suptitle(
        "Sample Images from Pizza vs Not Pizza Dataset",
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Classic Machine Learning
    """)
    return


@app.cell
def _(X_test_64, X_train_64):
    # Features flattening
    XL_flat = X_train_64.reshape(len(X_train_64), -1)
    XT_flat = X_test_64.reshape(len(X_test_64), -1)

    print("XL_flat: ", XL_flat.shape)
    print("XT_flat: ", XT_flat.shape)
    return XL_flat, XT_flat


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    To avoid long computation time due to complexity, we reduce the learning set to 100 elements per class (for example I've interrumpted LinearSVC after 10 minutes).
    """)
    return


@app.cell
def _(XL_flat, YL):
    from random import sample

    num_per_class = 200
    mask = []
    for c in [0, 1]:
        c_idx = np.where(YL == c)[0].tolist()
        mask += sample(c_idx, min(num_per_class, len(c_idx)))
    XL_sub = XL_flat[mask]
    YL_sub = YL[mask]
    return XL_sub, YL_sub


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Linear SVC
    """)
    return


@app.cell
def _(XL_sub, XT_flat, YL_sub, YT):
    from sklearn.svm import LinearSVC

    grid_lsvc = {"C": np.logspace(-2, 3, 5)}
    M_lsvc = GridSearchCV(
        estimator=LinearSVC(),
        param_grid=grid_lsvc,
        cv=5,
        scoring="accuracy",
        verbose=2,
    )
    M_lsvc.fit(XL_sub, YL_sub)
    Yp_lsvc = M_lsvc.predict(XT_flat)
    acc_lsvc = accuracy_score(YT, Yp_lsvc)
    cm_lsvc = confusion_matrix(YT, Yp_lsvc)
    return M_lsvc, acc_lsvc, cm_lsvc


@app.cell
def _(M_lsvc, acc_lsvc, cm_lsvc, overview_and_plot_cm):
    overview_and_plot_cm(acc_lsvc, M_lsvc.best_params_, cm_lsvc, "LinearSVC")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    So the linear model performs a little better than tossing a coin
    """)
    return


@app.cell
def _(M_lsvc, classes):
    _w = M_lsvc.best_estimator_.coef_[0]
    _w_img = _w.reshape(3, 64, 64)
    _w_mag = np.linalg.norm(_w_img, axis=0)

    _fig, _ax = plt.subplots(figsize=(6, 5))

    sns.heatmap(
        _w_mag,
        ax=_ax,
        cmap="RdBu_r",
        square=True,
        cbar=True,
        cbar_kws={"fraction": 0.046, "pad": 0.04},
        xticklabels=False,
        yticklabels=False,
    )

    _ax.set_title(
        f"Feature Weights: {classes[0]} (-) vs {classes[1]} (+)",
        fontsize=11,
    )
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _():
    mo.md(r"""
    ### SVM RBF
    """)
    return


@app.cell
def _(XL_sub, XT_flat, YL_sub, YT):
    from sklearn.svm import SVC

    grid_svm = {
        "C": np.logspace(-2, 3, 5),
        "kernel": ["rbf"],
        "gamma": np.logspace(-4, 1, 5),
    }
    M_svm = GridSearchCV(
        SVC(kernel="rbf"), grid_svm, cv=5, scoring="accuracy", verbose=2
    )

    M_svm.fit(XL_sub, YL_sub)
    Yp_svm = M_svm.predict(XT_flat)
    acc_svm = accuracy_score(YT, Yp_svm)
    cm_svm = confusion_matrix(YT, Yp_svm)
    return M_svm, acc_svm, cm_svm


@app.cell
def _(M_svm, acc_svm, cm_svm, overview_and_plot_cm):
    overview_and_plot_cm(acc_svm, M_svm.best_params_, cm_svm, "SVM RBF")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Random Forest
    """)
    return


@app.cell
def _(XL_sub, XT_flat, YL_sub, YT):
    from sklearn.ensemble import RandomForestClassifier

    grid_rf = {"n_estimators": [100], "max_features": [50, 100, 200]}
    M_rf = GridSearchCV(
        estimator=RandomForestClassifier(),
        param_grid=grid_rf,
        cv=5,
        scoring="accuracy",
        verbose=2,
    )
    M_rf.fit(XL_sub, YL_sub)
    Yp_rf = M_rf.predict(XT_flat)
    acc_rf = accuracy_score(YT, Yp_rf)
    cm_rf = confusion_matrix(YT, Yp_rf)
    return M_rf, acc_rf, cm_rf


@app.cell
def _(M_rf, acc_rf, cm_rf, overview_and_plot_cm):
    overview_and_plot_cm(acc_rf, M_rf.best_params_, cm_rf, "Random Forest")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Conclusions
    As expected, for image recognition the performance of classic machine learning models are not the best, we can also notice that the linear model has a more 'balanced' confusion matrix while the nonlinear models such as SVM RBF and RandomForest are more accurate on detectin `not_pizza` images and an higher error (wrong guesses) with `pizza` images
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Neural Network from scratch

    We'll follow the same approach we've seen in class for creating a NN for 64x64 images
    """)
    return


@app.cell
def _(NUM_CLASSES):
    from torch import nn
    import torch.nn.functional as F

    class CustomCNN(nn.Module):
        def __init__(self):
            super().__init__()
            # Input: (3, 64, 64)
            self.conv1 = nn.Conv2d(3, 32, kernel_size=3)  # -> (32, 64, 64)
            # MaxPool(2) -> (32, 32, 32)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3)  # -> (64, 32, 32)
            self.conv2_drop = nn.Dropout2d(p=0.5)
            # MaxPool(2) -> (64, 14, 14)
            self.fc1 = nn.Linear(64 * 14 * 14, 512)
            self.fc1_drop = nn.Dropout(p=0.5)
            self.fc2 = nn.Linear(512, NUM_CLASSES)

        def forward(self, x):
            x = self.conv1(x)
            x = F.max_pool2d(x, 2)
            x = torch.relu(x)
            x = self.conv2(x)
            x = self.conv2_drop(x)
            x = F.max_pool2d(x, 2)
            x = torch.relu(x)
            x = x.view(-1, x.size(1) * x.size(2) * x.size(2))
            x = self.fc1(x)
            x = self.fc1_drop(x)
            x = torch.relu(x)
            x = self.fc2(x)
            x = torch.softmax(
                x, dim=-1
            )  # probabilità: compatibile con CrossEntropyLoss (default skorch NeuralNetClassifier)
            return x

    return CustomCNN, nn


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    For Neural Networks we have to split into train and validation sets
    """)
    return


@app.cell
def _(X_train_64, YL, train_test_split):
    X_train_cnn_64, X_validation_cnn_64, y_train_cnn_64, y_val_cnn_64 = (
        train_test_split(X_train_64, YL, test_size=0.1)
    )

    print("X_train_cnn_64: ", len(X_train_cnn_64))
    print("X_validation_cnn_64: ", len(X_validation_cnn_64))
    return X_train_cnn_64, y_train_cnn_64


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    We can now train the network
    """)
    return


@app.cell
def _():
    from skorch.callbacks import LRScheduler

    lrs = LRScheduler(policy="StepLR", step_size=10, gamma=0.1)
    return (lrs,)


@app.cell
def _(CustomCNN, X_train_cnn_64, lrs, y_train_cnn_64):
    Alg_cnn = NeuralNetClassifier(
        CustomCNN,
        max_epochs=30,
        lr=0.001,
        optimizer=torch.optim.Adam,
        device=_device,
        callbacks=[lrs],
    )
    Alg_cnn.fit(X_train_cnn_64, y_train_cnn_64)
    return (Alg_cnn,)


@app.cell
def _(Alg_cnn, X_test_64, YT, overview_and_plot_cm):
    Yp_cnn = Alg_cnn.predict(X_test_64)
    acc_cnn = accuracy_score(YT, Yp_cnn)
    cm_cnn = confusion_matrix(YT, Yp_cnn)

    overview_and_plot_cm(acc_cnn, None, cm_cnn, "Custom Neural Network")
    return acc_cnn, cm_cnn


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    With the custom neural network we start to reach more meaningful results but there's still a huge gap, as also seen with traditional Machine Learning, as we can see with the confusion matrix, there's more accuracy for `non_pizza` than for `pizza` elements
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Transfer Learning (ResNet18)
    ### Without Freezing
    """)
    return


@app.cell
def _(NUM_CLASSES, nn):
    from torchvision import models

    # TODO: Add explanations
    class CustomRN18(nn.Module):
        def __init__(self):
            super(CustomRN18, self).__init__()
            pretrained_nn = models.resnet18(pretrained=True)
            d_encoding = pretrained_nn.fc.in_features
            pretrained_nn.fc = nn.Linear(d_encoding, NUM_CLASSES)
            self.model = pretrained_nn

        def forward(self, x):
            return self.model(x)

    return (CustomRN18,)


@app.cell
def _(X_train_224, YL, train_test_split):
    (
        X_train_rn18_224,
        X_validation_rn18_224,
        y_train_rn18_224,
        y_val_rn18_224,
    ) = train_test_split(X_train_224, YL, test_size=0.1)

    print("X_train_cnn_64: ", len(X_train_rn18_224))
    print("X_validation_cnn_64: ", len(X_validation_rn18_224))
    return X_train_rn18_224, y_train_rn18_224


@app.cell
def _(CustomRN18, X_train_rn18_224, lrs, nn, y_train_rn18_224):
    Alg_rn18 = NeuralNetClassifier(
        CustomRN18,
        criterion=nn.CrossEntropyLoss,
        max_epochs=30,
        lr=0.001,
        optimizer=torch.optim.Adam,
        # optimizer__momentum=0.9,
        device=device,
        callbacks=[lrs],
    )
    Alg_rn18.fit(X_train_rn18_224, y_train_rn18_224)
    return (Alg_rn18,)


@app.cell
def _(Alg_rn18, X_test_224, YT, overview_and_plot_cm):
    Yp_rn18 = Alg_rn18.predict(X_test_224)
    acc_rn18 = accuracy_score(YT, Yp_rn18)
    cm_rn18 = confusion_matrix(YT, Yp_rn18)

    overview_and_plot_cm(acc_rn18, None, cm_rn18, "Custom Neural Network")
    return acc_rn18, cm_rn18


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### With Freezing
    """)
    return


@app.cell
def _(CustomRN18, X_train_rn18_224, lrs, nn, y_train_rn18_224):
    from skorch.callbacks import Freezer

    frz18 = Freezer(lambda name: not name.startswith("model.fc"))

    Alg_rn18_frz = NeuralNetClassifier(
        CustomRN18,
        criterion=nn.CrossEntropyLoss,
        max_epochs=30,
        lr=0.001,
        optimizer=torch.optim.SGD,
        optimizer__momentum=0.9,
        device=device,
        callbacks=[lrs, frz18],
    )
    Alg_rn18_frz.fit(X_train_rn18_224, y_train_rn18_224)
    return (Alg_rn18_frz,)


@app.cell
def _(Alg_rn18_frz, X_test_224, YT, overview_and_plot_cm):
    Yp_rn18_frz = Alg_rn18_frz.predict(X_test_224)
    acc_rn18_frz = accuracy_score(YT, Yp_rn18_frz)
    cm_rn18_frz = confusion_matrix(YT, Yp_rn18_frz)

    overview_and_plot_cm(
        acc_rn18_frz, None, cm_rn18_frz, "ResNet18 with frozen hidden layers"
    )
    return acc_rn18_frz, cm_rn18_frz


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Results comparisons
    """)
    return


@app.cell
def _(
    NUM_CLASSES,
    acc_cnn,
    acc_lsvc,
    acc_rf,
    acc_rn18,
    acc_rn18_frz,
    acc_svm,
    classes,
    cm_cnn,
    cm_lsvc,
    cm_rf,
    cm_rn18,
    cm_rn18_frz,
    cm_svm,
):
    # Collect model results
    model_names = [
        "Linear SVC",
        "SVM RBF",
        "Random Forest",
        "Custom CNN",
        "ResNet18\n(no freeze)",
        "ResNet18\n(frozen)",
    ]

    accuracies = [acc_lsvc, acc_svm, acc_rf, acc_cnn, acc_rn18, acc_rn18_frz]
    confusion_matrices = [cm_lsvc, cm_svm, cm_rf, cm_cnn, cm_rn18, cm_rn18_frz]

    # Compute per-class accuracies from confusion matrices
    # cm layout: [[TN, FP], [FN, TP]]
    # Class 0 (not_pizza): TN / (TN + FP)
    # Class 1 (pizza): TP / (TP + FN)
    not_pizza_acc = []
    pizza_acc = []
    for cm in confusion_matrices:
        tn, fp = cm[0]
        fn, tp = cm[1]
        not_pizza_acc.append(tn / (tn + fp))
        pizza_acc.append(tp / (tp + fn))

    baseline = 1.0 / NUM_CLASSES

    # Create the comparison plot
    _fig, _ax = plt.subplots(figsize=(12, 7))

    x = np.arange(len(model_names))
    width = 0.25

    bars_overall = _ax.bar(
        x - width,
        accuracies,
        width,
        label="Overall Accuracy",
        color="#4C72B0",
        edgecolor="white",
        linewidth=0.7,
    )

    bars_not_pizza = _ax.bar(
        x,
        not_pizza_acc,
        width,
        label=f"Accuracy ({classes[0]})",
        color="#55A868",
        edgecolor="white",
        linewidth=0.7,
    )

    bars_pizza = _ax.bar(
        x + width,
        pizza_acc,
        width,
        label=f"Accuracy ({classes[1]})",
        color="#C44E52",
        edgecolor="white",
        linewidth=0.7,
    )

    # Baseline
    _ax.axhline(
        y=baseline,
        color="gray",
        linestyle="--",
        linewidth=1.5,
        alpha=0.8,
        label=f"Baseline (1/{NUM_CLASSES} = {baseline})",
    )

    # Formatting
    _ax.set_xticks(x)
    _ax.set_xticklabels(model_names, fontsize=11)
    _ax.set_ylabel("Accuracy", fontsize=13, fontweight="bold")
    _ax.set_title(
        "Model Performance Comparison",
        fontsize=16,
        fontweight="bold",
        pad=15,
    )
    _ax.set_ylim(0, 1.05)
    _ax.legend(loc="upper left", fontsize=11, framealpha=0.9)
    _ax.grid(axis="y", alpha=0.3, linestyle="--")

    # Add value labels on bars
    for bars in [bars_overall, bars_not_pizza, bars_pizza]:
        for bar in bars:
            height = bar.get_height()
            _ax.annotate(
                f"{height:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )

    plt.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    > [!TODO]
    >
    > * Add heatmap for linearSVC
    > * Add heatmap for a non-pizza image for CNNs
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
