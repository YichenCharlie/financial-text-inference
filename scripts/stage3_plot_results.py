from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent.parent


OUTPUT_DIR = (
    ROOT /
    "figures/stage3_bert"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================
# Figure 1
# Model Comparison
# =========================

def plot_model_comparison():

    models = [
        "TF-IDF\nLogistic",
        "BERT\nFine-tuning"
    ]

    accuracy = [
        0.7240,
        0.7873
    ]

    macro_f1 = [
        0.6211,
        0.7548
    ]


    x = np.arange(len(models))

    width = 0.35


    plt.figure(
        figsize=(8,5)
    )


    plt.bar(
        x-width/2,
        accuracy,
        width,
        label="Accuracy"
    )

    plt.bar(
        x+width/2,
        macro_f1,
        width,
        label="Macro-F1"
    )


    plt.xticks(
        x,
        models
    )

    plt.ylim(
        0,
        1
    )


    plt.ylabel(
        "Score"
    )

    plt.title(
        "Model Performance Comparison"
    )


    plt.legend()

    plt.tight_layout()


    plt.savefig(
        OUTPUT_DIR /
        "model_comparison.png",
        dpi=300
    )

    plt.close()



# =========================
# Figure 2
# Confusion Matrix
# =========================

def plot_confusion_matrix():

    tfidf = np.array(
        [
            [197,426],
            [110,1209]
        ]
    )


    bert = np.array(
        [
            [411,212],
            [201,1118]
        ]
    )


    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10,4)
    )


    matrices = [
        ("TF-IDF Logistic", tfidf),
        ("BERT Fine-tuning", bert)
    ]


    for ax, (title, matrix) in zip(
        axes,
        matrices
    ):

        im = ax.imshow(
            matrix
        )


        for i in range(2):

            for j in range(2):

                ax.text(
                    j,
                    i,
                    matrix[i,j],
                    ha="center",
                    va="center"
                )


        ax.set_xticks(
            [0,1],
            [
                "Pred 0\nNon-negative",
                "Pred 1\nNegative"
            ]
        )

        ax.set_yticks(
            [0,1],
            [
                "True 0\nNon-negative",
                "True 1\nNegative"
            ]
        )


        ax.set_title(
            title
        )


    plt.tight_layout()


    plt.savefig(
        OUTPUT_DIR /
        "confusion_matrix_comparison.png",
        dpi=300
    )

    plt.close()



# =========================
# Figure 3
# Training Curve
# =========================

def plot_training_curve():

    epochs = [
        1,
        2,
        3
    ]


    train_loss = [
        0.5027,
        0.4105,
        0.3312
    ]


    val_f1 = [
        0.7375,
        0.7485,
        0.7548
    ]


    fig, ax1 = plt.subplots(
        figsize=(8,5)
    )


    ax1.plot(
        epochs,
        train_loss,
        marker="o",
        label="Training Loss"
    )


    ax1.set_xlabel(
        "Epoch"
    )

    ax1.set_ylabel(
        "Training Loss"
    )


    ax2 = ax1.twinx()


    ax2.plot(
        epochs,
        val_f1,
        marker="s",
        label="Validation Macro-F1"
    )


    ax2.set_ylabel(
        "Validation Macro-F1"
    )


    plt.title(
        "BERT Fine-tuning Training Curve"
    )


    fig.tight_layout()


    plt.savefig(
        OUTPUT_DIR /
        "training_curve.png",
        dpi=300
    )


    plt.close()



# =========================
# Figure 4
# High Confidence Errors
# =========================

def plot_high_confidence_errors():

    categories = [
        "High Confidence\nFP",
        "High Confidence\nFN"
    ]


    values = [
        71,
        48
    ]


    plt.figure(
        figsize=(6,4)
    )


    plt.bar(
        categories,
        values
    )


    plt.ylabel(
        "Number of Samples"
    )


    plt.title(
        "High Confidence Error Analysis"
    )


    for i,v in enumerate(values):

        plt.text(
            i,
            v,
            str(v),
            ha="center",
            va="bottom"
        )


    plt.tight_layout()


    plt.savefig(
        OUTPUT_DIR /
        "high_confidence_errors.png",
        dpi=300
    )


    plt.close()



if __name__ == "__main__":

    plot_model_comparison()

    plot_confusion_matrix()

    plot_training_curve()

    plot_high_confidence_errors()


    print(
        "Figures saved to:",
        OUTPUT_DIR
    )