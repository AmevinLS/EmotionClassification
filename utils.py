from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from datasets import load_dataset
from nltk.stem import PorterStemmer


def plot_confusion_matrix(y_true, y_pred, display_labels=None, ax=None):
    conf_matrix = confusion_matrix(y_true, y_pred)
    conf_matrix_display = ConfusionMatrixDisplay(
        conf_matrix,
        display_labels=display_labels
    )
    conf_matrix_display.plot(ax=ax)


def preprocess_word(word: str, stem=True):
    stemmer = PorterStemmer()
    result = word
    if stem:
        result = stemmer.stem(result)
    return result


class EmotionDataset:
    def __init__(self):
        self.dataset = load_dataset("dair-ai/emotion")
        self.LABEL_TO_EMOTION_DICT = {
            0: "sadness",
            1: "joy",
            2: "love",
            3: "anger",
            4: "fear",
            5: "surprise"
        }
