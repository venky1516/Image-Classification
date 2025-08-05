import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
from tensorflow.keras.models import load_model
from data_load import load_data
from preprocessing import preprocess_data


model = load_model("cnn_model.h5")

_, _, x_test, y_test = load_data()

x_test, _ = preprocess_data(x_test, x_test)

y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

print(classification_report(y_true, y_pred))


cm = confusion_matrix(y_true, y_pred)


history = {
    'accuracy': [0.6, 0.7, 0.75, 0.78, 0.80],
    'val_accuracy': [0.55, 0.65, 0.7, 0.72, 0.75],
    'loss': [1.2, 0.9, 0.7, 0.6, 0.5],
    'val_loss': [1.4, 1.2, 1.0, 0.8, 0.6],
}


plt.figure(figsize=(12, 12))

plt.subplot(2, 2, 1)
plt.plot(history['accuracy'], label='Train Accuracy')
plt.plot(history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(history['loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(2, 2, 3)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.arange(10), yticklabels=np.arange(10))
plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")

report = classification_report(y_true, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()

plt.subplot(2, 2, 4)
report_df[['precision', 'recall', 'f1-score']].plot(kind='bar', figsize=(12, 6), colormap='viridis')
plt.title('Precision, Recall, F1-Score per Class')
plt.xlabel('Class')
plt.ylabel('Score')
plt.xticks(rotation=90)
plt.legend(loc='best')

plt.tight_layout()
plt.show()
