import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix

def createConfMatrix(targets, preds, set_type = "Validation", save_path = "../OutputImages/", count=0):
    save_path = f"{save_path}{set_type}ConfMatrix{count}"
    print(f"Accuracy on {set_type} data by : {accuracy_score(targets, preds)*100}")
    cf_matrix = confusion_matrix(targets, preds)
    plt.figure(figsize=(12,8))
    sns.heatmap(cf_matrix, annot=True, annot_kws={"size": 10})  # 添加fmt参数并调整annot_kws
    plt.title(f"Confusion Matrix for Classifier on {set_type} Data")
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()

def createLinearFig(accuracy_lis: list, save_path = "../OutputImages/FourEvaluateRates"):

    # print(accuracy_lis)
    labels = ["Accuracy", "Precision", "F1-Score"]
    x = range(1, len(accuracy_lis[0]) + 1)

    plt.figure(figsize=(14, 8))
    # plt.plot(x, accuracy_lis, marker='o')
    for sublist, label in zip(accuracy_lis, labels):
        print(f"labels : {labels}>>sublist: {sublist}")
        plt.plot(x, sublist, marker='o', label = label)

    plt.title("4 Evaluate Rates")
    plt.xlabel('EPOCH')
    plt.ylabel('Metric')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

    plt.grid(True)
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    # createLinearFig([[1, 2, 3, 4], [2, 5, 6, 7], [3, 5, 6, 7], [8, 8, 3, 2]])
    createLinearFig([[0.5555555555555556, 0.4983164983164983, 0.7003367003367004], [0.594458821357941, 0.5373507935733005, 0.7369091625180927], [0.5555555555555556, 0.4983164983164983, 0.7003367003367004], [0.5087823905444046, 0.44874276311624944, 0.6737597754506321]])

