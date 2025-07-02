import torch

def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)

def root_mean_squared_error(output, target):
    with torch.no_grad():
        rmse = torch.sqrt(torch.mean((output - target)**2))
    return rmse

def accuracy(output, target, threshold=0.2):
    with torch.no_grad():
        if output.min() < 0 or output.max() > 1: # 出力範囲でlogitか確率かを簡易判定
            output = torch.sigmoid(output)

        # 予測を0または1に二値化
        pred = (output > threshold).float()

        # 正しく分類された要素の数を数える
        # targetもfloat型であることを確認 (BCEWithLogitsLossを使う場合は通常float)
        correct = (pred == target).sum().item()

        # 全要素数で割る
        total_elements = target.numel() # numel() はテンソルの全要素数を返す
        
        acc = correct / total_elements
    return acc

def precision(output, target, threshold=0.2):
    TP, FP, FN, TN = _calculate_confusion_matrix_elements(output, target, threshold)
    return TP / (TP + FP + 1e-7) if (TP + FP) > 0 else 0.0

def recall(output, target, threshold=0.2):
    TP, FP, FN, TN = _calculate_confusion_matrix_elements(output, target, threshold)
    return TP / (TP + FN + 1e-7) if (TP + FN) > 0 else 0.0

def f1_score(output, target, threshold=0.2):
    prec = precision(output, target, threshold)
    rec = recall(output, target, threshold)
    return 2 * (prec * rec) / (prec + rec + 1e-7) if (prec + rec) > 0 else 0.0

# 混同行列要素を返すヘルパー関数 (上記関数の内部で使う)
def _calculate_confusion_matrix_elements(output, target, threshold):
    if output.min() < 0 or output.max() > 1:
        output = torch.sigmoid(output)
    pred = (output > threshold).float()
    target = target.float()

    TP = ((pred == 1) & (target == 1)).sum().item()
    FP = ((pred == 1) & (target == 0)).sum().item()
    FN = ((pred == 0) & (target == 1)).sum().item()
    TN = ((pred == 0) & (target == 0)).sum().item()
    return TP, FP, FN, TN