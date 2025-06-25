import torch


def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


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
        squared_diff = torch.square(output - target)

        sum_squared_diff_per_sample = torch.sum(squared_diff, dim=1)
        mse = torch.mean(sum_squared_diff_per_sample)
        rmse = torch.sqrt(mse)

    return rmse