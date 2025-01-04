# 导入包
import torch
import warnings
from tqdm.auto import tqdm
warnings.filterwarnings("ignore")

def conloss(attention1,attention3):
    attention1 = torch.mean(attention1,dim=1)
    attention3 = torch.mean(attention3,dim=1)
    return torch.norm(attention1 - attention3, p='fro')


# 训练函数
def train(model, train_loader, optimizer, criterion, fold, logger, device,sup,alpha):
    model.train()
    running_loss = 0.0
    total_correct = 0
    total_samples = 0
    # attention1_all =[]
    # attention2_all =[]
    # To_all = []
    # with torch.set_grad_enabled(True):
    #     with tqdm(total=len(train_loader), desc=f'Training Fold {fold}', leave=True) as pbar:
    for batch_idx, (indexs, X,PD_time,labels) in enumerate(train_loader):
        indexs = indexs.to(device)
        X = X.to(device)
        PD_time = PD_time.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        if sup:
            outputs,loss1 = model(indexs,X,PD_time,device)
            loss = criterion(outputs, labels)
            loss = loss + alpha*loss1
        else:
            outputs = model(indexs,X,PD_time,device)
            # print(outputs)
            # print(labels)
            loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        prob = outputs.softmax(1)
        total_correct += (predicted == labels).sum().item()
        total_samples += labels.size(0)

        logger.add(k=fold, pred=predicted.detach().cpu().numpy(),
                    true=labels.detach().cpu().numpy(),
                    prob=prob.detach().cpu().numpy())
                # attention1_all.append(attention1)
                # attention2_all.append(attention2)
                # To_all.append(To)
                # 更新进度条
                # pbar.update(1)
                # pbar.set_postfix(loss=(running_loss / (batch_idx + 1)), acc=(total_correct / total_samples))
    # attention1_all = torch.cat(attention1_all)
    # attention2_all = torch.cat(attention2_all)
    # To_all = torch.cat(To_all)
    train_metric = logger.evaluate(fold)
    # return train_metric, running_loss,To_all,attention1_all,attention2_all
    return train_metric,running_loss
# 验证函数
def validate(model, val_loader, criterion, fold, logger, device,sup,alpha):
    model.eval()
    total_correct = 0
    total_samples = 0
    total_loss = 0.0
    error_index = []
    # attention1_all =[]
    # attention2_all =[]
    # To_all = []
    # with torch.no_grad():
    #     with tqdm(total=len(val_loader), desc=f'Validating Fold {fold}', leave=True) as pbar:
    for batch_idx, (indexs, X,PD_time,labels) in enumerate(val_loader):
        indexs = indexs.to(device)
        X = X.to(device)
        PD_time = PD_time.to(device)
        labels = labels.to(device)
        if sup:
            outputs,loss1 = model(indexs,X,PD_time,device)
            loss = criterion(outputs, labels)
            loss = loss + alpha*loss1
        else:
            outputs = model(indexs, X,PD_time,device)
            # print(outputs)
            # print(labels)
            loss = criterion(outputs, labels)
        _, predicted = torch.max(outputs, 1)
        prob = outputs.softmax(1)
        total_loss += loss.item()
        total_correct += (predicted == labels).sum().item()
        total_samples += labels.size(0)
        error_index += (predicted != labels).cpu().numpy().tolist()

        logger.add(k=fold, pred=predicted.detach().cpu().numpy(),
                    true=labels.detach().cpu().numpy(),
                    prob=prob.detach().cpu().numpy())
                # attention1_all.append(attention1)
                # attention2_all.append(attention2)
                # To_all.append(To)
                # 更新进度条
                # pbar.update(1)
                # pbar.set_postfix(loss=(total_loss / (batch_idx + 1)), acc=(total_correct / total_samples))
    # attention1_all = torch.cat(attention1_all)
    # attention2_all = torch.cat(attention2_all)
    # To_all = torch.cat(To_all)
    val_metric = logger.evaluate(fold)
    # return val_metric, total_loss,error_index,To_all,attention1_all,attention2_all
    return val_metric,total_loss,error_index