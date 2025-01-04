# 导入包
import os
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
def fun_error_img(result_path,fold_num=5):
    for fold in range(fold_num):
        # 文件夹路径
        folder_path = result_path+'/error_img/fold '+str(fold)+'/'

        # 存储所有文本内容的列表
        text_list = []
        count_epoch = 0
        # 遍历文件夹中的文件
        for filename in os.listdir(folder_path):
            if filename.endswith('.txt'):  # 确保只读取文本文件
                count_epoch += 1
                file_path = os.path.join(folder_path, filename)
                content = []
                with open(file_path, 'r') as file:
                    for line in file:
                        content.append(line.strip())
                    text_list+=content
        img_error = pd.DataFrame({"img_path":pd.DataFrame(text_list)[0].value_counts().index.tolist(),
        "错误次数":pd.DataFrame(text_list)[0].value_counts().values.tolist(),
        "错误比例(除于总训练轮数)":pd.DataFrame(text_list)[0].value_counts().values/count_epoch})
        img_error.to_csv(result_path+'/error_img/error_img_fold '+str(fold)+'.csv',index=0)
def fun_result(result_path,save_path,fold_num=5):
    f=open(result_path, encoding='gbk')
    txt=[]
    for line in f:
        txt.append(line.strip())
    len_plot= int(len(txt)/fold_num)
    result_best_acc = []
    result_best_recall = []
    result_best_pre = []
    result_best_auc = []
    for fold in range(fold_num):
        pat = r'\d+\.\d+|\d+' # A|B，匹配A失败才匹配B
        d = pd.DataFrame(txt[len_plot*fold:len_plot*(fold+1)])[0].apply(lambda x:re.findall(pat,x))

        d = d.loc[pd.Index(list(range(1,len_plot+1,2)))]

        cols = ['train_loss','test_loss','train_acc','test_acc'
         ,'train_pre','test_pre','train_recall','test_recall','train_auc','test_auc']
        list_result=[]
        for i in range(10):
            list_result.append(d.apply(lambda x:x[i]).values.astype("float"))
        result_1 = pd.DataFrame(np.array(list_result).T,columns=cols)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.title("loss fold:"+str(fold))
        plt.plot(result_1['train_loss']/160,label='train_loss')
        plt.plot(result_1['test_loss']/40,label='test_loss')
        plt.ylabel("loss")
        plt.xlabel("epoch")
        plt.legend()
        plt.savefig(save_path+"fold_"+str(fold)+" loss.png")
        plt.close()

        plt.title("acc fold:"+str(fold))
        plt.plot(result_1['train_acc'],label='train_acc')
        plt.plot(result_1['test_acc'],label='test_acc')
        plt.ylabel("acc")
        plt.xlabel("epoch")
        plt.legend()
        plt.savefig(save_path+"fold_"+str(fold)+" acc.png")
        plt.close()

        plt.title("pre fold:"+str(fold))
        plt.plot(result_1['train_pre'],label='train_pre')
        plt.plot(result_1['test_pre'],label='test_pre')
        plt.ylabel("pre")
        plt.xlabel("epoch")
        plt.legend()
        plt.savefig(save_path+"fold_"+str(fold)+" pre.png")
        plt.close()

        plt.title("recall fold:"+str(fold))
        plt.plot(result_1['train_recall'],label='train_recall')
        plt.plot(result_1['test_recall'],label='test_recall')
        plt.ylabel("recall")
        plt.xlabel("epoch")
        plt.legend()
        plt.savefig(save_path+"fold_"+str(fold)+" recall.png")
        plt.close()

        plt.title("auc fold:"+str(fold))
        plt.plot(result_1['train_auc'],label='train_auc')
        plt.plot(result_1['test_auc'],label='test_auc')
        plt.ylabel("auc")
        plt.xlabel("epoch")
        plt.legend()
        plt.savefig(save_path+"fold_"+str(fold)+" auc.png")
        plt.close()

        result_save = result_1.loc[[result_1['test_acc'].idxmax(),
        result_1['test_recall'].idxmax(),
        result_1['test_pre'].idxmax(),
        result_1['test_auc'].idxmax()]]
        result_save.index = ["best_acc","best_recall","best_pre","best_auc"]
        result_save.to_csv(save_path+"fold_"+str(fold)+".csv")
        result_best_acc.append(result_save[['test_acc','test_pre','test_recall','test_auc']].loc['best_acc'].values)
        result_best_recall.append(result_save[['test_acc','test_pre','test_recall','test_auc']].loc['best_recall'].values)
        result_best_pre.append(result_save[['test_acc','test_pre','test_recall','test_auc']].loc['best_pre'].values)
        result_best_auc.append(result_save[['test_acc','test_pre','test_recall','test_auc']].loc['best_auc'].values)
    sns.heatmap(np.array(result_best_acc),annot=True)
    plt.xticks([0.5,1.5,2.5,3.5],["acc","pre","recall","auc"])
    plt.yticks([0.5,1.5,2.5,3.5,4.5],["fold_0","fold_1","fold_2","fold_3","fold_4"])
    plt.savefig(save_path+"Best_acc.png")
    plt.close()
    sns.heatmap(np.array(result_best_recall),annot=True)
    plt.xticks([0.5,1.5,2.5,3.5],["acc","pre","recall","auc"])
    plt.yticks([0.5,1.5,2.5,3.5,4.5],["fold_0","fold_1","fold_2","fold_3","fold_4"])
    plt.savefig(save_path+"Best_recall.png")
    plt.close()
    sns.heatmap(np.array(result_best_pre),annot=True)
    plt.xticks([0.5,1.5,2.5,3.5],["acc","pre","recall","auc"])
    plt.yticks([0.5,1.5,2.5,3.5,4.5],["fold_0","fold_1","fold_2","fold_3","fold_4"])
    plt.savefig(save_path+"Best_pre.png")
    plt.close()
    sns.heatmap(np.array(result_best_auc),annot=True)
    plt.xticks([0.5,1.5,2.5,3.5],["acc","pre","recall","auc"])
    plt.yticks([0.5,1.5,2.5,3.5,4.5],["fold_0","fold_1","fold_2","fold_3","fold_4"])
    plt.savefig(save_path+"Best_auc.png")
    plt.close()
    r_mean = pd.DataFrame({"Best_acc_mean":np.array(result_best_acc).mean(axis=0),
    "Best_recall_mean":np.array(result_best_recall).mean(axis=0),
    "Best_pre_mean":np.array(result_best_pre).mean(axis=0),
    "Best_auc_mean":np.array(result_best_auc).mean(axis=0)},index=['acc','pre','recall','auc'])
    r_mean.to_csv(save_path+"fold_mean.csv")