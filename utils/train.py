import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils.models import mlp_model
from utils.plots import plot_confusion


class EarlyStopping(object):

    def __init__(self, patience=2, save_path="model.pth", norm_min_max_dic="", cuda_vis_dev="0"):
        self._min_loss = np.inf
        self._patience = patience
        self._path = save_path
        self.__counter = 0
        self._norm_min_max_dic = norm_min_max_dic
        self._cuda_vis_dev = cuda_vis_dev
 
    def should_stop(self, model, loss):
        if loss < self._min_loss:
            self._min_loss = loss
            self.__counter = 0

            state_dict = model.state_dict()
            state_dict.update(self._norm_min_max_dic)

            torch.save(state_dict, self._path)

        elif loss > self._min_loss:
            self.__counter += 1
            if self.__counter >= self._patience:
                return True
            
        return False
   
    def load(self, model):
        device = torch.device(f"cuda:{self._cuda_vis_dev}") if torch.cuda.is_available() else "CPU"
        state_dict = torch.load(self._path, weights_only=True, map_location=device)
        
        model_state_dict = {k: v for k, v in state_dict.items() if k in model.state_dict()} 
        norm_min_max_dic = {k: v for k, v in state_dict.items() if k not in model.state_dict()}
        model.load_state_dict(model_state_dict)
        
        return model.to(device), norm_min_max_dic
    
    @property
    def counter(self):
        return self.__counter
    

def to_one_hot(number, num_classes):
    one_hot_vector = np.zeros(num_classes)
    one_hot_vector[number] = 1
    return one_hot_vector


def Inference(model, data, target):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # loss_fn = nn.CrossEntropyLoss()
    loss_fn = nn.BCELoss()

    with torch.no_grad():  
        model.eval()
        x, y = data.float().to(device), target.to(device)
        output = model(x)
        n_classes = output.shape[1]
        y = torch.from_numpy(np.array([to_one_hot(label, n_classes) for label in y.cpu().tolist()], dtype=np.float32)).to(device)
        loss = loss_fn(output, y).item()

        return output, loss


def Accuracy(output, target, class_names=None):
    softmax = nn.Softmax(dim=1)
    y_hat = softmax(output.cpu()).argmax(dim=1)
    labels = target.cpu()
    # Use provided class names, or fall back to generic "Class 0", "Class 1", ...
    if class_names is None:
        n_classes = output.shape[1]
        class_names = ['Control'] + [f'Class_{i}' for i in range(1, n_classes)]
    report_str  = classification_report(labels, y_hat, target_names=class_names, zero_division=0)
    report_dict = classification_report(labels, y_hat, target_names=class_names, zero_division=0, output_dict=True)
    return_values = [confusion_matrix(labels, y_hat), report_str, report_dict]
    
    return return_values


def loss_graph(train_loss_list, val_loss_list, test_loss_list, early_stop_epoch, cv_fold_number):
    plt.figure()
    plt.plot(train_loss_list, label="Train Loss")
    plt.plot(val_loss_list, label="Validation Loss")
    plt.plot(test_loss_list, label="Test Loss")
    plt.axvline(early_stop_epoch, color="grey", linestyle="--", label="Early Stop Epoch")
    
    plt.xlabel("Epoch")
    plt.legend()
    plt.savefig(f"Loss_graph/Loss_graph_CV-{cv_fold_number}")
    plt.show()  
    plt.close()


def training(x_data, y_data, min_max_dic, all_skfold,
             N_EPOCH=4000, early_stop_patience=500, min_epoch=100, learning_rate=0.000005, batch_num=64,
             export_path='./Export/CrossValidation', model_save_path="Trained_models",
             cuda_vis_dev=0, label_dim=None, label_dic=None):
    
    test_result_list = []
    cv_reports = []   # list of (fold, val_report_dict, test_report_dict)
    
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Training on {device}...')

    parameter_dic = {}
    init = 1
    for fold, (idx_train, idx_val_test) in enumerate(all_skfold.split(x_data, y_data)):
        idx_val = idx_val_test[:int(len(idx_val_test)/2)]
        idx_test = idx_val_test[int(len(idx_val_test)/2):]
    
        train_data = x_data[idx_train]
        train_target = y_data[idx_train]
        val_data = x_data[idx_val]
        val_target = y_data[idx_val]
        test_data = x_data[idx_test]
        test_target = y_data[idx_test]
        
        print('Number of data')
        print(f"\nTraining Data: {len(train_data)}  Validation Data: {len(val_data)} Test Data: {len(test_data)}")

        input_dim = x_data.shape[1]
        # Infer label_dim from data if not explicitly provided
        n_classes = label_dim if label_dim is not None else int(y_data.max().item()) + 1
        # Build label/class-name lookup used for Accuracy() and plot_confusion()
        _label_dic = label_dic if label_dic is not None else \
            {i: ('Control' if i == 0 else f'Class_{i}') for i in range(n_classes)}
        _class_names = list(_label_dic.values())
        model = mlp_model(dim1=input_dim, label_dim=n_classes).to(device)

        if init == 1:
            print(model)
            init = 0

        optimizer = torch.optim.NAdam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))#, eps=1e-8, weight_decay=4e-5)
        # loss_fn = nn.CrossEntropyLoss()
        loss_fn = nn.BCELoss()
    
        train_loss_list = []
        val_loss_list = []
        val_acc_list = []
        test_loss_list=[]
        
        # cm = np.zeros((N_EPOCH, 5, 5))
    
        early_stopper = EarlyStopping(patience=early_stop_patience, 
                                      save_path=f"./{model_save_path}/model_{fold+1}.pth", 
                                      norm_min_max_dic=min_max_dic, 
                                      cuda_vis_dev=cuda_vis_dev)  # The save path should be modified.
    
        early_stop=False
        if batch_num >= len(idx_train):
            batch_num = len(idx_train)
        
        print()
        for epoch in tqdm(range(N_EPOCH), desc=f"Cross Validation Fold: {fold + 1}", ncols=200):

            # training
            train_loss = 0.0
            model.train()

            if batch_num <= len(idx_train):
                iter = round(len(idx_train) / batch_num)
            else:
                iter=1
            
            for k in range(iter): # Divide the training data into ten batches (84 each) and train.
                start = k * batch_num
                end = (k + 1) * batch_num
                x, y = train_data[start:end].float().to(device), train_target[start:end].to(device)
                
                optimizer.zero_grad()
                output = model(x)
                y = torch.from_numpy(np.array([to_one_hot(label, n_classes) for label in y.cpu().tolist()], dtype=np.float32)).to(device)
                loss = loss_fn(output, y)
    
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                
            train_loss = train_loss / iter
            val_output, val_loss = Inference(model, val_data, val_target)
            val_accuracy = Accuracy(val_output, val_target, class_names=_class_names)
            
            val_acc_list.append(val_accuracy)
            train_loss_list.append(train_loss)
            val_loss_list.append(val_loss)
    
            #__
            test_output, test_loss = Inference(model, test_data, test_target)
            test_loss_list.append(test_loss)
            #___
    
            # cm[epoch] = update_cm(cm[epoch], y, output)

            # Determine early stop (early stopper starts after 100 epochs)
            if epoch > min_epoch and early_stopper.should_stop(model, val_loss): 
                
                early_stop = True
                early_stop_epoch = epoch - early_stopper.counter
                val_acc_final = val_acc_list[-early_stopper.counter-1]
                val_loss_final = val_loss_list[-early_stopper.counter-1]
    
                val_output, val_loss = Inference(model, val_data, val_target)
                val_accuracy = Accuracy(val_output, val_target, class_names=_class_names)

                test_model, min_max_dic = early_stopper.load(model)
                
                test_output, test_loss = Inference(test_model, test_data, test_target)
                test_accuracy = Accuracy(test_output, test_target, class_names=_class_names)

                break

        if early_stop == True:
            print(f"\nEarly Stop Epoch: {early_stop_epoch}")
            
            # Save model (same with that saved by Early Stopper but include accuracy info in the file name)____________
            test_result_list.append((fold+1, val_acc_final, val_loss_final, test_accuracy, test_loss))
            cv_reports.append((fold + 1, val_acc_final[2], test_accuracy[2]))
            
            print("Validation Confusion Matrix:\n")
            plot_confusion(val_accuracy[0], _label_dic)
            print("Validation Classification Report:\n")
            
            print(val_accuracy[1])
            print("Test Confusion Matrix:\n")
            plot_confusion(test_accuracy[0], _label_dic)
            print("Test Classification Report:\n")
            print(test_accuracy[1])
                
        if early_stop == False:
            val_output, val_loss = Inference(model, val_data, val_target)
            val_accuracy = Accuracy(val_output, val_target, class_names=_class_names)
            test_output, test_loss = Inference(model, test_data, test_target)
            test_accuracy = Accuracy(test_output, test_target, class_names=_class_names)
    
            early_stop_epoch = epoch

            cv_reports.append((fold + 1, val_accuracy[2], test_accuracy[2]))
            print(val_accuracy[0])
            print(val_accuracy[1])
            print(test_accuracy[0])
            print(test_accuracy[1])
            
        # Draw loss graph
        loss_graph(train_loss_list[1:], val_loss_list[1:], test_loss_list[1:], early_stop_epoch-1, fold+1)

        init=0
        for key, value in model.state_dict().items():
            if init == 0:
                parameter_dic[f"model_{fold+1}"] = value
                init = 1
    
    report_columns=["CV_Fold", "Val Acc", "Val Loss", "Test Acc", "Test Loss"]
    report_pd = pd.DataFrame(test_result_list, columns=report_columns)
    report_pd.to_csv(f"./{model_save_path}/{model_save_path}_test_report.csv", index=False)

    return parameter_dic, cv_reports

