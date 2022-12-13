import torch
import torch.nn as nn
import numpy
import itertools as it
from dataprocessor import DataProcessor
from torch.nn.functional import cross_entropy, mse_loss

def one_hot(x, class_count):
	# 第一构造一个[class_count, class_count]的对角线为1的向量
	# 第二保留label对应的行并返回
	return torch.eye(class_count)[x,:]

class Trainer():
    def __init__(self, 
        method: str = "baseline",
        epoch: int = 50,
        batch_size: int = 64,
        class_num: int = 10,
        instructor: nn.Module = None, 
        learner: nn.Module = None,
        optimizer_learner: torch.optim = None,
        optimizer_retrain_learner: torch.optim = None,
        optimizer_instructor: torch.optim = None,
        data_processor: DataProcessor = None,
        writer: torch.utils.tensorboard.SummaryWriter = None,
        device: str = "cpu"
    ):
        self.method = method
        self.epochs = epoch
        self.batch_size = batch_size
        self.class_num = class_num
        self.instructor = instructor
        self.learner = learner
        self.optimizer_learner = optimizer_learner
        self.optimizer_retrain_learner = optimizer_retrain_learner
        self.optimizer_instructor = optimizer_instructor
        self.data_processor = data_processor
        self.criterion = torch.nn.CrossEntropyLoss(reduction='none')
        self.device = device
        self.writer = writer

    def train(self):
        data_loader_train, data_loader_val, data_loader_test = self.data_processor.load_data()
        data_loader_error = self.data_processor.build_error_set()
        data_loader_train_and_error = self.data_processor.build_train_and_error_set()
        train_iteration = 0
        retrain_iteration = 0
        instructor_train_iteration = 0
        train_losses = [0.]
        retrain_losses = [0.]
        for epoch in range(self.epochs):
            # train learner model
            self.learner.train()
            for images, labels, indices in iter(data_loader_train):
                images = images.to(self.device)
                labels = labels.to(self.device)
                pred_y = self.learner(images)

                learner_loss = cross_entropy(pred_y, labels)
                self.optimizer_learner.zero_grad()
                learner_loss.backward()
                self.optimizer_learner.step()
                _, learner_id = torch.max(pred_y.data, 1)
                learner_nums_correct = torch.sum(learner_id == labels.data)
                
                self.writer.add_scalar("loss/train_learner", learner_loss.item(), train_iteration)
                self.writer.add_scalar("accuracy/train_learner", learner_nums_correct / len(images), train_iteration)
                train_iteration += 1
            # build the error set
            self.learner.eval()
            error_indices = []
            for images, labels, indices in iter(data_loader_val):
                images = images.to(self.device)
                labels = labels.to(self.device)
                with torch.no_grad():
                    pred_y = self.learner(images)
                _, id = torch.max(pred_y.data, 1)
                error_indices += indices[torch.nonzero(id != labels.data)].squeeze(1).tolist()
            print(f"The number of incorrect examples is {len(error_indices)}")
            self.writer.add_scalar("numbers/error_set", len(error_indices), epoch)
            self.data_processor.update_error_indices(error_indices)

            # train the instructor model
            self.learner.eval()
            self.instructor.train()
            for images, labels, indices in iter(data_loader_train_and_error):
                images = images.to(self.device)
                labels = labels.to(self.device)
                feature = self.learner.feature_output(images, labels, indices)
                logits = self.instructor(feature)
                loss = cross_entropy(logits, labels)
                _, id = torch.max(logits.data, 1)
                nums_correct = torch.sum(id == labels.data)
                self.writer.add_scalar("loss/train_instructor_on_t&e_set", loss.item(), instructor_train_iteration)
                self.writer.add_scalar("accuracy/instructor_on_t&e_set", nums_correct / len(images), instructor_train_iteration)
                self.optimizer_instructor.zero_grad()
                loss.backward()
                self.optimizer_instructor.step()
                instructor_train_iteration += 1

            target_matrix = torch.zeros([50000, self.class_num]).to(self.device)
            for images, labels, indices in iter(data_loader_train):
                with torch.no_grad():
                    target = self.instructor(self.learner.feature_output(images, labels, indices)).softmax(dim=1)
                target_matrix[indices] = target
            
            # retrain the learner
            self.learner.train()
            num = 0
            if self.method == "baseline":
                data_loader = data_loader_train_and_error
            elif self.method == "upper_bound":
                data_loader = data_loader_val
            elif self.method == "our_method":
                data_loader = data_loader_train
            else:
                data_loader = data_loader_train

            for images, labels, indices in iter(data_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                pred_y = self.learner(images, retrain=True)
                if self.method == "our_method":
                    loss = cross_entropy(pred_y, target_matrix[indices])
                else:
                    loss = cross_entropy(pred_y, labels)
                self.optimizer_learner.zero_grad()
                loss.backward()
                retrain_losses.append(loss.item())
                _, id = torch.max(pred_y.data, 1)
                nums_correct = torch.sum(id == labels.data)

                self.writer.add_scalar("loss/retrain_learner", loss.item(), retrain_iteration)
                self.writer.add_scalar("accuracy/retrain_learner", nums_correct / len(images), retrain_iteration)
                self.optimizer_learner.step()
                retrain_iteration += 1
                num += len(images)
            # test
            print(num)
            self.learner.eval()
            error_res = self.test(data_loader_error)
            test_res = self.test(data_loader_test)
            self.writer.add_scalar("accuracy/test_learner", test_res['acc'], epoch)
            self.writer.add_scalar("accuracy/error_learner", error_res['acc'], epoch)

    def test(self, data_loader):
        result = {}
        test_correct = 0
        num_dataset = 0
        for images, labels, indices in iter(data_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            with torch.no_grad():
                pred_y = self.learner(images)
                # loss = self.criterion(pred_y, labels).unsqueeze(1)
                # onehot = one_hot(labels, 10).to(self.device)
                # p_y = torch.softmax(pred_y, 1).gather(1, labels.unsqueeze(1)).squeeze()
                # top2 = torch.topk(torch.softmax(pred_y, 1), 2).values
                # cond = p_y == top2[:, 0]
                # p_y_ = torch.where(cond, top2[:, 1], top2[:, 0])
                # margin_p = (p_y - p_y_).unsqueeze(1)
                # model_feature = (torch.tensor([iteration, max(losses), max(val_accuracy)])*torch.ones((len(pred_y), 3))).to(self.device)
                # feature = torch.concat([pred_y, loss, margin_p], dim=1)
                # delta_logits = self.instructor(feature)
                # instructor_pred_y = pred_y + delta_logits
            _, id = torch.max(pred_y.data, 1)
            test_correct += torch.sum(id == labels.data)
            num_dataset += len(images)
        result['acc'] = test_correct / num_dataset
        return result


    def train_learner(self):
        data_loader_train, data_loader_val, data_loader_test = self.data_processor.load_data()
        for epoch in range(self.epochs):
            # train learner model
            for images, labels, indices in iter(data_loader_train):
                images = images.to(self.device)
                labels = labels.to(self.device)
                pred_y = self.learner(images)
                loss = self.criterion(pred_y, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            print(f"loss:{loss.item()}")
            # evaluate the performance on validation set
            self.learner.eval()
            test_correct = 0
            for images, labels, indices in iter(data_loader_test):
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.learner(images)
                _, id = torch.max(outputs.data, 1)
                test_correct += torch.sum(id == labels.data)
            print("correct:%.3f%%" % (test_correct / len(self.data_processor.data_test)))