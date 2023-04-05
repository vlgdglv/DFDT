import datetime as dt
import os
import time
import wandb
import torch.optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch import nn, tensor, Tensor
# from torch.utils.tensorboard import SummaryWriter
from model import TC, FocalLoss
from data import ST4Dataset

train_csv_path = './data/train_sample.csv'
test_csv_path = './data/test_sample.csv'
val_csv_path = './data/val_sample.csv'
log_path = "./data/ST4000/preprocessed/"
THRESHOLD = 0.8


class Solver:
    def __init__(self,
                 batch_size, epochs, lr,
                 train_dataloader, test_dataloader, val_dataloader,
                 model, optimizer, criterion, lr_schedular=None, grad_acc_step=3,
                 use_wandb=False, runtime_path="./run/", do_val=False,
                 save=False, log_inter=50
                 ):
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.grad_acc_step = grad_acc_step
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.val_dataloader = val_dataloader
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.lr_schedular = lr_schedular
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_wandb = use_wandb
        self.runtime_path = runtime_path
        self.model_weight_name = "{}_ckpt.pt".format(dt.datetime.strftime(dt.datetime.now(), '%Y-%m-%d-%H_%M_%S'))
        self.model_weight_path = self.runtime_path + self.model_weight_name
        self.log_path = runtime_path + "local.log"
        self.logger("New solver initialized...(:")
        self.logger("Optimizer: {}, Loss: {}, epochs:{}, lr:{}"
                    .format("Adam", self.criterion, self.epochs, self.lr))
        self.do_val = do_val
        self.save = save
        self.log_inter = log_inter

        if self.use_wandb:
            wandb.init(project='transformer classifier',
                       name=dt.datetime.strftime(dt.datetime.now(), '%Y-%m-%d %H:%M:%S'))
            wandb.config = {
                "learning_rate": self.lr,
                "epochs": self.epochs,
                "batch_size": self.batch_size
            }
            wandb.watch(self.model)

    def train(self):
        self.model.train()
        self.model = self.model.to(self.device)
        mes = "Training on: {}".format(self.device)
        print(mes)
        self.logger(mes)
        start_time = time.time()
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            print("Training on epoch {}/{}".format(epoch, self.epochs))
            total_loss = 0
            num_iters = len(self.train_dataloader)
            for src, tgt in tqdm(self.train_dataloader):
                src = src.float().to(self.device)
                tgt = tgt.float().to(self.device)
                out = self.model(src)
                loss = self.criterion(out, tgt)
                total_loss += loss.item()
                loss.backward()
                if (epoch+1) % self.grad_acc_step == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            if (epoch+1) % self.grad_acc_step == 0 and self.lr_schedular is not None:
                self.lr_schedular.step()

            print("Epoch: {}, Loss = {:.3f}".format(epoch, total_loss / num_iters))
            if self.use_wandb:
                wandb.log({'training loss per epoch': total_loss / num_iters})
                wandb.log({'learning rate per epoch': self.optimizer.state_dict()['param_groups'][0]['lr']})
            if (epoch - 1) % self.log_inter == 0:
                mes = "Epoch: {}, Loss = {:.3f}".format(epoch, total_loss / num_iters)
                self.logger(mes)

            if self.do_val:
                print("Validating...")
                # val:
                res_str, val_loss = self.eval(self.val_dataloader)
                res_str = "Epoch {} Val loss: {:.3f}, Val Result: {}"\
                    .format(epoch, val_loss/len(self.val_dataloader), res_str)
                print(res_str)
                if (epoch - 1) % self.log_inter == 0:
                    self.logger(res_str)

        duration = time.time() - start_time
        mes = "Training completed, time used: {} s".format(duration)
        print(mes)
        self.logger(mes)

        if self.save:
            self.save_model(self.model_weight_path)
            mes = "Model saved at: {}".format(self.model_weight_path)
            print(mes)
            self.logger(mes)

    def test(self):
        assert self.test_dataloader is not None, "you need a test dataloader"
        assert self.model is not None, "no model loaded"
        print("Testing...")
        res_str, _ = self.eval(self.test_dataloader)
        res_str = "Test Result: " \
                  "{}".format(res_str)
        print(res_str)
        self.logger(res_str)

    def eval(self, dataloader):
        assert dataloader is not None
        self.model.eval()
        self.model = self.model.to(self.device)
        val_total = len(dataloader)
        num_neg, num_pos, num_cor, num_tps, num_prp = 0, 0, 0, 0, 0
        val_loss, acc, prc, rec, f1 = .0, 0, 0, 0, 0
        for src, tgt in tqdm(dataloader):
            gt = tgt[:, 1] == 1
            num_gt_pos = int(gt.sum())
            len_gt = len(gt)

            num_pos += num_gt_pos
            num_neg += (len_gt - num_gt_pos)

            src = src.float().to(self.device)
            tgt = tgt.float().to(self.device)
            out = self.model(src)
            loss = self.criterion(out, tgt)
            val_loss += loss.item()

            pred = torch.argmax(out, dim=-1)
            pred = pred.cpu()
            tans = (pred == gt)
            num_cor += int(tans.sum())
            num_prp += int(pred.sum())
            num_tps += int((tans & gt).sum())

        val_total = num_pos + num_neg
        if val_total != 0:
            acc = num_cor / val_total
        if num_prp != 0:
            prc = num_tps / num_prp
        if num_pos != 0:
            rec = num_tps / num_pos
        if prc + rec != 0:
            f1 = 2 * prc * rec / (prc + rec)
        print("total={}".format(val_total))
        print("num_cor={}, num_tps={}, num_prp={}, num_pos={}, num_neg={}"
              .format(num_cor, num_tps, num_prp, num_pos, num_neg))
        res_str = "Acc: {:.3f}, Precision: {:.3f}, Recall: {:.3f}, F1: {:.3f}".format(acc, prc, rec, f1)
        return res_str, val_loss

    def save_model(self, path):
        torch.save(self.model, path)

    def load_model(self, path):
        self.model = torch.load(path)

    def logger(self, message):
        with open(self.log_path, "a") as f:
            time_str = dt.datetime.strftime(dt.datetime.now(), '%Y-%m-%d %H:%M:%S')
            f.write("[{}] {}\n".format(time_str, message))


if __name__ == "__main__":
    epochs = 1000
    bs = 32
    lr = 1e-4

    st4_train_dataset = ST4Dataset(train_csv_path, log_path)
    st4_test_dataset = ST4Dataset(test_csv_path, log_path)
    st4_val_dataset = ST4Dataset(val_csv_path, log_path)
    grad_acc_step = 1
    transformer_classifer = TC(
        d_model=12,
        nhead=1,
        # out_features=1,
    )
    Adam = torch.optim.Adam(transformer_classifer.parameters(), lr=lr)
    CALR = torch.optim.lr_scheduler.CosineAnnealingLR(Adam, int(epochs))
    BCEL = nn.BCELoss()
    MSEL = nn.MSELoss()
    FCLL = FocalLoss()

    solver = Solver(
        epochs=epochs,
        batch_size=bs,
        lr=lr,
        train_dataloader=DataLoader(st4_train_dataset, batch_size=bs, shuffle=True),
        test_dataloader=DataLoader(st4_test_dataset, batch_size=bs, shuffle=True),
        val_dataloader=DataLoader(st4_val_dataset, batch_size=bs, shuffle=True),
        model=transformer_classifer,
        optimizer=Adam,
        criterion=FCLL,
        lr_schedular=CALR,
        # use_wandb=True,
        do_val=True,
        grad_acc_step=grad_acc_step,
        # save=True,
        log_inter=1000
    )

    solver.train()
    # solver.load_model(solver.model_weight_path)
    solver.test()
