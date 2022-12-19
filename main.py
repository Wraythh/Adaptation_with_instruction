import torch
import argparse
import os
import datetime
import random
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from trainer import Trainer
from model.basic_model import Learner, Instructor
from dataprocessor import DataProcessor

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cifar-10")
    parser.add_argument("--dataset-split", type=float, default=0.5)
    parser.add_argument("--class-num", type=int, default=10)
    parser.add_argument("--feat-dim", type=int, default=2048)
    parser.add_argument("--exp-name", type=str, default="debug")
    parser.add_argument("--method", type=str, default="our_method")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--instructor-hidden-sizes", type=int, nargs="*", default=[64, 64])
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--con-temp", type=float, default=0.5)
    return parser.parse_args()

def main(args=get_args()):
    # random seed
    torch.manual_seed(args.seed) 
    torch.cuda.manual_seed(args.seed) 
    torch.cuda.manual_seed_all(args.seed)  
    np.random.seed(args.seed)  
    random.seed(args.seed)  
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # model
    data_processor = DataProcessor(batch_size=args.batch_size, dataset=args.dataset, dataset_split=args.dataset_split)
    learner = Learner(class_num=args.class_num, feat_dim=args.feat_dim, device=args.device)
    instructor = Instructor(
        class_num=args.class_num, 
        feat_dim=args.feat_dim,
        hidden_sizes=args.instructor_hidden_sizes, 
        device=args.device
    )
    optimizer_learner = torch.optim.Adam(list(learner.parameters()), lr=args.lr)
    optimizer_retrain_learner = torch.optim.Adam(list(learner.model.encoder.parameters()), lr=args.lr)
    optimizer_instructor = torch.optim.Adam(list(instructor.parameters()), lr=args.lr, weight_decay=1e-5)
    # log
    now = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    log_name = os.path.join(args.dataset, args.exp_name, str(args.seed), now)
    log_path = os.path.join(args.logdir, log_name)
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))

    trainer = Trainer(
        method=args.method,
        epoch=args.epochs,
        batch_size=args.batch_size,
        class_num=args.class_num,
        learner=learner,
        instructor=instructor,
        contrastive_temp=args.con_temp,
        optimizer_learner=optimizer_learner,
        optimizer_retrain_learner=optimizer_retrain_learner,
        optimizer_instructor=optimizer_instructor,
        data_processor=data_processor,
        writer=writer,
        device=args.device,
    )
    trainer.train()

if __name__ == "__main__":
    main()

