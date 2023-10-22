import os
import numpy as np
import torch
import argparse

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from datasets import RecWithContrastiveLearningDataset,DS

from trainers import ICSRecTrainer
from models import SASRecModel,GRUEncoder
from utils import EarlyStopping, get_user_seqs, check_path, set_seed


def show_args_info(args):
    print(f"--------------------Configure Info:------------")
    for arg in vars(args):
        print(f"{arg:<30} : {getattr(args, arg):>35}")


def main():
    parser = argparse.ArgumentParser()
    # system args
    parser.add_argument("--data_dir", default="../data/", type=str)
    parser.add_argument("--output_dir", default="output", type=str)
    parser.add_argument("--data_name", default="Sports_and_Outdoors", type=str)
    parser.add_argument("--encoder",default="SAS",type=str) # {"SAS":SASRec,"GRU":GRU4Rec}
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--model_idx", default=0, type=int, help="model idenfier 10, 20, 30...")
    parser.add_argument("--gpu_id", type=str, default="0", help="gpu_id")

    # robustness experiments
    parser.add_argument(
        "--noise_ratio",
        default=0.0,
        type=float,
        help="percentage of negative interactions in a sequence - robustness analysis",
    )

    ## contrastive learning task args
    parser.add_argument(
        "--temperature", default=1.0, type=float, help="softmax temperature (default:  1.0) - not studied."
    )

    parser.add_argument(
        "--intent_num",default=512,type=int,help="the multi intent nums!."
    )

    parser.add_argument(
        "--sim",default='dot',type=str,help="the calculate ways of the similarity."
    )

    # model args
    parser.add_argument("--model_name", default="ICSRec", type=str)
    parser.add_argument("--hidden_size", type=int, default=64, help="hidden size of transformer model")
    parser.add_argument("--num_hidden_layers", type=int, default=2, help="number of layers")
    parser.add_argument("--num_attention_heads", default=2, type=int)
    parser.add_argument("--hidden_act", default="gelu", type=str)  # gelu relu
    parser.add_argument("--attention_probs_dropout_prob", type=float, default=0.5, help="attention dropout p")
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.5, help="hidden dropout p")
    parser.add_argument("--initializer_range", type=float, default=0.02)
    parser.add_argument("--max_seq_length", default=50, type=int)

    # train args
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate of adam")
    parser.add_argument("--batch_size", type=int, default=256, help="number of batch_size")
    parser.add_argument("--epochs", type=int, default=300, help="number of epochs")
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--log_freq", type=int, default=1, help="per epoch print res")
    parser.add_argument("--seed", default=2022, type=int)
    # loss weight
    parser.add_argument("--rec_weight", type=float, default=1, help="weight of contrastive learning task")
    parser.add_argument("--lambda_0", type=float, default=0.1, help="weight of coarse-grain intent contrastive learning task")
    parser.add_argument("--beta_0", type=float, default=0.1, help="weight of fine-grain contrastive learning task")

    # ablation experiments
    parser.add_argument("--cl_mode",type=str,default='cf',help="contrastive mode")
    # {'cf':coarse-grain+fine-grain,'c':only coarse-grain,'f':only fine-grain}
    parser.add_argument("--f_neg", action="store_true", help="delete the FNM component (both in cicl and ficl)")

    # learning related
    parser.add_argument("--weight_decay", type=float, default=0.0, help="weight_decay of adam")

    args = parser.parse_args()
    set_seed(args.seed)
    check_path(args.output_dir)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    args.cuda_condition = torch.cuda.is_available() and not args.no_cuda
    print("Using Cuda:", torch.cuda.is_available())
    args.data_file = args.data_dir + args.data_name + ".txt"
    args.train_data_file = args.data_dir + args.data_name + "_1.txt"

    # construct supervisory signals via DS(·) operation
    if not os.path.exists(args.train_data_file):
        DS(args.data_file,args.train_data_file,args.max_seq_length)

    # training data
    _,train_user_seq, _, _, _ = get_user_seqs(args.train_data_file)
    # valid and test data
    _,user_seq, max_item, valid_rating_matrix, test_rating_matrix = get_user_seqs(args.data_file)

    args.item_size = max_item + 2
    args.mask_id = max_item + 1
    # save model args
    args_str = f"{args.model_name}-{args.encoder}-{args.data_name}-{args.model_idx}"
    args.log_file = os.path.join(args.output_dir, args_str + ".txt")

    show_args_info(args)

    with open(args.log_file, "a") as f:
        f.write(str(args) + "\n")

    # set item score in train set to `0` in validation
    args.train_matrix = valid_rating_matrix

    # save model
    checkpoint = args_str + ".pt"
    args.checkpoint_path = os.path.join(args.output_dir, checkpoint)

    # cluster
    cluster_dataset = RecWithContrastiveLearningDataset(
        args, train_user_seq, data_type="train"
    )
    cluster_sampler = SequentialSampler(cluster_dataset)
    cluster_dataloader = DataLoader(cluster_dataset, sampler=cluster_sampler, batch_size=args.batch_size)

    # training data
    train_dataset = RecWithContrastiveLearningDataset(
        args, train_user_seq, data_type="train"
    )
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size)

    eval_dataset = RecWithContrastiveLearningDataset(args, user_seq, data_type="valid")
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.batch_size)

    test_dataset = RecWithContrastiveLearningDataset(args, user_seq, data_type="test")
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.batch_size)


    if args.encoder=="SAS":
        model = SASRecModel(args=args)
    elif args.encoder=="GRU":
        model=GRUEncoder(args=args)
    trainer = ICSRecTrainer(model, train_dataloader,cluster_dataloader, eval_dataloader, test_dataloader, args)

    if args.do_eval:
        trainer.args.train_matrix = test_rating_matrix
        trainer.load(args.checkpoint_path)

        print(f"Load model from {args.checkpoint_path} for test!")
        scores, result_info = trainer.test(0, full_sort=True)

    else:
        print(f"Train ICSRec")
        early_stopping = EarlyStopping(args.checkpoint_path, patience=40, verbose=True)
        for epoch in range(args.epochs):
            trainer.train(epoch)
            # evaluate on NDCG@20
            scores, _ = trainer.valid(epoch, full_sort=True)
            early_stopping(np.array(scores[-1:]), trainer.model)
            if early_stopping.early_stop:
                print("Early stopping")
                break
        trainer.args.train_matrix = test_rating_matrix
        print("---------------Change to test_rating_matrix!-------------------")
        # load the best model
        trainer.model.load_state_dict(torch.load(args.checkpoint_path))
        scores, result_info = trainer.test(0, full_sort=True)

    print(args_str)
    print(result_info)
    with open(args.log_file, "a") as f:
        f.write(args_str + "\n")
        f.write(result_info + "\n")


main()
