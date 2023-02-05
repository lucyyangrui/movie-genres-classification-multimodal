import json
import logging
import time
import torch
from utils import evaluate, setup_logging, setup_seed, setup_device, build_optimizer, build_optimizer_vlbert
from sklearn.model_selection import StratifiedShuffleSplit

from config import parse_args
from data_helper import create_dataloader
from models.CCT_MMC_base1.model import Base1CttMmc
from models.MFMGR.model import MFMGP
from models.GMU_base2.model import GMU_MGC
from models.moviescope_base3.model import Base3Moviescope


def validation(model, val_dataloader, print_raw_value=False):
    model.eval()
    losses = []
    predictions_01 = []
    raw_predictions = []
    labels = []
    with torch.no_grad():
        t_begin = time.time()
        for step, batch in enumerate(val_dataloader):
            debug = False
            if step == 0:
                debug = True
            loss, accuracy, pred_label_id, raw_prediction = model(batch)

            loss = loss.mean()
            accuracy = accuracy.mean()
            predictions_01.extend(pred_label_id.cpu().numpy())
            raw_predictions.extend(raw_prediction.cpu().numpy())
            labels.extend(batch['label'].cpu().numpy())
            losses.append(loss)
            if step % 5 == 0:
                t = time.time()
                logging.info(f"step={step:4}/{len(val_dataloader)}|"
                             f"loss={loss:6.4}|acc={accuracy:0.4}|time={t - t_begin:.4}s")
                t_begin = time.time()

    loss = sum(losses) / len(losses)
    metrics = evaluate(predictions_01, labels, raw_predictions)
    if print_raw_value:
        print(raw_predictions[:3])
    return loss, metrics


def test_model(args, test_dataloader):
    logging.info('>>> loading model...')
    if args.model_name == 'base1':
        model = Base1CttMmc(args)
        optimizer, schedual = build_optimizer(args, model)
    elif args.model_name == 'mymodel':
        model = MFMGP(args)
        optimizer, schedual = build_optimizer_vlbert(args, model)
    elif args.model_name == 'base2':
        model = GMU_MGC(args)
        optimizer, schedual = build_optimizer(args, model)
    elif args.model_name == 'base3':
        model = Base3Moviescope(args)
        optimizer, schedual = build_optimizer(args, model)
    
    model_dict = torch.load(args.model_save_path + args.save_model_name, map_location='cpu')
    model.load_state_dict(model_dict['model_state_dict'])
    
    if args.device == 'cuda':
        model = torch.nn.parallel.DataParallel(model.to(args.device))
        print('>> using gpu to loading model...')
    
    val_loss, metrics = validation(model,test_dataloader)
    logging.info(f">>> val_loss: {val_loss}, matrics: {metrics}")
    return metrics['macro_f1'], metrics['micro_f1']


def train_and_validation(args):
    # TODO 考虑一下多标签的怎么做样本划分
    train_dataloader, val_dataloader, test_dataloader = create_dataloader(args)
    if args.model_name == 'base1':
        model = Base1CttMmc(args)
        optimizer, schedual = build_optimizer(args, model)
    elif args.model_name == 'mymodel':
        model = MFMGP(args)
        optimizer, schedual = build_optimizer_vlbert(args, model)
    elif args.model_name == 'base2':
        model = GMU_MGC(args)
        optimizer, schedual = build_optimizer(args, model)
    elif args.model_name == 'base3':
        model = Base3Moviescope(args)
        optimizer, schedual = build_optimizer(args, model)
    
    # model_dict = torch.load('./models/CCT_MMC_base1/save/model.bin', map_location='cpu')
    # model.load_state_dict(model_dict['model_state_dict'])  
    if args.device == 'cuda':
        model = torch.nn.parallel.DataParallel(model.to(args.device))
        print('>> using gpu to loading model...')

    loss_begin = 100.0
    micro_begin = 0.0
    no_decay_epoch = 0
    
    # logging.info('>>> begin validation...')
    # val_loss, metrics = validation(model, val_dataloader)
    # logging.info(f">>> val_loss: {val_loss}, matrics: {metrics}")
    
    logging.info('>>> begin training...')
    note_log = dict(
        note = args.note,
        best_epoch = 0,
        time_per_epoch = 0.0,
        valid_macro = 0.0,
        valid_micro = 0.0,
        micro_f1 = [],
        macro_f1 = []
        )
    for epoch in range(100):
        t_begin = time.time()
        t_log = t_begin
        train_loss = []
        for step, batch in enumerate(train_dataloader):
            model.train()
            optimizer.zero_grad()
            
            debug = False
            if step == 0:
                debug = True
            
            loss, accuracy, _, _ = model(batch)
            # model.hidden = model.init_hidden(args.train_batch_size // 4, 256, args.bs1_out_channel)
            loss = loss.mean()
            accuracy = accuracy.mean()
            loss.backward()

            optimizer.step()
            if args.model_name == 'mymodel':
                schedual.step()
            
            train_loss.append(loss)

            elap_t = time.time() - t_begin
            if step == 2 or step and step % 20 == 0:
                lr = optimizer.param_groups[0]['lr']
                logging.info(f"Epoch={epoch + 1}|step={step:4}/{len(train_dataloader)}|"
                             f"loss={loss:6.4}|lr={lr:0.8}|acc={accuracy:0.4}|time={elap_t:.4}s")
                t_begin = time.time()
        
        logging.info(f"train_loss={sum(train_loss) / len(train_loss):.4}")
        # validation
        print_raw_value = False
        logging.info('>>> begin validation...')
        val_loss, metrics = validation(model, val_dataloader, print_raw_value)
        logging.info(f">>> val_loss: {val_loss}, matrics: {metrics}")
        micro_f1 = metrics['micro_f1']
        macro_f1 = metrics['macro_f1']

        note_log['micro_f1'].append(micro_f1)
        note_log['macro_f1'].append(macro_f1)
        if note_log['time_per_epoch'] < 1:
            note_log['time_per_epoch'] = round(time.time() - t_log, 2)
            del t_log
        
        if micro_f1 - micro_begin > 0.001:
            micro_begin = micro_f1
            no_decay_epoch = 0
            logging.info('>> model saving...')
            torch.save({'epoch': epoch, 'model_state_dict': model.module.state_dict()},
                       args.model_save_path + '/' + args.save_model_name)
            note_log['best_epoch'] = epoch
        else:
            no_decay_epoch += 1
            if no_decay_epoch >= args.early_stop_epoch:
                logging.info('the micro_f1 is not increase over %d epoches, STOP Training!' % args.early_stop_epoch)
                break
                
    # 开始test
    logging.info('==========================================================')
    logging.info('>>> begin testing...')
    ma_f1, mi_f1 = test_model(args, test_dataloader)
    note_log['valid_macro'] = ma_f1
    note_log['valid_micro'] = mi_f1

    # 保存日志
    with open('./note_log.json', 'a+', encoding='utf-8') as f:
        json.dump(note_log, f)
        f.write('\n')


if __name__ == '__main__':
    args = parse_args()
    setup_seed(args)
    setup_device(args)
    setup_logging(args.model_name + '_' + args.ablation)
    logging.info("Training/evaluation parameters: %s", args)
    
    cls_freqs = {'动作': 744, '惊悚': 861, '冒险': 579, '剧情': 2244, '科幻': 403, '爱情': 591, '奇幻': 337, '喜剧': 1189, '恐怖': 414, '犯罪': 496}
    movie_num = 4063
    cls_weight = torch.FloatTensor([int(v) / movie_num for v in cls_freqs.values()]) ** -1
    print(cls_weight)
    args.cls_weight = cls_weight
    
    train_and_validation(args)
