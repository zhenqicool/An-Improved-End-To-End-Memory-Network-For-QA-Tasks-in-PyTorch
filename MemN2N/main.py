import os
from tqdm import tqdm
import datetime
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from model import MemN2N
from helpers import dataloader, get_fname, get_params


def train(train_iter, model, optimizer, epochs, max_clip, batch_size, lr, config, valid_iter=None):
    total_loss = 0
    loss_before = 20
    loss_after = 20
    valid_data = list(valid_iter)
    valid_loss = None
    next_epoch_to_report = 5
    pad = model.vocab.stoi['<pad>']

    # draw pic parameters
    loss_epoch5 = {
        'time': 0,
        'loss': [],
        'epoch': []
    }
    start_time = datetime.datetime.now()

    for epoch in tqdm(range(int(epochs)), desc="training:"):
        for _, batch in enumerate(train_iter, start=1):
            if batch.batch_size == batch_size:
                story = batch.story
                query = batch.query
                answer = batch.answer
                optimizer.zero_grad()
                if torch.cuda.is_available():
                    story = story.cuda()
                    query = query.cuda()
                    answer = answer.cuda()
                outputs = model(story, query)
                loss = F.nll_loss(outputs, answer.view(-1), ignore_index=pad, reduction='sum')
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_clip)
                optimizer.step()
                total_loss += loss.item()

                # linear start
                if model.use_ls:
                    loss = 0
                    for k, batch in enumerate(valid_data, start=1):
                        story = batch.story
                        query = batch.query
                        answer = batch.answer
                        outputs = model(story, query)
                        loss += F.nll_loss(outputs, answer.view(-1), ignore_index=pad, reduction='sum').item()
                    loss = loss / k
                    if valid_loss and valid_loss <= loss:
                        model.use_ls = False
                    else:
                        valid_loss = loss

                if loss_after < 0.001:
                    break
                if lr < 0.00001:
                    break
                if epoch == next_epoch_to_report:
                    print("#! epoch {:d} average batch loss: {:5.4f}".format(
                        int(epoch), total_loss / epoch))
                    next_epoch_to_report += 5
                    loss_after = total_loss / epoch
                    loss_epoch5['loss'].append(loss_after)
                    loss_epoch5['epoch'].append(epoch)
                    print('before', loss_before)
                    print('after', loss_after)

                if int(epoch) == epoch:
                    total_loss = 0
                if epoch == epochs:
                    break
                # loss_after > loss_before, scale down the learning rate
                if loss_after > loss_before:
                    lr = lr / 1.5
                    optimizer = optim.Adam(model.parameters(), lr)
                    print('lr', lr)
                loss_before = loss_after

            else:
                pass

    # write result
    end_time = datetime.datetime.now()
    loss_epoch5['time'] = (end_time - start_time).seconds
    if not os.path.isdir('/media/files/szq/MN_new/result/'):
        os.makedirs('/media/files/szq/MN_new/result/')
    with open('/media/files/szq/MN_new/result/' + get_fname(config) + '.json', 'w', encoding='utf-8') as file:
        file.write(json.dumps(loss_epoch5, indent=2, ensure_ascii=False))


def eval(test_iter, vocab_dict, model, batch_size):
    total_error = 0
    story_dict = {value:key for key, value in vocab_dict.fields['story'].vocab.stoi.items()}
    query_dict = {value:key for key, value in vocab_dict.fields['query'].vocab.stoi.items()}
    answer_dict = {value:key for key, value in vocab_dict.fields['answer'].vocab.stoi.items()}
    for k, batch in enumerate(test_iter, start=1):
        if batch.batch_size == batch_size:
            story = batch.story
            query = batch.query
            answer = batch.answer
            if torch.cuda.is_available():
                story = story.cuda()
                query = query.cuda()
                answer = answer.cuda()
            outputs = model(story, query)
            _, outputs = torch.max(outputs, -1)
            total_error += torch.mean((outputs != answer.view(-1)).float()).item()
            # 训练后 samples 展示
            print('test examples:')
            story_sent = ''
            for i in story.cpu().int().numpy()[0]:
                for j in i:
                    story_sent += query_dict[j] + '\t'
                story_sent += '\n'
            print('query:', story_sent)
            query_sent = ''
            for i in query.cpu().int().numpy()[0]:
                query_sent += query_dict[i] + '\t'
            print('query:', query_sent + '?')
            coran_sent = ''
            for i in answer.cpu().int().numpy()[0]:
                coran_sent += answer_dict[i] + '\t'
            print('correct answer:', coran_sent)
            print('predict answer:', answer_dict[outputs.cpu().numpy()[0]] + '\n')
        else:
            pass

    # 训练后错误率
    print("#! average error: {:5.1f}".format(total_error / k * 100))


def run(config):
    print("#! preparing data...")
    train_iter, valid_iter, test_iter, vocab = dataloader(config.batch_size, config.memory_size,
                                                          config.task, config.joint, config.tenk)

    print("#! instantiating model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MemN2N(get_params(config), vocab).to(device)

    if config.file:
        with open(os.path.join('/media/files/szq/MN_new/' + config.save_dir, get_fname(config)), 'rb') as f:
            if torch.cuda.is_available():
                state_dict = torch.load(f, map_location=lambda storage, loc: storage.cuda())
            else:
                state_dict = torch.load(f, map_location=lambda storage, loc: storage)
            model.load_state_dict(state_dict)

    if config.train:
        print("#! training...")
        optimizer = optim.Adam(model.parameters(), config.lr)
        train(train_iter, model, optimizer, config.num_epochs, config.max_clip, config.batch_size, config.lr, config, \
              valid_iter)
        if not os.path.isdir('/media/files/szq/MN_new/' + config.save_dir):
            os.makedirs('/media/files/szq/MN_new/' + config.save_dir)
        torch.save(model.state_dict(), os.path.join('/media/files/szq/MN_new/' + config.save_dir, get_fname(config)))

    print("#! testing...")
    with torch.no_grad():
        eval(test_iter, train_iter.dataset, model, config.batch_size)



