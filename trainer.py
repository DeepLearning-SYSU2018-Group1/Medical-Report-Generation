import time
import pickle
import argparse

import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.autograd import Variable

from utils.models import *
from utils.dataset import *
from utils.loss import *
from utils.logger import Logger

from DenseNet121 import DenseNet121
from HeatmapGenerator import HeatmapGenerator

class DebuggerBase:
    def __init__(self, args):
        self.args = args
        self.min_val_loss = 10000000000
        self.min_tag_loss = 1000000
        self.min_stop_loss = 1000000
        self.min_word_loss = 10000000

        self.min_train_loss = 10000000000

        self._init_model_path()
        self.model_dir = self._init_model_dir()
        self.writer = self._init_writer()
        self.train_transform = self._init_train_transform()
        self.val_transform = self._init_val_transform()
        self.vocab = self._init_vocab()
        self.model_state_dict = self._load_mode_state_dict()

        self.train_data_loader = self._init_data_loader(self.args.train_caption_json,self.args.train_file_list, self.train_transform)
        self.val_data_loader = self._init_data_loader(self.args.val_caption_json,self.args.val_file_list, self.val_transform)

        self.extractor = self._init_visual_extractor()
        self.mlc = self._init_mlc()
        self.co_attention = self._init_co_attention()
        self.sentence_model = self._init_sentence_model()
        self.word_model = self._init_word_model()

        self.heatamap_generator = HeatmapGenerator(self.extractor, self._init_heatmap_transform())

        self.ce_criterion = self._init_ce_criterion()
        self.mse_criterion = self._init_mse_criterion()

        if self.args.fix_encoder:
        	self._fix_encoder()

        if self.args.fix_decoder:
            self._fix_decoder()

        self.optimizer = self._init_optimizer()
        self.scheduler = self._init_scheduler()
        self.logger = self._init_logger()
        self.writer.write("{}\n".format(self.args))


    def train(self):
        for epoch_id in range(self.start_epoch, self.args.epochs):
            print("epoch training: %d " % epoch_id)
            train_tag_loss, train_stop_loss, train_word_loss, train_loss = self._epoch_train()
            val_tag_loss, val_stop_loss, val_word_loss, val_loss = self._epoch_val()

            if self.args.mode == 'train':
                self.scheduler.step(train_loss)
            else:
                self.scheduler.step(val_loss)
            self.writer.write(
                "[{} - Epoch {}] train loss:{} - val_loss:{} - lr:{}\n".format(self._get_now(),
                                                                               epoch_id,
                                                                               train_loss,
                                                                               val_loss,
                                                                               self.optimizer.param_groups[0]['lr']))
            self._save_model(epoch_id,
                             val_loss,
                             train_loss)
            self._log(train_tags_loss=train_tag_loss,
                      train_stop_loss=train_stop_loss,
                      train_word_loss=train_word_loss,
                      train_loss=train_loss,
                      val_tags_loss=val_tag_loss,
                      val_stop_loss=val_stop_loss,
                      val_word_loss=val_word_loss,
                      val_loss=val_loss,
                      lr=self.optimizer.param_groups[0]['lr'],
                      epoch=epoch_id)

    def _epoch_train(self):
        raise NotImplementedError

    def _epoch_val(self):
        raise NotImplementedError

    def _init_train_transform(self):
        transform = transforms.Compose([
            transforms.Resize(self.args.resize),
            transforms.RandomCrop(self.args.crop_size),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))])
        return transform

    def _init_val_transform(self):
        transform = transforms.Compose([
            transforms.Resize((self.args.crop_size, self.args.crop_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))])
        return transform

    def _init_heatmap_transform(self):
        transform = transforms.Compose([
            transforms.Resize((self.args.resize, self.args.resize)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))])
        return transform

    def _init_model_dir(self):
        model_dir = os.path.join(self.args.model_path, self.args.saved_model_name)

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        model_dir = os.path.join(model_dir, self._get_now())

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        return model_dir

    def _init_vocab(self):
        with open(self.args.vocab_path, 'rb') as f:
            vocab = pickle.load(f)

        self.writer.write("Vocab Size:{}\n".format(len(vocab)))

        return vocab

    def _load_mode_state_dict(self):
        self.start_epoch = 0
        print('load model', self.args.load_model_path)
        try:
            model_state = torch.load(self.args.load_model_path)
            print("[Load Model-",self.args.load_model_path," Succeed!]")
            self.start_epoch = model_state['epoch']
            self.writer.write("[Load Model-{} Succeed!]\n".format(self.args.load_model_path))
            self.writer.write("Load From Epoch {}\n".format(model_state['epoch']))
            return model_state
        except Exception as err:
            print("[Load Model Failed] {}\n".format(err))
            self.writer.write("[Load Model Failed] {}\n".format(err))
            return None

    def _init_visual_extractor(self):
        model = VisualFeatureExtractor(model_name=self.args.visual_model_name,
                                       pretrained=self.args.pretrained)

        #if self.model_state_dict is not None:
        '''checkpoint = './report_models/m-test-save-model-1.pth.tar'
        print('train model path:', checkpoint)
        modelCheckpoint = torch.load(checkpoint)
        model.model.load_state_dict(modelCheckpoint['CNN_state_dict'])
        return model'''

        # init cnn (densenet121) with chexnet pretrained cnn
        if self.args.chexnet_pretrained_model:
            print('init cnn (densenet121) with chexnet pretrained cnn')

            mydense121 = DenseNet121(14, True).cuda()
            mydense121 = torch.nn.DataParallel(mydense121,device_ids=[0]).cuda()
            checkpoint = './report_models/m-test-save-model-1.pth.tar'
            print('train model path:', checkpoint)
            modelCheckpoint = torch.load(checkpoint)
            mydense121.module.densenet121.features.load_state_dict(modelCheckpoint['CNN_state_dict'])

            with torch.no_grad():
                ct=0
                load_model_params_gen = mydense121.module.densenet121.features.parameters()
                for param in model.model.parameters():
                    load_param = next(load_model_params_gen)
                    if ct==0:
                        print(load_param.size(),param.size())
                        print('old param.data: ',param.data[0][0][0])
                        print('load param data: ',load_param[0][0][0])
                    ct+=1
                    param.data = load_param.data
                    if ct==1:
                        print('new param.data: ',param.data[0][0][0])
            print('_init_visual_extractor ct:',ct)

            for param in model.model.parameters():
                print('test param data changed\n',param.data[0][0][0])
                break

        if self.model_state_dict is not None:
            model.load_state_dict(self.model_state_dict['extractor'])

        if self.args.cuda:
            model = model.cuda()

        return model

    def _init_mlc(self):
        model = MLC(classes=self.args.classes,
                    sementic_features_dim=self.args.sementic_features_dim,
                    fc_in_features=self.extractor.out_features,
                    k=self.args.k)

        if self.model_state_dict is not None:
            model.load_state_dict(self.model_state_dict['mlc'])

        if self.args.cuda:
            model = model.cuda()
        return model

    def _init_co_attention(self):
        model = CoAttention(version=self.args.attention_version,
                            embed_size=self.args.embed_size,
                            hidden_size=self.args.hidden_size,
                            visual_size=self.extractor.out_features,
                            k=self.args.k,
                            momentum=self.args.momentum)

        if self.model_state_dict is not None:
            model.load_state_dict(self.model_state_dict['co_attention'])

        if self.args.cuda:
            model = model.cuda()
        return model

    def _init_sentence_model(self):
        raise NotImplementedError

    def _init_word_model(self):
        raise NotImplementedError

    def _init_data_loader(self, captions, file_list, transform):
        data_loader = get_loader(image_dir=self.args.image_dir,
                                 caption_json=captions,
                                 file_list=file_list,
                                 vocabulary=self.vocab,
                                 transform=transform,
                                 batch_size=self.args.batch_size,
                                 s_max=self.args.s_max,
                                 n_max=self.args.n_max,
                                 shuffle=True)
        return data_loader

    @staticmethod
    def _init_ce_criterion():
        return nn.CrossEntropyLoss(size_average=False, reduce=False)

    @staticmethod
    def _init_mse_criterion():
        return nn.MSELoss()

    def _fix_decoder(self):
        print('fix parameters of decoder')
        for param in self.sentence_model.parameters():
            param.requires_grad = False

        for param in self.word_model.parameters():
            param.requires_grad = False

    def _fix_encoder(self):     # add by pjl
        print('fix parameters of encoder')

        for param in self.extractor.parameters():
            param.requires_grad = False

        for param in self.mlc.parameters():
            param.requires_grad = False



        for param in self.co_attention.parameters():
            param.requires_grad = False

    def _init_optimizer(self):
        if self.args.fix_decoder == True:
        	params = list(self.extractor.parameters()) + \
        	         list(self.mlc.parameters()) + \
        	         list(self.co_attention.parameters())
        elif self.args.fix_encoder == True:
            params = list(self.word_model.parameters()) + \
                     list(self.sentence_model.parameters())
        else:
        	params = list(self.extractor.parameters()) + \
                     list(self.mlc.parameters()) + \
                     list(self.co_attention.parameters()) + \
                     list(self.word_model.parameters()) + \
                     list(self.sentence_model.parameters())

        return torch.optim.Adam(params=params, lr=self.args.learning_rate)

    def _log(self,
             train_tags_loss,
             train_stop_loss,
             train_word_loss,
             train_loss,
             val_tags_loss,
             val_stop_loss,
             val_word_loss,
             val_loss,
             lr,
             epoch):
        info = {
            'train tags loss': train_tags_loss,
            'train stop loss': train_stop_loss,
            'train word loss': train_word_loss,
            'train loss': train_loss,
            'val tags loss': val_tags_loss,
            'val stop loss': val_stop_loss,
            'val word loss': val_word_loss,
            'val loss': val_loss,
            'learning rate': lr
        }

        for tag, value in info.items():
            self.logger.scalar_summary(tag, value, epoch + 1)

    def _init_logger(self):
        logger = Logger(os.path.join(self.model_dir, 'logs'))
        return logger

    def _init_writer(self):
        writer = open(os.path.join(self.model_dir, 'logs.txt'), 'w')
        return writer

    def _to_var(self, x, requires_grad=True):
        if self.args.cuda:
            x = x.cuda()
        return Variable(x, requires_grad=requires_grad)

    def _get_date(self):
        return str(time.strftime('%Y%m%d', time.gmtime()))

    def _get_now(self):
        return str(time.strftime('%Y%m%d-%H:%M', time.gmtime()))

    def _init_scheduler(self):
        scheduler = ReduceLROnPlateau(self.optimizer, 'min', patience=self.args.patience, factor=0.1)
        return scheduler

    def _init_model_path(self):
        if not os.path.exists(self.args.model_path):
            os.makedirs(self.args.model_path)

    def _init_log_path(self):
        if not os.path.exists(self.args.log_path):
            os.makedirs(self.args.log_path)

    def _save_model(self, epoch_id, val_loss, train_loss):
        def save_model(_filename):
            self.writer.write("Saved Model in {}\n".format(_filename))
            torch.save({'extractor': self.extractor.state_dict(),
                        'mlc': self.mlc.state_dict(),
                        'co_attention': self.co_attention.state_dict(),
                        'sentence_model': self.sentence_model.state_dict(),
                        'word_model': self.word_model.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'epoch': epoch_id},
                       os.path.join(self.model_dir, "{}".format(_filename)))

        if val_loss < self.min_val_loss:
            # file_name = "val_best_loss_densenet121" + self._get_now() + ".pth.tar";
            file_name = "val_best_loss.pth.tar"
            save_model(file_name)
            self.min_val_loss = val_loss

        if train_loss < self.min_train_loss:
            # file_name = "train_best_loss_densenet121"+ self._get_now() + ".pth.tar";
            file_name = "train_best_loss.pth.tar"
            save_model(file_name)
            self.min_train_loss = train_loss


class LSTMDebugger(DebuggerBase):
    def _init_(self, args):
        DebuggerBase.__init__(self, args)
        self.args = args

    def _epoch_train(self):
        tag_loss, stop_loss, word_loss, loss = 0, 0, 0, 0
        self.extractor.train()
        self.mlc.train()
        self.co_attention.train()
        self.sentence_model.train()
        self.word_model.train()

        for i, (images, _, label, captions, prob) in enumerate(self.train_data_loader):
            print("train id : %d " % i)
            batch_tag_loss, batch_stop_loss, batch_word_loss, batch_loss = 0, 0, 0, 0

            #print('images size\n',images.size())
            images = self._to_var(images)

            visual_features, avg_features = self.extractor.forward(images)
            #print('heatmap')
            #print(visual_features.size())

            tags, semantic_features = self.mlc.forward(avg_features)

            batch_tag_loss = self.mse_criterion(tags, self._to_var(label, requires_grad=False)).sum()

            sentence_states = None
            prev_hidden_states = self._to_var(torch.zeros(images.shape[0], 1, self.args.hidden_size))

            context = self._to_var(torch.Tensor(captions).long(), requires_grad=False)
            prob_real = self._to_var(torch.Tensor(prob).long(), requires_grad=False)

            for sentence_index in range(captions.shape[1]):
                ctx, _, _ = self.co_attention.forward(avg_features,
                                                       semantic_features,
                                                       prev_hidden_states)

                topic, p_stop, hidden_states, sentence_states = self.sentence_model.forward(ctx,
                                                                                            prev_hidden_states,
                                                                                            sentence_states)

                batch_stop_loss += self.ce_criterion(p_stop.squeeze(), prob_real[:, sentence_index]).sum()

                # print("p_stop:{}".format(p_stop.squeeze()))
                # print("prob_real:{}".format(prob_real[:, sentence_index]))

                for word_index in range(1, captions.shape[2]):

                    '''if self.args.scheduled_sampling:
                        sample_prob = fc_feats.data.new(self.args.batch_size).uniform_(0, 1)
                        sample_mask = sample_prob < self.ss_prob
                        if sample_mask.sum() == 0:
                            words = self.word_model.forward(topic, context[:, sentence_index, :word_index])
                        else:
                            it = context[:, sentence_index, :word_index].clone()
                            
                            # words = self.word_model.forward(topic, )

                    else:'''
                    words = self.word_model.forward(topic, context[:, sentence_index, :word_index])

                    word_mask = (context[:, sentence_index, word_index] > 0).float()
                    batch_word_loss += (self.ce_criterion(words, context[:, sentence_index, word_index])
                                        * word_mask).sum()
                    # batch_word_loss += (self.ce_criterion(words, context[:, sentence_index, word_index])).sum()
                    # print("words:{}".format(torch.max(words, 1)[1]))
                    # print("real:{}".format(context[:, sentence_index, word_index]))
            
            # to fix encoder, we should comment self.args.lambda_tag * batch_tag_loss \ 
            # batch_loss = self.args.lambda_tag * batch_tag_loss \
            #              + self.args.lambda_stop * batch_stop_loss \
            #              + self.args.lambda_word * batch_word_loss

            # fix decoder
            if self.args.fix_decoder:
                batch_loss = self.args.lambda_tag * batch_tag_loss
            elif self.args.fix_encoder:
            	batch_loss = self.args.lambda_stop * batch_stop_loss + \
            	             self.args.lambda_word * batch_word_loss
            else:
                batch_loss = self.args.lambda_tag * batch_tag_loss \
                           + self.args.lambda_stop * batch_stop_loss \
                           + self.args.lambda_word * batch_word_loss

            self.optimizer.zero_grad()
            batch_loss.backward()
            if self.args.clip > 0:
                torch.nn.utils.clip_grad_norm(self.sentence_model.parameters(), self.args.clip)
                torch.nn.utils.clip_grad_norm(self.word_model.parameters(), self.args.clip)
            self.optimizer.step()

            tag_loss += self.args.lambda_tag * batch_tag_loss.data
            stop_loss += self.args.lambda_stop * batch_stop_loss.data
            word_loss += self.args.lambda_word * batch_word_loss.data
            loss += batch_loss.data

        return tag_loss, stop_loss, word_loss, loss

    def _epoch_val(self):
        tag_loss, stop_loss, word_loss, loss = 0, 0, 0, 0
        self.extractor.eval()
        self.mlc.eval()
        self.co_attention.eval()
        self.sentence_model.eval()
        self.word_model.eval()

        for i, (images, _, label, captions, prob) in enumerate(self.val_data_loader):
            batch_tag_loss, batch_stop_loss, batch_word_loss, batch_loss = 0, 0, 0, 0
            #print('val images size\n',images.size())
            images = self._to_var(images, requires_grad=False)

            visual_features, avg_features = self.extractor.forward(images)
            #print('val heatmap visual_features size')
            #print(visual_features.size())

            tags, semantic_features = self.mlc.forward(avg_features)

            batch_tag_loss = self.mse_criterion(tags, self._to_var(label, requires_grad=False)).sum()

            sentence_states = None
            prev_hidden_states = self._to_var(torch.zeros(images.shape[0], 1, self.args.hidden_size))

            context = self._to_var(torch.Tensor(captions).long(), requires_grad=False)
            prob_real = self._to_var(torch.Tensor(prob).long(), requires_grad=False)

            for sentence_index in range(captions.shape[1]):
                ctx, v_att, a_att = self.co_attention.forward(avg_features,
                                                       semantic_features,
                                                       prev_hidden_states)

                topic, p_stop, hidden_states, sentence_states = self.sentence_model.forward(ctx,
                                                                                            prev_hidden_states,
                                                                                            sentence_states)
                print("p_stop:{}".format(p_stop.squeeze()))
                print("prob_real:{}".format(prob_real[:, sentence_index]))

                batch_stop_loss += self.ce_criterion(p_stop.squeeze(), prob_real[:, sentence_index]).sum()

                for word_index in range(1, captions.shape[2]):
                    words = self.word_model.forward(topic, context[:, sentence_index, :word_index])
                    word_mask = (context[:, sentence_index, word_index] > 0).float()
                    batch_word_loss += (self.ce_criterion(words, context[:, sentence_index, word_index])
                                        * word_mask).sum()
                    print("words:{}".format(torch.max(words, 1)[1]))
                    print("real:{}".format(context[:, sentence_index, word_index]))

            batch_loss = self.args.lambda_tag * batch_tag_loss \
                         + self.args.lambda_stop * batch_stop_loss \
                         + self.args.lambda_word * batch_word_loss

            tag_loss += self.args.lambda_tag * batch_tag_loss.data
            stop_loss += self.args.lambda_stop * batch_stop_loss.data
            word_loss += self.args.lambda_word * batch_word_loss.data
            loss += batch_loss.data

        return tag_loss, stop_loss, word_loss, loss

    def _init_sentence_model(self):
        model = SentenceLSTM(version=self.args.sent_version,
                             embed_size=self.args.embed_size,
                             hidden_size=self.args.hidden_size,
                             num_layers=self.args.sentence_num_layers,
                             dropout=self.args.dropout,
                             momentum=self.args.momentum)

        if self.model_state_dict is not None:
            model.load_state_dict(self.model_state_dict['sentence_model'])

        if self.args.cuda:
            model = model.cuda()
        return model

    def _init_word_model(self):
        model = WordLSTM(vocab_size=len(self.vocab),
                         embed_size=self.args.embed_size,
                         hidden_size=self.args.hidden_size,
                         num_layers=self.args.word_num_layers,
                         n_max=self.args.n_max)

        if self.model_state_dict is not None:
            model.load_state_dict(self.model_state_dict['word_model'])

        if self.args.cuda:
            model = model.cuda()
        return model

    def _generate_heatmap(self):
        #ImagePath = 'heatmap/train/CXR3409_IM-1648-1001.png'
        #OutputHeatmapPath = 'heatmap/result/CXR3409_IM-1648-1001_heatmap.png'
        ImagePath = self.args.heatmap_input_img_path
        OutputHeatmapPath = self.args.heatmap_output_heatmap_path
        transCrop = self.args.crop_size
        self.heatamap_generator.generate_heatmap(ImagePath, OutputHeatmapPath, transCrop)
        print(ImagePath)
        print(OutputHeatmapPath)

    def _generate_heatmap_when_training(self):
        ImagePath = self.args.heatmap_input_img_path
        OutputHeatmapPath = self.args.heatmap_output_heatmap_path
        transCrop = self.args.crop_size
        self.heatamap_generator.generate_when_training(visual_features, ImagePath, OutputHeatmapPath, transCrop)
        print(ImagePath)
        print(OutputHeatmapPath)



if __name__ == '__main__':
    import warnings

    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser()

    """
    Data Argument
    """
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--mode', type=str, default='train')

    # Path Argument
    # todo : changing the name of the path / 2018/06/21
    parser.add_argument('--vocab_path', type=str, default='./data/data_using/vocab.pkl',
                        help='the path for vocabulary object')
    parser.add_argument('--image_dir', type=str, default='../../medical_report/images',
                        help='the path for images')
    parser.add_argument('--train_caption_json', type=str, default='./data/data_using/train_captions.json',
                        help='path for train captions')
    parser.add_argument('--val_caption_json', type=str, default='./data/data_using/val_captions.json',
                        help='path for validate captions')
    parser.add_argument('--test_caption_json', type=str, default='./data/data_using/test_captions.json',
                        help='path for test captions')
    parser.add_argument('--train_file_list', type=str, default='./data/data_using/train_data.txt',
                        help='the train array')
    parser.add_argument('--val_file_list', type=str, default='./data/data_using/val_data.txt',
                        help='the val array')
    # transforms argument
    parser.add_argument('--resize', type=int, default=224,
                        help='size for resizing images')
    parser.add_argument('--crop_size', type=int, default=224,
                        help='size for randomly cropping images')
    # Load/Save model argument
    parser.add_argument('--model_path', type=str, default='./report_models/model_using',
                        help='path for saving trained models')
    parser.add_argument('--load_model_path', type=str, default='./report_models/val_best_loss.pth.tar',
                        help='The path of loaded model')
    #parser.add_argument('--load_model_path', type=str, default='./report_models/model_using/training/20180630-03:29/val_best_loss_densenet12120180630-03:59.pth.tar', # todo:modified by pjl for retrain the model
                        # help='The path of loaded model')
    parser.add_argument('--saved_model_name', type=str, default='training',
                        help='The name of saved model')

    """
    Model Argument
    """
    parser.add_argument('--momentum', type=int, default=0.1)
    # VisualFeatureExtractor
    parser.add_argument('--visual_model_name', type=str, default='densenet121',
                        help='CNN model name')
    parser.add_argument('--pretrained', action='store_true', default=False,
                        help='not using pretrained model when training')
    # set True to init cnn (densenet121) with chexnet pretrained cnn
    parser.add_argument('--chexnet_pretrained_model', type=bool, default=True,
                        help='use chexnet pretrained densenet121 as cnn')

    # MLC
    parser.add_argument('--classes', type=int, default=325)
    parser.add_argument('--sementic_features_dim', type=int, default=512)
    parser.add_argument('--k', type=int, default=10)

    # Co-Attention
    parser.add_argument('--attention_version', type=str, default='v1')
    parser.add_argument('--embed_size', type=int, default=512)
    parser.add_argument('--hidden_size', type=int, default=512)

    # Sentence Model
    parser.add_argument('--sent_version', type=str, default='v1')
    parser.add_argument('--sentence_num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0)

    # Word Model
    parser.add_argument('--word_num_layers', type=int, default=1)

    """
    Training Argument
    """
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=int, default=0.001)
    parser.add_argument('--epochs', type=int, default=100)

    parser.add_argument('--clip', type=float, default=2,
                        help='gradient clip, -1 means no clip (default: 0.35)')
                        # todo: change clip as ta said to 5 by pjl
    parser.add_argument('--s_max', type=int, default=6)
    parser.add_argument('--n_max', type=int, default=30)

    # Loss Function
    parser.add_argument('--lambda_tag', type=float, default=10000)
    parser.add_argument('--lambda_stop', type=float, default=10)
    parser.add_argument('--lambda_word', type=float, default=1)

    # heatmap generation
    parser.add_argument('--heatmap_input_img_path', type=str, default='heatmap/train/CXR3_IM-1384-1001.png')
    parser.add_argument('--heatmap_output_heatmap_path', type=str, default='heatmap/result/CXR3_IM-1384-1001-heatmap.png')

    # training methods options
    parser.add_argument('--fix_encoder', type=bool, default=True,
    	                help='set True means fix encoder')
    parser.add_argument('--fix_decoder', type=bool, default=False,
                        help='set True to fix decoder')
    parser.add_argument('--scheduled_sampling', type=bool, default=False,
                        help='set True to use Scheduled Sampling')
    parser.add_argument('--ss_prob', type=float, default=0.25, 
                        help='scheduled sampling prob.')

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()

    debugger = LSTMDebugger(args)
    debugger.train()
    #debugger._generate_heatmap()
