import time
import pickle
import argparse
from tqdm import tqdm
from PIL import Image
import cv2

import os
import sys
import json

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable

from utils.models import *
from utils.dataset import *
from utils.loss import *
from utils.build_tag import *


class CaptionSampler(object):
    def __init__(self, args):
        self.args = args

        self.vocab = self.__init_vocab()
        self.tagger = self.__init_tagger()
        self.transform = self.__init_transform()
        self.data_loader = self.__init_data_loader(self.args.file_lits)
        self.model_state_dict = self.__load_mode_state_dict()

        self.extractor = self.__init_visual_extractor()
        self.mlc = self.__init_mlc()
        self.co_attention = self.__init_co_attention()
        self.sentence_model = self.__init_sentence_model()
        self.word_model = self.__init_word_model()

        self.ce_criterion = self._init_ce_criterion()
        self.mse_criterion = self._init_mse_criterion()

    @staticmethod
    def _init_ce_criterion():
        return nn.CrossEntropyLoss(size_average=False, reduce=False)

    @staticmethod
    def _init_mse_criterion():
        return nn.MSELoss()

    def sample(self, image_file):

        cam_dir = self.__init_cam_path(image_file)
        image_file = os.path.join(self.args.image_dir, image_file)

        imageData = Image.open(image_file).convert('RGB')
        imageData = self.transform(imageData)
        # images contains two input images
        imagesData = torch.stack([imageData, imageData])

        imageData = imageData.unsqueeze_(0)
        image = self.__to_var(imageData, requires_grad=False)
        images = self.__to_var(imagesData, requires_grad=False)

        visual_features, avg_features = self.extractor.forward(images)
        # avg_features.unsqueeze_(0)

        tags, semantic_features = self.mlc(avg_features)
        sentence_states = None
        # prev_hidden_states = self.__to_var(torch.zeros(1, 1, self.args.hidden_size))
        prev_hidden_states = self.__to_var(torch.zeros(images.shape[0], 1, self.args.hidden_size))

        pred_sentences = []

        for i in range(self.args.s_max):
            ctx, alpht_v, alpht_a = self.co_attention.forward(avg_features, semantic_features, prev_hidden_states)
            topic, p_stop, hidden_state, sentence_states = self.sentence_model.forward(ctx,
                                                                                       prev_hidden_states,
                                                                                       sentence_states)
            p_stop = p_stop.squeeze(1)
            p_stop = torch.max(p_stop, 1)[1].unsqueeze(1)

            start_tokens = np.zeros((topic.shape[0], 1))
            start_tokens[:, 0] = self.vocab('<start>')
            start_tokens = self.__to_var(torch.Tensor(start_tokens).long(), requires_grad=False)

            sampled_ids = self.word_model.sample(topic, start_tokens)
            prev_hidden_states = hidden_state
            sampled_ids = sampled_ids * p_stop

            pred_sentences.append(self.__vec2sent(sampled_ids.cpu().detach().numpy()[0]))

            cam = torch.mul(visual_features, alpht_v.view(alpht_v.shape[0], alpht_v.shape[1], 1, 1)).sum(1)
            cam.squeeze_()
            cam = cam.cpu().data.numpy()

            cam = cam[1]
            cam = cam / np.max(cam)
            cam = cv2.resize(cam, (self.args.cam_size, self.args.cam_size))
            cam = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

            imgOriginal = cv2.imread(image_file, 1)
            imgOriginal = cv2.resize(imgOriginal, (self.args.cam_size, self.args.cam_size))

            img = cam * 0.5 + imgOriginal
            cv2.imwrite(os.path.join(cam_dir, '{}.png'.format(i)), img)

        pred_sentences = list(filter(None, pred_sentences))
        return '. '.join(pred_sentences) + '.'

    def __init_cam_path(self, image_file):
        if not os.path.exists(self.args.generate_dir):
            os.makedirs(self.args.generate_dir)

        image_dir = os.path.join(self.args.generate_dir, image_file)

        if not os.path.exists(image_dir):
            os.makedirs(image_dir)
        return image_dir

    def __save_json(self, result):
        if not os.path.exists(self.args.result_path):
            os.makedirs(self.args.result_path)
        with open(os.path.join(self.args.result_path, '{}.json'.format(self.args.result_name)), 'w') as f:
            json.dump(result, f)

    def __load_mode_state_dict(self):
        try:
            model_state_dict = torch.load(self.args.load_model_path)
            print("[Load Model-{} Succeed!]".format(self.args.load_model_path))
            print("Load From Epoch {}".format(model_state_dict['epoch']))
            return model_state_dict
        except Exception as err:
            print("[Load Model Failed] {}".format(err))
            raise err

    def __init_tagger(self):
        return Tag()

    def __vec2sent(self, array):
        sampled_caption = []
        for word_id in array:
            word = self.vocab.get_word_by_id(word_id)
            if word == '<start>':
                continue
            if word == '<end>' or word == '':
                break
            sampled_caption.append(word)
        return ' '.join(sampled_caption)

    def __init_vocab(self):
        with open(self.args.vocab_path, 'rb') as f:
            vocab = pickle.load(f)
        return vocab

    def __init_data_loader(self, file_list):
        data_loader = get_loader(image_dir=self.args.image_dir,
                                 caption_json=self.args.caption_json,
                                 file_list=file_list,
                                 vocabulary=self.vocab,
                                 transform=self.transform,
                                 batch_size=self.args.batch_size,
                                 s_max=self.args.s_max,
                                 n_max=self.args.n_max,
                                 shuffle=True)
        return data_loader

    def __init_transform(self):
        transform = transforms.Compose([
            transforms.Resize((self.args.resize, self.args.resize)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))])
        return transform

    def __to_var(self, x, requires_grad=True):
        if self.args.cuda:
            x = x.cuda()
        return Variable(x, requires_grad=requires_grad)

    def __init_visual_extractor(self):
        model = VisualFeatureExtractor(model_name=self.args.visual_model_name,
                                       pretrained=self.args.pretrained)

        if self.model_state_dict is not None:
            print("Visual Extractor Loaded!")
            model.load_state_dict(self.model_state_dict['extractor'])

        if self.args.cuda:
            model = model.cuda()

        model.eval()
        return model

    def __init_mlc(self):
        model = MLC(classes=self.args.classes,
                    sementic_features_dim=self.args.sementic_features_dim,
                    fc_in_features=self.extractor.out_features,
                    k=self.args.k)

        if self.model_state_dict is not None:
            print("MLC Loaded!")
            model.load_state_dict(self.model_state_dict['mlc'])

        if self.args.cuda:
            model = model.cuda()

        model.eval()
        return model

    def __init_co_attention(self):
        model = CoAttention(embed_size=self.args.embed_size,
                            hidden_size=self.args.hidden_size,
                            visual_size=self.extractor.out_features,
                            k=self.args.k,
                            momentum=self.args.momentum)

        if self.model_state_dict is not None:
            print("Co-Attention Loaded!")
            model.load_state_dict(self.model_state_dict['co_attention'])

        if self.args.cuda:
            model = model.cuda()

        model.eval()
        return model

    def __init_sentence_model(self):
        model = SentenceLSTM(version=self.args.version,
                             embed_size=self.args.embed_size,
                             hidden_size=self.args.hidden_size,
                             num_layers=self.args.sentence_num_layers,
                             dropout=self.args.dropout,
                             momentum=self.args.momentum)

        if self.model_state_dict is not None:
            print("Sentence Model Loaded!")
            model.load_state_dict(self.model_state_dict['sentence_model'])

        if self.args.cuda:
            model = model.cuda()

        model.eval()
        return model

    def __init_word_model(self):
        model = WordLSTM(vocab_size=len(self.vocab),
                         embed_size=self.args.embed_size,
                         hidden_size=self.args.hidden_size,
                         num_layers=self.args.word_num_layers,
                         n_max=self.args.n_max)

        if self.model_state_dict is not None:
            print("Word Model Loaded!")
            model.load_state_dict(self.model_state_dict['word_model'])

        if self.args.cuda:
            model = model.cuda()

        model.eval()
        return model


def LoadSampler():
    import warnings
    warnings.filterwarnings("ignore")
    model_dir = "./report_models/model_using/training/20180707-04:15"
    generate_dir = '../mir_server/cam/'

    parser = argparse.ArgumentParser()

    """
    Data Argument
    """
    # Path Argument
    # parser.add_argument('--image_dir', type=str, default='../../medical_report/images',
    #                     help='the path for images')
    parser.add_argument('--image_dir', type=str, default='../mir_server/uploads',
                        help='the path for images')
    parser.add_argument('--caption_json', type=str, default='./data/data_using/test_captions.json',
                        help='path for captions')
    parser.add_argument('--vocab_path', type=str, default='./data/data_using/vocab.pkl',
                        help='the path for vocabulary object')
    parser.add_argument('--file_lits', type=str, default='./data/data_using/test_data.txt',
                        help='the path for test file list')
    parser.add_argument('--load_model_path', type=str, default=os.path.join(model_dir, 'val_best_loss.pth.tar'),
                        help='The path of loaded model')

    # transforms argument
    parser.add_argument('--resize', type=int, default=224,
                        help='size for resizing images')

    # CAM
    parser.add_argument('--cam_size', type=int, default=224)
    parser.add_argument('--generate_dir', type=str, default=generate_dir)

    # Saved result
    parser.add_argument('--result_path', type=str, default=os.path.join(model_dir, 'results'),
                        help='the path for storing results')
    parser.add_argument('--result_name', type=str, default='debugging',
                        help='the name of results')

    """
    Model argument
    """
    parser.add_argument('--momentum', type=int, default=0.1)
    # VisualFeatureExtractor
    parser.add_argument('--visual_model_name', type=str, default='densenet121',
                        help='CNN model name')
    parser.add_argument('--pretrained', action='store_true', default=False,
                        help='not using pretrained model when training')

    # MLC
    parser.add_argument('--classes', type=int, default=325)
    parser.add_argument('--sementic_features_dim', type=int, default=512)
    parser.add_argument('--k', type=int, default=10)

    # Co-Attention
    parser.add_argument('--embed_size', type=int, default=512)
    parser.add_argument('--hidden_size', type=int, default=512)

    # Sentence Model
    parser.add_argument('--version', type=str, default='v1')
    parser.add_argument('--sentence_num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0)

    # Word Model
    parser.add_argument('--word_num_layers', type=int, default=1)

    """
    Generating Argument
    """
    parser.add_argument('--s_max', type=int, default=6)
    parser.add_argument('--n_max', type=int, default=30)

    parser.add_argument('--batch_size', type=int, default=2)

    # Loss function
    parser.add_argument('--lambda_tag', type=float, default=10000)
    parser.add_argument('--lambda_stop', type=float, default=10)
    parser.add_argument('--lambda_word', type=float, default=1)

    # Image file
    parser.add_argument('--image_file', type=str, default='CXR1000_IM-0003-1001.png')

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()

    return CaptionSampler(args)

class Sampler(object):
    # 构造函数里加载模型，比如 tensorflow 的 graph, sess 等
    def __init__(self):
        self.sampler = LoadSampler()

    def sample(self, *params):

        # generate captions
        image_file = params[0]
        captions = self.sampler.sample(image_file)

        # generate heatmapImageUrls
        sentenceLength = 6
        heatmapImageUrls = []
        # imageName = params[0].split('.')[0]
        # imageExtension = params[0].split('.')[1]
        for i in range(sentenceLength):
            heatmapImageUrls.append("http://172.18.160.106:8080/image/" + image_file + "/" + str(i) + '.png')

        return dict(
            captions = captions,
            heatmapImageUrls = heatmapImageUrls
        )


print("生成 Model Sampler 实例.................")
model_sampler = Sampler()
print("Model Sampler 实例生成完成...............")
