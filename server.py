# coding: utf-8
# 用了 Python 的 Tornado 框架，当然也可以使用其它框架
# 英文文档 (v5.0) http://www.tornadoweb.org/en/stable/
# 中文文档 (v4.3) http://tornado-zh.readthedocs.io/zh/latest/
import json
import random
import os, time
import datetime

import tornado.httpserver
import tornado.ioloop
import tornado.options
import tornado.web
from tornado.options import define, options
from tornado.escape import json_decode, json_encode
from tornado.concurrent import Future

from utils.models import *
from utils.dataset import *
from utils.loss import *
from utils.build_tag import *

# 这里导入模型实例
from processor import model_sampler

# 定义端口默认值
define("port", default=8080, help="run on the given port", type=int)

UPLOADS_PATH = "../mir_server/uploads/"
IMAGE_PATH = "../mir_server/cam/"


class FileHandler(tornado.web.RequestHandler):
    def post(self):
        file1 = self.request.files['file'][0]
        original_fname = file1['filename']
        extension = os.path.splitext(original_fname)[1]
        fname = 'image' + str(int(round(time.time() * 1000))) + str(int(round(random.random() * 100)))
        final_filename = fname+extension
        output_file = open(UPLOADS_PATH + final_filename, 'wb')
        output_file.write(file1['body'])

        res = self.getMir(final_filename)

        self.set_status(res.get("statusCode"))
        self.set_header("Content-Type", "application/json")
        self.set_header("Access-Control-Allow-Origin", "*")
        self.set_header("Access-Control-Allow-Methods", "GET,POST")
        self.set_header("Access-Control-Allow-Headers", "Content-Type")
        self.write(json_encode(res))
        self.finish()

    def getMir(self, *params):
        data = model_sampler.sample(*params)
        res = dict(
            statusCode = 200,
            createAt = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            data = data
        )
        return res

if __name__ == "__main__":
    tornado.options.parse_command_line()
    app = tornado.web.Application(handlers=[
        (r"/api/mir", FileHandler),
        (r"/image/(.*)", tornado.web.StaticFileHandler, {'path': IMAGE_PATH})
    ])
    http_server = tornado.httpserver.HTTPServer(app)
    http_server.listen(options.port)
    tornado.ioloop.IOLoop.instance().start()
