# -*- coding: utf-8 -*-
#!/usr/bin/python3
import logging
import os
import time

def import_tf(device_id=-1, verbose=False):
    """
    import tensorflow, set tensorflow graph load device, set tensorflow log level, return tensorflow instance
    :param device_id: GPU id
    :param verbose: tensorflow logging level
    :return: tensorflow instance
    """
    # set visible gpu, -1 is cpu
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1' if device_id < 0 else str(device_id)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0' if verbose else '3'
    import tensorflow as tf
    tf.logging.set_verbosity(tf.logging.DEBUG if verbose else tf.logging.ERROR)
    return tf

def set_logger(context, log_dir, verbose=False):
    """
    set logger
    :param context: logger name
    :param log_dir: log file dir
    :param verbose: log level
    :return: python logging instance
    """
    return TNLog(log_dir, context, verbose)

class TNLog(object):
    def __init__(self, log_dir, context, verbose):
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)

        self.context = context
        self.root_log_dir = log_dir
        self.verbose = verbose

        self.formatter = logging.Formatter(
            '%(asctime)s-%(levelname)s:' + self.context + ':%(message)s',
            datefmt='%Y-%m-%d %H:%M:%S')

        # create log dir
        self.log_dir = time.strftime('%Y-%m-%d', time.localtime())

        self._create_logger()

    def _create_logger(self):
        """
        create logger
        :return:
        """
        self.loggers = {}

        if not os.path.exists(os.path.join(self.root_log_dir, self.log_dir)):
            os.mkdir(os.path.join(self.root_log_dir, self.log_dir))

        handlers = {logging.DEBUG: os.path.join(os.path.join(self.root_log_dir, self.log_dir), 'DEBUG.log'),
                    logging.INFO: os.path.join(os.path.join(self.root_log_dir, self.log_dir), 'INFO.log'),
                    logging.WARNING: os.path.join(os.path.join(self.root_log_dir, self.log_dir), 'WARNING.log'),
                    logging.ERROR: os.path.join(os.path.join(self.root_log_dir, self.log_dir), 'ERROR.log')}
        levels = handlers.keys()
        for level in levels:
            logger = logging.getLogger(str(level))
            logger.setLevel(level)

            file_handler = logging.FileHandler(handlers[level], encoding='utf-8')
            file_handler.setLevel(level)
            file_handler.setFormatter(self.formatter)
            logger.handlers = []
            logger.addHandler(file_handler)

            self.loggers[level] = logger

    def _change_file_handler(self):
        """
        change file handler
        :return:
        """
        if not time.strftime('%Y-%m-%d', time.localtime()) == self.log_dir:
            self.loggers = {}

            self.log_dir = time.strftime('%Y-%m-%d', time.localtime())
            self._create_logger()

    def info(self, msg):
        """
        log info level
        :param msg: log message
        :return:
        """
        print(self._time+'-INFO:%s:%s' % (self.context, msg), flush=True)
        self._change_file_handler()
        self.loggers[logging.INFO].info(msg)

    def error(self, msg):
        """
        log error level
        :param msg: log message
        :return:
        """
        print(self._time+'-ERROR:%s:%s' % (self.context, msg), flush=True)
        self._change_file_handler()
        self.loggers[logging.ERROR].error(msg)

    def debug(self, msg):
        """
        log debug level
        :param msg: log message
        :return:
        """
        if self.verbose:
            print(self._time+'-DEBUG:%s:%s' % (self.context, msg), flush=True)

        self._change_file_handler()
        self.loggers[logging.DEBUG].debug(msg)

    def warning(self, msg):
        """
        log warning level
        :param msg: log message
        :return:
        """
        print(self._time + '-ERROR:%s:%s' % (self.context, msg), flush=True)
        self._change_file_handler()
        self.loggers[logging.WARNING].error(msg)

    @property
    def _time(self):
        return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())