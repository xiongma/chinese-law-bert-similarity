# -*- coding: utf-8 -*-
#!/usr/bin/python3

import os

from termcolor import colored

from helper import import_tf, set_logger

__all__ = ['BertSim']

class BertSim(object):
    def __init__(self, gpu_no, log_dir, bert_sim_dir, verbose=False):
        self.bert_sim_dir = bert_sim_dir
        self.logger = set_logger(colored('BS', 'cyan'), log_dir, verbose)

        self.tf = import_tf(gpu_no, verbose)

        # add tokenizer
        from bert import tokenization
        self.tokenizer = tokenization.FullTokenizer(os.path.join(bert_sim_dir, 'vocab.txt'))
        # add placeholder
        self.input_ids = self.tf.placeholder(self.tf.int32, (None, 45), 'input_ids')
        self.input_mask = self.tf.placeholder(self.tf.int32, (None, 45), 'input_mask')
        self.input_type_ids = self.tf.placeholder(self.tf.int32, (None, 45), 'input_type_ids')
        # init graph
        self._init_graph()

    def _init_graph(self):
        """
        init bert graph
        """
        try:
            from bert import modeling
            bert_config = modeling.BertConfig.from_json_file(os.path.join(self.bert_sim_dir, 'bert_config.json'))
            self.model = modeling.BertModel(config=bert_config,
                                            is_training=False,
                                            input_ids=self.input_ids,
                                            input_mask=self.input_mask,
                                            token_type_ids=self.input_type_ids,
                                            use_one_hot_embeddings=False)

            # get output weights and output bias
            ckpt = self.tf.train.get_checkpoint_state(self.bert_sim_dir).all_model_checkpoint_paths[-1]
            reader = self.tf.train.NewCheckpointReader(ckpt)
            output_weights = reader.get_tensor('output_weights')
            output_bias = reader.get_tensor('output_bias')

            # get result op
            output_layer = self.model.get_pooled_output()
            logits = self.tf.matmul(output_layer, output_weights, transpose_b=True)
            logits = self.tf.nn.bias_add(logits, output_bias)
            self.probabilities = self.tf.nn.softmax(logits, axis=-1)

            sess_config = self.tf.ConfigProto(allow_soft_placement=True)
            sess_config.gpu_options.allow_growth = True

            graph = self.probabilities.graph
            saver = self.tf.train.Saver()
            self.sess = self.tf.Session(config=sess_config, graph=graph)
            self.sess.run(self.tf.global_variables_initializer())
            self.tf.reset_default_graph()
            saver.restore(self.sess, ckpt)

        except Exception as e:
            self.logger.error(e)

    def predict(self, request_list):
        """
        bert model predict
        :return: label, similarity
        :param request_list: request list, each element is text_a and text_b
        """
        # with self.sess.as_default():
        input_ids = []
        input_masks = []
        segment_ids = []

        for d in request_list:
            text_a = d[0]
            text_b = d[1]

            input_id, input_mask, segment_id = self._convert_single_example(text_a, text_b)
            input_ids.append(input_id)
            input_masks.append(input_mask)
            segment_ids.append(segment_id)

        predict_result = None
        try:
            predict_result = self.sess.run(self.probabilities, feed_dict={self.input_ids: input_ids,
                                                                          self.input_mask: input_masks,
                                                                          self.input_type_ids: segment_ids})
        except Exception as e:
            self.logger.error(e)
        finally:
            return predict_result

    def _convert_single_example(self, text_a, text_b):
        """
        convert text a and text b to id, padding [CLS] [SEP]
        :param text_a: text a
        :param text_b: text b
        :return: input ids, input mask, segment ids
        """
        tokens = []
        input_ids = []
        segment_ids = []
        input_mask = []
        try:
            text_a = self.tokenizer.tokenize(text_a)
            text_b = self.tokenizer.tokenize(text_b)
            self._truncate_seq_pair(text_a, text_b)

            tokens.append("[CLS]")
            segment_ids.append(0)

            for token in text_a:
                tokens.append(token)
                segment_ids.append(0)
            segment_ids.append(0)
            tokens.append("[SEP]")

            for token in text_b:
                tokens.append(token)
                segment_ids.append(1)
            tokens.append('[SEP]')
            segment_ids.append(1)

            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

            input_mask = [1] * len(input_ids)

            while len(input_ids) < 45:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)

        except:
            self.logger.error()

        finally:
            return input_ids, input_mask, segment_ids

    def _truncate_seq_pair(self, tokens_a, tokens_b):
        """
        Truncates a sequence pair in place to the maximum length.
        :param tokens_a: text a
        :param tokens_b: text b
        """
        try:
            while True:
                total_length = len(tokens_a) + len(tokens_b)

                if total_length <= 45 - 3:
                    break
                if len(tokens_a) > len(tokens_b):
                    tokens_a.pop()
                else:
                    tokens_b.pop()
        except:
            self.logger.error()


if __name__ == "__main__":
    bs = BertSim(gpu_no=0, log_dir='log', bert_sim_dir='bert_sim_model\\', verbose=True)
    text_a = '华为还准备起诉美国政府'
    text_b = '飞机出现后货舱火警信息'
    print(bs.predict([[text_a, text_b]]))