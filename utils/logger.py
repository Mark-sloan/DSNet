import tensorflow as tf
import os


class Logger():
    def __init__(self, sess, log_dir):
        self.sess = sess
        self.log_dir = log_dir
        self.summary_writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)
        
    # it can summarize scalers and images.
    def get_summary_op(self, summaries_dict, scope="summary"):
        self.summary_placeholders = {}
        self.summary_ops = {}
        """
        :param step: the step of the summary
        :param summerizer: use the train summary writer or the test one
        :param scope: variable scope
        :param summaries_dict: the dict of the summaries values (tag, value)
        :return:
        """
        #summary_writer = self.train_summary_writer if summerizer == "train" else self.test_summary_writer
        with tf.variable_scope(scope):
            if summaries_dict is not None:
                for tag, value in summaries_dict.items():
                    if 'scalers' in tag.lower():
                        tf.summary.scalar(tag, value)
                    elif 'images' in tag.lower():
                        tf.summary.image(tag, value, max_outputs=1)
            else:
                raise RuntimeError('summary dict is empty!!')
       
        summary_op = tf.summary.merge_all()
        
        return summary_op
        

                    
    def summary(self, summary, step):
        self.summary_writer.add_summary(summary=summary, global_step=step)
