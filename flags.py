import sys

from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_string('tags', help='', default='-')
flags.DEFINE_string('model_name', help='', default='gemini')
flags.DEFINE_string('task', help='', default='behaviour')
flags.DEFINE_string('mode', help='', default='test')
flags.DEFINE_string('dataset_name', help='', default='serengeti')
flags.DEFINE_integer('seed', help='', default=42)
flags.DEFINE_integer('num_epochs', help='', default=10)
flags.DEFINE_integer('batch_size', help='', default=2)
flags.DEFINE_integer('patience', help='', default=15)
flags.DEFINE_float('learning_rate', help='', default=0.001)
flags.DEFINE_string('train_filename', help='',
                    default='/data/luiz/dataset/partitions/behaviour-classifier/serengeti/train.csv')
flags.DEFINE_string('val_filename', help='',
                    default='/data/luiz/dataset/partitions/behaviour-classifier/serengeti/val.csv')
flags.DEFINE_string('test_filename', help='',
                    default='/data/luiz/dataset/partitions/behaviour-classifier/serengeti/test.csv')
flags.DEFINE_string('checkpoint_path', help='',
                    default='/data/luiz/dataset/models/behaviour-classifier/')
flags.DEFINE_string('results_path', help='',
                    default='/data/luiz/dataset/results/behaviour-classifier/')

flags.FLAGS(sys.argv)
