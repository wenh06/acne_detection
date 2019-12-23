# -*- coding: utf-8 -*-
import os
import io
import glob
import shutil
import random
import pandas as pd
import tensorflow as tf
from PIL import Image
import xml.etree.ElementTree as ET
from collections import namedtuple, OrderedDict
from typing import Union, Optional, NamedTuple, List, Tuple, NoReturn
from absl import flags

from object_detection.utils.dataset_util import (
  int64_feature, int64_list_feature,
  bytes_feature, bytes_list_feature,
  float_list_feature,
)


_ALL_LESION_TYPES = ['fore', 'cold_sore',]
_VALID_LEISION_TYPES = ['fore',]


def xml_to_csv(xml_dir:str, save_csv_path:Optional[str]=None) -> pd.DataFrame:
  """
  """
  xml_list = []
  for xml_file in glob.glob(os.path.join(xml_dir, '*.xml')):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    if len(root.findall('object')) == 0:
      print('{} has no acne annotation'.format(xml_file))
    for member in root.findall('object'):
      values = {
        'filename': root.find('filename').text if root.find('filename') is not None else '',
        'width': int(root.find('size').find('width').text),
        'height': int(root.find('size').find('height').text),
        'segmented': root.find('segmented').text if root.find('segmented') is not None else '',
        'class': member.find('name').text,
        'pose': member.find('pose').text if member.find('pose') is not None else '',
        'truncated': member.find('truncated').text if member.find('truncated') is not None else '',
        'difficult': member.find('difficult').text if member.find('difficult') is not None else '',
        'xmin': int(member.find('bndbox').find('xmin').text),
        'ymin': int(member.find('bndbox').find('ymin').text),
        'xmax': int(member.find('bndbox').find('xmax').text),
        'ymax': int(member.find('bndbox').find('ymax').text),
      }
      xml_list.append(values)
  column_names = ['filename', 'width', 'height', 'segmented', 'class', 'pose', 'truncated', 'difficult', 'xmin', 'ymin', 'xmax', 'ymax']
  xml_df = pd.DataFrame(xml_list)
  xml_df = xml_df[column_names]

  if save_csv_path is not None:
    xml_df.to_csv(save_csv_path, index=False)
    print('Converted xml to csv successfully .')

  return xml_df


def _split(df:pd.DataFrame, group:NamedTuple) -> List[NamedTuple]:
  """
  """
  data = namedtuple('data', ['filename', 'object'])
  gb = df.groupby(group)
  return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]

def _create_tf_example(group:NamedTuple, img_dir:str) -> tf.train.Example:
  """
  """
  with tf.gfile.GFile(os.path.join(img_dir, '{}'.format(group.filename)), 'rb') as fid:
    encoded_jpg = fid.read()
  encoded_jpg_io = io.BytesIO(encoded_jpg)  
  image = Image.open(encoded_jpg_io)  
  width, height = image.size
      
  filename = group.filename.encode('utf8')
  
  xmins = []
  xmaxs = []
  ymins = []
  ymaxs = []
  classes_text = []
  classes = []

  for index, row in group.object.iterrows():
    xmins.append(row['xmin'] / width)
    xmaxs.append(row['xmax'] / width)
    ymins.append(row['ymin'] / height)
    ymaxs.append(row['ymax'] / height)
    classes_text.append(row['class'].encode('utf8'))
    classes.append(1)

  tf_example = tf.train.Example(features=tf.train.Features(feature={
    'image/height': int64_feature(height),
    'image/width': int64_feature(width),
    'image/filename': bytes_feature(filename),
    'image/source_id': bytes_feature(filename),
    'image/encoded': bytes_feature(encoded_jpg),
    'image/format': bytes_feature(b'jpg'),
    'image/object/bbox/xmin': float_list_feature(xmins),
    'image/object/bbox/xmax': float_list_feature(xmaxs),
    'image/object/bbox/ymin': float_list_feature(ymins),
    'image/object/bbox/ymax': float_list_feature(ymaxs),
    'image/object/class/text': bytes_list_feature(classes_text),
    'image/object/class/label': int64_list_feature(classes),
  }))
  
  return tf_example

def csv_to_tfrecord(csv_input:Union[str, pd.DataFrame], img_dir:str, output_path:str):
  """
  """
  writer = tf.python_io.TFRecordWriter(output_path)
  if isinstance(csv_input, str):
    examples = pd.read_csv(csv_input)
  else:
    examples = csv_input.copy()
  
  grouped = split(examples, 'filename')
  for group in grouped:
    tf_example = create_tf_example(group, path)
    writer.write(tf_example.SerializeToString())
    nb_samples += 1

  writer.close()
  print('Successfully created the TFRecords: {}'.format(output_path))
  print('nb_samples =', nb_samples)
  

def train_test_split_dataframe(df:pd.DataFrame) -> Tuple[pd.DataFrame,pd.DataFrame]:
  """
  """
  raise NotImplementedError


def main():
  """
  """
  raise NotImplementedError


if __name__ == '__main__':
  main()
