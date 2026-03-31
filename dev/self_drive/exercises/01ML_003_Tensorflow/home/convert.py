import io
import os
import argparse
import logging
from enum import Enum

import sys
sys.path.append("/dev/self_drive/utils/waymo-open-dataset/src/waymo_open_dataset")

import tensorflow.compat.v1 as tf
from PIL import Image
from waymo_open_dataset import dataset_pb2 as open_dataset

from utils import parse_frame, int64_feature, int64_list_feature, bytes_feature
from utils import bytes_list_feature, float_list_feature

def get_minmax(center_coord, lenwid):
    return [center_coord - 0.5 * lenwid, center_coord + 0.5 * lenwid]

def create_tf_example(filename, encoded_jpeg, annotations):
    """
    convert to tensorflow object detection API format
    args:
    - filename [str]: name of the image
    - encoded_jpeg [bytes-likes]: encoded image
    - annotations [list]: bboxes and classes
    returns:
    - tf_example [tf.Example]
    """
    
    # Buffer image
    encoded_jpg_io = io.BytesIO(encoded_jpeg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    mapping = {1: "vehicle", 2: "pedestrian", 4: "cyclist"}
    img_fmt = b'jpg'
    filename = filename.encode('utf8')

    #xmins        = []
    #xmaxs        = []
    #ymins        = []
    #ymaxs        = []
    #classes_text = []
    #classes      = []

    #for ann in annotations:
    #    xmin ann.box.center_x - 0.5 * ann.box.length
    #    xmax ann.box.center_x + 0.5 * ann.box.length
    #    ymin ann.box.center_y - 0.5 * ann.box.width
    #    ymax ann.box.center_y + 0.5 * ann.box.width

    #    xmins.append(xmin / width)
    #    xmaxs.append(xmax / width)
    #    ymins.append(ymin / width)
    #    ymaxs.append(ymax / width)

    #    classes.append(ann.type)
    #    classes_text.append(mapping[ann.type].encode("utf8"))

    #tf_example = tf.train.Example(features=tf.train.Features(feature={
    #    "image/height"             : int64_feature(height),
    #    "image/width"              : int64_feature(width),
    #    "image/filename"           : bytes_feature(filename),
    #    "image/source_id"          : bytes_feature(filename),
    #    "image/encoded"            : bytes_feature(encoded_jpeg),
    #    "image/format"             : bytes_feature(image_format),
    #    "image/object/bbox/xmin"   : float_list_feature(xmins),
    #    "image/object/bbox/xmax"   : float_list_feature(xmaxs),
    #    "image/object/bbox/ymin"   : float_list_feature(ymins),
    #    "image/object/bbox/ymax"   : float_list_feature(ymaxs),
    #    "image/object/class/text"  : bytes_list_feature(classes_text),
    #    "image/object/class/label" : int64_list_feature(classes)}))

    #return tf_example



    
    class Mapping(Enum):
        VEHICLE    = 1,
        PEDESTRIAN = 2,
        CYCLIST    = 4
        
    xmns        = []
    xmxs        = []
    ymns        = []
    ymxs        = []
    classes     = []
    classes_txt = []
 
    mapping     = {Mapping.VEHICLE:'vehicle', Mapping.PEDESTRIAN:'pedestrian', Mapping.CYCLIST:'cyclist'}
    
    # ------------------------------------------------------------------
    # Process image
    # ------------------------------------------------------------------
    # buffer input image
    encoded_img_buffered = io.BytesIO(encoded_jpeg)
    # open image
    image = Image.open(encoded_img_buffered)
    w, h = image.size
    
    # ------------------------------------------------------------------
    # Process Annotations
    # ------------------------------------------------------------------
    for annotation in annotations:
        xmnmx = get_minmax(annotation.box.center_x, annotation.box.length)
        ymnmx = get_minmax(annotation.box.center_y, annotation.box.width)
        
        xmns.append(xmnmx[0]/w)
        xmxs.append(xmnmx[1]/w)
        ymns.append(ymnmx[0]/h)
        ymxs.append(ymnmx[1]/h)
        
        classes.append(annotation.type)
        classes_text.append(mapping[annotation.type].encode('utf8'))

    # ------------------------------------------------------------------
    # Roll into tf_example format
    # ------------------------------------------------------------------ 
    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height'             : int64_feature(height),
        'image/width'              : int64_feature(width),
        'image/filename'           : bytes_feature(filename),
        'image/source_id'          : bytes_feature(filename),
        'image/encoded'            : bytes_feature(encoded_jpeg),
        'image/format'             : bytes_feature(image_format),
        'image/object/bbox/xmin'   : float_list_feature(xmns),
        'image/object/bbox/xmax'   : float_list_feature(xmxs),
        'image/object/bbox/ymin'   : float_list_feature(ymns),
        'image/object/bbox/ymax'   : float_list_feature(ymxs),
        'image/object/class/text'  : bytes_list_feature(class_txt),
        'image/object/class/label' : int64_list_feature(classes),
    }))

    return tf_example


def process_tfr(path):
    """
    process a waymo tf record into a tf api tf record
    """
    # create processed data dir
    file_name = os.path.basename(path)

    logging.info(f'Processing {path}')
    writer = tf.python_io.TFRecordWriter(f'output/{file_name}')
    dataset = tf.data.TFRecordDataset(path, compression_type='')
    for idx, data in enumerate(dataset):
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        encoded_jpeg, annotations = parse_frame(frame)
        filename = file_name.replace('.tfrecord', f'_{idx}.tfrecord')
        tf_example = create_tf_example(filename, encoded_jpeg, annotations)
        writer.write(tf_example.SerializeToString())
    writer.close()


if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', required=True, type=str,
                        help='Waymo Open dataset tf record')
    args = parser.parse_args()  
    process_tfr(args.path)
