# Title: RADDet
# Authors: Ao Zhang, Erlik Nowruzi, Robert Laganiere
import tensorflow as tf
import numpy as np
import glob, os
import json
import random

import util.loader as loader
import util.helper as helper

class DataGenerator:
    def __init__(self, config_data, config_train, config_model, headoutput_shape, \
                anchors, anchors_cart=None, cart_shape=None):
        """ Data Generator:
            Data, Gt loader and generator, all sequences are based on the file
        PROJECT_ROOT/sequences.txt. 
        """
        self.input_size = config_model["input_shape"]
        self.config_data = config_data
        self.config_train = config_train
        self.config_model = config_model
        self.headoutput_shape = headoutput_shape
        self.cart_shape = cart_shape
        self.grid_strides = self.getGridStrides()
        self.anchor_boxes = anchors
        self.path = "/opt/dataset_ssd/radar1/Carrada/"
        with open("/opt/dataset_ssd/radar1/Carrada/annotations_frame_oriented.json", 'r') as fp:
            self.annotations = json.load(fp)
        self.RAD_sequences_train = self.readSequences(mode="train")
        random.shuffle(self.RAD_sequences_train)
        self.RAD_sequences_test = self.readSequences(mode="test")
        self.RAD_sequences_validate = self.readSequences(mode="val")

        self.batch_size = config_train["batch_size"]
        self.total_train_batches = (self.config_train["epochs"] * \
                                    4319) // self.batch_size
        self.total_test_batches = 1391
        self.total_validate_batches = 1483 // self.batch_size



    def getGridStrides(self, ):
        """ Get grid strides """
        strides = (np.array(self.config_model["input_shape"])[:3] / \
                            np.array(self.headoutput_shape[1:4]))
        return np.array(strides).astype(np.float32)

    @staticmethod
    def read_seq_ref(path):
        with open(os.path.join(path, "data_seq_ref.json")) as fp:
            seq_ref = json.load(fp)
        return seq_ref

    @staticmethod
    def read_dataset_sequences(path):
        with open(os.path.join(path, "selected_light_dataset_frame_oriented.json")) as fp:
            sequences = json.load(fp)
        return sequences

    def readSequences(self, mode):
        """ Read sequences from PROJECT_ROOT/sequences.txt """
        assert mode in ["train", "test", "val"]

        seq_ref = self.read_seq_ref(self.path)
        all_sequences = self.read_dataset_sequences(self.path)
        sequences = []
        for seq in seq_ref:
            if mode == "train":
                if seq_ref[seq]["split"] == "Train":
                    sequences.extend(glob.glob(os.path.join(self.path, seq, "RAD_numpy/*.npy")))
            elif mode == "test":
                if seq_ref[seq]["split"] == "Test":
                    sequences.extend(glob.glob(os.path.join(self.path, seq, "RAD_numpy/*.npy")))
            elif mode == "val":
                if seq_ref[seq]["split"] == "Validation":
                    sequences.extend(glob.glob(os.path.join(self.path, seq, "RAD_numpy/*.npy")))

        if len(sequences) == 0:
            raise ValueError("Cannot read sequences.txt. \
                        Please check if the file is organized properly.")
        return sequences

    """---------------------------------------------------------------------"""
    """-------------------- RAD 3D Boxes train/test set --------------------"""
    """---------------------------------------------------------------------"""
    def encodeToLabels(self, gt_instances):
        """ Transfer ground truth instances into Detection Head format """
        raw_boxes_xyzwhd = np.zeros((self.config_data["max_boxes_per_frame"], 7))
        ### initialize gronud truth labels as np.zeors ###
        gt_labels = np.zeros(list(self.headoutput_shape[1:4]) + \
                        [len(self.anchor_boxes)] + \
                        [len(self.config_data["all_classes"]) + 7])

        ### start transferring box to ground turth label format ###
        for i in range(len(gt_instances["classes"])):
            if i > self.config_data["max_boxes_per_frame"]:
                continue
            class_name = gt_instances["classes"][i]
            box_xyzwhd = gt_instances["boxes"][i]
            class_id = self.config_data["all_classes"].index(class_name)
            if i < self.config_data["max_boxes_per_frame"]:
                raw_boxes_xyzwhd[i, :6] = box_xyzwhd
                raw_boxes_xyzwhd[i, 6] = class_id
            class_onehot = helper.smoothOnehot(class_id, len(self.config_data["all_classes"]))
            
            exist_positive = False

            grid_strid = self.grid_strides
            anchor_stage = self.anchor_boxes
            box_xyzwhd_scaled = box_xyzwhd[np.newaxis, :].astype(np.float32)
            box_xyzwhd_scaled[:, :3] /= grid_strid
            anchorstage_xyzwhd = np.zeros([len(anchor_stage), 6])
            anchorstage_xyzwhd[:, :3] = np.floor(box_xyzwhd_scaled[:, :3]) + 0.5
            anchorstage_xyzwhd[:, 3:] = anchor_stage.astype(np.float32)

            iou_scaled = helper.iou3d(box_xyzwhd_scaled, anchorstage_xyzwhd, \
                                        self.input_size)
            ### NOTE: 0.3 is from YOLOv4, maybe this should be different here ###
            ### it means, as long as iou is over 0.3 with an anchor, the anchor
            ### should be taken into consideration as a ground truth label
            iou_mask = iou_scaled > 0.3

            if np.any(iou_mask):
                xind, yind, zind = np.floor(np.squeeze(box_xyzwhd_scaled)[:3]).\
                                    astype(np.int32)
                ### TODO: consider changing the box to raw yolohead output format ###
                gt_labels[xind, yind, zind, iou_mask, 0:6] = box_xyzwhd
                gt_labels[xind, yind, zind, iou_mask, 6:7] = 1.
                gt_labels[xind, yind, zind, iou_mask, 7:] = class_onehot
                exist_positive = True

            if not exist_positive:
                ### NOTE: this is the normal one ###
                ### it means take the anchor box with maximum iou to the raw
                ### box as the ground truth label
                anchor_ind = np.argmax(iou_scaled)
                xind, yind, zind = np.floor(np.squeeze(box_xyzwhd_scaled)[:3]).\
                                    astype(np.int32)
                gt_labels[xind, yind, zind, anchor_ind, 0:6] = box_xyzwhd
                gt_labels[xind, yind, zind, anchor_ind, 6:7] = 1.
                gt_labels[xind, yind, zind, anchor_ind, 7:] = class_onehot

        has_label = False
        for label_stage in gt_labels:
            if label_stage.max() != 0:
                has_label = True
        gt_labels = [np.where(gt_i == 0, 1e-16, gt_i) for gt_i in gt_labels]
        return gt_labels, has_label, raw_boxes_xyzwhd

    def trainData(self,):
        """ Generate train data with batch size """
        count = 0
        while count < len(self.RAD_sequences_train):
            RAD_filename = self.RAD_sequences_train[count] 
            RAD_data = loader.readRAD(RAD_filename)
            if RAD_data is None:
                raise ValueError("RAD file not found, please double check the path")
            ### NOTE: Gloabl Normalization ###
            RAD_data = np.log10(RAD_data ** 2 + 1)
            RAD_data = (RAD_data - self.config_data["global_mean_log"]) / \
                                self.config_data["global_variance_log"]
            ### load ground truth instances ###
            gt_instances = self.generate_gt_instances(RAD_filename)

            ### NOTE: decode ground truth boxes to YOLO format ###
            gt_labels, has_label, raw_boxes = self.encodeToLabels(gt_instances)

            if has_label:
                yield (RAD_data, gt_labels, raw_boxes)
            count += 1
            if count == len(self.RAD_sequences_train) - 1:
                # np.random.seed() # should I add seed here ?
                np.random.shuffle(self.RAD_sequences_train)

    def testData(self, ):
        """ Generate test data with batch size """
        count = 0
        while count < len(self.RAD_sequences_test):
            RAD_filename = self.RAD_sequences_test[count]
            RAD_data = loader.readRAD(RAD_filename)
            if RAD_data is None:
                raise ValueError("RAD file not found, please double check the path")
            ### NOTE: Gloabl Normalization ###
            RAD_data = np.log10(RAD_data ** 2 + 1)
            RAD_data = (RAD_data - self.config_data["global_mean_log"]) / \
                       self.config_data["global_variance_log"]
            ### load ground truth instances ###
            gt_instances = self.generate_gt_instances(RAD_filename)

            ### NOTE: decode ground truth boxes to YOLO format ###
            gt_labels, has_label, raw_boxes = self.encodeToLabels(gt_instances)

            if has_label:
                yield (RAD_data, gt_labels, raw_boxes)
            count += 1

    def validateData(self, ):
        """ Generate test data with batch size """
        count = 0
        while  count < len(self.RAD_sequences_validate):
            RAD_filename = self.RAD_sequences_validate[count]
            RAD_data = loader.readRAD(RAD_filename)
            if RAD_data is None:
                raise ValueError("RAD file not found, please double check the path")
            ### NOTE: Gloabl Normalization ###
            RAD_data = np.log10(RAD_data ** 2 + 1)
            RAD_data = (RAD_data - self.config_data["global_mean_log"]) / \
                       self.config_data["global_variance_log"]
            ### load ground truth instances ###
            gt_instances = self.generate_gt_instances(RAD_filename)

            ### NOTE: decode ground truth boxes to YOLO format ###
            gt_labels, has_label, raw_boxes = self.encodeToLabels(gt_instances)

            if has_label:
                yield (RAD_data, gt_labels, raw_boxes)
            count += 1

    def trainGenerator(self,):
        """ Building data generator using tf.data.Dataset.from_generator """
        return tf.data.Dataset.from_generator(self.trainData, \
                    output_types=(tf.float32, tf.float32, tf.float32), \
                    output_shapes=(tf.TensorShape(self.config_model["input_shape"]), \
                            tf.TensorShape(list(self.headoutput_shape[1:4]) + \
                            [len(self.anchor_boxes), \
                            7+len(self.config_data["all_classes"])]), \
                            tf.TensorShape([self.config_data["max_boxes_per_frame"], 7]) \
                            ), )

    def testGenerator(self,):
        """ Building data generator using tf.data.Dataset.from_generator """
        return tf.data.Dataset.from_generator(self.testData, \
                    output_types=(tf.float32, tf.float32, tf.float32), \
                    output_shapes=(tf.TensorShape(self.config_model["input_shape"]), \
                            tf.TensorShape(list(self.headoutput_shape[1:4]) + \
                            [len(self.anchor_boxes), \
                            7+len(self.config_data["all_classes"])]), \
                            tf.TensorShape([self.config_data["max_boxes_per_frame"], 7]) \
                            ), )
 
    def validateGenerator(self,):
        """ Building data generator using tf.data.Dataset.from_generator """
        return tf.data.Dataset.from_generator(self.validateData, \
                    output_types=(tf.float32, tf.float32, tf.float32), \
                    output_shapes=(tf.TensorShape(self.config_model["input_shape"]), \
                            tf.TensorShape(list(self.headoutput_shape[1:4]) + \
                            [len(self.anchor_boxes), \
                            7+len(self.config_data["all_classes"])]), \
                            tf.TensorShape([self.config_data["max_boxes_per_frame"], 7]) \
                            ), )

    @staticmethod
    def build_bbox2D(polygon, mode="RD"):
        x, y = [], []
        for x_, y_ in polygon:
            x.append(x_)
            y.append(y_)
        x1, y1, x2, y2 = min(x), min(y), max(x), max(y)
        if mode == "RA":
            bbox = [x1, y1, x2, y2]
        else:
            bbox = [256 - x2, y1, 256 - x1, y2]
        return bbox

    @staticmethod
    def build_bbox3D(rd_box, ra_box):
        xmin = rd_box[0]
        xmax = rd_box[2]
        ymin = ra_box[1]
        ymax = ra_box[3]
        zmin = rd_box[1]
        zmax = rd_box[3]
        return [xmin, xmax, ymin, ymax, zmin, zmax]

    def generate_gt_instances(self, RAD_filename):
        gt_instances = {"classes": [],
                        "boxes": []}
        seq, _, frame = RAD_filename.split(self.path)[1].split("/")
        frame = frame.split(".")[0]
        gt_dict = self.annotations[seq][frame]
        for instance in gt_dict:
            rd_dict = gt_dict[instance]["range_doppler"]
            ra_dict = gt_dict[instance]["range_angle"]
            ra_boxes = self.build_bbox2D(ra_dict["dense"], mode="RA")
            rd_boxes = self.build_bbox2D(rd_dict["dense"], mode="RD")
            boxes = self.build_bbox3D(rd_box=rd_boxes, ra_box=ra_boxes)
            boxes = helper.boxLocationsToWHD(np.array(boxes))
            gt_instances["classes"].append(self.config_data["all_classes"][rd_dict["label"]-1])
            gt_instances["boxes"].append(boxes)
        gt_instances["boxes"] = np.array(gt_instances["boxes"])
        return gt_instances
