from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import pickle
import pandas as pd
from keras.models import Model

from keras import backend as K
from keras_frcnn.config import Config
from keras_frcnn.resnet import *
from keras_frcnn.data_generators import *
from keras_frcnn.roi_helpers import *

import argparse


def format_img_size(img, C):
    """ formats the image size based on config """
    img_min_side = float(C.im_size)
    (height, width, _) = img.shape
    ratio = 1

    return img, ratio


def format_img_channels(img, C):
    """ formats the image channels based on config """
    img = img[:, :, (2, 1, 0)]
    img = img.astype(np.float32)
    img[:, :, 0] -= C.img_channel_mean[0]
    img[:, :, 1] -= C.img_channel_mean[1]
    img[:, :, 2] -= C.img_channel_mean[2]
    img /= C.img_scaling_factor
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img


def format_img(img, C):
    """ formats an image for model prediction based on config """
    img, ratio = format_img_size(img, C)
    img = format_img_channels(img, C)
    return img, ratio

def get_metrics(gt, preds, threshold=0.2):
    tp = 0.0
    fn = 0.0
    fp = 0.0
    for img, pred_coords_list in preds.items():
        #         image_with_results_path = os.path.join(result_dir,img)
        #         image_with_results = cv2.imread(image_with_results_path)
        df = pd.DataFrame(columns=["pred_bbox", "found_in_gt", "matched_gt_bbox", "iou"])
        df.found_in_gt = False
        gt_coords_list = gt[img]

        if not pred_coords_list:
            print("Coudn't find predicted coordinates for: ", img)
            continue

        for pred_bbox in pred_coords_list:
            got_gt = False
            append_dict = {}
            append_dict['found_in_gt'] = False

            pred_x1, pred_y1 = pred_bbox[0]
            pred_x2, pred_y2 = pred_bbox[1]
            append_dict["pred_bbox"] = pred_bbox

            for gt_bbox in gt_coords_list:
                gt_x1, gt_y1 = gt_bbox[0]
                gt_x2, gt_y2 = gt_bbox[1]

                iou_score = iou((gt_x1, gt_y1, gt_x2, gt_y2),
                                (pred_x1, pred_y1, pred_x2, pred_y2))

                if iou_score >= threshold:
                    #                     print('Predicted:{}, GroundTruth:{}'.format(pred_bbox,gt_bbox))
                    append_dict["found_in_gt"] = True
                    append_dict["matched_gt_bbox"] = gt_bbox
                    append_dict["iou"] = iou_score
                    #                     image_with_results = cv2.putText(image_with_results,'TP',(pred_x1,pred_y1),font,fontScale,fontColor,lineType)
                    #                     image_with_results = cv2.putText(image_with_results,str(round(iou_score,3)),(pred_x1,pred_y2),font,fontScale,fontColor,lineType)
                    #                     got_gt = True
                    break
            #                 else:
            #                     if iou_score>0:
            #                         image_with_results = cv2.putText(image_with_results,str(round(iou_score,3)),(pred_x1,pred_y2),font,fontScale,(255,0,0),lineType)
            #                     got_gt = False

            df = df.append(append_dict, ignore_index=True)
        df_no_dup = df.drop_duplicates(subset=['matched_gt_bbox'])

        #         print(df_no_dup)
        current_tp = df.found_in_gt.sum()
        current_fp = len(df) - current_tp
        current_fn = len(gt_coords_list) - len(df_no_dup.dropna(subset=['matched_gt_bbox']).matched_gt_bbox.unique())
        #         print(img)
        #         print('TP:{} FP:{} FN:{} Total_pred:{}'.format(current_tp,current_fp,current_fn,len(pred_coords_list)))

        tp = tp + current_tp
        fn += current_fn
        fp += current_fp
    #     if tp and fp and fn:
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f_score = 2 * (precision * recall) / (precision + recall)

    return tp, fp, fn, precision, recall, f_score


def predict(img_path, model, C, max_boxes=100):
    img = cv2.imread(img_path)

    X, ratio = format_img(img, C)
    X = np.transpose(X, (0, 2, 3, 1))

    [Y1, Y2, F] = model.predict(X)
    R = rpn_to_roi(Y1, Y2, C, K.image_dim_ordering(), overlap_thresh=0, max_boxes=max_boxes)
    R = R * C.rpn_stride
    R = R.astype('int')
    predictions = []
    pred_img = img.copy()
    #     plt.figure(figsize=(8,8))
    for pos in R:
        predictions.append(((pos[0], pos[1]), (pos[2], pos[3])))
        cv2.rectangle(pred_img, (pos[0], pos[1]), (pos[2], pos[3]), (0, 0, 255), 2)
    #     plt.grid()
    #     plt.imshow(pred_img)
    #     plt.show()
    return pred_img, predictions

def evaluate_model(model, config, gt_images_dict, eval_images_list, images_folder_path, output_folder_path,
                   threshold=0.3, model_type="keras_frcnn", max_boxes=100):
    pred_images_dict = {}

    for img in eval_images_list:
        img_path = os.path.join(images_folder_path, img)

        pred_img, predictions = predict(img_path, model, config, max_boxes)
        pred_images_dict[img] = predictions

    if not os.path.isdir(output_folder_path):
        os.mkdir(output_folder_path)

    with open(os.path.join(output_folder_path, "predictions_{}boxes.pkl".format(max_boxes)), "wb") as f:
        pickle.dump(pred_images_dict, f)

    tp, fp, fn, precision, recall, f_score = get_metrics(gt_images_dict, pred_images_dict, threshold=threshold)

    result = "Precision: {}\nRecall: {}\nF_score: {}\nTP: {}, FP: {}, FN: {}".format(precision, recall, f_score, tp, fp,
                                                                                     fn)
    #     print(result)

    with open(os.path.join(output_folder_path, "results_{}boxes.txt".format(max_boxes)), "w") as f:
        f.write(result)

    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluates the keras-frcnn models")
    parser.add_argument("--model_path")
    parser.add_argument("--config_path")
    parser.add_argument("--eval_list")
    parser.add_argument("--images_folder_path")
    parser.add_argument("--gt_bbox_dict_path")
    parser.add_argument("--output_folder_path")
    parser.add_argument("--iou_threshold", type = float)
    parser.add_argument("--max_bboxes", type = float)
    args = parser.parse_args()

    model_path = args.model_path
    config_path = args.config_path
    eval_files_list_path = args.eval_list
    images_folder_path = args.images_folder_path
    bbox_dict_path = args.gt_bbox_dict_path
    output_folder_path = args.output_folder_path
    iou_threshold = args.iou_threshold
    max_bboxes = args.max_bboxes

    if os.path.exists(config_path):
        with open(config_path, "rb") as f:
            C = pickle.load(f)
    else:
        C = Config()

    num_features = 512

    input_shape_img = (None, None, 3)
    input_shape_features = (None, None, num_features)

    img_input = Input(shape=input_shape_img)
    roi_input = Input(shape=(C.num_rois, 4))
    feature_map_input = Input(shape=input_shape_features)

    # define the base network (VGG here, can be Resnet50, Inception, etc)
    shared_layers = nn_base(img_input, trainable=False)

    # define the RPN, built on the base layers
    num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
    rpn_layers = rpn(shared_layers, num_anchors)

    # classifier = classifier_layer(feature_map_input, roi_input, C.num_rois, nb_classes=len(C.class_mapping))

    model_rpn = Model(img_input, rpn_layers)
    # model_classifier_only = Model([feature_map_input, roi_input], classifier)

    # model_classifier = Model([feature_map_input, roi_input], classifier)
    print('Loading weights from {}'.format(model_path))
    model_rpn.load_weights(model_path, by_name=True)
    # model_classifier.load_weights(C.model_path, by_name=True)

    model_rpn.compile(optimizer='sgd', loss='mse')

    with open("/home/devansh/X4/uhuru_data/eval.txt") as i:
        lines = i.read()
        eval_files = lines.split("\n")

    with open(bbox_dict_path, "rb") as f:
        bbox_coords_dict = pickle.load(f)

    result = evaluate_model(model_path, C, bbox_coords_dict, eval_files,
                            images_folder_path, output_folder_path, iou_threshold, max_bboxes)

    print(result)