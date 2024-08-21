import os
import numpy as np
import pandas as pd
import glob
import argparse
from compute_perf import (
    evaluate_predictions,
    evaluate_multiple_pred_folders,
    find_best_conf_threshold_and_plot,
)


def main(args):
    ds_name = args.dataset
    IMG_FOLDER = f"./data/{ds_name}/DS/images/val"
    gt_folder = f"./data/{ds_name}/DS/labels/val"
    imgs = glob.glob(f"{IMG_FOLDER}/*")
    imgs.sort()

    print(f"Total validation images: {len(imgs)}")

    pred_folders = glob.glob(f"./data/{ds_name}/DS/test_preds/**/labels")
    pred_folders.sort()
    pred_folders = [f for f in pred_folders if f"{ds_name}" in f]

    print(f"Total prediction folders: {len(pred_folders)}")

    # Define a range of possible confidence threshold values
    conf_thres_range = np.linspace(0.01, 0.50, 50)
    results_df = evaluate_multiple_pred_folders(pred_folders, gt_folder, conf_thres_range)
    results_df = results_df.sort_values(by="Best F1 Score", ascending=False)

    print(results_df.head())

    folder_name = results_df["Prediction Folder"].values[0]
    pred_folder = f"./data/test_preds/{folder_name}/labels"

    print(f"Evaluating best folder: {folder_name}")

    # Example usage
    best_conf_thres, best_f1_score, best_precision, best_recall = (
        find_best_conf_threshold_and_plot(pred_folder, gt_folder, conf_thres_range, True)
    )

    print(
        f"Best Confidence Threshold: {best_conf_thres}\nBest F1 Score: {best_f1_score}\nPrecision: {best_precision}\nRecall: {best_recall}"
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate prediction folders and find the best confidence threshold.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name, used to specify the folder structure.")
    
    args = parser.parse_args()
    
    main(args)
