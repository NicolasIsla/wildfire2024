import subprocess
import glob
import argparse

def main(dataset, model_directory):
    source = f"{dataset}"
    weights = glob.glob(f"{model_directory}/**/*pt", recursive=True)

    print(f"Number of weight files found: {len(weights)}")

    for weight in weights:
        model_name = weight.split('/')[-1].split('.')[0]
        cmd = f"yolo predict model={weight} iou=0.01 conf=0.01 source={source} save=False save_txt save_conf project=experimento/test_preds name={model_name}"
        print(f"* Command:\n{cmd}")
        subprocess.call(cmd, shell=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run YOLO predictions on validation set for multiple models.")
    parser.add_argument('--data_directory', type=str, required=True, help='Path to the dataset directory.')
    parser.add_argument('--model_directory', type=str, required=True, help='Path to the directory containing model weights.')

    args = parser.parse_args()
    main(args.data_directory, args.model_directory)
