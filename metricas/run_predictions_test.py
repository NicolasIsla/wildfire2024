import subprocess
import glob
import argparse

datasets_test =[
    #r"/data/nisla/2019a-smoke-full/DS/images/test/",
    # r"/data/nisla/AiForMankind/DS/images/test/",
    # # r"/data/nisla/total_Combine/DS/images/test/",
    # r"/data/nisla/SmokesFrames-2.4k/DS/images/test/",
    # # r"/data/nisla/Wilfire_2023/DS/images/test/",
    # r"/data/nisla/Nemo/DS/images/test/",
    # r"/data/nisla/DS_08_V1/DS/images/test/",
    # # r"/data/nisla/DS_08_V2/DS/images/test/"
    # r"/data/nisla/NemoV2/Nemo_yolo/DS/images/test/"
    # r"/data/nisla/NemoV3/DS/images/test/",
    # r"/data/nisla/Smoke50v3/DS/images/test/",
    # r"/data/nisla/TestSmoke/testsmoke/images/test/",
    r"/data/nisla/TestSmokev2/testsmokev2/images/test_10/",

]
def main(model_directory):
    weights = glob.glob(f"{model_directory}/**/*best.pt", recursive=True)

    print(f"Number of weight files found: {len(weights)}")
    for source in datasets_test:
        project = source.split('/')[-5]    
        for weight in weights:
            model_name = weight.split('/')[-3]
            cmd = f"yolo predict model={weight} iou=0.01 conf=0.01 source={source} save=False save_txt save_conf project=models_test/{project}/test_results name={model_name}"
            print(f"* Command:\n{cmd}")
            subprocess.call(cmd, shell=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run YOLO predictions on validation set for multiple models.")
    parser.add_argument('--model_directory', type=str, required=True, help='Path to the directory containing model weights.')


    args = parser.parse_args()
    main(args.model_directory)
