import argparse
import os
import pickle
import sys
from glob import glob

import cv2
import numpy as np
import torch
from tqdm import tqdm
from webdataset import WebLoader

sys.path.append(".")
from src.data import load_dataset
from src.model import VAE
from src.utils import video, vis, yaml_handler

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_root", type=str)
    parser.add_argument("-v", "--version", type=int, default=0)
    parser.add_argument("-ep", "--epoch", type=int, default=None)
    parser.add_argument("-g", "--gpu_id", type=int, default=None)
    args = parser.parse_args()
    data_root = args.data_root
    v = args.version
    ep = args.epoch
    gpu_id = args.gpu_id
    device = f"cuda:{gpu_id}"

    # data_dirs = sorted(glob(os.path.join(data_root, "*/")))
    data_dirs = [data_root]
    data_dir = data_dirs[0]

    checkpoint_dir = f"models/individual/vae/version_{v}"
    if ep is not None:
        checkpoint_path = f"{checkpoint_dir}/vae-seq_len90-stride30-256x192-last-epoch={ep}.ckpt"
    else:
        checkpoint_path = sorted(glob(f"{checkpoint_dir}/*.ckpt"))[-1]

    config = yaml_handler.load(f"{checkpoint_dir}/individual-vae.yaml")
    seq_len = config.seq_len
    stride = config.stride

    # load dataset
    dataset, n_samples = load_dataset(data_dirs, "individual", config, False)
    dataloader = WebLoader(dataset, num_workers=1, pin_memory=True)

    # load model
    model = VAE(config)
    model.configure_model()
    model = model.to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])

    # load video
    video_path = f"{data_dir}.mp4"
    cap = video.Capture(video_path)
    max_n_frame = cap.frame_count
    frame_size = cap.size

    # create writers
    wrt_x_vis = video.Writer(f"{data_dir}/pred_x_vis.mp4", cap.fps, cap.size)
    wrt_x_spc = video.Writer(f"{data_dir}/pred_x_spc.mp4", cap.fps, cap.size)
    wrt_cluster = video.Writer(f"{data_dir}/pred_cluster.mp4", cap.fps, cap.size)

    # pred
    mse_x_vis_dict = {}
    mse_x_spc_dict = {}
    latent_features = {"id": [], "label": [], "mu": []}
    save_dir = os.path.join(data_dir, "pred")
    os.makedirs(save_dir, exist_ok=True)

    model.eval()
    with torch.no_grad():
        pre_n_frame = seq_len
        results_tmp = []
        for batch in tqdm(dataloader, total=n_samples):
            results = model.predict_step(batch)
            for result in results:
                n_frame = int(result["key"].split("_")[1])
                key = result["key"]
                _id = result["id"]

                path = os.path.join(save_dir, f"{key}.pkl")
                with open(path, "wb") as f:
                    pickle.dump(result, f)

                # collect mse
                if _id not in mse_x_vis_dict:
                    mse_x_vis_dict[_id] = {}
                    mse_x_spc_dict[_id] = {}
                mse_x_vis_dict[_id][n_frame] = result["mse_x_vis"]
                mse_x_spc_dict[_id][n_frame] = result["mse_x_spc"]

                # collect latent features
                latent_features["id"].append(result["id"])
                latent_features["mu"].append(result["mu"])
                latent_features["label"].append(result["label"])

                # plot bboxs
                if pre_n_frame < n_frame:
                    for i in range(stride):
                        n_frame_tmp = pre_n_frame - stride + i
                        idx_data = seq_len - stride + i

                        frame = cap.read(n_frame_tmp)[1]
                        frame = cv2.putText(
                            frame,
                            f"frame:{n_frame_tmp}",
                            (10, 40),
                            cv2.FONT_HERSHEY_COMPLEX,
                            1.0,
                            (255, 255, 255),
                            1,
                        )

                        frame_vis = vis.plot_on_frame(
                            frame.copy(), results_tmp, idx_data, frame_size, "x_vis"
                        )
                        wrt_x_vis.write(frame_vis)
                        frame_spc = vis.plot_on_frame(
                            frame.copy(), results_tmp, idx_data, frame_size, "x_spc"
                        )
                        wrt_x_spc.write(frame_spc)
                        frame_cluster = vis.plot_on_frame(
                            frame.copy(), results_tmp, idx_data, frame_size, "cluster"
                        )
                        wrt_cluster.write(frame_cluster)

                    results_tmp = []
                    pre_n_frame = n_frame

                # add result in temporary result list
                results_tmp.append(result)

    del cap, wrt_x_vis, wrt_x_spc, wrt_cluster

    # plot mse
    vis.plot_mse(
        mse_x_vis_dict,
        max_n_frame,
        stride,
        0.05,
        "MSE_x_vis",
        f"{data_dir}/pred_x_vis.jpg",
    )
    vis.plot_mse(
        mse_x_spc_dict,
        max_n_frame,
        stride,
        0.05,
        "MSE_x_spc",
        f"{data_dir}/pred_x_spc.jpg",
    )

    # plot latent feature
    X = np.array(latent_features["mu"]).reshape(-1, config.latent_ndim * 19 * 2)
    labels = np.array(latent_features["label"])
    vis.plot_tsne(X, labels, 10, f"{data_dir}/pred_mu_tsne_label.jpg", cmap="tab10")

    # labels = np.array(latent_features["id"])
    # lut = len(np.unique(labels))
    # vis.plot_tsne(
    #     X, labels, 10, f"{data_dir}/pred_mu_tsne_id.jpg", cmap="gist_ncar", lut=lut
    # )
