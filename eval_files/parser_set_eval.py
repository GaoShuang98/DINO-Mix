import os
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description="Benchmarking Visual Geolocalization",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Training parameters

    parser.add_argument("--infer_batch_size", type=int, default=16,
                        help="Batch size for inference (caching and testing)")

    # Model parameters
    parser.add_argument("--backbone", type=str, default="dinov2_vitb14",
                        choices=["dinov2_vitb14", "dinov2_vits14", "dinov2_vitl14", "dinov2_vitg14", "alexnet", "vgg16",
                                 "resnet18conv4", "resnet18conv5", "resnet50conv4", "resnet50conv5", "resnet101conv4",
                                 "resnet101conv5", "cct384", "vit"], help="_")
    parser.add_argument("--l2", type=str, default="before_pool", choices=["before_pool", "after_pool", "none"],
                        help="When (and if) to apply the l2 norm with shallow aggregation layers")
    parser.add_argument("--aggregation", type=str, default="mixvpr",
                        choices=["mixvpr", "netvlad", "gem", "spoc", "mac", "rmac", "crn", "rrm", "cls", "seqpool",
                                 "none"])

    # Initialization parameters
    parser.add_argument("--seed", type=int, default=0)

    # Other parameters
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--num_workers", type=int, default=7, help="num_workers for all dataloaders")
    parser.add_argument('--resize', type=int, default=[224, 224], nargs=2, help="Resizing shape for images (HxW).")
    parser.add_argument('--test_method', type=str, default="hard_resize",
                        choices=["hard_resize", "single_query", "central_crop", "five_crops", "nearest_crop",
                                 "maj_voting"],
                        help="This includes pre/post-processing methods and prediction refinement")

    parser.add_argument("--efficient_ram_testing", action='store_true', help="_")
    parser.add_argument("--val_positive_dist_threshold", type=int, default=25, help="_")

    parser.add_argument('--recall_values', type=int, default=[1, 2, 3, 4, 5, 10, 15, 20, 25, 50, 100, 200], nargs="+",
                        help="Recalls to be computed, such as R@5.")

    # Paths parameters
    parser.add_argument("--datasets_folder", type=str, default=r'D:\Datasets', help="Path with all datasets")
    parser.add_argument("--dataset_name", type=str, default="pitts30k", help="Relative path of the dataset")
    parser.add_argument("--pca_dataset_folder", type=str, default=None,
                        help="Path with images to be used to compute PCA (ie: pitts30k/images/train")
    parser.add_argument("--save_dir", type=str, default="test_results",
                        help="Folder name of the current run (saved in ./logs/)")
    args = parser.parse_args()

    if args.datasets_folder == None:
        try:
            args.datasets_folder = os.environ['DATASETS_FOLDER']
        except KeyError:
            raise Exception("You should set the parameter --datasets_folder or export " +
                            "the DATASETS_FOLDER environment variable as such \n" +
                            "export DATASETS_FOLDER=../datasets_vg/datasets")

    return args

