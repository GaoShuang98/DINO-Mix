import torch
from os.path import join
from datetime import datetime

from eval_files import commons, datasets_ws, parser_set_eval, test


def main(model):

    ######################################### SETUP #########################################
    args = parser_set_eval.parse_arguments()
    start_time = datetime.now()
    args.save_dir = join("test_logs", args.save_dir, start_time.strftime('%Y-%m-%d_%H-%M-%S'))
    logging = commons.setup_logging(args.save_dir)
    commons.make_deterministic(args.seed)
    logging.info(f"Arguments: {args}")
    logging.info(f"The outputs are being saved in {args.save_dir}")

    ######################################### MODEL #########################################
    print(model)

    args.features_dim = 4096
    model = torch.nn.DataParallel(model)

    ######################################### DATASETS0 #########################################
    test_ds = datasets_ws.BaseDataset(args, args.datasets_folder, 'pitts30k', "test")
    logging.info(f"Test set: {test_ds}")

    ######################################### TEST on TEST SET #########################################
    recalls, recalls_str, flops, inference_time = test.test(args, test_ds, model, args.test_method, flops_times=True)
    logging.info(f"Recalls on {test_ds}: {recalls_str}")
    logging.info(f"FLOPs: {flops}, Inference Time: {inference_time}")


    ######################################### DATASETS1 #########################################
    test_ds1 = datasets_ws.BaseDataset(args, args.datasets_folder, 'pitts250k', "test")
    logging.info(f"Test set: {test_ds1}")

    ######################################### TEST on TEST SET #########################################
    recalls, recalls_str, flops, inference_time = test.test(args, test_ds1, model, args.test_method, flops_times=True)
    logging.info(f"Recalls on {test_ds1}: {recalls_str}")
    logging.info(f"FLOPs: {flops}, Inference Time: {inference_time}")


    ######################################### DATASETS2 #########################################
    test_ds2 = datasets_ws.BaseDataset(args, args.datasets_folder, 'sf-xl-val', "test")
    logging.info(f"Test set: {test_ds2}")

    ######################################### TEST on TEST SET #########################################
    recalls, recalls_str, flops, inference_time = test.test(args, test_ds2, model, args.test_method, flops_times=True)
    logging.info(f"Recalls on {test_ds2}: {recalls_str}")
    logging.info(f"FLOPs: {flops}, Inference Time: {inference_time}")

    ######################################### DATASETS3 #########################################
    test_ds3 = datasets_ws.BaseDataset(args, args.datasets_folder, 'tokyo247', "test")
    logging.info(f"Test set: {test_ds3}")

    ######################################### TEST on TEST SET #########################################
    recalls, recalls_str, flops, inference_time = test.test(args, test_ds3, model, args.test_method, flops_times=True)
    logging.info(f"Recalls on {test_ds3}: {recalls_str}")
    logging.info(f"FLOPs: {flops}, Inference Time: {inference_time}")

    ######################################### DATASETS4 #########################################
    test_ds4 = datasets_ws.BaseDataset(args, args.datasets_folder, 'nordland', "test")
    logging.info(f"Test set: {test_ds4}")

    ######################################### TEST on TEST SET #########################################
    recalls, recalls_str, flops, inference_time = test.test(args, test_ds4, model, args.test_method, flops_times=True)
    logging.info(f"Recalls on {test_ds4}: {recalls_str}")
    logging.info(f"FLOPs: {flops}, Inference Time: {inference_time}")

    ######################################### DATASETS5 #########################################
    test_ds5 = datasets_ws.BaseDataset(args, args.datasets_folder, 'sf-xl-testv1', "test")
    logging.info(f"Test set: {test_ds5}")

    ######################################### TEST on TEST SET #########################################
    recalls, recalls_str, flops, inference_time = test.test(args, test_ds5, model, args.test_method, flops_times=True)
    logging.info(f"Recalls on {test_ds5}: {recalls_str}")
    logging.info(f"FLOPs: {flops}, Inference Time: {inference_time}")

    logging.info(f"Finished in {str(datetime.now() - start_time)[:-7]}")

if __name__ == '__main__':
    model = torch.hub.load('GaoShuang98/DINO-Mix', 'dino_mix', pretrained=True)
    main(model)
