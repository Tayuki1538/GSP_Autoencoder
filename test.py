import argparse
import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser

import numpy as np
import matplotlib.pyplot as plt

import json

def main(config):
    logger = config.get_logger('test')

    # setup data_loader instances
    data_loader = getattr(module_data, config['data_loader']['type'])(
        data_path=config['data_loader']['args']['data_path'],
        batch_size=512,
        shuffle=False,
        validation_split=0.2,
        num_workers=2,
        normalization=None
    )

    # build model architecture
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['loss'])
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    # model.load_state_dict(state_dict)
    checkpoint = torch.load(config.resume)
    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = checkpoint["state_dict"]
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)
    print("Missing keys:", missing_keys)
    print("Unexpected keys:", unexpected_keys)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns)).to(device)


    if config['data_loader']['args']['is_positioning']:
        total_loss = 0.0
        total_metrics = torch.zeros(len(metric_fns))
        pos_list = []
        error_list = []
        n_samples = 0

        with torch.no_grad():
            for i, (data, target) in enumerate(tqdm(data_loader)):
                data, target = data.to(device), target.to(device)
                output = model(data)
                # print(target)

                #
                # save sample images, or do something with output here
                #

                # if config["arch"]["args"]["figure"] == True:
                #     plt.plot(output[0].flatten().cpu().numpy(), label="output")
                #     plt.plot(target[0].flatten().cpu().numpy(), label="target")
                #     plt.legend()
                #     plt.savefig(f"/dbfs/mnt/mnt_wg3-1/mita/positioning/{str(config.resume).split('/')[-2]}/output_{i}.png")

                # computing loss, metrics on test set
                loss = loss_fn(output, target)
                # print(output, target)
                batch_size = data.shape[0]
                total_loss += loss.item() * batch_size
                for j, metric in enumerate(metric_fns):
                    total_metrics[j] += metric(output.to(device), target.to(device)).to("cpu") * batch_size
                for j in range(batch_size):
                    n_samples += 1
                    error_list.append(float(np.linalg.norm(output[j].to("cpu").numpy() - target[j].to("cpu").numpy())))
                    pos_list.append(target[j].to("cpu").numpy())

        pos_list = np.array(pos_list) 
        # n_samples = len(data_loader.sampler)
        log = {'loss': total_loss / n_samples}
        log.update({
            met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
        })
        logger.info(log)

        print(f"median error: {np.median(error_list)} m, 90% error: {np.percentile(error_list, 90)}")

        y = np.linspace(0, 100, len(error_list))
        plt.plot(sorted(error_list), y)
        plt.title("CDF of positioning error\nerror: {}".format(error_list[j]))
        plt.xlabel("Positioning error [m]")
        plt.ylabel("Percentile")
        plt.savefig(f"{config['trainer']['save_dir']}/error_cdf.png")
        plt.close()

        plt.plot(sorted(error_list), y)
        plt.title("CDF of positioning error\nerror: {}".format(error_list[j]))
        plt.xlabel("Positioning error [m]")
        plt.ylabel("Percentile")
        plt.savefig(f"{config['trainer']['save_dir']}/error_cdf.pdf")
        plt.close()

        json.dump(error_list, open(f"{config['trainer']['save_dir']}/error.json", "w"))

        plt.scatter(2.2, -1, c="black", marker="o", label="speaker")
        plt.scatter(pos_list[:,0], pos_list[:,1], c=error_list, marker="o", cmap='viridis')

        plt.colorbar(label="Error Value")
        plt.xlim(-0.2, 3.2)
        plt.ylim(4.2, -1.2)

        plt.title("Scatter plot of error values")
        plt.savefig(f"{config['trainer']['save_dir']}/error_scatter.png")
        plt.close()

        plt.scatter(2.2, -1, c="black", marker="o", label="speaker")
        plt.scatter(pos_list[:,0], pos_list[:,1], c=error_list, marker="o", cmap='viridis')

        plt.colorbar(label="Error Value")
        plt.xlim(-0.2, 3.2)
        plt.ylim(4.2, -1.2)

        plt.title("Scatter plot of error values")
        plt.savefig(f"{config['trainer']['save_dir']}/error_scatter.pdf")
        plt.close()
        print("Plot saved at {}".format(config['trainer']['save_dir']))

    else:
        with torch.no_grad():
            total_loss = 0.0
            total_metrics = torch.zeros(len(metric_fns))
            error_list = []
            n_samples = 0

            fig, ax = plt.subplots(2, 1)
            for i, (data, target) in enumerate(tqdm(data_loader)):
                data, target = data.to(device), target.to(device)
                output = model(data)
                print(output.shape, target.shape)

                #
                # save sample images, or do something with output here
                #

                # computing loss, metrics on test set
                loss = loss_fn(output, target)
                batch_size = data.shape[0]
                total_loss += loss.item() * batch_size
                for j, metric in enumerate(metric_fns):
                    total_metrics[j] += metric(output.to(device), target.to(device)) * batch_size

                if output.min() < 0 or output.max() > 1:
                    output = torch.sigmoid(output)
                pred = (output > 0.20).float()

                for w in range(batch_size):
                    # print("batch size: ", batch_size)
                    n_samples += 1
                    # print(output.shape, target.shape)
                    error_list.append(float(np.linalg.norm(output[w].to("cpu").numpy() - target[w].to("cpu").numpy())))

                if i<2:
                    ax[i].plot(target[i].cpu().numpy().squeeze())
                    ax[i].plot(output[i].cpu().numpy().squeeze())
            plt.savefig("{}/plot.png".format(config['trainer']['save_dir']))
            print("Plot saved at {}".format(config['trainer']['save_dir']))

            # sorted_list = np.argsort(-np.array(error_list))
            # print(sorted_list)
            # print("10%: ", sorted_list[int(len(sorted_list)/10)], "50%: ", sorted_list[int(len(sorted_list)/2)], "90%: ", sorted_list[int(len(sorted_list)*0.9)])

        n_samples = len(data_loader.sampler)
        log = {'loss': total_loss / n_samples}
        log.update({
            met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
        })
        logger.info(log)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)
