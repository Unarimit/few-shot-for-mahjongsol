"""
Reproduce Omniglot results of Snell et al Prototypical networks.
"""
from torch.optim import Adam
from torch.utils.data import DataLoader
import argparse

import config
from few_shot.datasets import OmniglotDataset, MiniImageNet
from few_shot.models import get_few_shot_encoder, FewDecoderPro
from few_shot.core import NShotTaskSampler, EvaluateFewShot, prepare_nshot_task
from few_shot.proto import proto_net_episode
from few_shot.train import fit
from few_shot.callbacks import *
from few_shot.utils import setup_dirs
from config import PATH, DEVICE


if __name__ == '__main__':
    setup_dirs()
    device = DEVICE

    ##############
    # Parameters #
    ##############
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='miniImageNet')
    parser.add_argument('--distance', default='l2')
    parser.add_argument('--n-train', default=2, type=int)  # support
    parser.add_argument('--n-test', default=1, type=int)
    parser.add_argument('--k-train', default=15, type=int)  # class
    parser.add_argument('--k-test', default=37, type=int)  # class
    parser.add_argument('--q-train', default=5, type=int)
    parser.add_argument('--q-test', default=1, type=int)
    args = parser.parse_args()

    evaluation_episodes = 1000
    episodes_per_epoch = 100

    if args.dataset == 'omniglot':
        n_epochs = 40
        dataset_class = OmniglotDataset
        num_input_channels = 3
        drop_lr_every = 20
    elif args.dataset == 'miniImageNet':
        n_epochs = 60
        dataset_class = MiniImageNet
        args.k_test = 37
        args.k_train = 15
        num_input_channels = 3
        drop_lr_every = 40
    elif args.dataset == '19wminiNet':
        n_epochs = 60
        dataset_class = MiniImageNet
        args.k_test = 9
        args.k_train = 9
        num_input_channels = 3
        drop_lr_every = 40
    else:
        raise(ValueError, 'Unsupported dataset')

    param_str = f'{args.dataset}_nt={args.n_train}_kt={args.k_train}_qt={args.q_train}_' \
                f'nv={args.n_test}_kv={args.k_test}_qv={args.q_test}'

    print(param_str)

    ###################
    # Create datasets #
    ###################
    background = dataset_class('background', name=args.dataset)
    background_taskloader = DataLoader(
        background,
        batch_sampler=NShotTaskSampler(background, episodes_per_epoch, args.n_train, args.k_train, args.q_train),
        num_workers=0
    )
    evaluation = dataset_class('evaluation', name=args.dataset)
    evaluation_taskloader = DataLoader(
        evaluation,
        batch_sampler=NShotTaskSampler(evaluation, episodes_per_epoch, args.n_test, args.k_test, args.q_test),
        num_workers=0
    )


    #########
    # Model #
    #########
    model = get_few_shot_encoder(num_input_channels)
    # model = FewDecoderPro(num_input_channels)
    model.to(device, dtype=torch.double)


    ############
    # Training #
    ############
    print(f'Training Prototypical network on {args.dataset}...')
    optimiser = Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.NLLLoss().cuda()


    def lr_schedule(epoch, lr):
        # Drop lr every 2000 episodes
        if epoch % drop_lr_every == 0:
            return lr / 2
        else:
            return lr


    callbacks = [
        EvaluateFewShot(
            eval_fn=proto_net_episode,
            num_tasks=evaluation_episodes,  # ?? not used
            n_shot=args.n_test,
            k_way=args.k_test,
            q_queries=args.q_test,
            taskloader=evaluation_taskloader,
            prepare_batch=prepare_nshot_task(args.n_test, args.k_test, args.q_test),
            distance=args.distance
        ),
        ModelCheckpoint(
            filepath=PATH + f'/models/proto_nets/{param_str}.pth',
            monitor=f'val_{args.n_test}-shot_{args.k_test}-way_acc'
        ),
        LearningRateScheduler(schedule=lr_schedule),
        CSVLogger(PATH + f'/logs/proto_nets/{param_str}.csv'),
    ]

    fit(
        model,
        optimiser,
        loss_fn,
        epochs=n_epochs,
        dataloader=background_taskloader,
        prepare_batch=prepare_nshot_task(args.n_train, args.k_train, args.q_train),
        callbacks=callbacks,
        metrics=['categorical_accuracy'],
        fit_function=proto_net_episode,
        fit_function_kwargs={'n_shot': args.n_train, 'k_way': args.k_train, 'q_queries': args.q_train, 'train': True,
                             'distance': args.distance},
    )
