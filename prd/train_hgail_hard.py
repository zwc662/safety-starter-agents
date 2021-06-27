import argparse
import time
import datetime
import torch
import torch_ac
from torch_ac.utils import DictList
from shutil import copyfile

import tensorboardX
import sys
import os
import pickle

sys.path.append("{}".format(os.path.dirname(os.path.dirname(__file__))))

import utils
from model import ACModel, RewModel, DiscModel
from hgail_hard import HGAILAlgo
from progrew import RewardFn, Rewarder, RewarderWithMemory
from samplers import GenerativeSampler

# Parse arguments

def main():
    
    parser = argparse.ArgumentParser()
    
    ## General parameters
    parser.add_argument("--algo", required=True,
                        help="algorithm to use: a2c | ppo (REQUIRED)")
    parser.add_argument("--env", required=True,
                        help="name of the environment to train on (REQUIRED)")
    parser.add_argument("--model", default=None,
                        help="name of the model (default: {ENV}_{ALGO}_{TIME})")
    parser.add_argument("--seed", type=int, default=1,
                        help="random seed (default: 1)")
    parser.add_argument("--log-interval", type=int, default=1,
                        help="number of updates between two logs (default: 1)")
    parser.add_argument("--save-interval", type=int, default=10,
                        help="number of updates between two saves (default: 10, 0 means no saving)")
    parser.add_argument("--procs", type=int, default=1,
                        help="number of processes (default: 1)")
    parser.add_argument("--frames", type=int, default=10**8,
                        help="number of frames of training (default: 1e7)")

    ## Parameters for main algorithm
    parser.add_argument("--epochs", type=int, default=4,
                        help="number of epochs for PPO (default: 4)")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="batch size for PPO (default: 256)")
    parser.add_argument("--frames-per-proc", type=int, default=256, #None,
                        help="number of frames per process before update (default: 5 for A2C and 128 for PPO)")
    parser.add_argument("--discount", type=float, default=0.99,
                        help="discount factor (default: 0.99)")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate (default: 0.001)")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
                        help="lambda coefficient in GAE formula (default: 0.95, 1 means no gae)")
    parser.add_argument("--entropy-coef", type=float, default=0.01,
                        help="entropy term coefficient (default: 0.01)")
    parser.add_argument("--value-loss-coef", type=float, default=0.5,
                        help="value loss term coefficient (default: 0.5)")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
                        help="maximum norm of gradient (default: 0.5)")
    parser.add_argument("--optim-eps", type=float, default=1e-8,
                        help="Adam and RMSprop optimizer epsilon (default: 1e-8)")
    parser.add_argument("--optim-alpha", type=float, default=0.99,
                        help="RMSprop optimizer alpha (default: 0.99)")
    parser.add_argument("--clip-eps", type=float, default=0.2,
                        help="clipping epsilon for PPO (default: 0.2)")
    parser.add_argument("--ac-recurrence", type=int, default=4,
                        help="number of time-steps gradient is backpropagated (default: 1). If > 1, a LSTM is added to the model to have memory.")
    parser.add_argument("--ac-text", action="store_true", default=False,
                        help="add a GRU to the model to handle text input")
    parser.add_argument("--dc-recurrence", type=int, default=4,
                        help="number of time-steps gradient is backpropagated (default: 1). If > 1, a LSTM is added to the model to have memory.")
    parser.add_argument("--dc-memory", action="store_true", default=False,
                        help="add a GRU to the model to handle text input")
    parser.add_argument("--dc-text", action="store_true", default=False,
                        help="add a GRU to the model to handle text input")
    parser.add_argument("--dc-batch-size", type=int, default=256,
                        help="number of dsicriminator batch size.")
    parser.add_argument("--expert", type=str, default=None, 
                        help="only the name of the expert model, not the abspath")


    
    args = parser.parse_args()

    args.ac_mem = args.ac_recurrence > 1
    args.dc_mem = args.dc_recurrence > 1
    # Set run dir

    timestamp = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
    default_model_name = f"{args.env}_{args.algo}_ac_recurrence{args.ac_recurrence}_hgail_hard_dc_recurrence{args.dc_recurrence}_seed{args.seed}_{timestamp}"

    model_name = args.model or default_model_name
    model_dir = utils.get_model_dir(model_name)
    
    # Load loggers and Tensorboard writer

    txt_logger = utils.get_txt_logger(model_dir)
    csv_file, csv_logger = utils.get_csv_logger(model_dir)
    tb_writer = tensorboardX.SummaryWriter(model_dir)
    copyfile(os.path.join(os.path.dirname(__file__), "hgail_hard.py"), os.path.join(model_dir, "hgail_hard.py"))
    copyfile(os.path.join(os.path.dirname(__file__), "samplers.py"), os.path.join(model_dir, "samplers.py"))
    copyfile(os.path.join(os.path.dirname(__file__), "progrew.py"), os.path.join(model_dir, "progrew.py"))
    # Log command and all script arguments

    txt_logger.info("{}\n".format(" ".join(sys.argv)))
    txt_logger.info("{}\n".format(args))

    # Set seed for all randomness sources

    utils.seed(args.seed)

    # Set device

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    txt_logger.info(f"Device: {device}\n")
    print(device)

    # Load environments

    envs = []
    for i in range(args.procs):
        envs.append(utils.make_env(args.env, args.seed + 10000 * i))
    

    txt_logger.info("Environments loaded\n")

    # Load training status

    try:
        status = utils.get_status(model_dir)
    except OSError:
        status = {"num_frames": 0, "update": 0}
    txt_logger.info("Training status loaded\n")

    # Load observations preprocessor

    obs_space, preprocess_obss = utils.get_obss_preprocessor(envs[0].observation_space)
    if "vocab" in status:
        preprocess_obss.vocab.load_vocab(status["vocab"])
    txt_logger.info("Observations preprocessor loaded")

    # Load model
    acmodel = ACModel(obs_space, envs[0].action_space, args.ac_mem, args.ac_text)
    if "model_state" in status:
        acmodel.load_state_dict(status["model_state"])
    acmodel.to(device)
    txt_logger.info("Policy Model loaded\n")
    txt_logger.info("{}\n".format(acmodel))



        
    # Load Expert demonstrations
    if args.expert is None:
        args.expert = "MiniGrid-DoorKey-8x8-v0_ppo_recurrence8_seed1_reshape_True_21-03-30-19-19-31"

    demo_dir = os.path.join(utils.get_model_dir(args.expert), "demo.pt")
    with open(demo_dir, "rb") as demo_file:
        demos_list = pickle.load(demo_file)
    
    # Discriminator
    #dcmodel = RewModel(obs_space, args.dc_mem, args.dc_text)
    dcmodel = DiscModel(obs_space, envs[0].action_space)
    dcmodel.to(device)
    txt_logger.info("Discriminator Model loaded\n")
    txt_logger.info("{}\n".format(dcmodel))

    # Program Sampler
    rewardfn = RewardFn()
    sampler = GenerativeSampler("sampler", 20, rewardfn.num_holes, True, 64).to(device)
    txt_logger.info("Sampler loaded\n")
    txt_logger.info("{}\n".format(sampler))
    
    # Static reward function 
    reshape_reward = Rewarder(rewardfn, sampler, device)
    #reshape_reward = RewarderWithMemory(rewardfn, sampler, dcmodel, device, preprocess_obss)

    # Synthesizer
    dc_algo = HGAILAlgo(rewardfn, sampler, acmodel, dcmodel, device, batch_size = args.dc_batch_size, recurrence = args.dc_recurrence, preprocess_obss = preprocess_obss)
    demos = dc_algo.preprocess_demonstrations(demos_list)
    

    # Load algo
    if args.algo == "a2c":
        algo = torch_ac.A2CAlgo(envs, acmodel, device, args.frames_per_proc, args.discount, args.lr, args.gae_lambda,
                                args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.ac_recurrence,
                                args.optim_alpha, args.optim_eps, preprocess_obss, reshape_reward)
    elif args.algo == "ppo":
        algo = torch_ac.PPOAlgo(envs, acmodel, device, args.frames_per_proc, args.discount, args.lr, args.gae_lambda,
                                args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.ac_recurrence,
                                args.optim_eps, args.clip_eps, args.epochs, args.batch_size, preprocess_obss, reshape_reward)
    else:
        raise ValueError("Incorrect algorithm name: {}".format(args.algo))
        
    

    if "optimizer_state" in status:
        algo.optimizer.load_state_dict(status["optimizer_state"])
    txt_logger.info("Optimizer loaded\n")

    # Train model
    best_status = status.copy()
    best_data = []
    num_frames = status["num_frames"]
    update = status["update"]
    start_time = time.time()
    
    exps_state_prev = None
    exps_state_post = None
    exps_memory_prev = None
    while num_frames < args.frames:
        # Update model parameters

        update_start_time = time.time()
        exps, logs1 = algo.collect_experiences()
        exps_state_post = reshape_reward.suspend()

        logs2 = algo.update_parameters(exps)
        logs = {**logs1, **logs2}

        demos, demos_output, demos_prog_rews, demos_dc_rews, exps, exps_output, exps_prog_rews, exps_dc_rews, exps_prog_means = dc_algo.update_samples(demos, exps, exps_state_prev, exps_memory_prev)
        logs3 = dc_algo.update_sampler(demos_output, demos_prog_rews, demos_dc_rews, exps_output, exps_prog_rews, exps_dc_rews, exps_prog_means)
        logs4 = dc_algo.update_discriminator(demos, demos_prog_rews, demos_output, exps, exps_prog_rews, exps_output)
        exps_memory_prev = exps.memory[-1].detach().flatten()[:]
        exps_state_prev = exps_state_post.copy()
        reshape_reward.restore(exps_state_post)
        
        update_end_time = time.time()

        num_frames += logs["num_frames"]
        update += 1

        # Print logs

        if update % args.log_interval == 0:
            fps = logs["num_frames"]/(update_end_time - update_start_time)
            duration = int(time.time() - start_time)
            return_per_episode = utils.synthesize(logs["return_per_episode"])
            rreturn_per_episode = utils.synthesize(logs["reshaped_return_per_episode"])
            num_frames_per_episode = utils.synthesize(logs["num_frames_per_episode"])

            header = ["update", "frames", "FPS", "duration"]
            data = [update, num_frames, fps, duration]
            header += ["rreturn_" + key for key in rreturn_per_episode.keys()]
            data += rreturn_per_episode.values()
            header += ["sampler_" + key for key in logs3.keys()]
            data += [val for val in logs3.values()]
            header += ["dc_" + key for key in logs4.keys()]
            data += [val for val in logs4.values()]
            

            header += ["num_frames_" + key for key in num_frames_per_episode.keys()]
            data += num_frames_per_episode.values()
            header += ["entropy", "value", "policy_loss", "value_loss", "grad_norm"]
            data += [logs["entropy"], logs["value"], logs["policy_loss"], logs["value_loss"], logs["grad_norm"]]

            header += ["return_" + key for key in return_per_episode.keys()]
            data += return_per_episode.values()
            txt_logger.info(
                    "U {} | F {:06} | FPS {:04.0f} | D {} | rR:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | GenLs:  {:.2f} {:.2f} {:.2f} | DLs: {:.2f} {:.2f} {:.2f} {:2f}| F:μσmM {:.1f} {:.1f} {} {} | H {:.3f} | V {:.3f} | pL {:.3f} | vL {:.3f} | ∇ {:.3f} | R:μσmM {:.2f} {:.2f} {:.2f} {:.2f}"
                .format(*data))
            if update % (args.log_interval * args.log_interval) == 0:
                csv_logger.writerow(["holes"] + rewardfn.holes)
            if status["num_frames"] == 0:
                csv_logger.writerow(header)
                best_data = [i for i in data]
            csv_logger.writerow(data)
            csv_file.flush()

            for field, value in zip(header, data):
                tb_writer.add_scalar(field, value, num_frames)

        
        # Save status
        if args.save_interval > 0 and update % args.save_interval == 0:
            status = {"num_frames": num_frames, "update": update,
                    "model_state": acmodel.state_dict(), "policy_optimizer_state": algo.optimizer.state_dict(), 
                    "sampler_state": sampler.state_dict(), "sampler_optimizer_state": dc_algo.gen_optimizer.state_dict(),
                    "dc_state": dcmodel.state_dict(), "dc_optimizer_state":dc_algo.dc_optimizer.state_dict()}
            if hasattr(preprocess_obss, "vocab"):
                status["vocab"] = preprocess_obss.vocab.vocab
            utils.save_status(status, model_dir) 
            txt_logger.info("Status saved at {}".format(model_dir))
            

        # Save or load from the best status 
        if data[-4] > best_data[-4]:
            best_status = status.copy()
            best_data = [i for i in data]
            utils.save_status(best_status, model_dir, best = True)

            txt_logger.info("Best Status saved at {}".format(model_dir))
        elif False and num_frames > best_status["num_frames"] + args.frames / 10:
            if "model_state" in status:
                acmodel.load_state_dict(best_status["model_state"])
            txt_logger.info("Restart training from the best status\n")

    if len(best_data) > 0:
        csv_logger.writerow(["Final" for i in best_data])
        csv_logger.writerow(best_data)
        csv_file.flush()


if __name__ == '__main__':
    sys.argv = sys.argv + ['--algo', 'ppo', '--env', 'MiniGrid-DoorKey-8x8-v0', '--frames-per-proc', '512', '--ac-recurrence', '8', '--dc-recurrence', '8', '--save-interval', '10', '--frames', '500000', '--epochs', '4']
    main()
