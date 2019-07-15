from rl.agents.dqn import DQNAgent


def create_agent(network, processor, nb_actions, policy, memory, batch_size=32, nb_steps_warmup=32,
                 gamma=0.99, target_model_update=10000, train_interval=4, delta_clip=1.0, enable_double_dqn=False,
                 enable_dueling_network=False):

    return DQNAgent(model=network,
                    processor=processor,
                    nb_actions=nb_actions,
                    policy=policy,
                    memory=memory,
                    batch_size=batch_size,
                    enable_double_dqn=enable_double_dqn,
                    enable_dueling_network=enable_dueling_network,
                    nb_steps_warmup=nb_steps_warmup,
                    gamma=gamma,
                    target_model_update=target_model_update,
                    train_interval=train_interval,
                    delta_clip=delta_clip)
