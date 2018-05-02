
import tensorflow as tf
import cv2
import pong_game as game
import random
import numpy as np
from collections import deque

# Game name.
GAME = 'Pong'

# Number of valid actions.
ACTIONS = 3

# Size of minibatch.
BATCH = 32

# Learning Rate.
Lr = 1e-6

def affine_forward(x, w, b):
    out = tf.matmul(x, w) + b
    cache = (x, w, b)
    return out, cache

def affine_backward(dout, cache):
    x, w, b = cache
    dx, dw, db = None, None, None
    dx = tf.matmul(dout, w.T)
    dw = tf.matmul(x.T, dout)
    db = tf.matmul(dout.T, np.ones(N))
    return dx, dw, db

def relu_forward(z):
    out = tf.nn.relu(z)
    cache = (z)
    return out, cache

def relu_backward(dout, cache):
    dx, x = None, cache
    dx = tf.identity(dout)
    dx[x <= 0] = 0
    return dx

def weight_variable(shape):
    """ Initialize the weight variable."""
    initial = tf.truncated_normal(shape, stddev=1.)
    return tf.Variable(initial)

def bias_variable(shape):
    """ Initialize the bias variable."""
    initial = tf.constant(0., shape=shape)
    return tf.Variable(initial)

def createNetwork(x, w, b, y, test):
    """ Create a 3-layer network for estimating the Q value.
    Args: x, w, b, y, test
    Returns:
        classifications/loss
    """
    # Hidden layers.
    w_1, w_2, w_3 = w
    b_1, b_2, b_3 = b
    z_1, acache1 = affine_forward(x, w_1, b_1)
    a_1, rcache1 = relu_forward(z_1)
    z_2, acache2 = affine_forward(a_1, w_2, b_2)
    a_2, rcache2 = relu_forward(z_2)
    f, acache3 = affine_forward(a_2, w_3, b_3)
    if test == True:
        classifications = tf.nn.softmax_cross_entropy_with_logits(logits = tf.log(f))
        return tf.argmax(classifications)
    loss, df = tf.losses.softmax_cross_entropy(F, tf.log(y))
    da_2, dw_3, db_3 = affine_backward(df, acache3)
    dz_2 = relu_backward(da_2, rcache2)
    da_1, dw_2, db_2 = affine_backward(dz_2, acache2)
    dz_1 = relu_backward(da_1, rcache1)
    dx, dw_1, db_1 = affine_backward(dz_1, acache1)
    #update parameters
    x = x - dx
    w_1 = w_1 - dw_1
    w_2 = w_2 - dw_2
    w_3 = w_3 - dw_3
    b_1 = b_1 - db_1
    b_2 = b_2 - db_2
    b_3 = b_3 - db_3
    return loss

def minibatchGD(data, epoch):
    # Initialize the network weights and biases.
    w_1 = weight_variable([data[0].shape[1], data[0].shape[0]])
    b_1 = bias_variable(w_1.shape[1])

    w_2 = weight_variable([data[0].shape[1], data[0].shape[0]/2])
    b_2 = bias_variable(w_2.shape[1])

    w_3 = weight_variable([data[0].shape[1], 1])
    b_3 = bias_variable(w_3.shape[1])
    for e in range(epoch):
        data = tf.random_shuffle(data)
        for i in range(data[0].shape[0]):
            x, y = data[0], data[1]
            loss = createNetwork(x, [w_1, w_2, w_3],[b_1,b_2,b_3], y, test)
            return loss

def get_action_index(readout_t, epsilon, t):
    """ Choose an action epsilon-greedily.
    Details:
        choose an action randomly:
        (1) in the observation phase (t<OBSERVE).
        (2) beyond the observation phase with probability "epsilon".
        otherwise, choose the action with the highest Q-value.
    Args:
        readout_t: a vector with the Q-value associated with every action.
        epsilon: tempreture variable for exploration-exploitation.
        t: current number of iterations.
    Returns:
        index: the index of the action to be taken next.
    """

    action_index = np.random.choice([0,1,2],[1./3,1./3,1./3])

    return action_index


def trainNetwork(s, readout, sess):
    """ Train the artificial agent using Q-learning to play the pong game.
    Args:
        s: the current state formed by 4 frames of the playground.
        readout: the Q value for each passible action in the current state.
        sess: session
    """


    # Training operation.
    train_step = tf.train.AdamOptimizer(Lr).minimize(cost)

    # Open up a game state to communicate with emulator.
    game_state = game.GameState()

    # Initialize the replay memory.
    D = deque()

    # Initialize the action vector.
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1

    # Initialize the state of the game.
    s_t = np.array([0.5, 0.5, 0.03, 0.01, 0.5-paddle_height/2])

    # Save and load model checkpoints.
    saver = tf.train.Saver()
    sess.run(tf.initialize_all_variables())
    checkpoint = tf.train.get_checkpoint_state("saved_networks_q_learning")
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
    else:
        print("Could not find old network weights")

    # Initialize the epsilon value for the exploration phase.
    epsilon = INITIAL_EPSILON

    # Initialize the iteration counter.
    t = 0

    while True:
        # Choose an action epsilon-greedily.
        readout_t = readout.eval(feed_dict={s: [s_t]})[0]

        action_index = get_action_index(readout_t, epsilon, t)

        a_t = np.zeros([ACTIONS])

        a_t[action_index] = 1

        # Scale down epsilon during the exploitation phase.
        epsilon = scale_down_epsilon(epsilon, t)

        # Run the selected action and update the replay memeory
        for i in range(0, K):
            # Run the selected action and observe next state and reward.
            s_t1, r_t, terminal = run_selected_action(a_t, s_t, game_state)

            # Store the transition in the replay memory D.
            D.append((s_t, a_t, r_t, s_t1, terminal))
            if len(D) > REPLAY_MEMORY:
                D.popleft()

        # Start training once the observation phase is over.
        if (t > OBSERVE):

            # Sample a minibatch to train on.
            minibatch = random.sample(D, BATCH)

            # Get the batch variables.
            s_j_batch = [d[0] for d in minibatch]
            a_batch = [d[1] for d in minibatch]
            r_batch = [d[2] for d in minibatch]
            s_j1_batch = [d[3] for d in minibatch]
            terminal_batch = [d[4] for d in minibatch]

            # Compute the target Q-Value
            readout_j1_batch = readout.eval(feed_dict={s: s_j1_batch})
            target_q_batch = compute_target_q(r_batch, readout_j1_batch, terminal_batch)

            # Perform gradient step.
            train_step.run(feed_dict={
                y: target_q_batch,
                a: a_batch,
                s: s_j_batch})

        # Update the state.
        s_t = s_t1

        # Update the number of iterations.
        t += 1

        # Save a checkpoint every 10000 iterations.
        if t % 10000 == 0:
            saver.save(sess, 'saved_networks_q_learning/' + GAME + '-dqn', global_step=t)

        # Print info.
        state = ""
        if t <= OBSERVE:
            state = "observe"
        elif t > OBSERVE and t <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"
        print("TIMESTEP", t, "/ STATE", state, "/ EPSILON", epsilon, "/ ACTION", action_index, "/ REWARD", r_t, "/ Q_MAX %e" % np.max(readout_t))


def playGame():
    """Paly the pong game"""

    # Start an active session.
    sess = tf.InteractiveSession()
    myfile = open("expert_policy.txt")
    data = []
    for line in myfile:
        row = []
        for entry in line:
            row.append(entry)
        data.append(list(row))
    # Create the network.
    loss = minibatchGD(data, epoch)

def main():
    """ Main function """
    playGame()


if __name__ == "__main__":
    main()
