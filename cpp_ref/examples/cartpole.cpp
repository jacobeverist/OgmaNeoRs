// CartPole RL example using AOgmaNeo Sparse Predictive Hierarchies
//
// Demonstrates how to use the Hierarchy with an Actor IO type to learn
// a control policy for the classic CartPole balancing task.

#include <aogmaneo/hierarchy.h>

#include <cstdio>
#include <cmath>

// --- CartPole environment ---

struct CartPole {
    float x;         // cart position
    float x_dot;     // cart velocity
    float theta;     // pole angle (radians)
    float theta_dot; // pole angular velocity

    static constexpr float gravity = 9.8f;
    static constexpr float cart_mass = 1.0f;
    static constexpr float pole_mass = 0.1f;
    static constexpr float total_mass = cart_mass + pole_mass;
    static constexpr float pole_half_length = 0.5f;
    static constexpr float force_mag = 10.0f;
    static constexpr float dt = 0.02f;

    static constexpr float x_limit = 2.4f;
    static constexpr float theta_limit = 12.0f * 3.14159f / 180.0f; // ~0.2094 rad
    static constexpr int max_steps = 500;

    void reset() {
        x = 0.0f;
        x_dot = 0.0f;
        theta = 0.0f;
        theta_dot = 0.0f;
    }

    // returns true if episode is still alive
    bool step(int action) {
        float force = (action == 1) ? force_mag : -force_mag;

        float cos_theta = std::cos(theta);
        float sin_theta = std::sin(theta);

        float temp = (force + pole_mass * pole_half_length * theta_dot * theta_dot * sin_theta) / total_mass;
        float theta_acc = (gravity * sin_theta - cos_theta * temp) /
            (pole_half_length * (4.0f / 3.0f - pole_mass * cos_theta * cos_theta / total_mass));
        float x_acc = temp - pole_mass * pole_half_length * theta_acc * cos_theta / total_mass;

        // Euler integration
        x += x_dot * dt;
        x_dot += x_acc * dt;
        theta += theta_dot * dt;
        theta_dot += theta_acc * dt;

        return std::fabs(x) <= x_limit && std::fabs(theta) <= theta_limit;
    }
};

// --- State encoding ---

// Discretize a continuous value into a bin index in [0, num_bins-1]
int encode_bin(float value, float low, float high, int num_bins) {
    float normalized = (value - low) / (high - low); // [0, 1]

    if (normalized < 0.0f) normalized = 0.0f;
    if (normalized > 1.0f) normalized = 1.0f;

    int bin = static_cast<int>(normalized * (num_bins - 1) + 0.5f);

    return bin;
}

int main() {
    using namespace aon;

    const int num_bins = 16;
    const int num_actions = 2;
    const int num_episodes = 500;

    // --- Set up hierarchy ---

    // IO layer 0: observation (4 columns, 16 bins each)
    // IO layer 1: action (1 column, 2 possible actions)
    Array<Hierarchy::IO_Desc> io_descs(2);

    io_descs[0] = Hierarchy::IO_Desc(
        Int3(2, 2, num_bins),  // 2x2 = 4 columns for 4 state variables
        none,                  // input-only, no prediction needed
        4,                     // num_dendrites_per_cell
        2,                     // up_radius
        2                      // down_radius
    );

    io_descs[1] = Hierarchy::IO_Desc(
        Int3(1, 1, num_actions), // 1 column, 2 possible actions
        action,                  // RL action output
        4,                       // num_dendrites_per_cell (policy)
        2,                       // up_radius
        2,                       // down_radius
        64,                      // value_size
        4,                       // value_num_dendrites_per_cell
        256                      // history_capacity
    );

    // Two hidden layers
    Array<Hierarchy::Layer_Desc> layer_descs(2);

    layer_descs[0] = Hierarchy::Layer_Desc(
        Int3(4, 4, num_bins), // hidden size
        4,                    // num_dendrites_per_cell
        2,                    // up_radius
        2,                    // recurrent_radius (enable recurrence)
        2                     // down_radius
    );

    layer_descs[1] = Hierarchy::Layer_Desc(
        Int3(4, 4, num_bins), // hidden size
        4,                    // num_dendrites_per_cell
        2,                    // up_radius
        2,                    // recurrent_radius
        2                     // down_radius
    );

    Hierarchy hierarchy(io_descs, layer_descs);

    // Tune RL parameters
    hierarchy.params.ios[1].actor.discount = 0.99f;
    hierarchy.params.ios[1].actor.plr = 0.01f;
    hierarchy.params.ios[1].actor.vlr = 0.1f;

    // --- Training loop ---

    CartPole env;

    // Encoding ranges for each state variable
    const float ranges[][2] = {
        { -CartPole::x_limit,     CartPole::x_limit },     // x
        { -3.0f,                  3.0f },                   // x_dot
        { -CartPole::theta_limit, CartPole::theta_limit },  // theta
        { -3.0f,                  3.0f }                    // theta_dot
    };

    // Buffers
    Int_Buffer obs_cis(4); // 2*2 = 4 columns
    Int_Buffer act_cis(1); // 1*1 = 1 column

    printf("Training CartPole with AOgmaNeo SPH (%d episodes)...\n\n", num_episodes);
    printf("Episode | Steps | Avg(last 10)\n");
    printf("--------|-------|-------------\n");

    float avg_window[10] = {};
    int avg_idx = 0;

    for (int ep = 0; ep < num_episodes; ep++) {
        env.reset();
        hierarchy.clear_state();

        int steps = 0;

        for (int t = 0; t < CartPole::max_steps; t++) {
            // Encode observation into sparse column indices
            float state[4] = { env.x, env.x_dot, env.theta, env.theta_dot };

            for (int i = 0; i < 4; i++)
                obs_cis[i] = encode_bin(state[i], ranges[i][0], ranges[i][1], num_bins);

            // Get the action from previous step's prediction
            act_cis[0] = hierarchy.get_prediction_cis(1)[0];

            // Prepare inputs
            Array<Int_Buffer_View> inputs(2);
            inputs[0] = obs_cis;
            inputs[1] = act_cis;

            // Reward is 1.0 per step alive
            float reward = 1.0f;

            // Step the hierarchy
            hierarchy.step(inputs, true, reward, 0.0f);

            // Get the chosen action
            int chosen_action = hierarchy.get_prediction_cis(1)[0];

            // Step the environment
            bool alive = env.step(chosen_action);
            steps++;

            if (!alive)
                break;
        }

        // Track rolling average
        avg_window[avg_idx % 10] = static_cast<float>(steps);
        avg_idx++;

        int count = (avg_idx < 10) ? avg_idx : 10;
        float avg = 0.0f;
        for (int i = 0; i < count; i++)
            avg += avg_window[i];
        avg /= count;

        if ((ep + 1) % 10 == 0 || ep == 0)
            printf("%7d | %5d | %6.1f\n", ep + 1, steps, avg);
    }

    printf("\nDone.\n");

    return 0;
}
