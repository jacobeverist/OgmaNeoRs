// CartPole gymnasium example using AOgmaNeo Sparse Predictive Hierarchies (Rust port)
//
// Port of python_ref/cartpole_manual.py and python_ref/cartpole_env_runner.py
//
// Demonstrates SPH actor-critic RL on CartPole-v1 using Python's gymnasium
// library via PyO3.  Requires Python with `pip install gymnasium` installed.
//
// Run with:
//   cargo run --release --example cartpole_env_runner --features gymnasium-examples

use aogmaneo::helpers::Int3;
use aogmaneo::hierarchy::{Hierarchy, IoDesc, IoType, LayerDesc};
use pyo3::prelude::*;

/// Sigmoid squashing function: maps any real to (0, 1).
fn sigmoid(x: f32) -> f32 {
    (x * 0.5).tanh() * 0.5 + 0.5
}

/// Squash an observation value and bin it into [0, num_bins-1].
fn encode(val: f32, num_bins: i32) -> i32 {
    (sigmoid(val * 3.0) * (num_bins - 1) as f32 + 0.5) as i32
}

fn main() -> PyResult<()> {
    let input_resolution: i32 = 32;
    let num_actions: i32 = 2;
    let num_episodes = 1000;
    let max_steps = 500;

    // --- Hierarchy ---

    // IO 0: 4 obs floats → (2, 2, 32) grid, input-only
    // IO 1: action → (1, 1, 2), RL actor
    let io_descs = vec![
        IoDesc {
            size: Int3::new(2, 2, input_resolution),
            io_type: IoType::None,
            num_dendrites_per_cell: 4,
            up_radius: 2,
            down_radius: 2,
            value_size: 64,
            value_num_dendrites_per_cell: 4,
            history_capacity: 256,
        },
        IoDesc {
            size: Int3::new(1, 1, num_actions),
            io_type: IoType::Action,
            num_dendrites_per_cell: 4,
            up_radius: 2,
            down_radius: 2,
            value_size: 64,
            value_num_dendrites_per_cell: 4,
            history_capacity: 256,
        },
    ];

    // Two layers with exponential temporal memory
    let layer_descs = vec![
        LayerDesc {
            hidden_size: Int3::new(5, 5, 32),
            num_dendrites_per_cell: 4,
            up_radius: 2,
            recurrent_radius: -1,
            down_radius: 2,
            ticks_per_update: 1,
        },
        LayerDesc {
            hidden_size: Int3::new(5, 5, 32),
            num_dendrites_per_cell: 4,
            up_radius: 2,
            recurrent_radius: -1,
            down_radius: 2,
            ticks_per_update: 2,
        },
    ];

    let mut h = Hierarchy::new();
    h.init_random(&io_descs, &layer_descs);

    // Buffers for column indices
    let mut obs_cis = vec![0i32; 4]; // 2×2 = 4 columns
    let mut act_cis = vec![0i32; 1]; // 1×1 = 1 column

    // 10-episode rolling average tracker
    let mut window = [0.0f32; 10];
    let mut window_idx = 0usize;

    println!("CartPole-v1 with AOgmaNeo SPH ({num_episodes} episodes)");
    println!("Requires: pip install gymnasium");
    println!("{}", "-".repeat(50));

    Python::with_gil(|py| -> PyResult<()> {
        let gym = py.import_bound("gymnasium")?;
        let env = gym.call_method1("make", ("CartPole-v1",))?;

        for episode in 0..num_episodes {
            // Reset environment
            let reset_result = env.call_method0("reset")?;
            let (obs_arr, _): (Vec<f64>, Py<PyAny>) = reset_result.extract()?;

            let mut obs: Vec<f64> = obs_arr;
            let mut reward = 0.0f64;
            let mut steps = 0usize;

            for _ in 0..max_steps {
                // Encode 4 observation floats into 4 columns
                for i in 0..4 {
                    obs_cis[i] = encode(obs[i] as f32, input_resolution);
                }

                // Feed back the previous action prediction
                act_cis[0] = h.get_prediction_cis(1).first().copied().unwrap_or(0);

                h.step(&[&obs_cis, &act_cis], true, reward as f32, 0.0);

                // Read chosen action
                let action = h.get_prediction_cis(1)[0];

                // Step the gymnasium environment
                let step_result = env.call_method1("step", (action,))?;
                let (next_obs, _r, term, trunc, _): (Vec<f64>, f64, bool, bool, Py<PyAny>) =
                    step_result.extract()?;

                obs = next_obs;
                steps += 1;

                // Penalty on termination, 0.0 otherwise (matches cartpole_manual.py)
                reward = if term { -10.0 } else { 0.0 };

                if term || trunc {
                    break;
                }
            }

            // Rolling 10-episode average
            window[window_idx % 10] = steps as f32;
            window_idx += 1;
            let count = window_idx.min(10);
            let avg: f32 = window[..count].iter().sum::<f32>() / count as f32;

            println!("Episode {:>4} | steps: {:>3} | avg(10): {:>6.1}", episode + 1, steps, avg);
        }

        env.call_method0("close")?;
        Ok(())
    })?;

    println!("Done.");
    Ok(())
}
