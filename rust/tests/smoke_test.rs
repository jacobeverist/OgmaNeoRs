use aogmaneo::helpers::Int3;
use aogmaneo::hierarchy::{Hierarchy, IoDesc, IoType, LayerDesc};

fn make_random_cis(size: Int3) -> Vec<i32> {
    let cols = (size.x * size.y) as usize;
    let col_size = size.z as usize;
    (0..cols).map(|i| (i % col_size) as i32).collect()
}

#[test]
fn test_hierarchy_create_and_step() {
    // Minimal 1-IO, 1-layer hierarchy
    let io_descs = vec![IoDesc {
        size: Int3::new(4, 4, 16),
        io_type: IoType::Prediction,
        num_dendrites_per_cell: 2,
        up_radius: 2,
        down_radius: 2,
        value_size: 64,
        value_num_dendrites_per_cell: 2,
        history_capacity: 64,
    }];

    let layer_descs = vec![LayerDesc {
        hidden_size: Int3::new(4, 4, 16),
        num_dendrites_per_cell: 2,
        up_radius: 2,
        recurrent_radius: -1, // no recurrence
        down_radius: 2,
    }];

    let mut h = Hierarchy::new();
    h.init_random(&io_descs, &layer_descs);

    assert_eq!(h.get_num_layers(), 1);
    assert_eq!(h.get_num_io(), 1);
    assert_eq!(h.get_io_type(0), IoType::Prediction);

    // Run a few steps with dummy input
    let io_size = h.get_io_size(0);
    let input_cis = make_random_cis(io_size);

    for _ in 0..3 {
        h.step(&[&input_cis], true, 0.0, 0.0);
    }

    // Predictions should have correct length
    let pred_cis = h.get_prediction_cis(0);
    let expected_len = (io_size.x * io_size.y) as usize;
    assert_eq!(pred_cis.len(), expected_len);

    // All predicted column indices should be in [0, column_size)
    let col_size = io_size.z as i32;
    for &ci in pred_cis {
        assert!(ci >= 0 && ci < col_size, "ci={ci} out of range [0,{col_size})");
    }
}

#[test]
fn test_hierarchy_two_layers() {
    let io_descs = vec![IoDesc {
        size: Int3::new(4, 4, 16),
        io_type: IoType::Prediction,
        num_dendrites_per_cell: 2,
        up_radius: 2,
        down_radius: 2,
        value_size: 64,
        value_num_dendrites_per_cell: 2,
        history_capacity: 64,
    }];

    let layer_descs = vec![
        LayerDesc {
            hidden_size: Int3::new(4, 4, 16),
            num_dendrites_per_cell: 2,
            up_radius: 2,
            recurrent_radius: -1,
            down_radius: 2,
        },
        LayerDesc {
            hidden_size: Int3::new(3, 3, 16),
            num_dendrites_per_cell: 2,
            up_radius: 2,
            recurrent_radius: -1,
            down_radius: 2,
        },
    ];

    let mut h = Hierarchy::new();
    h.init_random(&io_descs, &layer_descs);

    assert_eq!(h.get_num_layers(), 2);

    let io_size = h.get_io_size(0);
    let input_cis = make_random_cis(io_size);

    for _ in 0..5 {
        h.step(&[&input_cis], true, 0.0, 0.0);
    }

    let pred_cis = h.get_prediction_cis(0);
    let col_size = io_size.z as i32;
    for &ci in pred_cis {
        assert!(ci >= 0 && ci < col_size);
    }
}

#[test]
fn test_hierarchy_clear_state() {
    let io_descs = vec![IoDesc {
        size: Int3::new(4, 4, 8),
        io_type: IoType::Prediction,
        num_dendrites_per_cell: 2,
        up_radius: 2,
        down_radius: 2,
        value_size: 32,
        value_num_dendrites_per_cell: 2,
        history_capacity: 32,
    }];

    let layer_descs = vec![LayerDesc {
        hidden_size: Int3::new(4, 4, 8),
        num_dendrites_per_cell: 2,
        up_radius: 2,
        recurrent_radius: -1,
        down_radius: 2,
    }];

    let mut h = Hierarchy::new();
    h.init_random(&io_descs, &layer_descs);

    let io_size = h.get_io_size(0);
    let input_cis = make_random_cis(io_size);

    h.step(&[&input_cis], true, 0.0, 0.0);
    h.clear_state();
    // Should still be able to step after clearing state
    h.step(&[&input_cis], true, 0.0, 0.0);
}

#[test]
fn test_multiple_io() {
    let io_descs = vec![
        IoDesc {
            size: Int3::new(4, 4, 8),
            io_type: IoType::Prediction,
            num_dendrites_per_cell: 2,
            up_radius: 2,
            down_radius: 2,
            value_size: 32,
            value_num_dendrites_per_cell: 2,
            history_capacity: 32,
        },
        IoDesc {
            size: Int3::new(4, 4, 8),
            io_type: IoType::Prediction,
            num_dendrites_per_cell: 2,
            up_radius: 2,
            down_radius: 2,
            value_size: 32,
            value_num_dendrites_per_cell: 2,
            history_capacity: 32,
        },
    ];

    let layer_descs = vec![LayerDesc {
        hidden_size: Int3::new(4, 4, 16),
        num_dendrites_per_cell: 2,
        up_radius: 2,
        recurrent_radius: -1,
        down_radius: 2,
    }];

    let mut h = Hierarchy::new();
    h.init_random(&io_descs, &layer_descs);

    assert_eq!(h.get_num_io(), 2);

    let io_size0 = h.get_io_size(0);
    let io_size1 = h.get_io_size(1);
    let input0 = make_random_cis(io_size0);
    let input1 = make_random_cis(io_size1);

    for _ in 0..3 {
        h.step(&[&input0, &input1], true, 0.0, 0.0);
    }

    for i in 0..2 {
        let pred = h.get_prediction_cis(i);
        let io_size = h.get_io_size(i);
        let col_size = io_size.z as i32;
        assert_eq!(pred.len(), (io_size.x * io_size.y) as usize);
        for &ci in pred {
            assert!(ci >= 0 && ci < col_size);
        }
    }
}
