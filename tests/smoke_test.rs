use aogmaneo::helpers::{Int3, VecWriter, SliceReader};
use aogmaneo::hierarchy::{Hierarchy, IoDesc, IoType, LayerDesc};
use aogmaneo::image_encoder::{ImageEncoder, VisibleLayerDesc as IeVLD};

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

#[test]
fn test_image_encoder_step_and_reconstruct() {
    let visible_layer_descs = vec![IeVLD {
        size: Int3::new(8, 8, 3), // 8x8 RGB
        radius: 2,
    }];

    let mut ie = ImageEncoder::default();
    ie.init_random(Int3::new(4, 4, 16), visible_layer_descs);

    assert_eq!(ie.get_hidden_size(), Int3::new(4, 4, 16));
    assert_eq!(ie.get_num_visible_layers(), 1);

    // Synthesize a simple gradient image (8*8*3 bytes)
    let num_pixels = 8 * 8 * 3;
    let inputs: Vec<u8> = (0..num_pixels).map(|i| (i % 256) as u8).collect();

    for _ in 0..5 {
        ie.step(&[&inputs], true, true);
    }

    // Hidden CIs must be valid
    let hidden_cis: Vec<i32> = ie.get_hidden_cis().to_vec();
    assert_eq!(hidden_cis.len(), 4 * 4);
    for &ci in &hidden_cis {
        assert!(ci >= 0 && ci < 16, "ci={ci} out of range [0,16)");
    }

    // reconstruct() should produce a byte image of the right size
    ie.reconstruct(&hidden_cis);
    let recon = ie.get_reconstruction(0);
    assert_eq!(recon.len(), 8 * 8 * 3);
}

#[test]
fn test_hierarchy_serialization_roundtrip() {
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
    let input_cis: Vec<i32> = (0..(io_size.x * io_size.y) as usize)
        .map(|i| (i % io_size.z as usize) as i32)
        .collect();

    // Run a few steps to build up state
    for _ in 0..3 {
        h.step(&[&input_cis], true, 0.0, 0.0);
    }

    // Capture predictions before serialization
    let preds_before: Vec<i32> = h.get_prediction_cis(0).to_vec();

    // Serialize
    let mut writer = VecWriter::new();
    h.write(&mut writer);
    let bytes = writer.data;

    // Deserialize into a fresh hierarchy
    let mut h2 = Hierarchy::new();
    let mut reader = SliceReader::new(&bytes);
    h2.read(&mut reader);

    // Predictions should match after loading
    let preds_after: Vec<i32> = h2.get_prediction_cis(0).to_vec();
    assert_eq!(preds_before, preds_after, "predictions differ after round-trip");

    // Should still be able to step after deserializing
    h2.step(&[&input_cis], true, 0.0, 0.0);
}
