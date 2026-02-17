// AOgmaNeo Rust port - Hierarchy (top-level orchestrator)
#![allow(clippy::needless_range_loop)]

use crate::helpers::*;
use crate::encoder::{Encoder, VisibleLayerDesc as EncoderVLD};
use crate::decoder::{Decoder, VisibleLayerDesc as DecoderVLD, Params as DecoderParams};
use crate::actor::{Actor, VisibleLayerDesc as ActorVLD, Params as ActorParams};

#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub enum IoType {
    None = 0,
    #[default]
    Prediction = 1,
    Action = 2,
}

impl From<u8> for IoType {
    fn from(v: u8) -> Self {
        match v {
            0 => IoType::None,
            1 => IoType::Prediction,
            2 => IoType::Action,
            _ => IoType::None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct IoDesc {
    pub size: Int3,
    pub io_type: IoType,
    pub num_dendrites_per_cell: usize,
    pub up_radius: i32,
    pub down_radius: i32,
    pub value_size: usize,
    pub value_num_dendrites_per_cell: usize,
    pub history_capacity: usize,
}

impl Default for IoDesc {
    fn default() -> Self {
        Self {
            size: Int3::new(5, 5, 16),
            io_type: IoType::Prediction,
            num_dendrites_per_cell: 4,
            up_radius: 2,
            down_radius: 2,
            value_size: 128,
            value_num_dendrites_per_cell: 2,
            history_capacity: 512,
        }
    }
}

#[derive(Clone, Debug)]
pub struct LayerDesc {
    pub hidden_size: Int3,
    pub num_dendrites_per_cell: usize,
    pub up_radius: i32,
    pub recurrent_radius: i32, // -1 to disable
    pub down_radius: i32,
}

impl Default for LayerDesc {
    fn default() -> Self {
        Self {
            hidden_size: Int3::new(5, 5, 16),
            num_dendrites_per_cell: 4,
            up_radius: 2,
            recurrent_radius: 0,
            down_radius: 2,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct LayerParams {
    pub decoder: DecoderParams,
    pub encoder: crate::encoder::Params,
    pub recurrent_importance: f32,
}

impl LayerParams {
    pub fn new() -> Self {
        Self {
            decoder: DecoderParams::default(),
            encoder: crate::encoder::Params::default(),
            recurrent_importance: 0.5,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct IoParams {
    pub decoder: DecoderParams,
    pub actor: ActorParams,
    pub importance: f32,
}

impl IoParams {
    pub fn new() -> Self {
        Self {
            decoder: DecoderParams::default(),
            actor: ActorParams::default(),
            importance: 1.0,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct Params {
    pub layers: Vec<LayerParams>,
    pub ios: Vec<IoParams>,
    pub anticipation: bool,
}

#[derive(Debug, Default)]
pub struct Hierarchy {
    encoders: Vec<Encoder>,
    decoders: Vec<Vec<Decoder>>, // decoders[layer][d_index]
    actors: Vec<Actor>,
    hidden_cis_prev: Vec<IntBuffer>,
    feedback_cis_prev: Vec<IntBuffer>,
    i_indices: IntBuffer,
    d_indices: IntBuffer,
    io_sizes: Vec<Int3>,
    io_types: Vec<u8>,
    pub params: Params,
}

impl Hierarchy {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn init_random(&mut self, io_descs: &[IoDesc], layer_descs: &[LayerDesc]) {
        let num_layers = layer_descs.len();
        let num_io = io_descs.len();

        self.encoders = vec![Encoder::default(); num_layers];
        self.decoders = vec![Vec::new(); num_layers];
        self.hidden_cis_prev = vec![Vec::new(); num_layers];
        self.feedback_cis_prev = vec![Vec::new(); num_layers.saturating_sub(1)];

        self.io_sizes = io_descs.iter().map(|d| d.size).collect();
        self.io_types = io_descs.iter().map(|d| d.io_type as u8).collect();

        let mut num_predictions = 0usize;
        let mut num_actions = 0usize;

        for d in io_descs {
            match d.io_type {
                IoType::Prediction => num_predictions += 1,
                IoType::Action => num_actions += 1,
                _ => {}
            }
        }

        self.i_indices = vec![0i32; num_io * 2];
        self.d_indices = vec![-1i32; num_io];
        self.actors = Vec::with_capacity(num_actions);

        for l in 0..num_layers {
            let mut e_vld: Vec<EncoderVLD> = Vec::new();

            if l == 0 {
                // First layer: one visible input per IO
                let recurrent = layer_descs[l].recurrent_radius > -1;
                e_vld.reserve(num_io + if recurrent { 1 } else { 0 });

                for i in 0..num_io {
                    e_vld.push(EncoderVLD {
                        size: io_descs[i].size,
                        radius: io_descs[i].up_radius,
                    });
                }

                // Build decoders (predictions)
                self.decoders[l] = Vec::with_capacity(num_predictions);
                let mut d_index = 0usize;

                for i in 0..num_io {
                    if io_descs[i].io_type == IoType::Prediction {
                        let num_d_vl = 1 + if l < num_layers - 1 { 1 } else { 0 };
                        let mut d_vld: Vec<DecoderVLD> = Vec::with_capacity(num_d_vl);
                        d_vld.push(DecoderVLD {
                            size: layer_descs[l].hidden_size,
                            radius: io_descs[i].down_radius,
                        });
                        if l < num_layers - 1 {
                            d_vld.push(d_vld[0].clone());
                        }

                        let mut dec = Decoder::default();
                        dec.init_random(
                            io_descs[i].size,
                            io_descs[i].num_dendrites_per_cell,
                            d_vld,
                        );
                        self.decoders[l].push(dec);

                        self.i_indices[d_index] = i as i32;
                        self.d_indices[i] = d_index as i32;
                        d_index += 1;
                    }
                }

                // Build actors
                let mut a_index = 0usize;
                for i in 0..num_io {
                    if io_descs[i].io_type == IoType::Action {
                        let num_a_vl = 1 + if l < num_layers - 1 { 1 } else { 0 };
                        let mut a_vld: Vec<ActorVLD> = Vec::with_capacity(num_a_vl);
                        a_vld.push(ActorVLD {
                            size: layer_descs[l].hidden_size,
                            radius: io_descs[i].down_radius,
                        });
                        if l < num_layers - 1 {
                            a_vld.push(a_vld[0].clone());
                        }

                        let mut actor = Actor::default();
                        actor.init_random(
                            io_descs[i].size,
                            io_descs[i].value_size,
                            io_descs[i].value_num_dendrites_per_cell,
                            io_descs[i].num_dendrites_per_cell,
                            io_descs[i].history_capacity,
                            a_vld,
                        );
                        self.actors.push(actor);

                        self.i_indices[num_io + a_index] = i as i32;
                        self.d_indices[i] = a_index as i32;
                        a_index += 1;
                    }
                }

                // Recurrent visible layer
                if recurrent {
                    e_vld.push(EncoderVLD {
                        size: layer_descs[l].hidden_size,
                        radius: layer_descs[l].recurrent_radius,
                    });
                }
            } else {
                // Higher layers: one visible input from previous layer
                let recurrent = layer_descs[l].recurrent_radius > -1;
                e_vld.reserve(1 + if recurrent { 1 } else { 0 });

                e_vld.push(EncoderVLD {
                    size: layer_descs[l - 1].hidden_size,
                    radius: layer_descs[l].up_radius,
                });

                // Single decoder per higher layer
                let num_d_vl = 1 + if l < num_layers - 1 { 1 } else { 0 };
                let mut d_vld: Vec<DecoderVLD> = Vec::with_capacity(num_d_vl);
                d_vld.push(DecoderVLD {
                    size: layer_descs[l].hidden_size,
                    radius: layer_descs[l].down_radius,
                });
                if l < num_layers - 1 {
                    d_vld.push(d_vld[0].clone());
                }

                let mut dec = Decoder::default();
                dec.init_random(
                    layer_descs[l - 1].hidden_size,
                    layer_descs[l].num_dendrites_per_cell,
                    d_vld,
                );
                self.decoders[l] = vec![dec];

                if recurrent {
                    e_vld.push(EncoderVLD {
                        size: layer_descs[l].hidden_size,
                        radius: layer_descs[l].recurrent_radius,
                    });
                }
            }

            self.encoders[l].init_random(layer_descs[l].hidden_size, e_vld);

            self.hidden_cis_prev[l] = self.encoders[l].get_hidden_cis().to_vec();

            if l < num_layers - 1 {
                self.feedback_cis_prev[l] = self.encoders[l].get_hidden_cis().to_vec();
            }
        }

        // init params
        self.params.layers = vec![LayerParams::new(); num_layers];
        self.params.ios = vec![IoParams::new(); num_io];
        self.params.anticipation = true;
    }

    pub fn step(
        &mut self,
        input_cis: &[&[i32]],
        learn_enabled: bool,
        reward: f32,
        mimic: f32,
    ) {
        let num_layers = self.encoders.len();
        let num_io = self.io_sizes.len();

        // Set importances from params
        for i in 0..num_io {
            self.encoders[0].get_visible_layer_mut(i).importance = self.params.ios[i].importance;
        }

        // --- Forward pass ---
        for l in 0..num_layers {
            // Copy previous hidden CIS
            self.hidden_cis_prev[l] = self.encoders[l].get_hidden_cis().to_vec();

            if l < num_layers - 1 {
                self.feedback_cis_prev[l] =
                    self.decoders[l + 1][0].get_hidden_cis().to_vec();
            }

            // Build input_cis for this layer's encoder
            let num_enc_vl = self.encoders[l].get_num_visible_layers();
            let mut layer_input_cis_owned: Vec<Vec<i32>> = Vec::with_capacity(num_enc_vl);

            if l == 0 {
                for i in 0..num_io {
                    layer_input_cis_owned.push(input_cis[i].to_vec());
                }
            } else {
                layer_input_cis_owned.push(self.encoders[l - 1].get_hidden_cis().to_vec());
            }

            // Add recurrent input
            let is_recurrent = self.is_layer_recurrent(l);
            if is_recurrent {
                layer_input_cis_owned.push(self.hidden_cis_prev[l].clone());
                // set recurrent importance
                let last_idx = num_enc_vl - 1;
                self.encoders[l].get_visible_layer_mut(last_idx).importance =
                    self.params.layers[l].recurrent_importance;
            }

            let layer_input_refs: Vec<&[i32]> =
                layer_input_cis_owned.iter().map(|v| v.as_slice()).collect();

            let enc_params = self.params.layers[l].encoder.clone();
            self.encoders[l].step(&layer_input_refs, learn_enabled, &enc_params);
        }

        // --- Backward pass ---
        for l in (0..num_layers).rev() {
            if learn_enabled {
                // Build learn input_cis (prev hidden state)
                let num_dec_vl = 1 + if l < num_layers - 1 { 1 } else { 0 };
                let mut layer_input_cis_owned: Vec<Vec<i32>> = Vec::with_capacity(num_dec_vl);
                layer_input_cis_owned.push(self.hidden_cis_prev[l].clone());

                if l < num_layers - 1 {
                    // learn on feedback
                    layer_input_cis_owned.push(self.feedback_cis_prev[l].clone());
                    let layer_input_refs: Vec<&[i32]> =
                        layer_input_cis_owned.iter().map(|v| v.as_slice()).collect();

                    for d in 0..self.decoders[l].len() {
                        let target_cis: Vec<i32>;
                        let dec_params;

                        if l == 0 {
                            let i_idx = self.i_indices[d] as usize;
                            target_cis = input_cis[i_idx].to_vec();
                            dec_params = self.params.ios[i_idx].decoder.clone();
                        } else {
                            target_cis = self.encoders[l - 1].get_hidden_cis().to_vec();
                            dec_params = self.params.layers[l].decoder.clone();
                        }

                        self.decoders[l][d].learn(&layer_input_refs, &target_cis, &dec_params);
                    }

                    if self.params.anticipation {
                        // learn on actual (current encoder output)
                        let actual = self.encoders[l].get_hidden_cis().to_vec();
                        layer_input_cis_owned[1] = actual;
                        let layer_input_refs2: Vec<&[i32]> =
                            layer_input_cis_owned.iter().map(|v| v.as_slice()).collect();

                        for d in 0..self.decoders[l].len() {
                            let target_cis: Vec<i32>;
                            let dec_params;

                            if l == 0 {
                                let i_idx = self.i_indices[d] as usize;
                                target_cis = input_cis[i_idx].to_vec();
                                dec_params = self.params.ios[i_idx].decoder.clone();
                            } else {
                                target_cis = self.encoders[l - 1].get_hidden_cis().to_vec();
                                dec_params = self.params.layers[l].decoder.clone();
                            }

                            self.decoders[l][d]
                                .activate(&layer_input_refs2, &dec_params);
                            self.decoders[l][d]
                                .learn(&layer_input_refs2, &target_cis, &dec_params);
                        }
                    }
                } else {
                    // top layer: no feedback second input
                    let layer_input_refs: Vec<&[i32]> =
                        layer_input_cis_owned.iter().map(|v| v.as_slice()).collect();

                    for d in 0..self.decoders[l].len() {
                        let target_cis: Vec<i32>;
                        let dec_params;

                        if l == 0 {
                            let i_idx = self.i_indices[d] as usize;
                            target_cis = input_cis[i_idx].to_vec();
                            dec_params = self.params.ios[i_idx].decoder.clone();
                        } else {
                            target_cis = self.encoders[l - 1].get_hidden_cis().to_vec();
                            dec_params = self.params.layers[l].decoder.clone();
                        }

                        self.decoders[l][d].learn(&layer_input_refs, &target_cis, &dec_params);
                    }
                }
            }

            // Build activate input_cis (current encoder output)
            let num_dec_vl = 1 + if l < num_layers - 1 { 1 } else { 0 };
            let mut layer_input_cis_owned: Vec<Vec<i32>> = Vec::with_capacity(num_dec_vl);
            layer_input_cis_owned.push(self.encoders[l].get_hidden_cis().to_vec());

            if l < num_layers - 1 {
                layer_input_cis_owned
                    .push(self.decoders[l + 1][0].get_hidden_cis().to_vec());
            }

            let layer_input_refs: Vec<&[i32]> =
                layer_input_cis_owned.iter().map(|v| v.as_slice()).collect();

            for d in 0..self.decoders[l].len() {
                let dec_params = if l == 0 {
                    let i_idx = self.i_indices[d] as usize;
                    self.params.ios[i_idx].decoder.clone()
                } else {
                    self.params.layers[l].decoder.clone()
                };
                self.decoders[l][d].activate(&layer_input_refs, &dec_params);
            }

            // actors (only at layer 0)
            if l == 0 {
                for d in 0..self.actors.len() {
                    let i_idx = self.i_indices[num_io + d] as usize;
                    let target_cis = input_cis[i_idx].to_vec();
                    let actor_params = self.params.ios[i_idx].actor.clone();
                    self.actors[d].step(
                        &layer_input_refs,
                        &target_cis,
                        learn_enabled,
                        reward,
                        mimic,
                        &actor_params,
                    );
                }
            }
        }
    }

    pub fn clear_state(&mut self) {
        for l in 0..self.encoders.len() {
            self.encoders[l].clear_state();
            for d in 0..self.decoders[l].len() {
                self.decoders[l][d].clear_state();
            }
        }
        for actor in &mut self.actors {
            actor.clear_state();
        }
    }

    pub fn is_layer_recurrent(&self, l: usize) -> bool {
        if l == 0 {
            self.encoders[l].get_num_visible_layers() > self.io_sizes.len()
        } else {
            self.encoders[l].get_num_visible_layers() > 1
        }
    }

    pub fn io_layer_exists(&self, i: usize) -> bool {
        self.d_indices[i] != -1
    }

    pub fn get_prediction_cis(&self, i: usize) -> &[i32] {
        let io_type = IoType::from(self.io_types[i]);
        if io_type == IoType::Action {
            self.actors[self.d_indices[i] as usize].get_hidden_cis()
        } else {
            self.decoders[0][self.d_indices[i] as usize].get_hidden_cis()
        }
    }

    pub fn get_prediction_acts(&self, i: usize) -> &[f32] {
        let io_type = IoType::from(self.io_types[i]);
        if io_type == IoType::Action {
            self.actors[self.d_indices[i] as usize].get_hidden_acts()
        } else {
            self.decoders[0][self.d_indices[i] as usize].get_hidden_acts()
        }
    }

    pub fn get_prediction_values(&self, i: usize) -> &[f32] {
        assert_eq!(IoType::from(self.io_types[i]), IoType::Action);
        self.actors[self.d_indices[i] as usize].get_hidden_values()
    }

    pub fn get_num_layers(&self) -> usize {
        self.encoders.len()
    }

    pub fn get_num_io(&self) -> usize {
        self.io_sizes.len()
    }

    pub fn get_io_size(&self, i: usize) -> Int3 {
        self.io_sizes[i]
    }

    pub fn get_io_type(&self, i: usize) -> IoType {
        IoType::from(self.io_types[i])
    }

    pub fn get_encoder(&self, l: usize) -> &Encoder {
        &self.encoders[l]
    }

    pub fn get_encoder_mut(&mut self, l: usize) -> &mut Encoder {
        &mut self.encoders[l]
    }

    pub fn get_decoder(&self, l: usize, i: usize) -> &Decoder {
        if l == 0 {
            &self.decoders[l][self.d_indices[i] as usize]
        } else {
            &self.decoders[l][i]
        }
    }

    pub fn get_decoder_mut(&mut self, l: usize, i: usize) -> &mut Decoder {
        if l == 0 {
            let d = self.d_indices[i] as usize;
            &mut self.decoders[l][d]
        } else {
            &mut self.decoders[l][i]
        }
    }

    pub fn get_actor(&self, i: usize) -> &Actor {
        &self.actors[self.d_indices[i] as usize]
    }

    pub fn get_actor_mut(&mut self, i: usize) -> &mut Actor {
        let d = self.d_indices[i] as usize;
        &mut self.actors[d]
    }

    pub fn get_i_indices(&self) -> &[i32] {
        &self.i_indices
    }

    pub fn get_d_indices(&self) -> &[i32] {
        &self.d_indices
    }

    // Serialization
    pub fn write(&self, writer: &mut dyn StreamWriter) {
        let num_layers = self.encoders.len() as i32;
        let num_io = self.io_sizes.len() as i32;
        let num_predictions = self.decoders[0].len() as i32;
        let num_actions = self.actors.len() as i32;

        writer.write_i32(num_layers);
        writer.write_i32(num_io);
        writer.write_i32(num_predictions);
        writer.write_i32(num_actions);

        for sz in &self.io_sizes {
            writer.write_int3(*sz);
        }
        writer.write_u8_slice(&self.io_types);
        writer.write_i32_slice(&self.i_indices);
        writer.write_i32_slice(&self.d_indices);

        for l in 0..self.encoders.len() {
            self.encoders[l].write(writer);
            for d in 0..self.decoders[l].len() {
                self.decoders[l][d].write(writer);
            }
        }

        for actor in &self.actors {
            actor.write(writer);
        }

        // params
        for lp in &self.params.layers {
            writer.write_f32(lp.decoder.scale);
            writer.write_f32(lp.decoder.lr);
            writer.write_f32(lp.encoder.choice);
            writer.write_f32(lp.encoder.vigilance);
            writer.write_f32(lp.encoder.lr);
            writer.write_f32(lp.encoder.active_ratio);
            writer.write_i32(lp.encoder.l_radius);
            writer.write_f32(lp.recurrent_importance);
        }

        for ip in &self.params.ios {
            writer.write_f32(ip.decoder.scale);
            writer.write_f32(ip.decoder.lr);
            // actor params
            writer.write_f32(ip.actor.vlr);
            writer.write_f32(ip.actor.plr);
            writer.write_f32(ip.actor.smoothing);
            writer.write_f32(ip.actor.discount);
            writer.write_f32(ip.actor.td_scale_decay);
            writer.write_f32(ip.actor.value_range);
            writer.write_i32(ip.actor.min_steps as i32);
            writer.write_i32(ip.actor.history_iters as i32);
            writer.write_f32(ip.importance);
        }

        writer.write_u8(if self.params.anticipation { 1 } else { 0 });
    }

    pub fn read(&mut self, reader: &mut dyn StreamReader) {
        let num_layers = reader.read_i32() as usize;
        let num_io = reader.read_i32() as usize;
        let num_predictions = reader.read_i32() as usize;
        let num_actions = reader.read_i32() as usize;

        self.io_sizes = (0..num_io).map(|_| reader.read_int3()).collect();
        self.io_types = (0..num_io).map(|_| reader.read_u8()).collect();

        self.i_indices = vec![0i32; num_io * 2];
        reader.read_i32_slice(&mut self.i_indices);

        self.d_indices = vec![0i32; num_io];
        reader.read_i32_slice(&mut self.d_indices);

        self.encoders = vec![Encoder::default(); num_layers];
        self.decoders = vec![Vec::new(); num_layers];
        self.hidden_cis_prev = vec![Vec::new(); num_layers];
        self.feedback_cis_prev = vec![Vec::new(); num_layers.saturating_sub(1)];

        for l in 0..num_layers {
            self.encoders[l].read(reader);

            let num_decoders = if l == 0 { num_predictions } else { 1 };
            self.decoders[l] = Vec::with_capacity(num_decoders);
            for _ in 0..num_decoders {
                let mut dec = Decoder::default();
                dec.read(reader);
                self.decoders[l].push(dec);
            }

            self.hidden_cis_prev[l] = self.encoders[l].get_hidden_cis().to_vec();
            if l < num_layers - 1 {
                self.feedback_cis_prev[l] = self.encoders[l].get_hidden_cis().to_vec();
            }
        }

        self.actors = Vec::with_capacity(num_actions);
        for _ in 0..num_actions {
            let mut actor = Actor::default();
            actor.read(reader);
            self.actors.push(actor);
        }

        // params
        self.params.layers = (0..num_layers)
            .map(|_| LayerParams {
                decoder: DecoderParams {
                    scale: reader.read_f32(),
                    lr: reader.read_f32(),
                },
                encoder: crate::encoder::Params {
                    choice: reader.read_f32(),
                    vigilance: reader.read_f32(),
                    lr: reader.read_f32(),
                    active_ratio: reader.read_f32(),
                    l_radius: reader.read_i32(),
                },
                recurrent_importance: reader.read_f32(),
            })
            .collect();

        self.params.ios = (0..num_io)
            .map(|_| IoParams {
                decoder: DecoderParams {
                    scale: reader.read_f32(),
                    lr: reader.read_f32(),
                },
                actor: ActorParams {
                    vlr: reader.read_f32(),
                    plr: reader.read_f32(),
                    smoothing: reader.read_f32(),
                    discount: reader.read_f32(),
                    td_scale_decay: reader.read_f32(),
                    value_range: reader.read_f32(),
                    min_steps: reader.read_i32() as usize,
                    history_iters: reader.read_i32() as usize,
                },
                importance: reader.read_f32(),
            })
            .collect();

        self.params.anticipation = reader.read_u8() != 0;
    }

    pub fn write_state(&self, writer: &mut dyn StreamWriter) {
        for l in 0..self.encoders.len() {
            self.encoders[l].write_state(writer);
            for d in 0..self.decoders[l].len() {
                self.decoders[l][d].write_state(writer);
            }
        }
        for actor in &self.actors {
            actor.write_state(writer);
        }
    }

    pub fn read_state(&mut self, reader: &mut dyn StreamReader) {
        for l in 0..self.encoders.len() {
            self.encoders[l].read_state(reader);
            for d in 0..self.decoders[l].len() {
                self.decoders[l][d].read_state(reader);
            }
        }
        for actor in &mut self.actors {
            actor.read_state(reader);
        }
    }

    pub fn write_weights(&self, writer: &mut dyn StreamWriter) {
        for l in 0..self.encoders.len() {
            self.encoders[l].write_weights(writer);
            for d in 0..self.decoders[l].len() {
                self.decoders[l][d].write_weights(writer);
            }
        }
        for actor in &self.actors {
            actor.write_weights(writer);
        }
    }

    pub fn read_weights(&mut self, reader: &mut dyn StreamReader) {
        for l in 0..self.encoders.len() {
            self.encoders[l].read_weights(reader);
            for d in 0..self.decoders[l].len() {
                self.decoders[l][d].read_weights(reader);
            }
        }
        for actor in &mut self.actors {
            actor.read_weights(reader);
        }
    }
}
