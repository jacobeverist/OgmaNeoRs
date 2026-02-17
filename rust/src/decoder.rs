// AOgmaNeo Rust port - Decoder (predictive reconstruction with multi-dendrite perceptrons)
#![allow(clippy::needless_range_loop)]

use rayon::prelude::*;
use crate::helpers::*;

#[derive(Clone, Debug)]
pub struct VisibleLayerDesc {
    pub size: Int3,
    pub radius: i32,
}

impl Default for VisibleLayerDesc {
    fn default() -> Self {
        Self {
            size: Int3::new(5, 5, 16),
            radius: 2,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct VisibleLayer {
    pub weights: SByteBuffer,
}

#[derive(Clone, Debug)]
pub struct Params {
    pub scale: f32,
    pub lr: f32,
}

impl Default for Params {
    fn default() -> Self {
        Self {
            scale: 8.0,
            lr: 0.1,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct Decoder {
    hidden_size: Int3,
    num_dendrites_per_cell: usize,
    hidden_cis: IntBuffer,
    hidden_acts: FloatBuffer,
    dendrite_acts: FloatBuffer,
    dendrite_deltas: IntBuffer,
    pub visible_layers: Vec<VisibleLayer>,
    pub visible_layer_descs: Vec<VisibleLayerDesc>,
}

// Per-column result of forward pass
struct ForwardResult {
    hidden_ci: i32,
    // dendrite_acts and hidden_acts written at column offset
    dendrite_acts: Vec<f32>,
    hidden_acts: Vec<f32>,
}

impl Decoder {
    #[allow(clippy::too_many_arguments)]
    fn compute_forward(
        column_pos: Int2,
        hidden_size: Int3,
        num_dendrites_per_cell: usize,
        visible_layers: &[VisibleLayer],
        visible_layer_descs: &[VisibleLayerDesc],
        input_cis: &[&[i32]],
        params: &Params,
    ) -> ForwardResult {
        let hidden_column_index = address2(column_pos, Int2::new(hidden_size.x, hidden_size.y));

        let num_hc = hidden_size.z as usize;
        let num_dendrites = num_hc * num_dendrites_per_cell;

        let mut dendrite_acts = vec![0.0f32; num_dendrites];
        let mut hidden_acts_col = vec![0.0f32; num_hc];

        let mut count = 0usize;

        for vli in 0..visible_layers.len() {
            let vl = &visible_layers[vli];
            let vld = &visible_layer_descs[vli];

            let diam = vld.radius * 2 + 1;
            let h_to_v = Float2::new(
                vld.size.x as f32 / hidden_size.x as f32,
                vld.size.y as f32 / hidden_size.y as f32,
            );
            let visible_center = project(column_pos, h_to_v);
            let field_lower_bound = Int2::new(
                visible_center.x - vld.radius,
                visible_center.y - vld.radius,
            );
            let iter_lower_bound =
                Int2::new(field_lower_bound.x.max(0), field_lower_bound.y.max(0));
            let iter_upper_bound = Int2::new(
                (visible_center.x + vld.radius).min(vld.size.x - 1),
                (visible_center.y + vld.radius).min(vld.size.y - 1),
            );

            count += ((iter_upper_bound.x - iter_lower_bound.x + 1)
                * (iter_upper_bound.y - iter_lower_bound.y + 1)) as usize;

            let vl_input_cis = input_cis[vli];

            for ix in iter_lower_bound.x..=iter_upper_bound.x {
                for iy in iter_lower_bound.y..=iter_upper_bound.y {
                    let visible_column_index =
                        address2(Int2::new(ix, iy), Int2::new(vld.size.x, vld.size.y));
                    let in_ci = vl_input_cis[visible_column_index] as usize;
                    let offset = Int2::new(ix - field_lower_bound.x, iy - field_lower_bound.y);

                    let wi_start_partial = num_hc
                        * (offset.y as usize
                            + diam as usize
                                * (offset.x as usize
                                    + diam as usize
                                        * (in_ci + vld.size.z as usize * hidden_column_index)));

                    for hc in 0..num_hc {
                        let dendrites_start = num_dendrites_per_cell * hc;
                        let wi_start = num_dendrites_per_cell * (hc + wi_start_partial);

                        for di in 0..num_dendrites_per_cell {
                            dendrite_acts[dendrites_start + di] +=
                                vl.weights[di + wi_start] as f32;
                        }
                    }
                }
            }
        }

        let half_num = num_dendrites_per_cell / 2;
        let dendrite_scale = (1.0f32 / count as f32).sqrt() / 127.0 * params.scale;
        let activation_scale = (1.0f32 / num_dendrites_per_cell as f32).sqrt();

        let mut max_index = 0usize;
        let mut max_activation = LIMIT_MIN;

        for hc in 0..num_hc {
            let dendrites_start = num_dendrites_per_cell * hc;

            let mut activation = 0.0f32;

            for di in 0..num_dendrites_per_cell {
                let act = dendrite_acts[dendrites_start + di] * dendrite_scale;
                dendrite_acts[dendrites_start + di] = sigmoidf(act); // store derivative
                activation += softplusf(act)
                    * (if di >= half_num { 2.0 } else { 0.0 } - 1.0);
            }

            activation *= activation_scale;
            hidden_acts_col[hc] = activation;

            if activation > max_activation {
                max_activation = activation;
                max_index = hc;
            }
        }

        // softmax
        let mut total = 0.0f32;
        for hc in 0..num_hc {
            hidden_acts_col[hc] = (hidden_acts_col[hc] - max_activation).exp();
            total += hidden_acts_col[hc];
        }
        let total_inv = 1.0 / LIMIT_SMALL.max(total);
        for hc in 0..num_hc {
            hidden_acts_col[hc] *= total_inv;
        }

        ForwardResult {
            hidden_ci: max_index as i32,
            dendrite_acts,
            hidden_acts: hidden_acts_col,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn learn_column(
        column_pos: Int2,
        hidden_size: Int3,
        num_dendrites_per_cell: usize,
        visible_layers: &mut [VisibleLayer],
        visible_layer_descs: &[VisibleLayerDesc],
        input_cis: &[&[i32]],
        hidden_target_cis: &[i32],
        hidden_acts: &[f32],
        dendrite_acts: &[f32],
        dendrite_deltas: &mut [i32],
        state: &mut u64,
        params: &Params,
    ) {
        let hidden_column_index = address2(column_pos, Int2::new(hidden_size.x, hidden_size.y));
        let hidden_cells_start = hidden_column_index * hidden_size.z as usize;

        let target_ci = hidden_target_cis[hidden_column_index] as usize;
        let num_hc = hidden_size.z as usize;
        let half_num = num_dendrites_per_cell / 2;

        // compute deltas
        for hc in 0..num_hc {
            let hidden_cell_index = hc + hidden_cells_start;
            let dendrites_start = num_dendrites_per_cell * hidden_cell_index;

            let error = params.lr * 127.0 * ((hc == target_ci) as i32 as f32 - hidden_acts[hidden_cell_index]);

            for di in 0..num_dendrites_per_cell {
                let sign = if di >= half_num { 2.0f32 } else { 0.0 } - 1.0;
                dendrite_deltas[dendrites_start + di] = rand_roundf_step(
                    error * sign * dendrite_acts[dendrites_start + di],
                    state,
                );
            }
        }

        // apply deltas to weights
        for vli in 0..visible_layers.len() {
            let vld = &visible_layer_descs[vli];
            let diam = vld.radius * 2 + 1;

            let h_to_v = Float2::new(
                vld.size.x as f32 / hidden_size.x as f32,
                vld.size.y as f32 / hidden_size.y as f32,
            );
            let visible_center = project(column_pos, h_to_v);
            let field_lower_bound = Int2::new(
                visible_center.x - vld.radius,
                visible_center.y - vld.radius,
            );
            let iter_lower_bound =
                Int2::new(field_lower_bound.x.max(0), field_lower_bound.y.max(0));
            let iter_upper_bound = Int2::new(
                (visible_center.x + vld.radius).min(vld.size.x - 1),
                (visible_center.y + vld.radius).min(vld.size.y - 1),
            );

            let vl_input_cis = input_cis[vli];
            let vl = &mut visible_layers[vli];

            for ix in iter_lower_bound.x..=iter_upper_bound.x {
                for iy in iter_lower_bound.y..=iter_upper_bound.y {
                    let visible_column_index =
                        address2(Int2::new(ix, iy), Int2::new(vld.size.x, vld.size.y));
                    let in_ci = vl_input_cis[visible_column_index] as usize;
                    let offset = Int2::new(ix - field_lower_bound.x, iy - field_lower_bound.y);

                    let wi_start_partial = num_hc
                        * (offset.y as usize
                            + diam as usize
                                * (offset.x as usize
                                    + diam as usize
                                        * (in_ci + vld.size.z as usize * hidden_column_index)));

                    for hc in 0..num_hc {
                        let hidden_cell_index = hc + hidden_cells_start;
                        let dendrites_start = num_dendrites_per_cell * hidden_cell_index;
                        let wi_start = num_dendrites_per_cell * (hc + wi_start_partial);

                        for di in 0..num_dendrites_per_cell {
                            let delta = dendrite_deltas[dendrites_start + di];
                            vl.weights[di + wi_start] = (vl.weights[di + wi_start] as i32 + delta)
                                .clamp(-127, 127) as i8;
                        }
                    }
                }
            }
        }
    }

    pub fn init_random(
        &mut self,
        hidden_size: Int3,
        num_dendrites_per_cell: usize,
        visible_layer_descs: Vec<VisibleLayerDesc>,
    ) {
        self.visible_layer_descs = visible_layer_descs;
        self.hidden_size = hidden_size;
        self.num_dendrites_per_cell = num_dendrites_per_cell;

        let num_hidden_columns = (hidden_size.x * hidden_size.y) as usize;
        let num_hidden_cells = num_hidden_columns * hidden_size.z as usize;
        let num_dendrites = num_hidden_cells * num_dendrites_per_cell;

        self.visible_layers = self
            .visible_layer_descs
            .iter()
            .map(|vld| {
                let diam = vld.radius * 2 + 1;
                let area = (diam * diam) as usize;
                let weights_size = num_dendrites * area * vld.size.z as usize;

                let weights: SByteBuffer = (0..weights_size)
                    .map(|_| {
                        ((global_rand() % (INIT_WEIGHT_NOISEI + 1)) as i32
                            - INIT_WEIGHT_NOISEI as i32 / 2) as i8
                    })
                    .collect();

                VisibleLayer { weights }
            })
            .collect();

        self.hidden_cis = vec![0i32; num_hidden_columns];
        self.hidden_acts = vec![0.0f32; num_hidden_cells];
        self.dendrite_acts = vec![0.0f32; num_dendrites];
        self.dendrite_deltas = vec![0i32; num_dendrites];
    }

    pub fn activate(&mut self, input_cis: &[&[i32]], params: &Params) {
        let num_hidden_columns = (self.hidden_size.x * self.hidden_size.y) as usize;
        let hidden_size = self.hidden_size;
        let num_dendrites_per_cell = self.num_dendrites_per_cell;

        let results: Vec<ForwardResult> = (0..num_hidden_columns)
            .into_par_iter()
            .map(|i| {
                let column_pos = Int2::new(
                    (i / hidden_size.y as usize) as i32,
                    (i % hidden_size.y as usize) as i32,
                );
                Self::compute_forward(
                    column_pos,
                    hidden_size,
                    num_dendrites_per_cell,
                    &self.visible_layers,
                    &self.visible_layer_descs,
                    input_cis,
                    params,
                )
            })
            .collect();

        for (i, res) in results.into_iter().enumerate() {
            self.hidden_cis[i] = res.hidden_ci;
            let cells_start = i * hidden_size.z as usize;
            let dend_start = i * hidden_size.z as usize * num_dendrites_per_cell;
            self.hidden_acts[cells_start..cells_start + hidden_size.z as usize]
                .copy_from_slice(&res.hidden_acts);
            self.dendrite_acts[dend_start..dend_start + hidden_size.z as usize * num_dendrites_per_cell]
                .copy_from_slice(&res.dendrite_acts);
        }
    }

    pub fn learn(
        &mut self,
        input_cis: &[&[i32]],
        hidden_target_cis: &[i32],
        params: &Params,
    ) {
        let num_hidden_columns = (self.hidden_size.x * self.hidden_size.y) as usize;
        let hidden_size = self.hidden_size;
        let num_dendrites_per_cell = self.num_dendrites_per_cell;

        let base_state = global_rand() as u64;

        for i in 0..num_hidden_columns {
            let column_pos = Int2::new(
                (i / hidden_size.y as usize) as i32,
                (i % hidden_size.y as usize) as i32,
            );

            let mut state = rand_get_state(base_state + i as u64 * RAND_SUBSEED_OFFSET);

            Self::learn_column(
                column_pos,
                hidden_size,
                num_dendrites_per_cell,
                &mut self.visible_layers,
                &self.visible_layer_descs,
                input_cis,
                hidden_target_cis,
                &self.hidden_acts,
                &self.dendrite_acts,
                &mut self.dendrite_deltas,
                &mut state,
                params,
            );
        }
    }

    pub fn clear_state(&mut self) {
        self.hidden_cis.fill(0);
        self.hidden_acts.fill(0.0);
    }

    pub fn get_hidden_cis(&self) -> &[i32] {
        &self.hidden_cis
    }

    pub fn get_hidden_acts(&self) -> &[f32] {
        &self.hidden_acts
    }

    pub fn get_hidden_size(&self) -> Int3 {
        self.hidden_size
    }

    pub fn get_num_visible_layers(&self) -> usize {
        self.visible_layers.len()
    }

    pub fn get_visible_layer(&self, i: usize) -> &VisibleLayer {
        &self.visible_layers[i]
    }

    pub fn get_visible_layer_desc(&self, i: usize) -> &VisibleLayerDesc {
        &self.visible_layer_descs[i]
    }

    // Serialization
    pub fn write(&self, writer: &mut dyn StreamWriter) {
        writer.write_int3(self.hidden_size);
        writer.write_i32(self.num_dendrites_per_cell as i32);
        writer.write_i32_slice(&self.hidden_cis);
        writer.write_f32_slice(&self.hidden_acts);
        writer.write_f32_slice(&self.dendrite_acts);
        writer.write_i32(self.visible_layers.len() as i32);

        for (vl, vld) in self.visible_layers.iter().zip(self.visible_layer_descs.iter()) {
            writer.write_int3(vld.size);
            writer.write_i32(vld.radius);
            writer.write_i8_slice(&vl.weights);
        }
    }

    pub fn read(&mut self, reader: &mut dyn StreamReader) {
        self.hidden_size = reader.read_int3();
        self.num_dendrites_per_cell = reader.read_i32() as usize;

        let num_hidden_columns = (self.hidden_size.x * self.hidden_size.y) as usize;
        let num_hidden_cells = num_hidden_columns * self.hidden_size.z as usize;
        let num_dendrites = num_hidden_cells * self.num_dendrites_per_cell;

        self.hidden_cis = vec![0i32; num_hidden_columns];
        reader.read_i32_slice(&mut self.hidden_cis);

        self.hidden_acts = vec![0.0f32; num_hidden_cells];
        reader.read_f32_slice(&mut self.hidden_acts);

        self.dendrite_acts = vec![0.0f32; num_dendrites];
        reader.read_f32_slice(&mut self.dendrite_acts);

        self.dendrite_deltas = vec![0i32; num_dendrites];

        let num_visible_layers = reader.read_i32() as usize;
        self.visible_layers = Vec::with_capacity(num_visible_layers);
        self.visible_layer_descs = Vec::with_capacity(num_visible_layers);

        for _ in 0..num_visible_layers {
            let size = reader.read_int3();
            let radius = reader.read_i32();
            let vld = VisibleLayerDesc { size, radius };

            let diam = vld.radius * 2 + 1;
            let area = (diam * diam) as usize;
            let weights_size = num_dendrites * area * vld.size.z as usize;

            let mut weights = vec![0i8; weights_size];
            reader.read_i8_slice(&mut weights);

            self.visible_layers.push(VisibleLayer { weights });
            self.visible_layer_descs.push(vld);
        }
    }

    pub fn write_state(&self, writer: &mut dyn StreamWriter) {
        writer.write_i32_slice(&self.hidden_cis);
        writer.write_f32_slice(&self.hidden_acts);
        writer.write_f32_slice(&self.dendrite_acts);
    }

    pub fn read_state(&mut self, reader: &mut dyn StreamReader) {
        reader.read_i32_slice(&mut self.hidden_cis);
        reader.read_f32_slice(&mut self.hidden_acts);
        reader.read_f32_slice(&mut self.dendrite_acts);
    }

    pub fn write_weights(&self, writer: &mut dyn StreamWriter) {
        for vl in &self.visible_layers {
            writer.write_i8_slice(&vl.weights);
        }
    }

    pub fn read_weights(&mut self, reader: &mut dyn StreamReader) {
        for vl in &mut self.visible_layers {
            reader.read_i8_slice(&mut vl.weights);
        }
    }
}
