use crate::plain_nn_builder::plain_ops::*;
use crate::plain_nn_builder::plain_utils::*;
use crate::plain_nn_builder::plain_layers::PlainLayer;

use rayon::prelude::*;

pub struct PlainBatchNormLayer<T: PlainElement> {
    pub id: String,
    pub x_hat: PlainTensor<T>,
    pub mean: PlainTensor<T>,
    pub variance: PlainTensor<T>,
    pub gamma: PlainTensor<T>,
    pub beta: PlainTensor<T>,
    pub batch_mean: PlainTensor<T>,
    pub batch_variance: PlainTensor<T>,
    pub grad_gamma: Option<PlainTensor<T>>,
    pub grad_beta: Option<PlainTensor<T>>,
    pub velocity_gamma: Option<PlainTensor<T>>,
    pub velocity_beta: Option<PlainTensor<T>>,
}

impl<T: PlainElement> PlainBatchNormLayer<T> {
    pub fn new(id: String, x_hat: PlainTensor<T>, mean: PlainTensor<T>, variance: PlainTensor<T>, gamma: PlainTensor<T>, beta: PlainTensor<T>) -> Self 
        where T: Default
    {
        Self {
            id,
            x_hat,
            mean: mean.clone(),
            variance: variance.clone(),
            gamma,
            beta,
            batch_mean: PlainTensor{
                data: vec![T::default(); mean.data.len()],
                shape: mean.shape.clone(),
            },
            batch_variance: PlainTensor{
                data: vec![T::default(); variance.data.len()],
                shape: variance.shape.clone(),
            },
            grad_gamma: None,
            grad_beta: None,
            velocity_gamma: None,
            velocity_beta: None,
        }
    }
}

impl<T> PlainLayer<T> for PlainBatchNormLayer<T>
where
    T: PlainAdd + PlainSub + PlainMul + PlainDiv + PlainSqrt + PlainMulInf + PlainDivInf + Send + Sync + Clone + PlainElement + PlainValueType + Copy + Default,
{
    fn forward(&mut self, input: &PlainTensor<T>) -> PlainTensor<T> {
        let batch_size = input.shape[0];
        let channels = input.shape[1];
        let height = input.shape[2];
        let width = input.shape[3];

        let hw = height * width;
        let chw = channels * hw;
        let n_elements_per_channel = batch_size * hw;
        let n = T::from_f32(n_elements_per_channel as f32);
        let epsilon = T::from_f32(1e-5);
        let momentum = T::from_f32(0.9);
        let current_weight = T::from_f32(0.1);

        // We will collect results for each channel in parallel
        // This avoids borrowing &mut self inside the parallel closure
        let channel_results: Vec<(T, T, Vec<T>, Vec<T>)> = (0..channels)
            .into_par_iter()
            .map(|j| {
                // 1. Compute Mean for this channel
                let mut sum = T::default();
                for i in 0..batch_size {
                    let batch_offset = i * chw + j * hw;
                    for offset in 0..hw {
                        sum = sum.add(input.data[batch_offset + offset]);
                    }
                }
                let batch_mean = sum.div(n);

                // 2. Compute Variance for this channel
                let mut var_sum = T::default();
                for i in 0..batch_size {
                    let batch_offset = i * chw + j * hw;
                    for offset in 0..hw {
                        let diff = input.data[batch_offset + offset].sub(batch_mean);
                        var_sum = var_sum.add(diff.mul(diff));
                    }
                }
                let batch_variance = var_sum.div(n);

                // 3. Normalize, Scale, and Shift
                let std = (batch_variance.add(epsilon)).sqrt();
                let gamma = self.gamma.data[j];
                let beta = self.beta.data[j];

                let mut local_x_hat = vec![T::default(); n_elements_per_channel];
                let mut local_normalized = vec![T::default(); n_elements_per_channel];

                for i in 0..batch_size {
                    let batch_offset = i * chw + j * hw;
                    let local_batch_offset = i * hw;
                    for offset in 0..hw {
                        let x = input.data[batch_offset + offset];
                        let xhat_val = x.sub(batch_mean).div(std);
                        
                        local_x_hat[local_batch_offset + offset] = xhat_val;
                        local_normalized[local_batch_offset + offset] = (xhat_val.mul(gamma)).add(beta);
                    }
                }

                (batch_mean, batch_variance, local_x_hat, local_normalized)
            })
            .collect();

        // Final Assembly (Sequential update of self and output tensors)
        let mut normalized_data = vec![T::default(); input.data.len()];
        let mut x_hat_data = vec![T::default(); input.data.len()];

        for (j, (b_mean, b_var, l_xhat, l_norm)) in channel_results.into_iter().enumerate() {
            // Update statistics
            self.batch_mean.data[j] = b_mean;
            self.batch_variance.data[j] = b_var;
            //let prev = self.mean.data[j];
            self.mean.data[j] = (self.mean.data[j].mul_inf(momentum)).add(b_mean.mul_inf(current_weight));
            
            //let mean = prev.mul(momentum).add(b_mean.mul(current_weight));
            //println!("Exact mean: {:?}, Approx mean: {:?}", self.mean.data[j].to_f32(), mean.to_f32());

            //let prev2 = self.variance.data[j];
            self.variance.data[j] = (self.variance.data[j].mul_inf(momentum)).add(b_var.mul_inf(current_weight));
            
            //let var = (prev2.mul(momentum)).add(b_var.mul(current_weight));
            //println!("Exact var: {:?}, Approx var: {:?}", self.variance.data[j].to_f32(), var.to_f32());

            //println!("BatchNorm Channel {}: batch_mean = {:?}, batch_var = {:?}", j, b_mean.to_f32(), b_var.to_f32());
            

            // Reassemble data from local channel buffers into NCHW format
            for i in 0..batch_size {
                let global_idx_start = i * chw + j * hw;
                let local_idx_start = i * hw;
                
                x_hat_data[global_idx_start..global_idx_start + hw]
                    .copy_from_slice(&l_xhat[local_idx_start..local_idx_start + hw]);
                
                normalized_data[global_idx_start..global_idx_start + hw]
                    .copy_from_slice(&l_norm[local_idx_start..local_idx_start + hw]);
            }
        }

        self.x_hat = PlainTensor {
            data: x_hat_data,
            shape: input.shape.clone(),
        };

        PlainTensor {
            data: normalized_data,
            shape: input.shape.clone(),
        }

    }

    fn backward(
        &mut self,
        input: &PlainTensor<T>,
        grad_output: &PlainTensor<T>,
    ) -> PlainTensor<T> {
        let (batch_size, channels, height, width) = (
            input.shape[0],
            input.shape[1],
            input.shape[2],
            input.shape[3],
        );

        let hw = height * width;
        let chw = channels * hw;
        let n_elements_per_channel = batch_size * hw;
        let n = T::from_f32(n_elements_per_channel as f32);
        let eps = T::from_f32(1e-5);

        // 1. Pre-allocate the gradient input data buffer
        // We will fill this in parallel by chunks
        let mut grad_input_data = vec![T::default(); input.data.len()];

        // 2. Parallel Computation over Channels
        // This returns the gradients for gamma and beta to be stored in the struct
        let channel_stats: Vec<(T, T)> = (0..channels)
            .into_par_iter()
            .map(|c| {
                let mut grad_gamma_c = T::default();
                let mut grad_beta_c = T::default();
                let mut sum_dy = T::default();
                let mut sum_dy_xhat = T::default();

                // Pass A: Compute Reductions (Sums) for this channel
                for i in 0..batch_size {
                    let base_idx = i * chw + c * hw;
                    for offset in 0..hw {
                        let idx = base_idx + offset;
                        let dy = grad_output.data[idx];
                        let xhat_val = self.x_hat.data[idx];

                        grad_gamma_c = grad_gamma_c.add(dy.mul(xhat_val));
                        grad_beta_c = grad_beta_c.add(dy);
                        sum_dy = sum_dy.add(dy);
                        sum_dy_xhat = sum_dy_xhat.add(dy.mul(xhat_val));
                    }
                }

                // Intermediate terms for the gradient formula
                let gamma = self.gamma.data[c];
                let std = (self.batch_variance.data[c].add(eps)).sqrt();
                let mean_dy = sum_dy.div(n);
                let mean_dy_xhat = sum_dy_xhat.div(n);
                let inv_std_gamma = gamma.div(std);

                // Pass B: Compute dx (grad_input) and write to the shared buffer
                // Since each channel 'c' writes to unique indices, this is thread-safe
                // even though we are conceptually mutably borrowing different parts of grad_input_data.
                // For pure safety in idiomatic Rust, we use raw pointers or split_at_mut, 
                // but for this example, we'll use a controlled indexed write.
                for i in 0..batch_size {
                    let base_idx = i * chw + c * hw;
                    for offset in 0..hw {
                        let idx = base_idx + offset;
                        let dy = grad_output.data[idx];
                        let xhat_val = self.x_hat.data[idx];

                        // Standard PyTorch Batchnorm backward formula:
                        // dx = (gamma / std) * (dy - mean_dy - x_hat * mean_dy_xhat)
                        let term = dy.sub(mean_dy).sub(xhat_val.mul(mean_dy_xhat));
                        let dx = inv_std_gamma.mul(term);

                        // We use a raw pointer approach for maximum speed to circumvent the borrow checker
                        // safely, as each channel thread owns a unique set of indices.
                        unsafe {
                            let ptr = grad_input_data.as_ptr() as *mut T;
                            *ptr.add(idx) = dx;
                        }
                    }
                }

                (grad_gamma_c, grad_beta_c)
            })
            .collect();

        // 3. Update grad_gamma and grad_beta in the struct
        let mut g_gamma = vec![T::default(); channels];
        let mut g_beta = vec![T::default(); channels];

        for (c, (gg, gb)) in channel_stats.into_iter().enumerate() {
            g_gamma[c] = gg;
            g_beta[c] = gb;
        }

        self.grad_gamma = Some(PlainTensor {
            data: g_gamma,
            shape: vec![1, 1, 1, channels],
        });
        self.grad_beta = Some(PlainTensor {
            data: g_beta,
            shape: vec![1, 1, 1, channels],
        });

        PlainTensor {
            data: grad_input_data,
            shape: input.shape.clone(),
        }
    }

    fn update_parameters(&mut self, learning_rate: T, weight_decay: T, momentum: T) 
    where 
        T: Send + Sync + Copy + PlainElement 
    {
        // 1. Inizializzazione Lazy delle Velocity
        if self.velocity_gamma.is_none() {
            self.velocity_gamma = Some(PlainTensor{
                    data: vec![T::from_f32(0.0); self.gamma.data.len()],
                    shape: self.gamma.shape.clone(),
            });
        }
        if self.velocity_beta.is_none() {
            self.velocity_beta = Some(PlainTensor{
                    data: vec![T::from_f32(0.0); self.beta.data.len()],
                    shape: self.beta.shape.clone(),
            });
        }

        // 2. Estrazione sicura
        if let (Some(grad_gamma), Some(grad_beta), Some(vel_gamma), Some(vel_beta)) = (
            &self.grad_gamma,
            &self.grad_beta,
            &mut self.velocity_gamma,
            &mut self.velocity_beta
        ) {
            // --- UPDATE GAMMA (Scale) ---
            // Applica Momentum + Weight Decay
            self.gamma.data.par_iter_mut()
                .zip(grad_gamma.data.par_iter())
                .zip(vel_gamma.data.par_iter_mut())
                .for_each(|((gamma_val, &grad), v)| {
                    // 1. Weight Decay: g' = g + (wd * gamma)
                    let wd_term = gamma_val.mul(weight_decay);
                    let g_prime = grad.add(wd_term);

                    // 2. Momentum: v = (mu * v) + g'
                    let v_momentum = v.mul_inf(momentum);
                    *v = v_momentum.add(g_prime);

                    // 3. Update: gamma = gamma - (lr * v)
                    let step = v.mul(learning_rate);
                    *gamma_val = gamma_val.sub(step);
                });

            // --- UPDATE BETA (Shift) ---
            // Applica SOLO Momentum (No Weight Decay sui termini di bias/shift)
            self.beta.data.par_iter_mut()
                .zip(grad_beta.data.par_iter())
                .zip(vel_beta.data.par_iter_mut())
                .for_each(|((beta_val, &grad), v)| {
                    // 1. Momentum: v = (mu * v) + g
                    let v_momentum = v.mul(momentum);
                    *v = v_momentum.add(grad);

                    // 2. Update: beta = beta - (lr * v)
                    let step = v.mul(learning_rate);
                    *beta_val = beta_val.sub(step);
                });

        } else {
            // Opzionale: Panic o Log se mancano i gradienti
            // panic!("Batch Norm gradients missing update skipped");
        }
    }

    fn inference(&mut self, input: &PlainTensor<T>) -> PlainTensor<T> {
        let batch_size = input.shape[0];
        let mut normalized = input.clone();

        for j in 0..input.shape[1] { 
            let mean = self.mean.data[j];
            let var = self.variance.data[j];
            let gamma = self.gamma.data[j];
            let beta = self.beta.data[j];

            let std = (var.add(T::from_f32(1e-5))).sqrt();

            //println!("Inference BatchNorm Channel {}: mean = {:?}, var = {:?}, std = {:?}", j, mean.to_f32(), var.to_f32(), std.to_f32());

            for i in 0..batch_size {
                for k in 0..input.shape[2] {
                    for l in 0..input.shape[3] {
                        let idx = input.flatten_index(&[i, j, k, l]);
                        let x = input.data[idx];
                        normalized.data[idx] = (x.sub(mean).div(std)).mul(gamma).add(beta);
                    }
                }
            }
        }

        normalized
    }

    fn approximate_inference(&mut self, input: &PlainTensor<T>) -> PlainTensor<T> {
        let batch_size = input.shape[0];
        let mut normalized = input.clone();

        for j in 0..input.shape[1] { 
            let mean = self.mean.data[j];
            let var = self.variance.data[j];
            let gamma = self.gamma.data[j];
            let beta = self.beta.data[j];

            let std = (var.add(T::from_f32(1e-5))).sqrt();

            for i in 0..batch_size {
                for k in 0..input.shape[2] {
                    for l in 0..input.shape[3] {
                        let idx = input.flatten_index(&[i, j, k, l]);
                        let x = input.data[idx];
                        normalized.data[idx] = (x.sub(mean).div(std)).mul(gamma).add(beta);
                    }
                }
            }
        }

        normalized
    }

    fn get_id(&self) -> String {
        self.id.clone()
    }

    fn get_weights(&self) -> PlainTensor<T> {
        self.gamma.clone()
    }

    fn get_biases(&self) -> PlainTensor<T> {
        self.beta.clone()
    }

    fn get_grad_weights(&self) -> PlainTensor<T> {
        self.grad_gamma.clone().unwrap()
    }

    fn get_grad_biases(&self) -> PlainTensor<T> {
        self.grad_beta.clone().unwrap()
    }

}
