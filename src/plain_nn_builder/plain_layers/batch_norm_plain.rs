use crate::plain_nn_builder::plain_ops::*;
use crate::plain_nn_builder::plain_utils::*;
use crate::plain_nn_builder::plain_layers::PlainLayer;

use rayon::iter::*;
use rayon::prelude::*;
use rayon::scope;


pub struct PlainBatchNormLayer<T: PlainElement> {
    pub id: String,
    pub x_hat: PlainTensor<T>,
    pub mean: PlainTensor<T>,
    pub variance: PlainTensor<T>,
    pub gamma: PlainTensor<T>,
    pub beta: PlainTensor<T>,
    pub batch_mean: PlainTensor<T>,
    pub batch_variance: PlainTensor<T>,
    pub grad_mean: Option<PlainTensor<T>>,
    pub grad_variance: Option<PlainTensor<T>>,
    pub grad_gamma: Option<PlainTensor<T>>,
    pub grad_beta: Option<PlainTensor<T>>,
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
            grad_mean: None,
            grad_variance: None,
            grad_gamma: None,
            grad_beta: None,
        }
    }
}

impl<T> PlainLayer<T> for PlainBatchNormLayer<T>
where
    T: PlainAdd + PlainSub + PlainMul + PlainDiv + PlainSqrt + PlainMulInf + PlainDivInf + Send + Sync + Clone + PlainElement + PlainValueType + Copy + Default,
{
    fn forward(&mut self, input: &PlainTensor<T>) -> PlainTensor<T> {
        let batch_size = input.shape[0];
        let mut normalized = input.clone();
        let mut x_hat = PlainTensor{
            data: vec![T::default(); input.data.len()],
            shape: input.shape.clone(),
        }; 

        for j in 0..input.shape[1] { // loop over channels
            // Compute mean for this channel from current batch
            let mut sum = T::default();
            let n = T::from_f32((batch_size * input.shape[2] * input.shape[3]) as f32);
            
            for i in 0..batch_size {
                for k in 0..input.shape[2] {
                    for l in 0..input.shape[3] {
                        let idx = input.flatten_index(&[i, j, k, l]);
                        sum = sum.add(input.data[idx]);
                    }
                }
            }
            let batch_mean = sum.div(n);
            self.batch_mean.data[j] = batch_mean.clone();
            
            // Compute variance for this channel from current batch
            let mut var_sum = T::default();
            for i in 0..batch_size {
                for k in 0..input.shape[2] {
                    for l in 0..input.shape[3] {
                        let idx = input.flatten_index(&[i, j, k, l]);
                        let diff = input.data[idx].sub(batch_mean);
                        var_sum = var_sum.add(diff.mul(diff));
                    }
                }
            }
            let batch_variance = var_sum.div(n);
            self.batch_variance.data[j] = batch_variance.clone();
            
            // Update running statistics with momentum (0.9 old, 0.1 new)
            let momentum = T::from_f32(0.9);
            let current_weight = T::from_f32(0.1);
            self.mean.data[j] = self.mean.data[j].mul(momentum).add(batch_mean.mul(current_weight));
            self.variance.data[j] = self.variance.data[j].mul(momentum).add(batch_variance.mul(current_weight));

            let gamma = self.gamma.data[j];
            let beta = self.beta.data[j];

            let std = (batch_variance.clone().add(T::from_f32(1e-5))).sqrt();

            for i in 0..batch_size {
                for k in 0..input.shape[2] {
                    for l in 0..input.shape[3] {
                        let idx = input.flatten_index(&[i, j, k, l]);
                        let x = input.data[idx];
                        let xhat_val = x.sub(batch_mean).div(std);
                        x_hat.data[idx] = xhat_val;
                        normalized.data[idx] = (xhat_val.mul(gamma)).add(beta);
                    }
                }
            }
        }

        self.x_hat = x_hat; 
        normalized
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

        let n_elements = (batch_size * height * width) as f32;
        let n = T::from_f32(n_elements);

        let reshaped_grad_output = if grad_output.shape[3] == channels * height * width {
            // Flattened case
            PlainTensor {
                data: grad_output.data.clone(),
                shape: vec![batch_size, channels, height, width],
            }
        } else {
            grad_output.clone()
        };

        let mut grad_input = PlainTensor {
            data: vec![T::default(); input.data.len()],
            shape: input.shape.clone(),
        };

        // Initialize gamma and beta gradients
        self.grad_gamma = Some(PlainTensor {
            data: vec![T::default(); channels],
            shape: vec![1, 1, 1, channels],
        });
        self.grad_beta = Some(PlainTensor {
            data: vec![T::default(); channels],
            shape: vec![1, 1, 1, channels],
        });

        let x_hat = &self.x_hat;

        for c in 0..channels {
            let mut grad_gamma_c = T::default();
            let mut grad_beta_c = T::default();
            let mut sum_dy = T::default();
            let mut sum_dy_xhat = T::default();

            // First pass: compute sums for mean
            for i in 0..batch_size {
                for h in 0..height {
                    for w in 0..width {
                        let idx = i * channels * height * width + c * height * width + h * width + w;
                        let dy = reshaped_grad_output.data[idx];
                        let xhat_val = x_hat.data[idx];

                        grad_gamma_c = grad_gamma_c.add(dy.mul(xhat_val));
                        grad_beta_c = grad_beta_c.add(dy);

                        sum_dy = sum_dy.add(dy);
                        sum_dy_xhat = sum_dy_xhat.add(dy.mul(xhat_val));
                        /* 
                        if sum_dy_xhat.add(dy.mul(xhat_val)).to_f32().abs() > 100.0 || sum_dy_xhat.add(dy.mul(xhat_val)).to_f32().is_nan() {
                            panic!("sum_dy_xhat too large at BN - backward {:?}, dy: {:?}, xhat_val: {:?}", sum_dy_xhat.to_f32(), dy.to_f32(), xhat_val.to_f32());
                        }
                        */
                    }
                }
            }

            // Store gamma and beta gradients
            if let Some(ref mut grad_gamma) = self.grad_gamma {
                grad_gamma.data[c] = grad_gamma_c;
            }
            if let Some(ref mut grad_beta) = self.grad_beta {
                grad_beta.data[c] = grad_beta_c;
            }

            let gamma = self.gamma.data[c];
            let std = (self.batch_variance.data[c].add(T::from_f32(1e-5))).sqrt();

            // Second pass: compute grad_input per element using PyTorch formula
            for i in 0..batch_size {
                for h in 0..height {
                    for w in 0..width {
                        let idx = i * channels * height * width + c * height * width + h * width + w;
                        let dy = reshaped_grad_output.data[idx];
                        let xhat_val = x_hat.data[idx];

                        // Compute per-channel mean terms
                        let mean_dy = sum_dy.div(n);
                        let mean_dy_xhat = sum_dy_xhat.div(n);
                        /* 
                        if mean_dy_xhat.to_f32().abs() > 100.0 {
                            panic!("mean_dy_xhat too large at BN - backward {:?}, sum_dy_xhat: {:?}, n: {:?}", mean_dy_xhat.to_f32(), sum_dy_xhat.to_f32(), n.to_f32());
                        }
                        */

                        // PyTorch formula: dx = gamma / std * (dy - mean(dy) - xhat * mean(dy * xhat))
                        let dx = (gamma.div(std)).mul((dy.sub(mean_dy)).sub(xhat_val.mul(mean_dy_xhat)));
                        /* 
                        if dx.to_f32() > 100.0 {
                            panic!("Gradient too large at BN - backward {:?}, gamma: {:?}, std: {:?}, dy: {:?}, mean_dy: {:?}, xhat_val: {:?}, mean_dy_xhat: {:?}",  dx.to_f32(), gamma.to_f32(), std.to_f32(), dy.to_f32(), mean_dy.to_f32(), xhat_val.to_f32(), mean_dy_xhat.to_f32());
                        }
                        */
                        grad_input.data[idx] = dx;
                    }
                }
            }
        }

        grad_input
    }

    fn update_parameters(&mut self, learning_rate: T) {
        if let (Some(grad_gamma), Some(grad_beta)) = (&self.grad_gamma, &self.grad_beta) {
            for i in 0..self.gamma.data.len() {
                self.gamma.data[i] = self.gamma.data[i].sub(grad_gamma.data[i].mul(learning_rate));
                self.beta.data[i] = self.beta.data[i].sub(grad_beta.data[i].mul(learning_rate));
            }
        }
    }

    fn inference(&mut self, input: &PlainTensor<T>) -> PlainTensor<T> {
        let batch_size = input.shape[0];
        let mut normalized = input.clone();

        for j in 0..input.shape[1] { // loop over channels
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

    fn approximate_inference(&mut self, input: &PlainTensor<T>) -> PlainTensor<T> {
        let batch_size = input.shape[0];
        let mut normalized = input.clone();

        for j in 0..input.shape[1] { // loop over channels
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
                        normalized.data[idx] = (x.sub(mean).div_inf(std)).mul_inf(gamma).add(beta);
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
