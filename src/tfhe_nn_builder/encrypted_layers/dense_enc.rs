use crate::tfhe_nn_builder::encrypted_utils::tensor::EncryptedTensor;
use crate::tfhe_nn_builder::encrypted_utils::encrypted_context::EncryptedContext;
use crate::tfhe_nn_builder::encrypted_utils::server_key_trait::ServerKeyTrait;
use crate::tfhe_nn_builder::encrypted_utils::encrypted_types::{EncryptableValueType, EncryptedElement};
use crate::tfhe_nn_builder::encrypted_ops::{EncryptedAdd, EncryptedMul, EncryptedNegate};
use crate::tfhe_nn_builder::encrypted_layers::EncryptedLayer;

use rayon::prelude::*;
use rayon::scope;

use std::time::Instant;

pub struct EncryptedDenseLayer<T: EncryptedElement> {
    pub id: String,
    pub weights: EncryptedTensor<T>, // shape: [1, 1, input_dim, output_dim]
    pub biases: EncryptedTensor<T>,  // shape: [1, 1, 1, output_dim]
    pub grad_weights: Option<EncryptedTensor<T>>,
    pub grad_biases: Option<EncryptedTensor<T>>,
}

impl<T: EncryptedElement> EncryptedDenseLayer<T> {
    pub fn _new(id: String, weights: EncryptedTensor<T>, biases: EncryptedTensor<T>) -> Self {
        Self {
            id,
            weights, // expect shape [1, 1, input_dim, output_dim]
            biases,  // expect shape [1, 1, 1, output_dim]
            grad_weights: None,
            grad_biases: None,
        }
    }
}

impl<K, T> EncryptedLayer<K, T> for EncryptedDenseLayer<T>
where
    K: ServerKeyTrait + EncryptedAdd<K, T> + EncryptedMul<K, T> + EncryptedNegate<K, T>,
    T: Clone + EncryptedElement + EncryptableValueType,
{
    fn forward(&mut self, input: &EncryptedTensor<T>, ctx: &EncryptedContext<K, T>) -> EncryptedTensor<T> {
        let start = Instant::now();
        let flatten_input = input.flatten_hw_to_1d();
        let mut weighted_sum = flatten_input.matmul(&self.weights.transpose(), ctx); 

        let batch_size = input.shape[0];
        let output_dim = self.biases.shape[3];
        let bias_data = &self.biases.data;

        let repeated: Vec<T> = (0..batch_size)
            .into_par_iter()
            .flat_map(|_| bias_data.par_iter().cloned())
            .collect();

        let expanded_biases = EncryptedTensor {
            data: repeated,
            shape: vec![batch_size, 1, 1, output_dim],
        };

        weighted_sum = weighted_sum.add(&expanded_biases, ctx);
        println!("Time: {:?}", start.elapsed());
        weighted_sum
    }

    fn backward(
        &mut self,
        input: &EncryptedTensor<T>,          // [batch_size, input_dim]
        grad_output: &EncryptedTensor<T>,    // [batch_size, output_dim]
        ctx: &EncryptedContext<K, T>,
    ) -> EncryptedTensor<T>
    where
        K: ServerKeyTrait + EncryptedMul<K, T> + Send + Sync,
        T: Clone + Send + Sync,
    {
        let mut grad_weights_opt = None;
        let mut grad_biases_opt = None;
        let mut grad_input_opt = None;

        scope(|s| {
            s.spawn(|_| {
                let flatten_input = input.flatten_hw_to_1d();
                let grad_weights = grad_output.transpose().matmul(&flatten_input, ctx).sum_on_first_axis(ctx);
                grad_weights_opt = Some(grad_weights);
            });
    
            s.spawn(|_| {
                let grad_biases = grad_output.sum_on_first_axis(ctx);
                grad_biases_opt = Some(grad_biases);
            });
    
            s.spawn(|_| {
                let grad_input = grad_output.matmul(&self.weights, ctx);
                grad_input_opt = Some(grad_input);
            });
        });
    
        // Unwrap results (these will always be Some because the spawns run synchronously)
        let grad_weights = grad_weights_opt.expect("grad_weights not computed");
        let grad_biases = grad_biases_opt.expect("grad_biases not computed");
        let grad_input = grad_input_opt.expect("grad_input not computed");
    
        self.grad_weights = Some(grad_weights.clone());
        self.grad_biases = Some(grad_biases.clone());

        grad_input
    }

    fn update_parameters(&mut self, learning_rate: T, ctx: &EncryptedContext<K, T>)
    where
        K: ServerKeyTrait + EncryptedMul<K, T> + Send + Sync,
        T: Clone + Send + Sync,
    {
        if let (Some(grad_w), Some(grad_b)) = (&self.grad_weights, &self.grad_biases) {
            let mut lr_grad_w_opt = None;
            let mut lr_grad_b_opt = None;
    
            // Compute scalar multiplications in parallel
            scope(|s| {
                s.spawn(|_| {
                    lr_grad_w_opt = Some(grad_w.mul_scalar(&learning_rate, ctx));
                });
                s.spawn(|_| {
                    lr_grad_b_opt = Some(grad_b.mul_scalar(&learning_rate, ctx));
                });
            });
    
            let lr_grad_w = lr_grad_w_opt.expect("lr_grad_w not computed");
            let lr_grad_b = lr_grad_b_opt.expect("lr_grad_b not computed");
    
            let mut new_weights = None;
            let mut new_biases = None;
    
            // Subtractions can also be parallelized
            scope(|s| {
                s.spawn(|_| {
                    new_weights = Some(self.weights.sub(&lr_grad_w, ctx));
                });
                s.spawn(|_| {
                    new_biases = Some(self.biases.sub(&lr_grad_b, ctx));
                });
            });
    
            self.weights = new_weights.expect("weights update failed");
            self.biases = new_biases.expect("biases update failed");
        }
    }

    fn get_weights(&self) -> EncryptedTensor<T> {
        self.weights.clone()
    }

    fn get_biases(&self) -> EncryptedTensor<T> {
        self.biases.clone()
    }

    fn get_grad_weights(&self) -> EncryptedTensor<T> {
        self.grad_weights.clone().unwrap()
    }

    fn get_grad_biases(&self) -> EncryptedTensor<T> {
        self.grad_biases.clone().unwrap()
    }

    fn get_id(&self) -> String {
        self.id.clone()
    }
}

