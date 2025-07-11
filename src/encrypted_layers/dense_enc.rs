use crate::encrypted_utils::tensor::EncryptedTensor;
use crate::encrypted_utils::encrypted_context::EncryptedContext;
use crate::encrypted_utils::server_key_trait::ServerKeyTrait;
use crate::encrypted_utils::encrypted_types::EncryptedElement;
use crate::encrypted_ops::{EncryptedAdd, EncryptedMul, EncryptedNegate};
use crate::encrypted_layers::EncryptedLayer;

use rayon::prelude::*;

pub struct EncryptedDenseLayer<T: EncryptedElement> {
    pub id: String,
    pub weights: EncryptedTensor<T>,
    pub biases: EncryptedTensor<T>,
    pub grad_weights: Option<EncryptedTensor<T>>,
    pub grad_biases: Option<EncryptedTensor<T>>,
}

impl<T: EncryptedElement> EncryptedDenseLayer<T> {
    pub fn new(id:String, weights: EncryptedTensor<T>, biases: EncryptedTensor<T>) -> Self {
        Self {
            id,
            weights,
            biases,
            grad_weights: None,
            grad_biases: None,
        }
    }
}

impl<K, T> EncryptedLayer<K, T> for EncryptedDenseLayer<T>
where
    K: ServerKeyTrait + EncryptedAdd<K, T> + EncryptedMul<K, T> + EncryptedNegate<K, T>,
    T: Clone + EncryptedElement,
{
    fn forward(&self, input: &EncryptedTensor<T>, ctx: &EncryptedContext<K, T>) -> EncryptedTensor<T> {
        let weighted_sum = input.matmul(&self.weights, ctx);
        let row = &self.biases.data;
        
        let batch_size = input.shape[0];
        let feature_len = row.len();

        let repeated: Vec<T> = (0..batch_size)
            .into_par_iter()
            .flat_map(|_| row.clone().into_par_iter())
            .collect();
        let expanded_biases = EncryptedTensor {
            data: repeated,
            shape: vec![input.shape[0], self.biases.shape[1]],
        };
        weighted_sum.add(&expanded_biases, ctx)
    }

    fn backward(
        &mut self,
        input: &EncryptedTensor<T>,
        grad_output: &EncryptedTensor<T>,
        ctx: &EncryptedContext<K, T>,
    ) -> EncryptedTensor<T> {
        let input_t = input.transpose();
        let grad_weights = input_t.matmul(&grad_output, ctx);
        let grad_biases = grad_output;
        let weights_t = self.weights.transpose();
        let grad_input = grad_output.matmul(&weights_t, ctx);
        self.grad_weights = Some(grad_weights.clone());
        self.grad_biases = Some(grad_biases.clone());

        grad_input
    }

    fn update_parameters(
            &mut self,
            learning_rate: T,
            ctx: &EncryptedContext<K, T>,
    ) {
        if let (Some(grad_w), Some(grad_b)) = (&self.grad_weights, &self.grad_biases) {
            // Element-wise multiplication: grad_weights * learning_rate
            let mut lr_grad_w = grad_w.mul_scalar(&learning_rate, ctx);
            let mut lr_grad_b = grad_b.mul_scalar(&learning_rate, ctx);
            // Subtract from current weights and biases
            self.weights = self.weights.sub(&lr_grad_w, ctx);
            let reduced_grad_b = lr_grad_b.sum_axis(0, ctx);
            self.biases = self.biases.sub(&reduced_grad_b, ctx);
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
