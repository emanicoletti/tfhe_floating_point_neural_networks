use std::time::Instant;
use rayon::prelude::*;

use crate::tfhe_nn_builder::encrypted_utils::tensor::*;
use crate::tfhe_nn_builder::encrypted_context::*;
use crate::tfhe_nn_builder::encrypted_ops::*;
use crate::tfhe_nn_builder::encrypted_types::*;
use crate::tfhe_nn_builder::encrypted_layers::EncryptedLayer;
use crate::tfhe_nn_builder::server_key_trait::ServerKeyTrait;

pub struct EncryptedReLUActivation<T: EncryptedElement> {
    pub id: String,
    pub derivatives: EncryptedTensor<T>,
}

impl<T: EncryptedElement> EncryptedReLUActivation<T> {
    pub fn _new(id: String, derivatives: EncryptedTensor<T>) -> Self {
        Self {
            id,
            derivatives,
        }
    }
}

impl<K, T> EncryptedLayer<K, T> for EncryptedReLUActivation<T>
where
    K: ServerKeyTrait + EncryptedAdd<K, T> + EncryptedMul<K, T> + EncryptedNegate<K, T> + EncryptedReLU<K, T>+ EncryptedBackwardRelu<K, T>,
    T: Clone + EncryptedElement + EncryptableValueType<>,
{
    fn forward(&mut self, input: &EncryptedTensor<T>, ctx: &EncryptedContext<K, T>) -> EncryptedTensor<T> {
        let start = Instant::now();
        let (activations, derivatives) = input.relu(&ctx);
        println!("ReLU activation forward time: {:?}", start.elapsed());
        self.derivatives = derivatives;
        activations
    }

    fn backward(
        &mut self,
        _input: &EncryptedTensor<T>,
        grad_output: &EncryptedTensor<T>,
        ctx: &EncryptedContext<K, T>,
    ) -> EncryptedTensor<T>
    where
        K: ServerKeyTrait + EncryptedMul<K, T> + EncryptedBackwardRelu<K, T>,
        T: Clone + EncryptableValueType<>,
    {
        // grad_input = grad_output * derivative
        let grad_input_data: Vec<T> = grad_output
            .data
            .par_iter()
            .zip(self.derivatives.data.par_iter())
            .map(|(g, d)| ctx.server_key.backward_relu(g.clone(), d.clone(), ctx))
            .collect();

        EncryptedTensor::new(grad_input_data, grad_output.shape.clone())
    }

    fn update_parameters(
        &mut self,
        _learning_rate: T,
        _ctx: &EncryptedContext<K, T>,
    ) {
        // No parameters to update in ReLU activation
    }

    fn get_biases(&self) -> EncryptedTensor<T> {
        // No biases in ReLU activation
        self.derivatives.clone()
    }

    fn get_weights(&self) -> EncryptedTensor<T> {
        // No weights in ReLU activation
        self.derivatives.clone()
    }

    fn get_grad_weights(&self) -> EncryptedTensor<T> {
        // No gradients for weights in ReLU activation
        self.derivatives.clone()
    }

    fn get_grad_biases(&self) -> EncryptedTensor<T> {
        // No gradients for biases in ReLU activation
        self.derivatives.clone()
    }

    fn get_id(&self) -> String {
        self.id.clone()
    }
}

