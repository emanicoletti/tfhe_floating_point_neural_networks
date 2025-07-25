use crate::encrypted_utils::tensor::EncryptedTensor;
use crate::encrypted_utils::encrypted_context::EncryptedContext;
use crate::encrypted_utils::server_key_trait::ServerKeyTrait;
use crate::encrypted_utils::encrypted_types::{EncryptableValueType, EncryptedElement};
use crate::encrypted_ops::{EncryptedAdd, EncryptedMul, EncryptedNegate, EncryptedTanh};
use crate::encrypted_layers::EncryptedLayer;

pub struct EncryptedTanhActivation<T: EncryptedElement> {
    pub id: String,
    pub derivatives: EncryptedTensor<T>, // same shape as input, stores derivative per element
    pub ranges: Vec<(T, T, T, T, T)>,    // piecewise segments: (min, max, a, b, derivative)
}

impl<T: EncryptedElement> EncryptedTanhActivation<T> {
    pub fn new(id: String, derivatives: EncryptedTensor<T>, ranges: Vec<(T, T, T, T, T)>) -> Self {
        Self {
            id,
            derivatives,
            ranges
        }
    }
}

impl<K, T> EncryptedLayer<K, T> for EncryptedTanhActivation<T>
where
    K: ServerKeyTrait + EncryptedAdd<K, T> + EncryptedMul<K, T> + EncryptedNegate<K, T> + EncryptedTanh<K, T>,
    T: Clone + EncryptedElement + EncryptableValueType<>, {

    fn forward(&mut self, input: &EncryptedTensor<T>, ctx: &EncryptedContext<K, T>) -> EncryptedTensor<T> {
        let (activations, derivatives ) = input.tanh(&ctx);
        self.derivatives = derivatives;
        activations
    }

    fn backward(
        &mut self,
        input: &EncryptedTensor<T>,
        grad_output: &EncryptedTensor<T>,
        ctx: &EncryptedContext<K, T>,
    ) -> EncryptedTensor<T> {
        self.derivatives.clone()
    }

    fn update_parameters(
            &mut self,
            learning_rate: T,
            ctx: &EncryptedContext<K, T>,
        ) {
        // No parameters to update in tanh activation
    }

    fn get_biases(&self) -> EncryptedTensor<T> {
        // No biases in tanh activation
        self.derivatives.clone()
    }

    fn get_grad_biases(&self) -> EncryptedTensor<T> {
        // No gradients for biases in tanh activation
        self.derivatives.clone()
    }   

    fn get_grad_weights(&self) -> EncryptedTensor<T> {
        // No gradients for weights in tanh activation
        self.derivatives.clone()
    }

    fn get_weights(&self) -> EncryptedTensor<T> {
        // No weights in tanh activation
        self.derivatives.clone()
    }

    fn get_id(&self) -> String {
        self.id.clone()
    }
}
