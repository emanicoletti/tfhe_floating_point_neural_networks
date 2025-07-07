use crate::encrypted_utils::tensor::EncryptedTensor;
use crate::encrypted_utils::encrypted_context::EncryptedContext;
use crate::encrypted_utils::server_key_trait::ServerKeyTrait;
use crate::encrypted_utils::encrypted_types::EncryptedElement;
use crate::encrypted_ops::{EncryptedAdd, EncryptedMul};
use crate::encrypted_layers::EncryptedLayer;


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
    K: ServerKeyTrait + EncryptedAdd<K, T> + EncryptedMul<K, T>,
    T: Clone + EncryptedElement,
{
    fn forward(&self, input: &EncryptedTensor<T>, ctx: &EncryptedContext<K, T>) -> EncryptedTensor<T> {
        let weighted_sum = self.weights.matmul(input, ctx);
        weighted_sum.add(&self.biases, ctx)
    }

    fn backward(
        &mut self,
        input: &EncryptedTensor<T>,
        grad_output: &EncryptedTensor<T>,
        ctx: &EncryptedContext<K, T>,
    ) -> EncryptedTensor<T> {
        let input_t = input.transpose();
        let grad_weights = grad_output.transpose().matmul(&input, ctx);
        let grad_biases = grad_output.sum_axis(0, ctx);
        let weights_t = self.weights.transpose();
        let grad_input = grad_output.matmul(&weights_t, ctx);

        self.grad_weights = Some(grad_weights);
        self.grad_biases = Some(grad_biases);

        grad_input
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
