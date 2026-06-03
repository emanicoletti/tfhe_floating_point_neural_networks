use crate::tfhe_nn_builder::encrypted_utils::encrypted_context::EncryptedContext;
use crate::tfhe_nn_builder::encrypted_utils::encrypted_types::EncryptedElement;
use crate::tfhe_nn_builder::encrypted_utils::server_key_trait::ServerKeyTrait;
use crate::tfhe_nn_builder::encrypted_utils::tensor::EncryptedTensor;

pub trait EncryptedLayer<K, T>: Send + Sync
where
    K: ServerKeyTrait,
    T: EncryptedElement,
{
    fn forward(
        &mut self,
        input: &EncryptedTensor<T>,
        ctx: &EncryptedContext<K, T>,
    ) -> EncryptedTensor<T>;

    fn backward(
        &mut self,
        input: &EncryptedTensor<T>,
        grad_output: &EncryptedTensor<T>,
        ctx: &EncryptedContext<K, T>,
    ) -> EncryptedTensor<T>;

    fn update_parameters(&mut self, learning_rate: T, ctx: &EncryptedContext<K, T>);

    fn get_weights(&self) -> EncryptedTensor<T>;

    fn get_biases(&self) -> EncryptedTensor<T>;

    #[allow(dead_code)]
    // Debugging method
    fn get_grad_weights(&self) -> EncryptedTensor<T>;

    #[allow(dead_code)]
    // Debugging method
    fn get_grad_biases(&self) -> EncryptedTensor<T>;

    fn get_id(&self) -> String;
}
