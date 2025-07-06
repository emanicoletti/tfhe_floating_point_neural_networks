use crate::encrypted_utils::tensor::EncryptedTensor;
use crate::encrypted_utils::encrypted_context::EncryptedContext;
use crate::encrypted_utils::server_key_trait::ServerKeyTrait;
use crate::encrypted_utils::encrypted_types::EncryptedElement;


pub trait EncryptedLayer<K, T>: Send + Sync
where
    K: ServerKeyTrait,
    T: EncryptedElement,
{
    fn forward(&self, input: &EncryptedTensor<T>, ctx: &EncryptedContext<K, T>) -> EncryptedTensor<T>;

    fn backward(
        &mut self,
        input: &EncryptedTensor<T>,
        grad_output: &EncryptedTensor<T>,
        ctx: &EncryptedContext<K, T>,
    ) -> EncryptedTensor<T>;
}