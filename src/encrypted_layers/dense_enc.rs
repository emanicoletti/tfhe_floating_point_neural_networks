use crate::encrypted_utils::tensor::EncryptedTensor;
use crate::encrypted_utils::encrypted_context::EncryptedContext;
use crate::encrypted_utils::server_key_trait::ServerKeyTrait;
use crate::encrypted_utils::encrypted_types::EncryptedElement;
use crate::encrypted_ops::{EncryptedAdd, EncryptedMul};


pub struct EncryptedDenseLayer<T:EncryptedElement> {
    weights: EncryptedTensor<T>, 
    biases: EncryptedTensor<T>,       
}

impl<T: EncryptedElement> EncryptedDenseLayer<T> {

    pub fn forward<K>(
        &self,
        input: &EncryptedTensor<T>,
        ctx: &EncryptedContext<K, T>,
    ) -> EncryptedTensor<T>
    where
        K: ServerKeyTrait + EncryptedAdd<K, T> + EncryptedMul<K, T>,
    {
        let weighted_sum = self.weights.matmul(input, ctx);
        let output = weighted_sum.add(&self.biases, ctx);

        EncryptedTensor {
            data: output.data,
            shape: output.shape,
        }
    }

}
