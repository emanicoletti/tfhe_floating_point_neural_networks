use crate::encrypted_utils::tensor::EncryptedTensor;
use crate::encrypted_utils::encrypted_context::EncryptedContext;

pub trait LossFunction<K, T> 
where
    K: ServerKeyTrait,
{
    fn compute_loss(
        &self,
        predicted: &EncryptedTensor<T>,
        target: &EncryptedTensor<T>,
        ctx: &EncryptedContext<K, T>,
    ) -> T;

    fn gradient(
        &self,
        predicted: &EncryptedTensor<T>,
        target: &EncryptedTensor<T>,
        ctx: &EncryptedContext<K, T>,
    ) -> EncryptedTensor<T>;
}

pub struct MseLoss;

impl<K, T> LossFunction<K, T> for MseLoss
where
    K: ServerKeyTrait,
    T: Clone + EncryptedAdd<K, T> + EncryptedMul<K, T>,   // Assuming EncryptedOps trait defined with K and T
{
    fn compute_loss(
        &self,
        predicted: &EncryptedTensor<T>,
        target: &EncryptedTensor<T>,
        ctx: &EncryptedContext<K, T>,
    ) -> T {
        unimplemented!()
    }

    fn gradient(
        &self,
        predicted: &EncryptedTensor<T>,
        target: &EncryptedTensor<T>,
        ctx: &EncryptedContext<K, T>,
    ) -> EncryptedTensor<T> {
        // Gradient is 2 * (predicted - target) / N
        unimplemented!()
    }
}