use crate::encrypted_utils::tensor::EncryptedTensor;
use crate::encrypted_utils::encrypted_context::EncryptedContext;
use crate::encrypted_utils::server_key_trait::ServerKeyTrait;
use crate::encrypted_utils::encrypted_types::{EncryptedElement, EncryptableValueType};
use crate::encrypted_ops::*;

use tfhe::prelude::FheTryEncrypt;
use tfhe::ClientKey;

pub trait LossFunction<K, T> 
where
    K: ServerKeyTrait,
    T: EncryptedElement,
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
    K: ServerKeyTrait + EncryptedAdd<K, T> + EncryptedMul<K, T> + EncryptedDiv<K, T> + EncryptedNegate<K, T>, 
    T: Clone + FheTryEncrypt<<T as EncryptableValueType>::Plain, ClientKey> + EncryptedElement + EncryptableValueType,
{
    fn compute_loss(
        &self,
        predicted: &EncryptedTensor<T>,
        target: &EncryptedTensor<T>,
        ctx: &EncryptedContext<K, T>,
    ) -> T {
        let mut sum = ctx.encrypted_zero.clone();

        for (p, t) in predicted.data.iter().zip(&target.data) {
            let t_negate = ctx.server_key.negate(t.clone());
            let diff = ctx.server_key.add(p.clone(), t_negate.clone(), ctx);
            let squared = ctx.server_key.mul(diff.clone(), diff, ctx);
            sum = ctx.server_key.add(sum, squared, ctx);
        }

        let n = predicted.data.len();
        let n_enc = T::try_encrypt_plain(n, &ctx.client_key).expect("Failed to encrypt n");

        ctx.server_key.div(sum, n_enc, ctx)
    }

    fn gradient(
        &self,
        predicted: &EncryptedTensor<T>,
        target: &EncryptedTensor<T>,
        ctx: &EncryptedContext<K, T>,
    ) -> EncryptedTensor<T> {
        let n = predicted.data.len();
        let n_enc = T::try_encrypt_plain(n, &ctx.client_key)
            .expect("Failed to encrypt length");
    
        let two = T::try_encrypt_plain(2, &ctx.client_key)
            .expect("Failed to encrypt scalar 2");
    
        let mut grad_data = Vec::with_capacity(n);
    
        for (p, t) in predicted.data.iter().zip(&target.data) {
            let t_neg = ctx.server_key.negate(t.clone());
            let diff = ctx.server_key.add(p.clone(), t_neg, ctx);     
            let double_diff = ctx.server_key.mul(diff, two.clone(), ctx); 
            let grad = ctx.server_key.div(double_diff, n_enc.clone(), ctx); 
            grad_data.push(grad);
        }
    
        EncryptedTensor {
            data: grad_data,
            shape: predicted.shape.clone(),
        }
    }
}