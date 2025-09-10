use super::add::*;
use super::mul::*;
use super::negate::*;
use super::div::*;
use super::tanh::*;
use super::max::*;
use super::grad_if_equal::*;
use super::same_sign_add::*;
use super::relu::*;
use crate::tfhe_nn_builder::encrypted_utils::encrypted_context::EncryptedContext;
use crate::tfhe_nn_builder::encrypted_utils::server_key_trait::ServerKeyTrait;
use tfhe::FheBool;
use tfhe::{FheUint8, FheUint16, FheUint32, FheUint64, ServerKey, CudaServerKey};


pub trait EncryptedAdd<K, T>
where
    K: ServerKeyTrait + Clone + Send + Sync, // or your ServerKeyTrait if already defined
{
    fn add(&self, a: T, b: T, ctx: &EncryptedContext<K, T>) -> T;
}

pub trait EncryptedMul<K, T>
where
    K: ServerKeyTrait + Clone + Send + Sync,
{
    fn mul(&self, a: T, b: T, ctx: &EncryptedContext<K, T>) -> T;
}

pub trait EncryptedDiv<K,T>
where 
    K: ServerKeyTrait + Clone + Send + Sync,
{
    fn div(&self, a: T, b: T, ctx: &EncryptedContext<K, T>) -> T;
}

pub trait EncryptedNegate<K, T>
where
    K: ServerKeyTrait + Clone + Send + Sync,
{
    fn negate(&self, a: T) -> T;
}

pub trait EncryptedTanh<K, T>
where
    K: ServerKeyTrait + Clone + Send + Sync,
{
    fn tanh(&self, a: T, ctx: &EncryptedContext<K, T>) -> (T, T);
}

pub trait EncryptedMax<K, T>
where
    K: ServerKeyTrait + Clone + Send + Sync,
{
    fn max(&self, a: T, b: T, ctx: &EncryptedContext<K, T>) -> T;
}

pub trait EncryptedGradIfEqual<K, T>
where
    K: ServerKeyTrait + Clone + Send + Sync,
{
    fn grad_if_equal(&self, a: T, b: T, zero: T, grad: T, ctx: &EncryptedContext<K, T>) -> T;
}

pub trait EncryptedSameSignAdd<K, T>
where
    K: ServerKeyTrait + Clone + Send + Sync,
{
    fn same_sign_add(&self, a: T, b: T, ctx: &EncryptedContext<K, T>) -> T;
}

pub trait EncryptedReLU<K, T>
where
    K: ServerKeyTrait + Clone + Send + Sync,
{
    fn relu(&self, a: T, ctx: &EncryptedContext<K, T>) -> T;
}

pub trait EncryptedBackwardRelu<K, T>
where
    K: ServerKeyTrait + Clone + Send + Sync,
{
    fn backward_relu(
        &self,
        grad_output: T,
        derivative: T,
        ctx: &EncryptedContext<K, T>,
    ) -> T;
}

impl EncryptedAdd<CudaServerKey, FheUint8> for CudaServerKey {
    fn add(&self, a: FheUint8, b: FheUint8, ctx: &EncryptedContext<Self, FheUint8>) -> FheUint8 {
        fhe_add8_gpu(a, b, ctx.encrypted_mask.clone(), ctx.encrypted_zero.clone(), self.clone())
    }
}

impl EncryptedAdd<ServerKey, FheUint8> for ServerKey {
    fn add(&self, a: FheUint8, b: FheUint8, ctx: &EncryptedContext<Self, FheUint8>) -> FheUint8 {
        fhe_add8_cpu(a, b, ctx.encrypted_mask.clone(), ctx.encrypted_zero.clone(), self.clone())
    }
}

impl EncryptedAdd<CudaServerKey, FheUint16> for CudaServerKey {
    fn add(&self, a: FheUint16, b: FheUint16, ctx: &EncryptedContext<Self, FheUint16>) -> FheUint16 {
        fhe_add16_gpu(a, b, ctx.encrypted_mask.clone(), ctx.encrypted_zero.clone(), self.clone())
    }
}

impl EncryptedAdd<ServerKey, FheUint16> for ServerKey {
    fn add(&self, a: FheUint16, b: FheUint16, ctx: &EncryptedContext<Self, FheUint16>) -> FheUint16 {
        fhe_add16_cpu(a, b, ctx.encrypted_mask.clone(), ctx.encrypted_zero.clone(), self.clone())
    }
}

impl EncryptedAdd<CudaServerKey, FheUint32> for CudaServerKey {
    fn add(&self, a: FheUint32, b: FheUint32, ctx: &EncryptedContext<Self, FheUint32>) -> FheUint32 {
        fhe_add32_gpu(a, b, ctx.encrypted_mask.clone(), ctx.encrypted_zero.clone(), self.clone())
    }
}

impl EncryptedAdd<ServerKey, FheUint32> for ServerKey {
    fn add(&self, a: FheUint32, b: FheUint32, ctx: &EncryptedContext<Self, FheUint32>) -> FheUint32 {
        fhe_add32_cpu(a, b, ctx.encrypted_mask.clone(), ctx.encrypted_zero.clone(), self.clone())
    }
}

impl EncryptedAdd<CudaServerKey, FheUint64> for CudaServerKey {
    fn add(&self, a: FheUint64, b: FheUint64, ctx: &EncryptedContext<Self, FheUint64>) -> FheUint64 {
        fhe_add64_gpu(a, b,  ctx.encrypted_mask.clone(), ctx.encrypted_zero.clone(), self.clone())
    }
}

impl EncryptedAdd<ServerKey, FheUint64> for ServerKey {
    fn add(&self, a: FheUint64, b: FheUint64, ctx: &EncryptedContext<Self, FheUint64>) -> FheUint64 {
        fhe_add64_cpu(a, b, ctx.encrypted_mask.clone(), ctx.encrypted_zero.clone(), self.clone())
    }
}

impl EncryptedMul<CudaServerKey, FheUint8> for CudaServerKey {
    fn mul(&self, a: FheUint8, b: FheUint8, ctx: &EncryptedContext<Self, FheUint8>) -> FheUint8 {
        fhe_lmul8_gpu(a, b, ctx.encrypted_zero.clone(), self.clone())
    }
}

impl EncryptedMul<ServerKey, FheUint8> for ServerKey {
    fn mul(&self, a: FheUint8, b: FheUint8, ctx: &EncryptedContext<Self, FheUint8>) -> FheUint8 {
        fhe_lmul8_cpu(a, b, ctx.encrypted_zero.clone(), self.clone())
    }
}

impl EncryptedMul<CudaServerKey, FheUint16> for CudaServerKey {
    fn mul(&self, a: FheUint16, b: FheUint16, ctx: &EncryptedContext<Self, FheUint16>) -> FheUint16 {
        fhe_lmul16_gpu(a, b, ctx.encrypted_zero.clone(), self.clone())
    }
}

impl EncryptedMul<ServerKey, FheUint16> for ServerKey {
    fn mul(&self, a: FheUint16, b: FheUint16, ctx: &EncryptedContext<Self, FheUint16>) -> FheUint16 {
        fhe_lmul16_cpu(a, b, ctx.encrypted_zero.clone(), self.clone())
    }
}

impl EncryptedMul<CudaServerKey, FheUint32> for CudaServerKey {
    fn mul(&self, a: FheUint32, b: FheUint32, ctx: &EncryptedContext<Self, FheUint32>) -> FheUint32 {
        fhe_lmul32_gpu(a, b, ctx.encrypted_zero.clone(), self.clone())
    }
}

impl EncryptedMul<ServerKey, FheUint32> for ServerKey {
    fn mul(&self, a: FheUint32, b: FheUint32, ctx: &EncryptedContext<Self, FheUint32>) -> FheUint32 {
        fhe_lmul32_cpu(a, b, ctx.encrypted_zero.clone(), self.clone())
    }
}

impl EncryptedMul<CudaServerKey, FheUint64> for CudaServerKey {
    fn mul(&self, a: FheUint64, b: FheUint64, ctx: &EncryptedContext<Self, FheUint64>) -> FheUint64 {
        fhe_lmul64_gpu(a, b, ctx.encrypted_zero.clone(), self.clone())
    }
}

impl EncryptedMul<ServerKey, FheUint64> for ServerKey {
    fn mul(&self, a: FheUint64, b: FheUint64, ctx: &EncryptedContext<Self, FheUint64>) -> FheUint64 {
        fhe_lmul64_cpu(a, b, ctx.encrypted_zero.clone(), self.clone())
    }
}

impl EncryptedDiv<CudaServerKey, FheUint8> for CudaServerKey {
    fn div(&self, a: FheUint8, b: FheUint8, ctx: &EncryptedContext<Self, FheUint8>) -> FheUint8 {
        fhe_ldiv8_gpu(a, b, ctx.encrypted_zero.clone(), self.clone())
    }
}

impl EncryptedDiv<ServerKey, FheUint8> for ServerKey {
    fn div(&self, a: FheUint8, b: FheUint8, ctx: &EncryptedContext<Self, FheUint8>) -> FheUint8 {
        fhe_ldiv8_cpu(a, b, ctx.encrypted_zero.clone(), self.clone())
    }
}

impl EncryptedDiv<CudaServerKey, FheUint16> for CudaServerKey {
    fn div(&self, a: FheUint16, b: FheUint16, ctx: &EncryptedContext<Self, FheUint16>) -> FheUint16 {
        fhe_ldiv16_gpu(a, b, ctx.encrypted_zero.clone(), self.clone())
    }
}

impl EncryptedDiv<ServerKey, FheUint16> for ServerKey {
    fn div(&self, a: FheUint16, b: FheUint16, ctx: &EncryptedContext<Self, FheUint16>) -> FheUint16 {
        fhe_ldiv16_cpu(a, b, ctx.encrypted_zero.clone(), self.clone())
    }
}

impl EncryptedDiv<CudaServerKey, FheUint32> for CudaServerKey {
    fn div(&self, a: FheUint32, b: FheUint32, ctx: &EncryptedContext<Self, FheUint32>) -> FheUint32 {
        fhe_ldiv32_gpu(a, b, ctx.encrypted_zero.clone(), self.clone())
    }
}

impl EncryptedDiv<ServerKey, FheUint32> for ServerKey {
    fn div(&self, a: FheUint32, b: FheUint32, ctx: &EncryptedContext<Self, FheUint32>) -> FheUint32 {
        fhe_ldiv32_cpu(a, b, ctx.encrypted_zero.clone(), self.clone())
    }
}

impl EncryptedDiv<CudaServerKey, FheUint64> for CudaServerKey {
    fn div(&self, a: FheUint64, b: FheUint64, ctx: &EncryptedContext<Self, FheUint64>) -> FheUint64 {
        fhe_ldiv64_gpu(a, b, ctx.encrypted_zero.clone(), self.clone())
    }
}

impl EncryptedDiv<ServerKey, FheUint64> for ServerKey {
    fn div(&self, a: FheUint64, b: FheUint64, ctx: &EncryptedContext<Self, FheUint64>) -> FheUint64 {
        fhe_ldiv64_cpu(a, b, ctx.encrypted_zero.clone(), self.clone())
    }
}

impl EncryptedNegate<CudaServerKey, FheUint8> for CudaServerKey {
    fn negate(&self, a: FheUint8) -> FheUint8 {
        fhe_negate8_gpu(a, self.clone())
    }
}

impl EncryptedNegate<ServerKey, FheUint8> for ServerKey {
    fn negate(&self, a: FheUint8) -> FheUint8 {
        fhe_negate8_cpu(a, self.clone())
    }
}

impl EncryptedNegate<CudaServerKey, FheUint16> for CudaServerKey {
    fn negate(&self, a: FheUint16) -> FheUint16 {
        fhe_negate16_gpu(a, self.clone())
    }
}

impl EncryptedNegate<ServerKey, FheUint16> for ServerKey {
    fn negate(&self, a: FheUint16,) -> FheUint16 {
        fhe_negate16_cpu(a, self.clone())
    }
}

impl EncryptedNegate<CudaServerKey, FheUint32> for CudaServerKey {
    fn negate(&self, a: FheUint32) -> FheUint32 {
        fhe_negate32_gpu(a, self.clone())
    }
}

impl EncryptedNegate<ServerKey, FheUint32> for ServerKey {
    fn negate(&self, a: FheUint32) -> FheUint32 {
        fhe_negate32_cpu(a,self.clone())
    }
}

impl EncryptedNegate<CudaServerKey, FheUint64> for CudaServerKey {
    fn negate(&self, a: FheUint64) -> FheUint64 {
        fhe_negate64_gpu(a, self.clone())
    }
}

impl EncryptedNegate<ServerKey, FheUint64> for ServerKey {
    fn negate(&self, a: FheUint64) -> FheUint64 {
        fhe_negate64_cpu(a, self.clone())
    }
}

impl EncryptedTanh<CudaServerKey, FheUint16> for CudaServerKey {
    fn tanh(&self, a: FheUint16, ctx: &EncryptedContext<Self, FheUint16>) -> (FheUint16, FheUint16) {
        fhe_lmul_tanh16_gpu(a, self.clone(), ctx.encrypted_zero.clone(), ctx.encrypted_mask.clone(), &ctx.ranges)
    }
}

impl EncryptedTanh<ServerKey, FheUint16> for ServerKey {
    fn tanh(&self, a: FheUint16, ctx: &EncryptedContext<Self, FheUint16>) -> (FheUint16, FheUint16) {
        fhe_lmul_tanh16_cpu(a, self.clone(), ctx.encrypted_zero.clone(), ctx.encrypted_mask.clone(), &ctx.ranges)
    }
}

impl EncryptedTanh<CudaServerKey, FheUint32> for CudaServerKey {
    fn tanh(&self, a: FheUint32, ctx: &EncryptedContext<Self, FheUint32>) -> (FheUint32, FheUint32) {
        fhe_lmul_tanh32_gpu(a, self.clone(), ctx.encrypted_zero.clone(), ctx.encrypted_mask.clone(), &ctx.ranges)
    }
}

impl EncryptedTanh<ServerKey, FheUint32> for ServerKey {
    fn tanh(&self, a: FheUint32, ctx: &EncryptedContext<Self, FheUint32>) -> (FheUint32, FheUint32) {
        fhe_lmul_tanh32_cpu(a, self.clone(), ctx.encrypted_zero.clone(), ctx.encrypted_mask.clone(), &ctx.ranges)
    }
}

impl EncryptedMax<CudaServerKey, FheUint16> for CudaServerKey {
    fn max(&self, a: FheUint16, b: FheUint16, ctx: &EncryptedContext<Self, FheUint16>) -> FheUint16 {
        fhe_max16_gpu(a, b, self.clone())
    }
}

impl EncryptedMax<CudaServerKey, FheUint32> for CudaServerKey {
    fn max(&self, a: FheUint32, b: FheUint32, ctx: &EncryptedContext<Self, FheUint32>) -> FheUint32 {
        fhe_max32_gpu(a, b, self.clone())
    }
}

impl EncryptedMax<ServerKey, FheUint32> for ServerKey {
    fn max(&self, a: FheUint32, b: FheUint32, ctx: &EncryptedContext<Self, FheUint32>) -> FheUint32 {
        fhe_max32_cpu(a, b, self.clone())
    }
}

impl EncryptedGradIfEqual<CudaServerKey, FheUint16> for CudaServerKey {
    fn grad_if_equal(&self, a: FheUint16, b: FheUint16, zero: FheUint16, grad: FheUint16, ctx: &EncryptedContext<Self, FheUint16>) -> FheUint16 {
        fhe_grad_if_equal16_gpu(a, b, zero, grad, self.clone())
    }
}

impl EncryptedGradIfEqual<CudaServerKey, FheUint32> for CudaServerKey {
    fn grad_if_equal(&self, a: FheUint32, b: FheUint32, zero: FheUint32, grad: FheUint32, ctx: &EncryptedContext<Self, FheUint32>) -> FheUint32 {
        fhe_grad_if_equal32_gpu(a, b, zero, grad, self.clone())
    }
}

impl EncryptedGradIfEqual<ServerKey, FheUint32> for ServerKey {
    fn grad_if_equal(&self, a: FheUint32, b: FheUint32, zero: FheUint32, grad: FheUint32, ctx: &EncryptedContext<Self, FheUint32>) -> FheUint32 {
        fhe_grad_if_equal32_cpu(a, b, zero, grad, self.clone())
    }
}

impl EncryptedSameSignAdd<CudaServerKey, FheUint16> for CudaServerKey {
    fn same_sign_add(&self, a: FheUint16, b: FheUint16, ctx: &EncryptedContext<Self, FheUint16>) -> FheUint16 {
        fhe_ss_add16_gpu(a, b, ctx.encrypted_mask.clone(), ctx.encrypted_zero.clone(), self.clone())
    }
}

impl EncryptedReLU<CudaServerKey, FheUint16> for CudaServerKey {
    fn relu(&self, a: FheUint16, ctx: &EncryptedContext<Self, FheUint16>) -> FheUint16 {
        fhe_relu16_gpu(a, ctx.encrypted_zero.clone(), self.clone())
    }
}

impl EncryptedReLU<CudaServerKey, FheUint32> for CudaServerKey {
    fn relu(&self, a: FheUint32, ctx: &EncryptedContext<Self, FheUint32>) -> FheUint32 {
        fhe_relu32_gpu(a, ctx.encrypted_zero.clone(), self.clone())
    }
}

impl EncryptedBackwardRelu<CudaServerKey, FheUint16> for CudaServerKey {
    fn backward_relu(
        &self,
        grad_output: FheUint16,
        derivative: FheUint16,
        ctx: &EncryptedContext<Self, FheUint16>,
    ) -> FheUint16 {
        backward_relu16_gpu(grad_output, derivative, ctx.encrypted_zero.clone(), self.clone())
    }
}

impl EncryptedBackwardRelu<CudaServerKey, FheUint32> for CudaServerKey {
    fn backward_relu(
        &self,
        grad_output: FheUint32,
        derivative: FheUint32,
        ctx: &EncryptedContext<Self, FheUint32>,
    ) -> FheUint32 {
        backward_relu32_gpu(grad_output, derivative, ctx.encrypted_zero.clone(), self.clone())
    }
}
