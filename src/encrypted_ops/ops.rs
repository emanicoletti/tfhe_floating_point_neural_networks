use super::add::*;
use super::mul::*;
use super::negate::*;
use crate::encrypted_utils::encrypted_context::EncryptedContext;
use crate::encrypted_utils::server_key_trait::ServerKeyTrait;
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

pub trait EncryptedNegate<K, T>
where
    K: ServerKeyTrait + Clone + Send + Sync,
{
    fn negate(&self, a: T) -> T;
}

impl EncryptedAdd<CudaServerKey, FheUint8> for CudaServerKey {
    fn add(&self, a: FheUint8, b: FheUint8, ctx: &EncryptedContext<Self, FheUint8>) -> FheUint8 {
        fhe_add8_gpu(a, b, ctx.encrypted_mask.clone(), self.clone())
    }
}

impl EncryptedAdd<ServerKey, FheUint8> for ServerKey {
    fn add(&self, a: FheUint8, b: FheUint8, ctx: &EncryptedContext<Self, FheUint8>) -> FheUint8 {
        fhe_add8_cpu(a, b, ctx.encrypted_mask.clone(), self.clone())
    }
}

impl EncryptedAdd<CudaServerKey, FheUint16> for CudaServerKey {
    fn add(&self, a: FheUint16, b: FheUint16, ctx: &EncryptedContext<Self, FheUint16>) -> FheUint16 {
        fhe_add16_gpu(a, b, ctx.encrypted_mask.clone(), self.clone())
    }
}

impl EncryptedAdd<ServerKey, FheUint16> for ServerKey {
    fn add(&self, a: FheUint16, b: FheUint16, ctx: &EncryptedContext<Self, FheUint16>) -> FheUint16 {
        fhe_add16_cpu(a, b, ctx.encrypted_mask.clone(), self.clone())
    }
}

impl EncryptedAdd<CudaServerKey, FheUint32> for CudaServerKey {
    fn add(&self, a: FheUint32, b: FheUint32, ctx: &EncryptedContext<Self, FheUint32>) -> FheUint32 {
        fhe_add32_gpu(a, b, ctx.encrypted_mask.clone(), self.clone())
    }
}

impl EncryptedAdd<ServerKey, FheUint32> for ServerKey {
    fn add(&self, a: FheUint32, b: FheUint32, ctx: &EncryptedContext<Self, FheUint32>) -> FheUint32 {
        fhe_add32_cpu(a, b, ctx.encrypted_mask.clone(), self.clone())
    }
}

impl EncryptedAdd<CudaServerKey, FheUint64> for CudaServerKey {
    fn add(&self, a: FheUint64, b: FheUint64, ctx: &EncryptedContext<Self, FheUint64>) -> FheUint64 {
        fhe_add64_gpu(a, b,  ctx.encrypted_mask.clone(), self.clone())
    }
}

impl EncryptedAdd<ServerKey, FheUint64> for ServerKey {
    fn add(&self, a: FheUint64, b: FheUint64, ctx: &EncryptedContext<Self, FheUint64>) -> FheUint64 {
        fhe_add64_cpu(a, b, ctx.encrypted_mask.clone(), self.clone())
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

