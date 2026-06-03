use tfhe::ClientKey;
use tfhe::prelude::*;
use tfhe::{FheUint8, FheUint16, FheUint32, FheUint64};

use half::f16;

pub trait EncryptedElement: Clone + Send + Sync {}

impl EncryptedElement for FheUint8 {}
impl EncryptedElement for FheUint16 {}
impl EncryptedElement for FheUint32 {}
impl EncryptedElement for FheUint64 {}

pub trait EncryptableValueType {
    type Plain;

    fn try_encrypt_plain(value: usize, key: &ClientKey) -> Result<Self, ()>
    where
        Self: Sized;

    fn decrypt(&self, key: &ClientKey) -> Self::Plain;

    fn n_bits(n: f32) -> usize;
}

impl EncryptableValueType for FheUint8 {
    type Plain = u8;

    fn try_encrypt_plain(value: usize, key: &ClientKey) -> Result<Self, ()> {
        FheUint8::try_encrypt(value as u8, key).map_err(|_| ())
    }

    fn decrypt(&self, key: &ClientKey) -> Self::Plain {
        FheDecrypt::decrypt(self, key)
    }

    fn n_bits(_n: f32) -> usize {
        unimplemented!()
    }
}

impl EncryptableValueType for FheUint16 {
    type Plain = u16;

    fn try_encrypt_plain(value: usize, key: &ClientKey) -> Result<Self, ()> {
        FheUint16::try_encrypt(value as u16, key).map_err(|_| ())
    }

    fn decrypt(&self, key: &ClientKey) -> Self::Plain {
        FheDecrypt::decrypt(self, key)
    }

    fn n_bits(n: f32) -> usize {
        let n_f16 = f16::from_f32(n);
        n_f16.to_bits() as usize
    }
}

impl EncryptableValueType for FheUint32 {
    type Plain = u32;

    fn try_encrypt_plain(value: usize, key: &ClientKey) -> Result<Self, ()> {
        FheUint32::try_encrypt(value as u32, key).map_err(|_| ())
    }

    fn decrypt(&self, key: &ClientKey) -> Self::Plain {
        FheDecrypt::decrypt(self, key)
    }

    fn n_bits(n: f32) -> usize {
        n.to_bits() as usize
    }
}

impl EncryptableValueType for FheUint64 {
    type Plain = u64;

    fn try_encrypt_plain(value: usize, key: &ClientKey) -> Result<Self, ()> {
        FheUint64::try_encrypt(value as u64, key).map_err(|_| ())
    }

    fn decrypt(&self, key: &ClientKey) -> Self::Plain {
        FheDecrypt::decrypt(self, key)
    }

    fn n_bits(_n: f32) -> usize {
        unimplemented!()
    }
}
