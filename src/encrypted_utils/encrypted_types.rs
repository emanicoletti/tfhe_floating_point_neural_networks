use tfhe::{FheUint8, FheUint16, FheUint32, FheUint64};
use tfhe::prelude::*;
use tfhe::ClientKey;

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
}

impl EncryptableValueType for FheUint8 {
    type Plain = u8;

    fn try_encrypt_plain(value: usize, key: &ClientKey) -> Result<Self, ()> {
        FheUint8::try_encrypt(value as u8, key).map_err(|_| ())
    }
}

impl EncryptableValueType for FheUint16 {
    type Plain = u16;

    fn try_encrypt_plain(value: usize, key: &ClientKey) -> Result<Self, ()> {
        FheUint16::try_encrypt(value as u16, key).map_err(|_| ())
    }
}

impl EncryptableValueType for FheUint32 {
    type Plain = u32;

    fn try_encrypt_plain(value: usize, key: &ClientKey) -> Result<Self, ()> {
        FheUint32::try_encrypt(value as u32, key).map_err(|_| ())
    }
}

impl EncryptableValueType for FheUint64 {
    type Plain = u64;

    fn try_encrypt_plain(value: usize, key: &ClientKey) -> Result<Self, ()> {
        FheUint64::try_encrypt(value as u64, key).map_err(|_| ())
    }
}