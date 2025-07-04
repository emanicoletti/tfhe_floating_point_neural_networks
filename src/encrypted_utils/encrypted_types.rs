use tfhe::{FheUint8, FheUint16, FheUint32, FheUint64};

pub trait EncryptedElement: Clone + Send + Sync {}

impl EncryptedElement for FheUint8 {}
impl EncryptedElement for FheUint16 {}
impl EncryptedElement for FheUint32 {}
impl EncryptedElement for FheUint64 {}