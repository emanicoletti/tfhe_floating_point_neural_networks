use tfhe::{ServerKey, CudaServerKey};

pub trait ServerKeyTrait: Clone + Send + Sync {}

impl ServerKeyTrait for ServerKey {}

impl ServerKeyTrait for CudaServerKey {}