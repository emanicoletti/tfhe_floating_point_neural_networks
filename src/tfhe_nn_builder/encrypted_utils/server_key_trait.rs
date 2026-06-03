use tfhe::{CudaServerKey, ServerKey};

pub trait ServerKeyTrait: Clone + Send + Sync {}

impl ServerKeyTrait for ServerKey {}

impl ServerKeyTrait for CudaServerKey {}
