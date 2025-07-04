use tfhe::{ServerKey, CudaServerKey};

pub trait ServerKeyTrait: Clone + Send + Sync {
    // Put common methods or markers here if needed
}

impl ServerKeyTrait for ServerKey {}

impl ServerKeyTrait for CudaServerKey {}